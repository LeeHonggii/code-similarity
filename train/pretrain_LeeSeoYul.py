import os, io, math, time, random, tokenize
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
from einops import rearrange

# =============================
# 설정
# =============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 경로
PAIR_PATH = "./data/pairs_balanced_v4.parquet"
TOKENIZER_DIR = "./data/tokenizer_unigram_LeeSeoYul"
SAVE_DIR = "./models/siamese_contrastive"

# 모델 하이퍼파라미터
MAX_LEN = 512
D_MODEL = 512
FF_DIM = 1024
N_LAYER = 6
N_HEAD = 8
DROPOUT = 0.1

# 위치 인코딩 (하나만 True)
USE_ROPE = True
USE_ALIBI = False
USE_ABS_POS = False

# 학습 하이퍼파라미터
BATCH_SIZE = 16
EPOCHS = 10
GRAD_ACCUM_STEPS = 2
STEP_LOG = 50
LR_BASE = 1e-4
LR_TAU = 1e-5
WARMUP_RATIO = 0.05
CLIP_NORM = 1.0
HARD_WEIGHT = 1.5
SEMI_WEIGHT = 1.3
TAU_INIT = 0.07
TAU_LEARNABLE = True

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================
# 전처리 함수
# =========================================
def code_to_marked_text(code: str) -> str:
    """Python 코드에 마커 삽입"""
    INDENT_TOK, DEDENT_TOK, NEWLINE_TOK = "<indent>", "<dedent>", "<newline>"
    out = []
    try:
        g = tokenize.generate_tokens(io.StringIO(code).readline)
        for tt, ts, *_ in g:
            if tt == tokenize.INDENT:
                out.append(f" {INDENT_TOK} ")
            elif tt == tokenize.DEDENT:
                out.append(f" {DEDENT_TOK} ")
            elif tt in (tokenize.NEWLINE, tokenize.NL):
                out.append(f" {NEWLINE_TOK} ")
            elif tt == tokenize.COMMENT:
                continue
            else:
                out.append(" " + ts + " ")
    except Exception:
        out = [" " + str(code).replace("\n", f" {NEWLINE_TOK} ") + " "]
    return " ".join("".join(out).split())


# =========================================
# 데이터셋
# =========================================
class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        ta = code_to_marked_text(str(r["text_a"]))
        tb = code_to_marked_text(str(r["text_b"]))

        ea = self.tok(ta, truncation=True, max_length=self.max_len,
                      padding="max_length", return_tensors="pt")
        eb = self.tok(tb, truncation=True, max_length=self.max_len,
                      padding="max_length", return_tensors="pt")
        return {
            "ids_a": ea["input_ids"].squeeze(0),
            "mask_a": ea["attention_mask"].squeeze(0),
            "ids_b": eb["input_ids"].squeeze(0),
            "mask_b": eb["attention_mask"].squeeze(0),
            "ptype": r["pair_type"],
        }


# =========================================
# RoPE & ALiBi
# =========================================
def apply_rope(q, k):
    """Rotary Position Embedding"""
    Dh = q.shape[-1]
    half = Dh // 2
    freq = torch.arange(half, device=q.device, dtype=q.dtype)
    freq = 1.0 / (10000 ** (freq / half))
    L = q.shape[2]
    pos = torch.arange(L, device=q.device, dtype=q.dtype)
    angles = torch.einsum('l,d->ld', pos, freq)
    sin = torch.sin(angles)[None, None, :, :]
    cos = torch.cos(angles)[None, None, :, :]

    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rot(q), rot(k)


_alibi_cache = {}


def build_alibi_bias(n_head, L, device, dtype):
    """ALiBi bias 생성 (캐시)"""
    key = (n_head, L, device)
    if key in _alibi_cache:
        return _alibi_cache[key]

    import math as _m

    def get_slopes(n):
        def power_of_two_slopes(n):
            start = 2 ** (-2 ** -(_m.log2(n) - 3))
            return [start * (start ** i) for i in range(n)]

        if _m.log2(n).is_integer():
            return torch.tensor(power_of_two_slopes(n))
        closest = 2 ** _m.floor(_m.log2(n))
        slopes = power_of_two_slopes(closest)
        extra = power_of_two_slopes(2 * closest)[0::2][:n - len(slopes)]
        return torch.tensor(slopes + extra)

    slopes = get_slopes(n_head).to(device=device, dtype=dtype)
    dist = torch.arange(L, device=device)
    bias = (dist[None, :] - dist[:, None]).clamp(min=0).to(dtype)
    bias = -bias[None, None, :, :] * slopes[:, None, None]
    _alibi_cache[key] = bias
    return bias


# =========================================
# 모델 컴포넌트
# =========================================
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, rope=False, alibi=False):
        super().__init__()
        assert not (rope and alibi), "RoPE와 ALiBi는 동시에 사용 불가"
        assert d_model % n_head == 0
        self.nh = n_head
        self.dh = d_model // n_head
        self.rope = rope
        self.alibi = alibi
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad_mask):
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.nh)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.nh)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.nh)

        if self.rope:
            if q.shape[-1] % 2 != 0:
                q = F.pad(q, (0, 1))
                k = F.pad(k, (0, 1))
            q, k = apply_rope(q, k)

        att = torch.einsum('b h i d, b h j d -> b h i j', q, k) / (q.shape[-1] ** 0.5)

        if self.alibi:
            bias = build_alibi_bias(self.nh, L, x.device, x.dtype)
            att = att + bias

        if key_pad_mask is not None:
            mask = key_pad_mask[:, None, None, :]
            att = att.masked_fill(mask, float('-inf'))

        w = F.softmax(att, dim=-1)
        w = self.drop(w)
        h = torch.einsum('b h i j, b h j d -> b h i d', w, v)
        h = rearrange(h, 'b h l d -> b l (h d)')
        return self.o(h)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, p, rope=False, alibi=False, ff_dim=2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiheadSelfAttention(d_model, n_head, p, rope=rope, alibi=alibi)
        self.drop = nn.Dropout(p)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x, key_pad_mask):
        h = self.sa(self.ln1(x), key_pad_mask)
        x = x + self.drop(h)
        h = self.ff(self.ln2(x))
        x = x + self.drop(h)
        return x


class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, p_drop, pad_id,
                 use_abs_pos=False, rope=False, alibi=False, ff_dim=2048):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.use_abs_pos = use_abs_pos
        if self.use_abs_pos:
            self.pos = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_head, p_drop, rope=rope, alibi=alibi, ff_dim=ff_dim)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        B, L = x.size()
        h = self.tok(x)
        if self.use_abs_pos:
            pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            h = h + self.pos(pos)
        key_pad_mask = (attn_mask == 0)
        for blk in self.blocks:
            h = blk(h, key_pad_mask)
        h = self.ln(h)
        # Mean Pool (pad 제외)
        denom = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        return F.normalize(pooled, dim=-1)


class SiameseModel(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.enc = CodeEncoder(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            p_drop=DROPOUT,
            pad_id=pad_id,
            use_abs_pos=USE_ABS_POS,
            rope=USE_ROPE,
            alibi=USE_ALIBI,
            ff_dim=FF_DIM
        )
        self.tau = nn.Parameter(torch.tensor(TAU_INIT, dtype=torch.float32),
                                requires_grad=TAU_LEARNABLE)

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        za = self.enc(ids_a, mask_a)
        zb = self.enc(ids_b, mask_b)
        return za, zb


# =========================================
# Loss & Metrics
# =========================================
def weighted_info_nce(za, zb, pair_type, tau):
    """Weighted InfoNCE Loss"""
    logits = torch.matmul(za, zb.T) / tau.clamp(min=1e-3)
    labels = torch.arange(len(za), device=za.device)
    li = F.cross_entropy(logits, labels, reduction="none")
    lj = F.cross_entropy(logits.T, labels, reduction="none")
    loss = (li + lj) / 2

    wmap = {
        "hard_negative": HARD_WEIGHT,
        "semi_hard_negative": SEMI_WEIGHT,
    }
    weights = torch.tensor([wmap.get(p, 1.0) for p in pair_type],
                           device=za.device, dtype=loss.dtype)
    return (loss * weights).mean(), logits


def recall_at_k(logits, k):
    """Recall@K"""
    target = torch.arange(logits.size(0), device=logits.device)
    topk = logits.topk(k, dim=1).indices
    return (topk == target.unsqueeze(1)).any(dim=1).float().mean().item()


def mean_reciprocal_rank(logits):
    """Mean Reciprocal Rank"""
    target = torch.arange(logits.size(0), device=logits.device)
    ranks = (logits.argsort(dim=1, descending=True) == target.unsqueeze(1)).nonzero()[:, 1] + 1
    return (1.0 / ranks.float()).mean().item()


# =========================================
# Main
# =========================================
def main():
    print("=" * 70)
    print("Contrastive Learning Pre-training (Siamese Model)")
    print("이서율 - From-scratch Transformer with RoPE")
    print("=" * 70)

    print(f"\n✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 설정 출력
    print(f"\n설정:")
    print(f"  Pair Data: {PAIR_PATH}")
    print(f"  Tokenizer: {TOKENIZER_DIR}")
    print(f"  Save Dir: {SAVE_DIR}")
    print(f"  D_MODEL: {D_MODEL}, N_LAYER: {N_LAYER}, N_HEAD: {N_HEAD}")
    print(f"  RoPE: {USE_ROPE}, ALiBi: {USE_ALIBI}, Abs Pos: {USE_ABS_POS}")
    print(f"  Batch: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR_BASE}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 데이터 로드
    print(f"\n{'=' * 70}")
    print("데이터 로드")
    print(f"{'=' * 70}")

    df = pd.read_parquet(PAIR_PATH)
    print(f"✓ 총 페어: {len(df):,}개")

    # 문제 단위 split
    problems = df["problem_id_a"].unique()
    np.random.shuffle(problems)
    train_ids, val_ids, test_ids = np.split(problems,
                                             [int(.8 * len(problems)), int(.9 * len(problems))])
    train_df = df[df.problem_id_a.isin(train_ids)]
    val_df = df[df.problem_id_a.isin(val_ids)]
    test_df = df[df.problem_id_a.isin(test_ids)]

    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # 토크나이저
    print(f"\n{'=' * 70}")
    print("토크나이저 로드")
    print(f"{'=' * 70}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    VOCAB_SIZE = tokenizer.vocab_size

    print(f"✓ Vocab Size: {VOCAB_SIZE:,}")
    print(f"  PAD ID: {PAD_ID}")

    # 데이터로더
    train_dl = DataLoader(PairDataset(train_df, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(PairDataset(val_df, tokenizer, MAX_LEN),
                        batch_size=BATCH_SIZE, shuffle=False)

    # 모델
    print(f"\n{'=' * 70}")
    print("모델 초기화")
    print(f"{'=' * 70}")

    model = SiameseModel(VOCAB_SIZE, PAD_ID).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 총 파라미터: {total_params:,}")

    # Optimizer & Scheduler
    if TAU_LEARNABLE:
        tau_params = [model.tau]
        base_params = [p for n, p in model.named_parameters() if n != 'tau']
        opt = torch.optim.AdamW([
            {'params': base_params, 'lr': LR_BASE},
            {'params': tau_params, 'lr': LR_TAU},
        ])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR_BASE)

    total_steps = len(train_dl) * EPOCHS
    warmup_steps = max(100, int(WARMUP_RATIO * total_steps))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"✓ Optimizer: AdamW")
    print(f"  Total Steps: {total_steps}, Warmup: {warmup_steps}")

    # 학습
    print(f"\n{'=' * 70}")
    print("학습 시작")
    print(f"{'=' * 70}")

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        opt.zero_grad()

        for step, batch in enumerate(train_dl, 1):
            ids_a = batch["ids_a"].to(device)
            mask_a = batch["mask_a"].to(device)
            ids_b = batch["ids_b"].to(device)
            mask_b = batch["mask_b"].to(device)
            ptype = batch["ptype"]

            za, zb = model(ids_a, mask_a, ids_b, mask_b)
            loss, lg = weighted_info_nce(za, zb, ptype, model.tau)

            (loss / GRAD_ACCUM_STEPS).backward()

            with torch.no_grad():
                preds = lg.argmax(dim=1)
                acc = (preds == torch.arange(len(preds), device=preds.device)).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

            if step % STEP_LOG == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch + 1}/{EPOCHS} | Step {step:>4} | lr={lr_now:.2e}] "
                      f"loss={loss.item():.4f} | acc={acc:.4f}")

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()
                scheduler.step()
                opt.zero_grad()
                global_step += 1

        print(f"[Train] epoch {epoch + 1}: "
              f"loss={total_loss / n_batches:.4f} | acc={total_acc / n_batches:.4f} | tau={model.tau.item():.4f}")

        # Validation
        model.eval()
        val_loss, val_acc, val_n = 0.0, 0.0, 0
        all_r1, all_r5, all_r10, all_mrr = [], [], [], []
        with torch.no_grad():
            for batch in val_dl:
                ids_a = batch["ids_a"].to(device)
                mask_a = batch["mask_a"].to(device)
                ids_b = batch["ids_b"].to(device)
                mask_b = batch["mask_b"].to(device)
                ptype = batch["ptype"]

                za, zb = model(ids_a, mask_a, ids_b, mask_b)
                v_loss, lg = weighted_info_nce(za, zb, ptype, model.tau)

                preds = lg.argmax(dim=1)
                v_acc = (preds == torch.arange(len(preds), device=preds.device)).float().mean().item()

                val_loss += v_loss.item()
                val_acc += v_acc
                val_n += 1

                all_r1.append(recall_at_k(lg, 1))
                all_r5.append(recall_at_k(lg, 5))
                all_r10.append(recall_at_k(lg, 10))
                all_mrr.append(mean_reciprocal_rank(lg))

        print(f"[Val] epoch {epoch + 1}: "
              f"loss={val_loss / val_n:.4f} | acc={val_acc / val_n:.4f} "
              f"| R@1={np.mean(all_r1):.3f} R@5={np.mean(all_r5):.3f} R@10={np.mean(all_r10):.3f} "
              f"| MRR={np.mean(all_mrr):.3f} | tau={model.tau.item():.4f}")

        # 체크포인트 저장
        torch.save(model.state_dict(), f"{SAVE_DIR}/ckpt_epoch{epoch + 1}.pt")

    # 최종 저장
    torch.save(model.state_dict(), f"{SAVE_DIR}/siamese_contrastive_final.pt")
    print(f"\n✅ 최종 모델 저장: {SAVE_DIR}/siamese_contrastive_final.pt")


if __name__ == "__main__":
    main()
