import os, io, re, tokenize
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm
import ast as _pyast
from einops import rearrange

# =============================
# 설정
# =============================
TEST_CSV = "./data/test.csv"
TOKENIZER_DIR = "./data/tokenizer_unigram_LeeSeoYul"
CKPT_PATH = "./models/siamese_contrastive/siamese_contrastive_final.pt"
SUBMIT_PATH = "./inference/submission_LeeSeoYul.csv"

MAX_LEN = 512
D_MODEL = 512
FF_DIM = 1024
N_LAYER = 6
N_HEAD = 8
DROPOUT = 0.1
PAD_ID = 0
BATCH_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================
# 전처리
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
# 모델 컴포넌트
# =========================================
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.nh = n_head
        self.dh = d_model // n_head
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

        att = torch.einsum('b h i d, b h j d -> b h i j', q, k) / (q.shape[-1] ** 0.5)

        if key_pad_mask is not None:
            mask = key_pad_mask[:, None, None, :]
            att = att.masked_fill(mask, float('-inf'))

        w = F.softmax(att, dim=-1)
        w = self.drop(w)
        h = torch.einsum('b h i j, b h j d -> b h i d', w, v)
        h = rearrange(h, 'b h l d -> b l (h d)')
        return self.o(h)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, p, ff_dim=2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiheadSelfAttention(d_model, n_head, p)
        self.drop = nn.Dropout(p)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x, key_pad_mask):
        x = x + self.drop(self.sa(self.ln1(x), key_pad_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, pad_id, p_drop):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_head, p_drop, ff_dim=d_model * 4)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        B, L = x.size()
        h = self.tok(x)
        key_pad_mask = (attn_mask == 0)
        for blk in self.blocks:
            h = blk(h, key_pad_mask)
        h = self.ln(h)
        denom = attn_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        return F.normalize(pooled, dim=-1)


class ASTMiniGNN(nn.Module):
    """Shallow AST-GNN"""

    def __init__(self, type_vocab_size=256, hid=128, out=128):
        super().__init__()
        self.type_emb = nn.Embedding(type_vocab_size, hid)
        self.lin1 = nn.Linear(hid, hid)
        self.lin2 = nn.Linear(hid, out)
        self._node_type2id = {"<unk>": 0}

    def _edges_and_types(self, code: str):
        try:
            tree = _pyast.parse(code)
        except Exception:
            return [], [0]
        nodes, edges = [], []
        q = [(tree, -1)]
        while q:
            node, parent = q.pop(0)
            nid = len(nodes)
            ntype = type(node).__name__
            tid = self._node_type2id.setdefault(ntype, len(self._node_type2id))
            nodes.append(tid)
            if parent >= 0:
                edges.append((parent, nid))
                edges.append((nid, parent))
            for ch in _pyast.iter_child_nodes(node):
                q.append((ch, nid))
        if not nodes:
            nodes = [0]
        return edges, nodes

    def forward_one(self, code: str, device):
        edges, type_ids = self._edges_and_types(code)
        N = len(type_ids)
        X = self.type_emb(torch.tensor(type_ids, device=device))
        if N == 1:
            return self.lin2(F.relu(self.lin1(X))).mean(dim=0)
        adj = torch.zeros((N, N), device=device)
        for i, j in edges:
            if 0 <= i < N and 0 <= j < N:
                adj[i, j] = 1.0
        adj = adj + torch.eye(N, device=device)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        D = torch.diag(deg_inv_sqrt)
        A = D @ adj @ D
        h = A @ X
        h = F.relu(self.lin1(h))
        z = A @ h
        z = self.lin2(z)
        return z.mean(dim=0)

    def encode_batch(self, codes, device):
        zs = [self.forward_one(c, device) for c in codes]
        return F.normalize(torch.stack(zs, dim=0), dim=-1)


class FusionSiamese(nn.Module):
    """Fusion: Text Encoder + AST-GNN"""

    def __init__(self, vocab_size, d_model, n_head, n_layer, pad_id, p_drop, ast_out=128):
        super().__init__()
        self.enc = CodeEncoder(vocab_size, d_model, n_head, n_layer, pad_id, p_drop)
        self.ast = ASTMiniGNN(out=ast_out)
        self.proj = nn.Linear(d_model + ast_out, d_model)
        self.tau = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)

    def fuse(self, ids, mask, raw_codes, device):
        z_txt = self.enc(ids, mask)
        z_ast = self.ast.encode_batch(raw_codes, device)
        z = torch.cat([z_txt, z_ast], dim=1)
        return F.normalize(self.proj(z), dim=-1)

    def forward(self, ids_a, mask_a, ids_b, mask_b, raw_a, raw_b):
        device = ids_a.device
        za = self.fuse(ids_a, mask_a, raw_a, device)
        zb = self.fuse(ids_b, mask_b, raw_b, device)
        return za, zb


# =========================================
# 데이터셋
# =========================================
class TestPairs(Dataset):
    def __init__(self, frame, tokenizer, max_len=512):
        self.df = frame.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = r["pair_id"]
        code1 = "" if pd.isna(r["code1"]) else str(r["code1"])
        code2 = "" if pd.isna(r["code2"]) else str(r["code2"])
        ta = code_to_marked_text(code1)
        tb = code_to_marked_text(code2)

        ea = self.tok(ta, truncation=True, max_length=self.max_len,
                      padding="max_length", return_tensors="pt")
        eb = self.tok(tb, truncation=True, max_length=self.max_len,
                      padding="max_length", return_tensors="pt")
        return {
            "pair_id": pid,
            "ids_a": ea["input_ids"].squeeze(0),
            "mask_a": ea["attention_mask"].squeeze(0),
            "ids_b": eb["input_ids"].squeeze(0),
            "mask_b": eb["attention_mask"].squeeze(0),
            "raw_a": code1,
            "raw_b": code2,
        }


# =========================================
# Main
# =========================================
def main():
    print("=" * 70)
    print("Contrastive Learning Inference (Fusion Model)")
    print("이서율 - Text + AST-GNN")
    print("=" * 70)

    print(f"\n✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 설정
    print(f"\n설정:")
    print(f"  Test CSV: {TEST_CSV}")
    print(f"  Tokenizer: {TOKENIZER_DIR}")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Submit: {SUBMIT_PATH}")

    # 테스트 데이터 로드
    print(f"\n{'=' * 70}")
    print("테스트 데이터 로드")
    print(f"{'=' * 70}")

    test_df = pd.read_csv(TEST_CSV)
    if {"code1_norm", "code2_norm"}.issubset(test_df.columns):
        test_df = test_df.rename(columns={"code1_norm": "code1", "code2_norm": "code2"})

    print(f"✓ 테스트 데이터: {len(test_df):,}개")
    print(f"  컬럼: {test_df.columns.tolist()}")

    # 토크나이저
    print(f"\n{'=' * 70}")
    print("토크나이저 로드")
    print(f"{'=' * 70}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    VOCAB_SIZE = tokenizer.vocab_size

    print(f"✓ Vocab Size: {VOCAB_SIZE:,}")

    # 데이터로더
    test_dl = DataLoader(
        TestPairs(test_df, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 모델 로드
    print(f"\n{'=' * 70}")
    print("모델 로드")
    print(f"{'=' * 70}")

    model = FusionSiamese(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        pad_id=PAD_ID,
        p_drop=DROPOUT,
        ast_out=128
    ).to(device)

    # 체크포인트 로드
    try:
        sd = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"✓ 체크포인트 로드 완료: {CKPT_PATH}")
    except Exception as e:
        print(f"⚠️ 체크포인트 로드 실패: {e}")
        print(f"  랜덤 초기화 모델 사용")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {total_params:,}")

    # 추론
    print(f"\n{'=' * 70}")
    print("추론 시작")
    print(f"{'=' * 70}")

    USE_AMP = torch.cuda.is_available()
    pair_ids, probs = [], []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Predicting"):
            ids_a = batch["ids_a"].to(device, non_blocking=True)
            mask_a = batch["mask_a"].to(device, non_blocking=True)
            ids_b = batch["ids_b"].to(device, non_blocking=True)
            mask_b = batch["mask_b"].to(device, non_blocking=True)
            raw_a = batch["raw_a"]
            raw_b = batch["raw_b"]

            with autocast(enabled=USE_AMP):
                za, zb = model(ids_a, mask_a, ids_b, mask_b, raw_a, raw_b)
                tau = model.tau if hasattr(model, "tau") else torch.tensor(0.1, device=za.device)
                logits = (za * zb).sum(dim=1) / torch.clamp(tau, min=1e-3)
                prob = torch.sigmoid(logits).detach().float().cpu()

            pair_ids.extend(batch["pair_id"])
            probs.extend(prob.tolist())

    print(f"\n✓ 추론 완료: {len(probs):,}개")

    # 제출 파일 생성
    print(f"\n{'=' * 70}")
    print("제출 파일 생성")
    print(f"{'=' * 70}")

    sub = pd.DataFrame({"pair_id": pair_ids, "similar": probs})

    # 확률 -> 레이블 (threshold=0.5)
    sub["similar"] = (sub["similar"] >= 0.5).astype(int)

    print(f"\n예측 분포:")
    print(sub['similar'].value_counts())
    print(f"  - similar=0: {(sub['similar'] == 0).sum():,}개 ({(sub['similar'] == 0).sum() / len(sub) * 100:.1f}%)")
    print(f"  - similar=1: {(sub['similar'] == 1).sum():,}개 ({(sub['similar'] == 1).sum() / len(sub) * 100:.1f}%)")

    # 저장
    os.makedirs(os.path.dirname(SUBMIT_PATH), exist_ok=True)
    sub.to_csv(SUBMIT_PATH, index=False)

    print(f"\n✓ 제출 파일 저장: {SUBMIT_PATH}")

    print(f"\n미리보기:")
    print(sub.head(10))

    print(f"\n{'=' * 70}")
    print("✓ 모든 작업 완료!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
