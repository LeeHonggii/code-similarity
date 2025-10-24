"""
RoleBERT Pre-training (From Scratch)
오정탁 - BERT with Role Embeddings (KEYWORD, IDENTIFIER, OP, LITERAL)

Usage:
    python train/pretrain_rolebert_OhJeongTak.py
"""

import os, io, re, math, json, random, warnings, tokenize
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.normalizers import NFKC
except Exception:
    raise RuntimeError("pip install tokenizers")

warnings.filterwarnings("ignore", category=UserWarning)

# =============================
# 설정
# =============================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Special Tokens
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
PAD, CLS, SEP, MASK, UNK = SPECIAL_TOKENS

# Role Schema
ROLE2ID = {
    "PAD": 0,
    "UNK": 1,
    "KEYWORD": 2,
    "IDENTIFIER": 3,
    "OP": 4,
    "LITERAL": 5,
    "SEP": 6,
}
ROLE_VOCAB_SIZE = len(ROLE2ID)

# 경로
PARQUET_PATH = "./data/code_corpus_preprocessed.parquet"
SAVE_ROOT = "./models/rolebert"
TOKENIZER_SAVE = os.path.join(SAVE_ROOT, "tokenizer")
CHECKPOINT_SAVE = os.path.join(SAVE_ROOT, "checkpoints")

# 하이퍼파라미터
VOCAB_SIZE = 32000
MAX_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 1
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MLM_PROB = 0.15
LAMBDA_MLM = 1.0
LAMBDA_ROLE = 1.0
MAX_GRAD_NORM = 1.0

# 모델 설정
HIDDEN_SIZE = 768
NUM_LAYERS = 6
NUM_HEADS = 12
INTERMEDIATE_SIZE = 3072
MAX_POSITION = 512


# =========================================
# 유틸리티
# =========================================
def simple_format_code_py(code: str) -> str:
    """주석 제거 + 공백 정리"""
    try:
        out_tokens = []
        tokgen = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok_type, tok_str, *_ in tokgen:
            if tok_type != tokenize.COMMENT:
                out_tokens.append((tok_type, tok_str))
        code = tokenize.untokenize(out_tokens)
    except Exception:
        pass
    code = code.replace("\t", "    ")
    code = "\n".join(ln.rstrip() for ln in code.splitlines())
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code


def heuristic_role_tagging(tokens: List[str]) -> List[int]:
    """토큰에 Role ID 부여 (KEYWORD, IDENTIFIER, OP, LITERAL)"""
    py_keywords = {
        "def", "class", "return", "if", "elif", "else", "for", "while", "try",
        "except", "finally", "with", "as", "import", "from", "pass", "break",
        "continue", "lambda", "yield", "global", "nonlocal", "assert", "raise",
        "del", "in", "is", "not", "and", "or", "true", "false", "none",
    }
    ops = set(list("=+-*/%<>!&|^~:.,;()[]{}@")) | {
        "**", "//", "==", "!=", ">=", "<=", "->", "+=", "-=", "*=", "/=",
        "%=", "**=", "//=", "<<", ">>", "<<=", ">>=", "&=", "|=", "^=", "@="
    }

    role_ids = []
    for tk in tokens:
        low = tk.lower()
        if tk in (PAD, CLS, SEP, MASK, UNK):
            role_ids.append(ROLE2ID["SEP"])
        elif low in py_keywords:
            role_ids.append(ROLE2ID["KEYWORD"])
        elif re.fullmatch(r"\d+(\.\d+)?", tk):
            role_ids.append(ROLE2ID["LITERAL"])
        elif (len(tk) >= 2 and tk[0] in ("'", '"')) or re.fullmatch(r"['\"].*['\"]", tk):
            role_ids.append(ROLE2ID["LITERAL"])
        elif (tk in ops) or re.fullmatch(r"[\(\)\[\]\{\}\,\:\.\;\+\-\*\/\%\=\!\<\>\&\|\^\~@]+", tk):
            role_ids.append(ROLE2ID["OP"])
        else:
            role_ids.append(ROLE2ID["IDENTIFIER"])
    return role_ids


# =========================================
# 토크나이저 학습
# =========================================
def iter_corpus_texts(parquet_path: str, text_col_hint: str = "text"):
    """Parquet에서 코드 텍스트 로드"""
    df = pd.read_parquet(parquet_path)
    if text_col_hint not in df.columns:
        cands = [c for c in df.columns if "code" in c.lower() or "text" in c.lower()]
        assert cands, f"No text/code column in: {list(df.columns)}"
        text_col_hint = cands[0]
    for x in df[text_col_hint].astype(str).tolist():
        yield simple_format_code_py(x)


def train_bpe_tokenizer(
    parquet_path: str,
    vocab_size: int,
    save_dir: str
) -> str:
    """BPE 토크나이저 학습"""
    print(f"\n{'=' * 70}")
    print("BPE 토크나이저 학습")
    print(f"{'=' * 70}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    tok = Tokenizer(BPE(unk_token=UNK))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tmp_txt = os.path.join(save_dir, "corpus.tmp.txt")
    print(f"코퍼스 파일 생성 중...")
    with open(tmp_txt, "w", encoding="utf-8") as f:
        for i, text in enumerate(iter_corpus_texts(parquet_path)):
            f.write(text.replace("\n", " ") + "\n")
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1} 라인 처리...")

    print(f"\n토크나이저 학습 중...")
    tok.train(files=[tmp_txt], trainer=trainer)

    tok_path = os.path.join(save_dir, "tokenizer.json")
    tok.save(tok_path)
    
    os.remove(tmp_txt)
    
    print(f"✓ 토크나이저 저장: {tok_path}")
    print(f"  Vocab Size: {tok.get_vocab_size():,}")
    
    return tok_path


# =========================================
# 데이터셋
# =========================================
class CodeDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer_json: str,
        text_col: str = "text",
        max_len: int = 256,
        do_format: bool = True,
    ):
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        if text_col not in self.df.columns:
            cands = [c for c in self.df.columns if "code" in c.lower() or "text" in c.lower()]
            assert cands, f"No text/code column: {list(self.df.columns)}"
            text_col = cands[0]
        self.text_col = text_col
        self.max_len = max_len
        self.do_format = do_format

        self.tokenizer = Tokenizer.from_file(tokenizer_json)
        self.pad_id = self.tokenizer.token_to_id(PAD)
        self.cls_id = self.tokenizer.token_to_id(CLS)
        self.sep_id = self.tokenizer.token_to_id(SEP)
        self.mask_id = self.tokenizer.token_to_id(MASK)
        self.unk_id = self.tokenizer.token_to_id(UNK)

    def encode(self, text: str) -> Dict[str, List[int]]:
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        # [CLS] ... [SEP]
        ids = [self.cls_id] + ids[: self.max_len - 2] + [self.sep_id]
        seg = [0] * len(ids)
        return {"input_ids": ids, "token_type_ids": seg}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        code = str(self.df.iloc[idx][self.text_col])
        if self.do_format:
            code = simple_format_code_py(code)
        
        out = self.encode(code)
        tokens = [self.tokenizer.id_to_token(i) for i in out["input_ids"]]
        role_ids = heuristic_role_tagging(tokens)
        
        return {
            "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
            "token_type_ids": torch.tensor(out["token_type_ids"], dtype=torch.long),
            "role_ids": torch.tensor(role_ids, dtype=torch.long),
        }


@dataclass
class CollatorMLMRole:
    """MLM + Role Labeling Collator"""
    pad_id: int
    mask_id: int
    vocab_size: int
    mlm_prob: float = 0.15

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad_tensor(t: torch.Tensor, pad_val: int):
            out = torch.full((max_len,), pad_val, dtype=t.dtype)
            out[: len(t)] = t
            return out

        input_ids = torch.stack([pad_tensor(f["input_ids"], self.pad_id) for f in features])
        token_type_ids = torch.stack([pad_tensor(f["token_type_ids"], 0) for f in features])
        role_ids = torch.stack([pad_tensor(f["role_ids"], ROLE2ID["PAD"]) for f in features])
        attention_mask = (input_ids != self.pad_id).long()

        # MLM labels
        labels = input_ids.clone()
        special_mask = (input_ids == self.pad_id)
        prob = torch.full_like(input_ids, fill_value=self.mlm_prob, dtype=torch.float)
        prob[special_mask] = 0.0
        masked = torch.bernoulli(prob).bool()

        labels[~masked] = -100

        # 80% [MASK]
        replace_mask = masked & (torch.rand_like(prob) < 0.8)
        input_ids[replace_mask] = self.mask_id

        # 10% random
        random_mask = masked & (~replace_mask) & (torch.rand_like(prob) < 0.5)
        random_words = torch.randint(low=0, high=self.vocab_size, size=input_ids.size(), dtype=torch.long)
        input_ids[random_mask] = random_words[random_mask]

        # Role labels
        role_labels = role_ids.clone()
        role_labels[(input_ids == self.pad_id)] = -100

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "role_ids": role_ids,
            "role_labels": role_labels,
        }


# =========================================
# 모델
# =========================================
class RoleBertConfig:
    def __init__(
        self,
        vocab_size: int,
        role_vocab_size: int = ROLE_VOCAB_SIZE,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.role_vocab_size = role_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob


class RoleBertForPretraining(nn.Module):
    """BERT with Role Embeddings"""
    
    def __init__(self, cfg: RoleBertConfig, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        # Embeddings
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=pad_id)
        self.pos_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.role_embeddings = nn.Embedding(cfg.role_vocab_size, cfg.hidden_size)

        self.emb_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.emb_drop = nn.Dropout(cfg.hidden_dropout_prob)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_attention_heads,
            dim_feedforward=cfg.intermediate_size,
            dropout=cfg.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_hidden_layers)

        # MLM Head
        self.mlm_transform = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
        )
        self.mlm_decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.mlm_decoder.bias = self.mlm_bias

        # Role Classifier
        self.role_classifier = nn.Linear(cfg.hidden_size, cfg.role_vocab_size)

        # Weight tying
        self.mlm_decoder.weight = self.word_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _embed(self, input_ids, token_type_ids, role_ids):
        bsz, seqlen = input_ids.size()
        pos_ids = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)

        x = (
            self.word_embeddings(input_ids)
            + self.pos_embeddings(pos_ids)
            + self.seg_embeddings(token_type_ids)
            + self.role_embeddings(role_ids.clamp(0, self.cfg.role_vocab_size - 1))
        )
        x = self.emb_ln(x)
        x = self.emb_drop(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        role_labels: Optional[torch.Tensor] = None,
        lambda_mlm: float = 1.0,
        lambda_role: float = 1.0,
    ):
        x = self._embed(input_ids, token_type_ids, role_ids)
        key_padding_mask = (attention_mask == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # MLM
        mlm_hidden = self.mlm_transform(h)
        logits = self.mlm_decoder(mlm_hidden)

        # Role
        role_logits = self.role_classifier(h)

        loss = None
        mlm_loss = None
        role_loss = None
        
        if labels is not None:
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        if role_labels is not None:
            role_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                role_logits.view(-1, role_logits.size(-1)), role_labels.view(-1)
            )
        if (mlm_loss is not None) or (role_loss is not None):
            loss = (lambda_mlm * (mlm_loss if mlm_loss is not None else 0.0)) + \
                   (lambda_role * (role_loss if role_loss is not None else 0.0))

        return {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "role_loss": role_loss,
            "logits": logits,
            "role_logits": role_logits,
            "last_hidden_state": h,
        }


# =========================================
# Scheduler
# =========================================
class CosineWithWarmup:
    """Cosine decay with warmup"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.step_num <= self.warmup_steps:
                lr = base_lr * self.step_num / self.warmup_steps
            else:
                progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
            group["lr"] = lr


# =========================================
# Main
# =========================================
def main():
    print("=" * 70)
    print("RoleBERT Pre-training (From Scratch)")
    print("오정탁 - BERT with Role Embeddings")
    print("=" * 70)
    
    print(f"\n✅ Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(CHECKPOINT_SAVE, exist_ok=True)
    
    # ========================================
    # 토크나이저 학습
    # ========================================
    tok_path = train_bpe_tokenizer(
        parquet_path=PARQUET_PATH,
        vocab_size=VOCAB_SIZE,
        save_dir=TOKENIZER_SAVE
    )
    
    # ========================================
    # 데이터셋
    # ========================================
    print(f"\n{'=' * 70}")
    print("데이터셋 구축")
    print(f"{'=' * 70}")
    
    ds = CodeDataset(
        parquet_path=PARQUET_PATH,
        tokenizer_json=tok_path,
        text_col="text",
        max_len=MAX_LEN,
        do_format=True,
    )
    print(f"✓ 총 샘플: {len(ds):,}개")
    
    tk = Tokenizer.from_file(tok_path)
    pad_id = tk.token_to_id(PAD)
    mask_id = tk.token_to_id(MASK)
    vocab_size = tk.get_vocab_size()
    
    collate = CollatorMLMRole(
        pad_id=pad_id,
        mask_id=mask_id,
        vocab_size=vocab_size,
        mlm_prob=MLM_PROB
    )
    
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        pin_memory=(DEVICE == "cuda")
    )
    
    # ========================================
    # 모델
    # ========================================
    print(f"\n{'=' * 70}")
    print("모델 초기화")
    print(f"{'=' * 70}")
    
    cfg = RoleBertConfig(
        vocab_size=vocab_size,
        role_vocab_size=ROLE_VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_POSITION,
        hidden_dropout_prob=0.1,
    )
    
    model = RoleBertForPretraining(cfg, pad_id=pad_id)
    model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 총 파라미터: {total_params:,}")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Heads: {NUM_HEADS}")
    
    # ========================================
    # Optimizer & Scheduler
    # ========================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = CosineWithWarmup(optimizer, warmup_steps, total_steps)
    
    print(f"\n✓ Optimizer: AdamW")
    print(f"  Total Steps: {total_steps}, Warmup: {warmup_steps}")
    
    # ========================================
    # 학습
    # ========================================
    print(f"\n{'=' * 70}")
    print("학습 시작")
    print(f"{'=' * 70}")
    
    model.train()
    step = 0
    
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            role_ids = batch["role_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            role_labels = batch["role_labels"].to(DEVICE)

            out = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                role_ids=role_ids,
                labels=labels,
                role_labels=role_labels,
                lambda_mlm=LAMBDA_MLM,
                lambda_role=LAMBDA_ROLE,
            )
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            mlm_l = float(out["mlm_loss"]) if out["mlm_loss"] is not None else 0.0
            role_l = float(out["role_loss"]) if out["role_loss"] is not None else 0.0
            
            pbar.set_postfix({
                "loss": f"{float(loss):.4f}",
                "mlm": f"{mlm_l:.4f}",
                "role": f"{role_l:.4f}"
            })
            
            step += 1
    
    # ========================================
    # 저장
    # ========================================
    print(f"\n{'=' * 70}")
    print("모델 저장")
    print(f"{'=' * 70}")
    
    ckpt_path = os.path.join(CHECKPOINT_SAVE, "rolebert_scratch.pt")
    torch.save(model.state_dict(), ckpt_path)
    
    print(f"✓ 모델 저장: {ckpt_path}")
    print(f"✓ 토크나이저: {tok_path}")
    
    print(f"\n{'=' * 70}")
    print("✓ 학습 완료!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
