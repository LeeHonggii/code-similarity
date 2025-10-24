import os, math, random, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm.auto import tqdm

# =============================
# 설정
# =============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 경로
PRETRAINED_CKPT = "./models/rolebert/rolebert_scratch.pt"
TOKENIZER_DIR = "./models/rolebert"
TRAIN_CSV = "./data/train_pairs.csv"
OUT_DIR = "./models/rolebert_finetune"

# 하이퍼파라미터
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LR_BACKBONE = 2e-5
LR_HEAD = 1e-4
WARMUP_RATIO = 0.05
VAL_RATIO = 0.1
GRAD_ACCUM = 1
WEIGHT_DECAY = 0.01
USE_AMP = True
BACKBONE_SKELETON = "bert-base-uncased"  # 구조 스켈레톤

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================
# 데이터셋
# =========================================
REQUIRED_COLS = {"code_a_norm", "code_b_norm", "similar"}


def load_pairs_csv(path):
    """CSV 로드 및 컬럼 정규화"""
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    
    need = {}
    for k in REQUIRED_COLS:
        if k in cols_lower:
            need[k] = cols_lower[k]
        else:
            raise ValueError(f"CSV에 '{k}' 컬럼이 필요합니다. 현재: {list(df.columns)}")
    
    df = df.rename(columns={
        need["code_a_norm"]: "code_a_norm",
        need["code_b_norm"]: "code_b_norm",
        need["similar"]: "similar",
    })
    df["similar"] = df["similar"].astype(int)
    return df


class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tk = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        a = str(self.df.loc[i, "code_a_norm"])
        b = str(self.df.loc[i, "code_b_norm"])
        y = float(self.df.loc[i, "similar"])
        
        enc = self.tk(
            a, b,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # 허용 키만 추출
        allowed = {
            k: v.squeeze(0) for k, v in enc.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        allowed["labels"] = torch.tensor(y, dtype=torch.float)
        return allowed


# =========================================
# 모델
# =========================================
class CrossEncoder(nn.Module):
    """Cross-Encoder: 코드 페어를 직접 분류"""
    
    def __init__(self, backbone_name_or_dir):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name_or_dir)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = out.last_hidden_state[:, 0]
        logit = self.head(cls).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logit, labels)
        
        return {"loss": loss, "logits": logit}


# =========================================
# 평가
# =========================================
def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def evaluate(model, loader, best_thr=None):
    """평가 및 최적 threshold 탐색"""
    model.eval()
    probs, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            for k in list(batch.keys()):
                if hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device)
            
            inputs = {
                k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids"]
                if k in batch
            }
            out = model(**inputs)
            logits = out["logits"].detach().cpu().numpy()
            probs.append(sigmoid_np(logits))
            labels.append(batch["labels"].detach().cpu().numpy())
    
    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    
    thr = 0.5 if best_thr is None else best_thr
    if best_thr is None:
        cand = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(labels, (probs >= t).astype(int)) for t in cand]
        thr = float(cand[int(np.argmax(f1s))])
    
    preds = (probs >= thr).astype(int)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "thr": float(thr),
    }


# =========================================
# Main
# =========================================
def main():
    print("=" * 70)
    print("Cross-Encoder Fine-tuning")
    print("정탁 - RoBERTa-based Cross-Encoder")
    print("=" * 70)
    
    print(f"\n✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 설정 출력
    print(f"\n설정:")
    print(f"  Pretrained: {PRETRAINED_CKPT}")
    print(f"  Tokenizer: {TOKENIZER_DIR}")
    print(f"  Train CSV: {TRAIN_CSV}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR_BACKBONE}/{LR_HEAD}")
    
    # ========================================
    # 데이터 로드
    # ========================================
    print(f"\n{'=' * 70}")
    print("데이터 로드")
    print(f"{'=' * 70}")
    
    df_all = load_pairs_csv(TRAIN_CSV)
    print(f"✓ 총 샘플: {len(df_all):,}개")
    
    # Split
    n_total = len(df_all)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val
    
    train_df, val_df = random_split(
        df_all,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_df = df_all.iloc[train_df.indices].reset_index(drop=True)
    val_df = df_all.iloc[val_df.indices].reset_index(drop=True)
    
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}")
    
    # ========================================
    # 토크나이저
    # ========================================
    print(f"\n{'=' * 70}")
    print("토크나이저 로드")
    print(f"{'=' * 70}")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"✓ Vocab Size: {len(tokenizer):,}")
    
    # ========================================
    # 데이터로더
    # ========================================
    train_ds = PairDataset(train_df, tokenizer, MAX_LEN)
    val_ds = PairDataset(val_df, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # ========================================
    # 모델 생성
    # ========================================
    print(f"\n{'=' * 70}")
    print("모델 초기화")
    print(f"{'=' * 70}")
    
    model = CrossEncoder(BACKBONE_SKELETON)
    
    # 토크나이저 크기 맞추기
    try:
        model.backbone.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    
    # 프리트레인 가중치 로드
    print(f"\n프리트레인 체크포인트 로드: {PRETRAINED_CKPT}")
    try:
        ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        new_sd = model.state_dict()
        loaded = 0
        
        for k, v in state_dict.items():
            mk = k[7:] if k.startswith("module.") else k
            if mk in new_sd and isinstance(v, torch.Tensor) and new_sd[mk].shape == v.shape:
                new_sd[mk] = v
                loaded += 1
        
        model.load_state_dict(new_sd, strict=False)
        print(f"✓ Backbone 파라미터 로드: {loaded}개")
    except Exception as e:
        print(f"⚠️ 체크포인트 로드 실패: {e}")
        print(f"  랜덤 초기화로 진행")
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {total_params:,}")
    
    # ========================================
    # Optimizer & Scheduler
    # ========================================
    head_params = list(model.head.parameters())
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("head.")
    ]
    
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD}
        ],
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    print(f"\n✓ Optimizer: AdamW")
    print(f"  Total Steps: {total_steps}, Warmup: {warmup_steps}")
    
    # ========================================
    # 학습
    # ========================================
    print(f"\n{'=' * 70}")
    print("학습 시작")
    print(f"{'=' * 70}")
    
    best_f1, best_thr = -1.0, 0.5
    best_state = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}")
        
        for step, batch in pbar:
            for k in list(batch.keys()):
                if hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device)
            
            inputs = {
                k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids"]
                if k in batch
            }
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(**inputs)
                loss = nn.BCEWithLogitsLoss()(out["logits"], batch["labels"])
            
            scaler.scale(loss).backward()
            
            if step % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            running += loss.item()
            pbar.set_postfix({"loss": running / step})
        
        print(f"[Train] epoch={epoch} | loss={running / len(train_loader):.4f}")
        
        # Validation
        val_metrics = evaluate(model, val_loader, best_thr=None)
        print(f"[Val] epoch={epoch} | acc={val_metrics['acc']:.4f} "
              f"f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f} "
              f"thr={val_metrics['thr']:.2f}")
        
        # Best 저장
        if val_metrics["f1"] > best_f1:
            best_f1, best_thr = val_metrics["f1"], val_metrics["thr"]
            best_state = {
                "model": model.state_dict(),
                "config": {
                    "max_len": MAX_LEN,
                    "thr": best_thr,
                    "backbone_skeleton": BACKBONE_SKELETON
                },
            }
            torch.save(best_state, os.path.join(OUT_DIR, "best_crossencoder.pt"))
            print(f"✓ Best 모델 저장 (F1={best_f1:.4f}, thr={best_thr:.2f})")
            
            # HF 저장
            try:
                model.backbone.save_pretrained(os.path.join(OUT_DIR, "hf_backbone"))
                tokenizer.save_pretrained(os.path.join(OUT_DIR, "hf_tokenizer"))
            except Exception:
                pass
    
    print(f"\n{'=' * 70}")
    print(f"✓ 학습 완료!")
    print(f"{'=' * 70}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best Threshold: {best_thr:.2f}")
    print(f"모델 저장: {OUT_DIR}/best_crossencoder.pt")


if __name__ == "__main__":
    main()
