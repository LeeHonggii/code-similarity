"""
ELECTRA Pre-training (Generator + Discriminator)
병률 - Replaced Token Detection

Usage:
    python train/pretrain_electra_ByeongRyul.py
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    PreTrainedTokenizerFast,
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# =============================
# 설정
# =============================
SEED = 42

# 경로
BASE_DIR = Path("./")
DATA_PATH = BASE_DIR / "data" / "codenet_python_dataset.parquet"
TOKENIZER_DIR = BASE_DIR / "data" / "tokenizers"
OUTPUT_DIR = BASE_DIR / "models" / "custom_electra"

# 토크나이저 선택
TOKENIZER_OPTIONS = {
    "bpe": TOKENIZER_DIR / "bpe_tokenizer.json",
    "wordpiece": TOKENIZER_DIR / "wordpiece_tokenizer.json",
    "unigram": TOKENIZER_DIR / "unigram.model",
}
TOKENIZER_TYPE = "bpe"

# 데이터 설정
SAMPLE_SIZE = None
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_LENGTH = 256
MLM_PROB = 0.15

# 학습 설정
BATCH_SIZE = 48
GRAD_ACCUM = 8
MAX_STEPS = 30000
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.02
DISC_LOSS_WEIGHT = 50.0

# Generator 설정 (작은 모델)
GEN_HIDDEN = 128
GEN_LAYERS = 6
GEN_HEADS = 4
GEN_INTERMEDIATE = 512

# Discriminator 설정 (큰 모델)
DISC_HIDDEN = 256
DISC_LAYERS = 12
DISC_HEADS = 8
DISC_INTERMEDIATE = 1024

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# Utilities
# =============================
def set_seed(seed: int) -> None:
    """시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def normalize_status(value: Any) -> str:
    """상태 정규화"""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return "AC" if int(value) == 4 else str(value)
    text = str(value).strip().upper()
    if text in {"AC", "ACCEPTED"}:
        return "AC"
    return text


def load_codenet_dataframe(path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    """CodeNet 데이터 로드"""
    df = pd.read_parquet(path)
    
    # Status 필터링
    status_col = None
    for candidate in ["status", "Status", "judge_status"]:
        if candidate in df.columns:
            status_col = candidate
            break
    
    if status_col:
        df = df[df[status_col].map(normalize_status).isin({"AC"})]
    
    # 텍스트 컬럼 확인
    text_col = None
    for col in ["processed_code", "code", "text"]:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("텍스트 컬럼을 찾을 수 없습니다")
    
    if text_col != "processed_code":
        df = df.rename(columns={text_col: "processed_code"})
    
    df = df.dropna(subset=["processed_code"]).reset_index(drop=True)
    
    # 샘플링
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=SEED)
    
    # problem_id 생성
    if "problem_id" not in df.columns:
        df["problem_id"] = df.index.astype(str)
    
    return df


def disjoint_problem_split(
    frame: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int
):
    """
    문제 ID 기반 Disjoint Split
    Train/Val/Test 간 문제 겹치지 않음
    """
    assert val_ratio + test_ratio < 1.0
    
    problems = frame["problem_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(problems)

    total = len(problems)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    
    if val_ratio > 0 and val_count == 0:
        val_count = 1
    if test_ratio > 0 and test_count == 0:
        test_count = 1
    if val_count + test_count >= total:
        test_count = max(0, total - val_count - 1)

    val_ids = set(problems[:val_count])
    test_ids = set(problems[val_count:val_count + test_count])
    train_ids = set(problems[val_count + test_count:])

    train_df = frame[frame["problem_id"].isin(train_ids)].reset_index(drop=True)
    val_df = frame[frame["problem_id"].isin(val_ids)].reset_index(drop=True)
    test_df = frame[frame["problem_id"].isin(test_ids)].reset_index(drop=True)

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    print(f"  Problem IDs -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}, Overlap: {len(overlap)}")

    return train_df, val_df, test_df


def load_tokenizer(tokenizer_key: str) -> PreTrainedTokenizerFast:
    """토크나이저 로드"""
    path = TOKENIZER_OPTIONS[tokenizer_key]
    if not path.exists():
        raise FileNotFoundError(f"토크나이저를 찾을 수 없습니다: {path}")
    
    if path.suffix == ".json":
        tokenizer = BertTokenizerFast(tokenizer_file=str(path))
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path))
    
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.padding_side = "right"
    
    special_tokens = {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }
    tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer


# =========================================
# Data Collator
# =========================================
class ElectraDataCollator:
    """ELECTRA용 Data Collator (MLM + Original 보존)"""
    
    def __init__(self, tokenizer: PreTrainedTokenizerFast, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        input_ids = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        token_type_ids = batch.get("token_type_ids")

        original_input_ids = input_ids.clone()
        generator_input_ids, generator_labels = self.mask_tokens(input_ids)

        result = {
            "generator_input_ids": generator_input_ids,
            "generator_labels": generator_labels,
            "original_input_ids": original_input_ids,
            "attention_mask": attention_mask,
        }
        
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids
        
        return result

    def mask_tokens(self, inputs: torch.Tensor):
        """MLM 마스킹"""
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=inputs.device)

        # Special tokens 제외
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
            for val in labels
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=inputs.device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Padding 제외
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8, device=inputs.device)
        ).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% random
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5, device=inputs.device)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device
        )
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


# =========================================
# ELECTRA Model
# =========================================
class ElectraPretrainingModel(torch.nn.Module):
    """
    ELECTRA Pre-training Model
    - Generator: 작은 MLM 모델
    - Discriminator: 큰 Replaced Token Detection 모델
    """
    
    def __init__(
        self,
        generator_config: ElectraConfig,
        discriminator_config: ElectraConfig,
        disc_loss_weight: float = 50.0,
    ) -> None:
        super().__init__()
        self.generator = ElectraForMaskedLM(generator_config)
        self.discriminator = ElectraForPreTraining(discriminator_config)
        self.disc_loss_weight = disc_loss_weight
        self.pad_token_id = generator_config.pad_token_id

        special_tokens = [
            getattr(generator_config, "pad_token_id", None),
            getattr(generator_config, "mask_token_id", None),
            getattr(generator_config, "cls_token_id", None),
            getattr(generator_config, "sep_token_id", None),
            getattr(generator_config, "unk_token_id", None),
        ]
        self.special_token_ids = [tid for tid in special_tokens if tid is not None]

        # Embedding 공유
        shared_embeddings = self.generator.get_input_embeddings()
        self.discriminator.set_input_embeddings(shared_embeddings)

    def sample_generator_predictions(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 20,
    ) -> torch.Tensor:
        """Generator 예측 샘플링 (top-k sampling)"""
        with torch.no_grad():
            vocab_size = logits.size(-1)
            masked_logits = logits.clone()
            
            # Special tokens 제외
            for tid in self.special_token_ids:
                if 0 <= tid < vocab_size:
                    masked_logits[..., tid] = -1e9
            
            if top_k > 0 and top_k < vocab_size:
                topk_vals, topk_idx = torch.topk(
                    masked_logits / temperature, k=top_k, dim=-1
                )
                probs = torch.softmax(topk_vals, dim=-1)
                sampled = torch.multinomial(probs.view(-1, top_k), num_samples=1)
                sampled = sampled.view(*probs.shape[:-1], 1)
                predictions = torch.gather(topk_idx, -1, sampled).squeeze(-1)
            else:
                probs = torch.softmax(masked_logits / temperature, dim=-1)
                predictions = torch.multinomial(
                    probs.view(-1, vocab_size), num_samples=1
                )
                predictions = predictions.view(*logits.shape[:-1])
            
            return predictions

    def forward(
        self,
        generator_input_ids: torch.Tensor,
        generator_labels: torch.Tensor,
        original_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Generator: MLM
        generator_outputs = self.generator(
            input_ids=generator_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=generator_labels,
        )
        gen_loss = generator_outputs.loss
        gen_logits = generator_outputs.logits

        # Generator 예측으로 입력 대체
        generator_predictions = self.sample_generator_predictions(gen_logits)
        replaced_input_ids = generator_input_ids.clone()
        mask_positions = generator_labels != -100
        replaced_input_ids[mask_positions] = generator_predictions[mask_positions]

        # Discriminator labels: replaced vs original
        disc_labels = (replaced_input_ids != original_input_ids).long()
        
        if attention_mask is not None:
            disc_labels = disc_labels.masked_fill(attention_mask == 0, -100)
        
        for tid in self.special_token_ids:
            disc_labels = disc_labels.masked_fill(original_input_ids == tid, -100)

        # Discriminator: Replaced Token Detection
        disc_outputs = self.discriminator(
            input_ids=replaced_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=disc_labels,
        )
        disc_loss = disc_outputs.loss

        total_loss = gen_loss + self.disc_loss_weight * disc_loss

        return {
            "loss": total_loss,
            "generator_loss": gen_loss,
            "discriminator_loss": disc_loss,
        }


# =========================================
# Main
# =========================================
def main():
    print("=" * 70)
    print("ELECTRA Pre-training")
    print("병률 - Generator + Discriminator (Replaced Token Detection)")
    print("=" * 70)
    
    print(f"\n✅ Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n설정:")
    print(f"  Data: {DATA_PATH}")
    print(f"  Tokenizer: {TOKENIZER_TYPE} -> {TOKENIZER_OPTIONS[TOKENIZER_TYPE]}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Max Length: {MAX_LENGTH}")
    
    # ========================================
    # 데이터 로드
    # ========================================
    print(f"\n{'=' * 70}")
    print("데이터 로드")
    print(f"{'=' * 70}")
    
    codenet_df = load_codenet_dataframe(DATA_PATH, SAMPLE_SIZE)
    print(f"✓ 로드 완료: {len(codenet_df):,}개")
    print(f"  Problem IDs: {codenet_df['problem_id'].nunique():,}개")
    
    # Split
    train_df, val_df, test_df = disjoint_problem_split(
        codenet_df, VAL_RATIO, TEST_RATIO, SEED
    )
    print(f"\n✓ Split 완료")
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    # ========================================
    # 토크나이저
    # ========================================
    print(f"\n{'=' * 70}")
    print("토크나이저 로드")
    print(f"{'=' * 70}")
    
    tokenizer = load_tokenizer(TOKENIZER_TYPE)
    print(f"✓ Vocab Size: {len(tokenizer):,}")
    print(f"  Special Tokens: {tokenizer.special_tokens_map}")
    
    # ========================================
    # 데이터셋 구축
    # ========================================
    print(f"\n{'=' * 70}")
    print("데이터셋 토크나이징")
    print(f"{'=' * 70}")
    
    def df_to_dataset(frame: pd.DataFrame) -> Dataset:
        data = {
            "code": frame["processed_code"].tolist(),
            "problem_id": frame["problem_id"].tolist(),
        }
        return Dataset.from_dict(data)
    
    raw_datasets = DatasetDict({
        "train": df_to_dataset(train_df),
        "validation": df_to_dataset(val_df),
    })
    if len(test_df) > 0:
        raw_datasets["test"] = df_to_dataset(test_df)
    
    def tokenize_function(batch):
        return tokenizer(
            batch["code"],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["code", "problem_id"],
    )
    
    print(f"✓ 토크나이징 완료")
    print(tokenized_datasets)
    
    # ========================================
    # Data Collator
    # ========================================
    data_collator = ElectraDataCollator(tokenizer, mlm_probability=MLM_PROB)
    
    # ========================================
    # 모델
    # ========================================
    print(f"\n{'=' * 70}")
    print("모델 초기화")
    print(f"{'=' * 70}")
    
    vocab_size = len(tokenizer)
    
    special_token_kwargs = {
        key: value
        for key, value in {
            "pad_token_id": tokenizer.pad_token_id,
            "mask_token_id": tokenizer.mask_token_id,
            "cls_token_id": tokenizer.cls_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "unk_token_id": tokenizer.unk_token_id,
        }.items()
        if value is not None
    }
    
    # Generator (작은 모델)
    generator_config = ElectraConfig(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=GEN_HIDDEN,
        num_hidden_layers=GEN_LAYERS,
        num_attention_heads=GEN_HEADS,
        intermediate_size=GEN_INTERMEDIATE,
        max_position_embeddings=MAX_LENGTH,
        **special_token_kwargs,
    )
    
    # Discriminator (큰 모델)
    discriminator_config = ElectraConfig(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=DISC_HIDDEN,
        num_hidden_layers=DISC_LAYERS,
        num_attention_heads=DISC_HEADS,
        intermediate_size=DISC_INTERMEDIATE,
        max_position_embeddings=MAX_LENGTH,
        **special_token_kwargs,
    )
    
    model = ElectraPretrainingModel(
        generator_config,
        discriminator_config,
        disc_loss_weight=DISC_LOSS_WEIGHT
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 모델 생성 완료")
    print(f"  총 파라미터: {total_params / 1e6:.2f}M")
    print(f"\n  Generator:")
    print(f"    Hidden: {GEN_HIDDEN}, Layers: {GEN_LAYERS}, Heads: {GEN_HEADS}")
    print(f"  Discriminator:")
    print(f"    Hidden: {DISC_HIDDEN}, Layers: {DISC_LAYERS}, Heads: {DISC_HEADS}")
    
    # ========================================
    # Training Arguments
    # ========================================
    print(f"\n{'=' * 70}")
    print("학습 설정")
    print(f"{'=' * 70}")
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=100,
        eval_steps=1000,
        save_steps=2000,
        save_strategy="steps",
        save_total_limit=3,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_steps=MAX_STEPS,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        tf32=True,
        dataloader_num_workers=8,
        save_safetensors=True,
        report_to=[],
        eval_strategy="steps",
        logging_strategy="steps",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    print(f"✓ 학습 설정 완료")
    print(f"  Max Steps: {MAX_STEPS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Grad Accum: {GRAD_ACCUM}")
    print(f"  Learning Rate: {LR}")
    print(f"  Warmup Ratio: {WARMUP_RATIO}")
    
    # ========================================
    # Trainer
    # ========================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    # ========================================
    # 학습
    # ========================================
    print(f"\n{'=' * 70}")
    print("학습 시작")
    print(f"{'=' * 70}")
    
    train_result = trainer.train()
    
    # ========================================
    # 저장
    # ========================================
    print(f"\n{'=' * 70}")
    print("모델 저장")
    print(f"{'=' * 70}")
    
    trainer.save_model(str(OUTPUT_DIR / "model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "tokenizer"))
    
    print(f"✓ 모델 저장: {OUTPUT_DIR / 'model'}")
    print(f"✓ 토크나이저 저장: {OUTPUT_DIR / 'tokenizer'}")
    
    # ========================================
    # 평가
    # ========================================
    print(f"\n{'=' * 70}")
    print("최종 평가")
    print(f"{'=' * 70}")
    
    metrics = trainer.evaluate(tokenized_datasets["validation"])
    print(f"✓ Eval Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    
    print(f"\n{'=' * 70}")
    print("✓ 모든 작업 완료!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
