import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from transformers import (
    PreTrainedTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# =============================
# 설정
# =============================
SEED = 42

# 경로
BASE_DIR = Path("./")
DATA_PATH = BASE_DIR / "data" / "codenet_python_dataset.parquet"
TOKENIZER_DIR = BASE_DIR / "data" / "tokenizers"
OUTPUT_DIR = BASE_DIR / "models" / "custom_bert"

# 토크나이저 선택
TOKENIZER_OPTIONS = {
    "bpe": TOKENIZER_DIR / "bpe_tokenizer.json",
    "wordpiece": TOKENIZER_DIR / "wordpiece_tokenizer.json",
    "unigram": TOKENIZER_DIR / "unigram.model",
}
TOKENIZER_TYPE = "bpe"  # 'bpe', 'wordpiece', 또는 'unigram'

# 데이터 설정
SAMPLE_SIZE = None  # None: 전체 사용, 정수: 샘플 수
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_LENGTH = 128

# 학습 설정
BATCH_SIZE = 32
GRAD_ACCUM = 4
NUM_EPOCHS = 10
LR = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MLM_PROB = 0.15

# 모델 설정
HIDDEN_SIZE = 512
NUM_LAYERS = 12
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048

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


def stratified_split(
    frame: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int
):
    """
    문제 ID 기반 Stratified Split
    """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1
    assert (val_ratio + test_ratio) < 1
    
    val_test_ratio = val_ratio + test_ratio

    # Stratify label
    strat_labels = None
    if "problem_id" in frame.columns:
        problem_counts = frame["problem_id"].value_counts()
        if frame["problem_id"].nunique() > 1 and problem_counts.min() >= 2:
            strat_labels = frame["problem_id"]

    # Train / Temp split
    try:
        train_df, temp_df = train_test_split(
            frame,
            test_size=val_test_ratio,
            stratify=strat_labels,
            random_state=seed,
            shuffle=True,
        )
    except ValueError:
        print("Stratified split 실패 -> random split")
        train_df, temp_df = train_test_split(
            frame,
            test_size=val_test_ratio,
            stratify=None,
            random_state=seed,
            shuffle=True,
        )

    # Val / Test split
    strat_temp = None
    if strat_labels is not None:
        temp_counts = temp_df["problem_id"].value_counts()
        if temp_df["problem_id"].nunique() > 1 and temp_counts.min() >= 2:
            strat_temp = temp_df["problem_id"]

    test_fraction = test_ratio / val_test_ratio
    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_fraction,
            stratify=strat_temp,
            random_state=seed,
            shuffle=True,
        )
    except ValueError:
        print("2차 Stratified split 실패 -> random split")
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_fraction,
            stratify=None,
            random_state=seed,
            shuffle=True,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def load_tokenizer(tokenizer_key: str) -> PreTrainedTokenizerFast:
    """토크나이저 로드"""
    target_path = TOKENIZER_OPTIONS[tokenizer_key]
    suffix = target_path.suffix.lower()
    
    special_tokens = {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }
    
    if suffix == ".json":
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(target_path),
            **special_tokens
        )
    elif suffix == ".model":
        tokenizer_obj = Tokenizer.from_file(str(target_path))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            **special_tokens
        )
    else:
        raise ValueError(f"지원하지 않는 확장자: {suffix}")

    # Special tokens 추가
    missing_specials = [
        tok for tok in special_tokens.values()
        if tok not in tokenizer.get_vocab()
    ]
    if missing_specials:
        tokenizer.add_special_tokens({
            "additional_special_tokens": missing_specials
        })

    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.padding_side = "right"
    
    return tokenizer


# =========================================
# Main
# =========================================
def main():
    print("=" * 70)
    print("Custom BERT Pre-training (MLM)")
    print("병률 - BPE/WordPiece/Unigram + BERT")
    print("=" * 70)
    
    print(f"\n✅ Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n설정:")
    print(f"  Data: {DATA_PATH}")
    print(f"  Tokenizer: {TOKENIZER_TYPE} -> {TOKENIZER_OPTIONS[TOKENIZER_TYPE]}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Sample Size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full'}")
    print(f"  Max Length: {MAX_LENGTH}")
    
    # ========================================
    # 데이터 로드
    # ========================================
    print(f"\n{'=' * 70}")
    print("데이터 로드")
    print(f"{'=' * 70}")
    
    try:
        df = pd.read_parquet(DATA_PATH)
        initial_count = len(df)
        print(f"✓ 로드 완료: {initial_count:,}개 샘플")
    except Exception as e:
        print(f"✗ 로드 실패: {e}")
        return
    
    # Accepted만 필터
    if "status" in df.columns:
        df = df.loc[df["status"] == "Accepted"].copy()
        df = df.drop(columns=["status"], errors="ignore")
        print(f"  Accepted: {len(df):,}개")
    
    # 텍스트 컬럼 확인
    text_col = None
    for col in ["processed_code", "code", "text"]:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print("✗ 텍스트 컬럼을 찾을 수 없습니다")
        return
    
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    
    # problem_id 확인
    if "problem_id" not in df.columns:
        df["problem_id"] = df.index.astype(str)
    
    print(f"  텍스트 컬럼: {text_col}")
    print(f"  Problem IDs: {df['problem_id'].nunique():,}개")
    
    # 샘플링
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
        print(f"  샘플링: {len(df):,}개")
    
    df = df.reset_index(drop=True)
    
    # Split
    train_df, val_df, test_df = stratified_split(df, VAL_RATIO, TEST_RATIO, SEED)
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
    
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })
    
    def tokenize_function(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=MAX_LENGTH,
        )
    
    remove_columns = dataset_dict["train"].column_names
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns,
    )
    
    print(f"✓ 토크나이징 완료")
    print(tokenized_datasets)
    
    # 샘플 확인
    sample_ids = tokenized_datasets["train"][0]["input_ids"][:40]
    print(f"\n샘플 디코딩: {tokenizer.decode(sample_ids)}")
    
    # ========================================
    # 모델
    # ========================================
    print(f"\n{'=' * 70}")
    print("모델 초기화")
    print(f"{'=' * 70}")
    
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_LENGTH,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    model = BertForMaskedLM(config)
    model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 모델 생성 완료")
    print(f"  파라미터: {total_params / 1e6:.2f}M")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Heads: {NUM_HEADS}")
    
    # ========================================
    # Data Collator
    # ========================================
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=MLM_PROB,
        pad_to_multiple_of=8,
    )
    
    # ========================================
    # Training Arguments
    # ========================================
    print(f"\n{'=' * 70}")
    print("학습 설정")
    print(f"{'=' * 70}")
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        logging_steps=2000,
        save_steps=5000,
        save_strategy="epoch",
        save_total_limit=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        bf16=torch.cuda.is_available(),
        tf32=True,
        report_to=[],
        seed=SEED,
        dataloader_num_workers=4,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
    )
    
    print(f"✓ 학습 설정 완료")
    print(f"  Epochs: {NUM_EPOCHS}")
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
