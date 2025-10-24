"""
RoBERTa-small Fine-tuning for Code Similarity
황호성 - 코드 유사도 분류 Fine-tuning
- 프리트레이닝된 RoBERTa-small 사용
- 로컬 모델 또는 허깅페이스 모델 선택 가능

Usage:
    # 로컬 프리트레이닝 모델 사용
    python train/finetune_roberta_HwangHosung.py --model local
    
    # 허깅페이스 프리트레이닝 모델 사용
    python train/finetune_roberta_HwangHosung.py --model huggingface
    
    # 허깅페이스 파인튜닝 모델 로드만
    python train/finetune_roberta_HwangHosung.py --model huggingface --mode load
"""

import pandas as pd
import numpy as np
import torch
import os
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


# ========================================
# SafeCollator (토큰 범위 검증)
# ========================================
class SafeCollator:
    """
    토큰 ID 범위를 검증하고 token_type_ids를 제거하는 안전한 Collator
    RoBERTa는 token_type_ids를 사용하지 않음
    """
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.inner = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features):
        batch = self.inner(features)
        
        # RoBERTa: token_type_ids 사용 안 함 (섞여 있으면 out-of-range 유발)
        batch.pop("token_type_ids", None)

        # 토큰 id 범위 검증
        ids = batch["input_ids"]
        v = self.tokenizer.vocab_size
        if torch.any(ids.ge(v)) or torch.any(ids.lt(0)):
            bad = torch.nonzero((ids>=v) | (ids<0), as_tuple=False)[:10].tolist()
            mx, mn = int(ids.max()), int(ids.min())
            raise RuntimeError(
                f"out-of-range token id in batch: min={mn}, max={mx}, "
                f"vocab={v}, bad_positions={bad}"
            )
        return batch


# ========================================
# 데이터 전처리
# ========================================
def preprocess_function(batch, tokenizer, max_len=512):
    """코드 페어를 토크나이징"""
    c1 = [x for x in batch['code1']]
    c2 = [x for x in batch['code2']]
    enc = tokenizer(
        c1, c2,
        max_length=max_len,
        padding="max_length",
        truncation=True
    )
    enc['labels'] = batch['similar']
    return enc


def minmax_ids(tokenized_ds, split):
    """토큰 ID 범위 확인"""
    arr = tokenized_ds[split]["input_ids"]
    mn = min(min(row) for row in arr)
    mx = max(max(row) for row in arr)
    return mn, mx, len(arr[0])


# ========================================
# 메인 함수
# ========================================
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="RoBERTa-small Fine-tuning for Code Similarity")
    parser.add_argument(
        "--model",
        type=str,
        choices=["local", "huggingface"],
        default="local",
        help="local: 로컬 프리트레이닝 모델, huggingface: 허깅페이스 모델"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load"],
        default="train",
        help="train: 파인튜닝 학습, load: 파인튜닝 모델 로드만"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("RoBERTa-small Fine-tuning - 코드 유사도 분류")
    print("황호성 - Unigram Tokenizer")
    print("=" * 70)
    
    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # ========================================
    # Mode: Load (파인튜닝 모델만 로드)
    # ========================================
    if args.mode == "load":
        print(f"\n{'='*70}")
        print("허깅페이스 파인튜닝 모델 로드")
        print(f"{'='*70}")
        
        HF_MODEL = "hosung1/code-sim-roberta-small"
        
        try:
            print(f"\n모델 로드 중: {HF_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
            
            print(f"✓ 파인튜닝 모델 로드 완료!")
            print(f"  Vocab Size: {len(tokenizer):,}")
            print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Num Labels: {model.config.num_labels}")
            
            print(f"\n사용법:")
            print(f"  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
            print(f"  tokenizer = AutoTokenizer.from_pretrained('{HF_MODEL}')")
            print(f"  model = AutoModelForSequenceClassification.from_pretrained('{HF_MODEL}')")
            
            # 테스트
            code1 = "def add(a, b):\n    return a + b"
            code2 = "def sum(x, y):\n    return x + y"
            
            inputs = tokenizer(code1, code2, return_tensors="pt")
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            
            print(f"\n테스트 예측:")
            print(f"  Code1: {code1}")
            print(f"  Code2: {code2}")
            print(f"  예측: {'유사' if pred == 1 else '다름'} (label={pred})")
            
        except Exception as e:
            print(f"✗ 모델 로드 실패: {e}")
        
        return
    
    # ========================================
    # Mode: Train (파인튜닝 학습)
    # ========================================
    print(f"\n{'='*70}")
    print("파인튜닝 학습 모드")
    print(f"{'='*70}")
    
    # 설정
    CSV_PATH = "./data/train_pairs.csv"
    MAX_LEN = 512
    EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5
    
    # 모델/토크나이저 경로 설정
    if args.model == "local":
        TOKENIZER_DIR = "./models/roberta_small_mlm"
        MODEL_DIR = "./models/roberta_small_mlm"
        OUTPUT_DIR = "./models/roberta_small_finetune"
        print(f"\n로컬 프리트레이닝 모델 사용")
    else:  # huggingface
        TOKENIZER_DIR = "hosung1/roberta_small_mlm_from_scratch"
        MODEL_DIR = "hosung1/roberta_small_mlm_from_scratch"
        OUTPUT_DIR = "./models/roberta_small_finetune_hf"
        print(f"\n허깅페이스 프리트레이닝 모델 사용")
    
    print(f"\n설정:")
    print(f"  데이터: {CSV_PATH}")
    print(f"  토크나이저: {TOKENIZER_DIR}")
    print(f"  프리트레이닝 모델: {MODEL_DIR}")
    print(f"  출력: {OUTPUT_DIR}")
    print(f"  Max Length: {MAX_LEN}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ========================================
    # 1. 데이터 로드
    # ========================================
    print(f"\n{'='*70}")
    print("학습 데이터 로드")
    print(f"{'='*70}")
    
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ 데이터 로드 완료: {len(df):,}개 샘플")
        print(f"  컬럼: {df.columns.tolist()}")
        print(f"\nLabel 분포:")
        print(df['similar'].value_counts())
    except FileNotFoundError:
        print(f"✗ 데이터 파일을 찾을 수 없습니다: {CSV_PATH}")
        return
    except Exception as e:
        print(f"✗ 데이터 로드 실패: {e}")
        return
    
    # ========================================
    # 2. 토크나이저 로드
    # ========================================
    print(f"\n{'='*70}")
    print("토크나이저 로드")
    print(f"{'='*70}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        # Pair classification: 끝에서부터 truncate
        tokenizer.truncation_side = "left"
        
        print(f"✓ 토크나이저 로드 완료: {TOKENIZER_DIR}")
        print(f"  Vocab Size: {len(tokenizer):,}")
        print(f"  Special Tokens: {tokenizer.special_tokens_map}")
        print(f"  Truncation Side: {tokenizer.truncation_side}")
    except Exception as e:
        print(f"✗ 토크나이저 로드 실패: {e}")
        return
    
    # ========================================
    # 3. 데이터 분할 & 전처리
    # ========================================
    print(f"\n{'='*70}")
    print("데이터 분할 & 전처리")
    print(f"{'='*70}")
    
    train_df, valid_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df['similar']
    )
    
    print(f"✓ 데이터 분할 완료")
    print(f"  Train: {len(train_df):,}개")
    print(f"  Val: {len(valid_df):,}개")
    
    # Dataset 생성
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))
    raw_ds = DatasetDict(train=train_ds, validation=valid_ds)
    
    # 토크나이징
    print(f"\n토크나이징 중...")
    tokenized_ds = raw_ds.map(
        lambda batch: preprocess_function(batch, tokenizer, MAX_LEN),
        batched=True,
        remove_columns=['code1', 'code2', 'similar'],
        desc="Tokenizing"
    )
    
    print(f"✓ 토크나이징 완료")
    
    # 토큰 범위 검증
    print(f"\n토큰 ID 범위 확인:")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  cls_token_id: {tokenizer.cls_token_id}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    
    mn_tr, mx_tr, L_tr = minmax_ids(tokenized_ds, "train")
    mn_va, mx_va, L_va = minmax_ids(tokenized_ds, "validation")
    print(f"  [train] min/max/length = {mn_tr}/{mx_tr}/{L_tr}")
    print(f"  [valid] min/max/length = {mn_va}/{mx_va}/{L_va}")
    
    # ========================================
    # 4. 모델 로드
    # ========================================
    print(f"\n{'='*70}")
    print("모델 로드")
    print(f"{'='*70}")
    
    try:
        num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            num_labels=num_labels
        )
        
        print(f"✓ 모델 로드 완료: {MODEL_DIR}")
        print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Num Labels: {num_labels}")
        print(f"  Max Position Embeddings: {getattr(model.config, 'max_position_embeddings', None)}")
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        return
    
    # ========================================
    # 5. 평가 메트릭 설정
    # ========================================
    print(f"\n{'='*70}")
    print("평가 메트릭 설정")
    print(f"{'='*70}")
    
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
    
    print(f"✓ 평가 메트릭: Accuracy, F1 (macro)")
    
    # ========================================
    # 6. Training Arguments
    # ========================================
    print(f"\n{'='*70}")
    print("학습 설정")
    print(f"{'='*70}")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        logging_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        optim="adamw_torch" if not torch.cuda.is_available() else "adamw_torch_fused",
        adam_epsilon=1e-6,
        report_to=[],
    )
    
    print(f"✓ Training Arguments 설정 완료")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  FP16: {training_args.fp16}")
    
    # ========================================
    # 7. Trainer 설정
    # ========================================
    print(f"\n{'='*70}")
    print("Trainer 설정")
    print(f"{'='*70}")
    
    safe_collator = SafeCollator(tokenizer, pad_to_multiple_of=8)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        tokenizer=tokenizer,
        data_collator=safe_collator,
        compute_metrics=compute_metrics,
    )
    
    print(f"✓ Trainer 설정 완료")
    
    # ========================================
    # 8. 학습 시작
    # ========================================
    print(f"\n{'='*70}")
    print("파인튜닝 시작")
    print(f"{'='*70}")
    
    train_result = trainer.train()
    
    # ========================================
    # 9. 평가
    # ========================================
    print(f"\n{'='*70}")
    print("최종 평가")
    print(f"{'='*70}")
    
    metrics = trainer.evaluate()
    
    print(f"\n평가 결과:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # ========================================
    # 10. 모델 저장
    # ========================================
    print(f"\n{'='*70}")
    print("모델 저장")
    print(f"{'='*70}")
    
    best_dir = Path(OUTPUT_DIR) / "best"
    best_dir.mkdir(exist_ok=True, parents=True)
    
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    
    print(f"✓ Best 모델 저장 완료: {best_dir}")
    
    # ========================================
    # 11. 완료
    # ========================================
    print(f"\n{'='*70}")
    print("✓ 모든 작업 완료!")
    print(f"{'='*70}")
    print(f"모델 저장 위치: {best_dir}")
    print(f"\n최종 평가 결과:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n허깅페이스 비교 모델:")
    print(f"  hosung1/code-sim-roberta-small")


if __name__ == "__main__":
    main()
