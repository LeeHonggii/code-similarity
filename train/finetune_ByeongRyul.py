"""
Fine-tuning for Code Similarity Classification
병률 - Sequence Classification (Binary)

Usage:
    python train/finetune_similarity_ByeongRyul.py
"""

import os
import ast
import json
import random
from pathlib import Path
from datetime import datetime
from itertools import combinations
from functools import lru_cache
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
    ElectraConfig,
    ElectraForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =============================
# 설정
# =============================
SEED = 42

# 경로
BASE_DIR = Path("./")
CODE_ROOT = BASE_DIR / "data" / "code"  # 코드 파일 디렉토리
TRAIN_CSV_PATH = BASE_DIR / "data" / "train.csv"
TEST_CSV_PATH = BASE_DIR / "data" / "test.csv"
OUTPUT_DIR = BASE_DIR / "models" / "finetuned"
CACHE_PARQUET_PATH = OUTPUT_DIR / "train_pairs.parquet"

# 데이터 생성 설정
USE_CODE_DIRECTORY = False  # True: 코드 디렉토리에서 페어 생성, False: CSV 사용
MAX_PROBLEMS = None
MAX_FILES_PER_PROBLEM = None
MAX_POSITIVE_PAIRS_PER_PROBLEM = 250
NEGATIVE_PAIR_RATIO = 1.0

# 학습 설정
VAL_RATIO = 0.1
MAX_LENGTH = 512
THRESHOLD = 0.5

# 모델 선택
SELECTED_MODELS = [
    # "custom_bert",
    # "bert_base_uncased",
    # "codebert_base",
    "custom_electra",
]

# 모델 레지스트리
MODEL_REGISTRY = {
    "custom_bert": {
        "display_name": "custom_bert",
        "model_path": "./models/custom_bert/model",
        "tokenizer_path": "./models/custom_bert/tokenizer",
        "is_custom_bert": True,
    },
    "bert_base_uncased": {
        "display_name": "bert-base-uncased",
        "model_path": "google-bert/bert-base-uncased",
        "tokenizer_path": "google-bert/bert-base-uncased",
        "is_custom_bert": False,
    },
    "codebert_base": {
        "display_name": "codebert-base",
        "model_path": "microsoft/codebert-base",
        "tokenizer_path": "microsoft/codebert-base",
        "is_custom_bert": False,
    },
    "custom_electra": {
        "display_name": "custom_electra",
        "model_path": "./models/custom_electra/model",
        "tokenizer_path": "./models/custom_electra/tokenizer",
        "is_custom_bert": False,
    },
}

# 학습 하이퍼파라미터
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 128,
    "per_device_eval_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "eval_steps": 50,
    "save_steps": 50,
    "save_total_limit": 5,
    "lr_scheduler_type": "cosine",
    "evaluation_strategy": "steps",
    "logging_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "dataloader_num_workers": 8,
    "fp16": True,
    "bf16": False,
    "gradient_checkpointing": True,
    "report_to": [],
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


# =========================================
# 코드 전처리
# =========================================
class CodePreprocessor:
    """주석 제거, Docstring 제거, 공백 정리"""
    
    def __init__(self, remove_docstrings: bool = True, normalize_whitespace: bool = True):
        self.remove_docstrings = remove_docstrings
        self.normalize_whitespace = normalize_whitespace

    def remove_comments(self, code: str) -> str:
        """주석 제거"""
        lines = code.split("\n")
        cleaned = []
        for line in lines:
            in_string = False
            string_char = ""
            buffer = []
            i = 0
            while i < len(line):
                ch = line[i]
                if not in_string and ch in ("'", '"'):
                    in_string = True
                    string_char = ch
                    buffer.append(ch)
                elif in_string and ch == string_char:
                    if i == 0 or line[i - 1] != '\\':
                        in_string = False
                        string_char = ""
                    buffer.append(ch)
                elif not in_string and ch == '#':
                    break
                else:
                    buffer.append(ch)
                i += 1
            cleaned.append("".join(buffer).rstrip())
        return "\n".join(cleaned)

    def remove_docstrings_ast(self, code: str) -> str:
        """AST 기반 Docstring 제거"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        node.body = node.body[1:]
                if isinstance(node, ast.Module):
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        node.body = node.body[1:]
            return ast.unparse(tree)
        except Exception:
            return code

    def normalize_whitespace_text(self, code: str) -> str:
        """공백 정리"""
        lines = []
        for line in code.split("\n"):
            line = line.expandtabs(4).rstrip()
            if line.strip():
                lines.append(line)
        return "\n".join(lines)

    def preprocess(self, code: str) -> str:
        """전처리 파이프라인"""
        processed = self.remove_comments(code)
        if self.remove_docstrings:
            processed = self.remove_docstrings_ast(processed)
        if self.normalize_whitespace:
            processed = self.normalize_whitespace_text(processed)
        return processed


preprocessor = CodePreprocessor(remove_docstrings=True, normalize_whitespace=True)


@lru_cache(maxsize=None)
def preprocess_cached(text: str) -> str:
    """캐싱된 전처리"""
    return preprocessor.preprocess(text)


# =========================================
# 데이터 로딩
# =========================================
def read_code_file(path: Path) -> str:
    """코드 파일 읽기"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def load_code_repository(
    code_root: Path,
    max_problems=None,
    max_files_per_problem=None
) -> pd.DataFrame:
    """코드 리포지토리에서 데이터 로드"""
    problem_dirs = sorted([p for p in code_root.iterdir() if p.is_dir()])
    if max_problems is not None:
        problem_dirs = problem_dirs[:max_problems]
    
    records = []
    for problem_dir in problem_dirs:
        files = sorted(problem_dir.glob("*.py"))
        if max_files_per_problem is not None:
            files = files[:max_files_per_problem]
        
        for file_path in files:
            code_text = read_code_file(file_path)
            processed_text = preprocess_cached(code_text)
            records.append({
                "problem_id": problem_dir.name,
                "file_path": str(file_path),
                "code": code_text,
                "processed_code": processed_text,
            })
    
    df = pd.DataFrame(records)
    if not df.empty:
        print(f"✓ {len(df):,}개 코드 로드 ({df['problem_id'].nunique()}개 문제)")
    return df


def build_pairs_from_repository(
    snippets: pd.DataFrame,
    max_positive_pairs_per_problem: int,
    negative_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """코드 페어 생성 (positive + negative)"""
    rng = random.Random(seed)
    positive_examples = []
    
    # Positive pairs (같은 문제)
    for problem_id, group in snippets.groupby("problem_id"):
        processed_list = group["processed_code"].tolist()
        source_list = group["file_path"].tolist()
        indices = list(range(len(processed_list)))
        combos = list(combinations(indices, 2))
        rng.shuffle(combos)
        
        if max_positive_pairs_per_problem is not None:
            combos = combos[:max_positive_pairs_per_problem]
        
        for i_idx, j_idx in combos:
            positive_examples.append({
                "code1_processed": processed_list[i_idx],
                "code2_processed": processed_list[j_idx],
                "source1": source_list[i_idx],
                "source2": source_list[j_idx],
                "similar": 1,
            })
    
    print(f"  Positive pairs: {len(positive_examples):,}")

    # Negative pairs (다른 문제)
    negative_examples = []
    all_records = snippets[["processed_code", "file_path", "problem_id"]].reset_index(drop=True)
    negatives_needed = int(len(positive_examples) * negative_ratio)
    
    while len(negative_examples) < negatives_needed and len(all_records) >= 2:
        i_idx, j_idx = rng.sample(range(len(all_records)), 2)
        if all_records.loc[i_idx, "problem_id"] == all_records.loc[j_idx, "problem_id"]:
            continue
        negative_examples.append({
            "code1_processed": all_records.loc[i_idx, "processed_code"],
            "code2_processed": all_records.loc[j_idx, "processed_code"],
            "source1": all_records.loc[i_idx, "file_path"],
            "source2": all_records.loc[j_idx, "file_path"],
            "similar": 0,
        })
    
    print(f"  Negative pairs: {len(negative_examples):,}")

    all_pairs = positive_examples + negative_examples
    rng.shuffle(all_pairs)
    return pd.DataFrame(all_pairs)


def load_pairs_from_csv(csv_path: Path) -> pd.DataFrame:
    """CSV에서 페어 로드"""
    df = pd.read_csv(csv_path)
    df["code1"] = df["code1"].fillna("").astype(str)
    df["code2"] = df["code2"].fillna("").astype(str)
    df["similar"] = df["similar"].astype(int)
    df["code1_processed"] = df["code1"].map(preprocess_cached)
    df["code2_processed"] = df["code2"].map(preprocess_cached)
    df["source1"] = "train_csv"
    df["source2"] = "train_csv"
    return df[["code1_processed", "code2_processed", "similar", "source1", "source2"]]


def build_training_dataframe() -> pd.DataFrame:
    """학습 데이터 구축"""
    if USE_CODE_DIRECTORY:
        if CACHE_PARQUET_PATH.exists():
            pairs_df = pd.read_parquet(CACHE_PARQUET_PATH)
            print(f"✓ 캐시에서 로드: {CACHE_PARQUET_PATH}")
        else:
            snippets = load_code_repository(
                CODE_ROOT,
                max_problems=MAX_PROBLEMS,
                max_files_per_problem=MAX_FILES_PER_PROBLEM,
            )
            pairs_df = build_pairs_from_repository(
                snippets,
                max_positive_pairs_per_problem=MAX_POSITIVE_PAIRS_PER_PROBLEM,
                negative_ratio=NEGATIVE_PAIR_RATIO,
                seed=SEED,
            )
            if not pairs_df.empty:
                pairs_df.to_parquet(CACHE_PARQUET_PATH, index=False)
                print(f"✓ 캐시 저장: {CACHE_PARQUET_PATH}")
    else:
        pairs_df = load_pairs_from_csv(TRAIN_CSV_PATH)
        print(f"✓ CSV 로드: {len(pairs_df):,}개")

    pairs_df = pairs_df.drop_duplicates(
        subset=["code1_processed", "code2_processed", "similar"]
    ).reset_index(drop=True)
    
    print(f"✓ 총 페어: {len(pairs_df):,}개 (중복 제거 후)")
    return pairs_df


# =========================================
# 모델 로딩
# =========================================
def load_model_bundle(model_key: str):
    """모델 + 토크나이저 로드"""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_key}")
    
    meta = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer_path"])

    if meta.get("is_custom_bert", False):
        bert_config = BertConfig.from_pretrained(meta["model_path"])
        bert_config.num_labels = 2
        model = BertForSequenceClassification.from_pretrained(
            meta["model_path"],
            config=bert_config,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            meta["model_path"],
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model, meta


# =========================================
# 데이터셋
# =========================================
def df_to_dataset(frame: pd.DataFrame) -> Dataset:
    """DataFrame → HuggingFace Dataset"""
    export_df = frame[["code1_processed", "code2_processed", "similar"]].rename(
        columns={
            "code1_processed": "text1",
            "code2_processed": "text2",
            "similar": "labels"
        }
    )
    return Dataset.from_pandas(export_df, preserve_index=False)


def tokenize_dataset_dict(
    dataset_dict: DatasetDict,
    tokenizer,
    max_length: int,
    return_token_type_ids: bool,
) -> DatasetDict:
    """데이터셋 토크나이징"""
    def _tokenize(batch):
        return tokenizer(
            batch["text1"],
            batch["text2"],
            truncation=True,
            max_length=max_length,
            return_token_type_ids=return_token_type_ids,
        )

    tokenized = DatasetDict()
    for split, dataset in dataset_dict.items():
        remove_cols = [col for col in dataset.column_names if col in ["text1", "text2"]]
        tokenized[split] = dataset.map(
            _tokenize,
            batched=True,
            remove_columns=remove_cols,
        )
    return tokenized


def tokenize_test_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    return_token_type_ids: bool,
) -> Dataset:
    """테스트 데이터셋 토크나이징"""
    def _tokenize(batch):
        return tokenizer(
            batch["text1"],
            batch["text2"],
            truncation=True,
            max_length=max_length,
            return_token_type_ids=return_token_type_ids,
        )
    
    remove_cols = [col for col in dataset.column_names if col in ("text1", "text2")]
    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=remove_cols,
    )


# =========================================
# 평가
# =========================================
def compute_metrics(eval_pred):
    """평가 메트릭"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def prepare_training_arguments(output_dir: Path, run_name: str) -> TrainingArguments:
    """TrainingArguments 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs = dict(TRAINING_CONFIG)
    training_kwargs.update({
        "output_dir": str(output_dir),
        "run_name": run_name,
        "seed": SEED,
        "overwrite_output_dir": True,
    })
    return TrainingArguments(**training_kwargs)


# =========================================
# 추론
# =========================================
def predict_and_export(
    trainer: Trainer,
    tokenizer,
    raw_test_ds: Dataset,
    pair_ids: List[int],
    model_label: str,
    return_token_type_ids: bool,
    max_length: int,
) -> Tuple[Path, Dict[str, Any], np.ndarray]:
    """테스트 추론 및 제출 파일 생성"""
    tokenized_test = tokenize_test_dataset(
        raw_test_ds,
        tokenizer,
        max_length=max_length,
        return_token_type_ids=return_token_type_ids,
    )
    
    predictions = trainer.predict(tokenized_test)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    positives = (probs[:, 1] >= THRESHOLD).astype(int)
    
    submission = pd.DataFrame({"pair_id": pair_ids, "similar": positives})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"submission_{model_label}_{timestamp}.csv"
    submission.to_csv(output_path, index=False)
    
    return output_path, predictions.metrics, probs[:, 1]


# =========================================
# Main
# =========================================
def main():
    print("=" * 70)
    print("Fine-tuning for Code Similarity Classification")
    print("병률 - Sequence Classification (Binary)")
    print("=" * 70)
    
    print(f"\n✅ Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n설정:")
    print(f"  Train CSV: {TRAIN_CSV_PATH}")
    print(f"  Test CSV: {TEST_CSV_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Models: {SELECTED_MODELS}")
    
    # ========================================
    # 데이터 로드
    # ========================================
    print(f"\n{'=' * 70}")
    print("학습 데이터 구축")
    print(f"{'=' * 70}")
    
    pairs_df = build_training_dataframe()
    
    # Split
    stratify_labels = pairs_df["similar"] if pairs_df["similar"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        pairs_df,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=stratify_labels,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"\n✓ Split 완료")
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}")
    print(f"  Train similar=1: {(train_df['similar'] == 1).sum():,}")
    print(f"  Val similar=1: {(val_df['similar'] == 1).sum():,}")
    
    raw_datasets = DatasetDict({
        "train": df_to_dataset(train_df),
        "validation": df_to_dataset(val_df),
    })
    
    # 테스트 데이터
    print(f"\n{'=' * 70}")
    print("테스트 데이터 로드")
    print(f"{'=' * 70}")
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df["code1"] = test_df["code1"].fillna("").astype(str)
    test_df["code2"] = test_df["code2"].fillna("").astype(str)
    test_df["code1_processed"] = test_df["code1"].map(preprocess_cached)
    test_df["code2_processed"] = test_df["code2"].map(preprocess_cached)
    test_pair_ids = test_df["pair_id"].tolist()
    
    raw_test_dataset = Dataset.from_pandas(
        test_df[["code1_processed", "code2_processed"]].rename(
            columns={"code1_processed": "text1", "code2_processed": "text2"}
        ),
        preserve_index=False,
    )
    
    print(f"✓ 테스트: {len(raw_test_dataset):,}개")
    
    # ========================================
    # 모델별 학습
    # ========================================
    experiment_log = []
    trained_models = {}
    
    for model_key in SELECTED_MODELS:
        print(f"\n{'=' * 70}")
        print(f"학습 시작: {model_key}")
        print(f"{'=' * 70}")
        
        tokenizer, model, meta = load_model_bundle(model_key)
        model_label = meta["display_name"]
        
        return_token_type_ids = getattr(model.config, "type_vocab_size", 0) > 1
        model_max_len = getattr(model.config, "max_position_embeddings", MAX_LENGTH)
        effective_max_len = min(MAX_LENGTH, model_max_len)
        
        if effective_max_len < MAX_LENGTH:
            print(f"⚠️ Max length: {MAX_LENGTH} → {effective_max_len}")
        
        # 토크나이징
        tokenized_ds = tokenize_dataset_dict(
            raw_datasets,
            tokenizer,
            effective_max_len,
            return_token_type_ids=return_token_type_ids,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        
        # Training Arguments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"finetune-{model_label}-{timestamp}"
        training_args = prepare_training_arguments(
            OUTPUT_DIR / f"runs_{model_label}_{timestamp}",
            run_name
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # 학습
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        
        print(f"\n✓ 학습 완료: {model_label}")
        print(f"  Eval Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        print(f"  Accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
        print(f"  F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
        
        experiment_log.append({
            "model_key": model_key,
            "model_label": model_label,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "return_token_type_ids": return_token_type_ids,
            "max_length": effective_max_len,
        })
        
        trained_models[model_key] = {
            "trainer": trainer,
            "tokenizer": tokenizer,
            "model_label": model_label,
            "return_token_type_ids": return_token_type_ids,
            "max_length": effective_max_len,
        }
    
    # ========================================
    # 추론
    # ========================================
    print(f"\n{'=' * 70}")
    print("테스트 추론")
    print(f"{'=' * 70}")
    
    submission_records = []
    
    for model_key, payload in trained_models.items():
        trainer = payload["trainer"]
        tokenizer = payload["tokenizer"]
        model_label = payload["model_label"]
        return_token_type_ids = payload["return_token_type_ids"]
        effective_max_len = payload["max_length"]
        
        output_path, predict_metrics, probas = predict_and_export(
            trainer,
            tokenizer,
            raw_test_dataset,
            test_pair_ids,
            model_label,
            return_token_type_ids=return_token_type_ids,
            max_length=effective_max_len,
        )
        
        print(f"✓ 제출 파일: {output_path}")
        print(f"  Positive rate: {np.mean(probas >= THRESHOLD) * 100:.1f}%")
        
        submission_records.append({
            "model_key": model_key,
            "model_label": model_label,
            "submission_path": str(output_path),
            "predict_metrics": predict_metrics,
            "positive_rate": float(np.mean(probas >= THRESHOLD)),
            "max_length": effective_max_len,
        })
    
    # ========================================
    # 요약
    # ========================================
    print(f"\n{'=' * 70}")
    print("실험 요약")
    print(f"{'=' * 70}")
    
    summary_df = pd.DataFrame([
        {
            "model": row["model_label"],
            "max_length": row.get("max_length"),
            **{f"eval_{k}": v for k, v in row["eval_metrics"].items()}
        }
        for row in experiment_log
    ])
    print(summary_df.to_string(index=False))
    
    print(f"\n{'=' * 70}")
    print("✓ 모든 작업 완료!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
