"""
Inference for Code Similarity Classification
병률 - Sequence Classification Inference

Usage:
    python inference/inference_similarity_ByeongRyul.py
"""

import ast
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    ElectraForSequenceClassification,
    DataCollatorWithPadding,
)
from tqdm.auto import tqdm

# =============================
# 설정
# =============================
# 경로
BASE_DIR = Path("./")
CHECKPOINT_DIR = BASE_DIR / "models" / "finetuned" / "checkpoint-best"
TEST_CSV_PATH = BASE_DIR / "data" / "test.csv"
SAMPLE_SUBMISSION_PATH = BASE_DIR / "data" / "sample_submission.csv"
OUTPUT_DIR = BASE_DIR / "inference"

# 추론 설정
BATCH_SIZE = 256
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================
# 코드 전처리
# =========================================
class CodePreprocessor:
    """주석 제거, Docstring 제거, 공백 정리"""
    
    def __init__(self, remove_docstrings=True, normalize_whitespace=True):
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
                    if i == 0 or line[i - 1] != "\\":
                        in_string = False
                        string_char = ""
                    buffer.append(ch)
                elif not in_string and ch == "#":
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
# Main
# =========================================
def main():
    print("=" * 70)
    print("Inference for Code Similarity Classification")
    print("병률 - Sequence Classification Inference")
    print("=" * 70)
    
    print(f"\n✅ Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n설정:")
    print(f"  Checkpoint: {CHECKPOINT_DIR}")
    print(f"  Test CSV: {TEST_CSV_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Threshold: {THRESHOLD}")
    
    # ========================================
    # 모델 & 토크나이저 로드
    # ========================================
    print(f"\n{'=' * 70}")
    print("모델 로드")
    print(f"{'=' * 70}")
    
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    cfg = AutoConfig.from_pretrained(CHECKPOINT_DIR)
    
    print(f"  Model Type: {cfg.model_type}")
    print(f"  Architectures: {cfg.architectures}")
    
    # 모델 타입별 로드
    if cfg.model_type == "bert":
        model = BertForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    elif cfg.model_type == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    elif cfg.model_type == "electra":
        model = ElectraForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {cfg.model_type}")
    
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ 모델 로드 완료")
    
    # 토크나이저 설정
    return_token_type_ids = getattr(model.config, "type_vocab_size", 0) > 1
    model_max_len = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
    effective_max_len = min(tokenizer.model_max_length, model_max_len)
    
    print(f"  Max Length: {effective_max_len}")
    print(f"  Return Token Type IDs: {return_token_type_ids}")
    
    # ========================================
    # 테스트 데이터 로드
    # ========================================
    print(f"\n{'=' * 70}")
    print("테스트 데이터 로드")
    print(f"{'=' * 70}")
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df["code1"] = test_df["code1"].fillna("").astype(str)
    test_df["code2"] = test_df["code2"].fillna("").astype(str)
    
    print(f"✓ 테스트 샘플: {len(test_df):,}개")
    
    # 전처리
    print(f"\n전처리 중...")
    test_df["code1_processed"] = test_df["code1"].map(preprocess_cached)
    test_df["code2_processed"] = test_df["code2"].map(preprocess_cached)
    
    raw_test_dataset = Dataset.from_pandas(
        test_df[["code1_processed", "code2_processed"]],
        preserve_index=False,
    )
    
    # 토크나이징
    print(f"토크나이징 중...")
    
    def tokenize_batch(batch):
        return tokenizer(
            batch["code1_processed"],
            batch["code2_processed"],
            truncation=True,
            max_length=effective_max_len,
            return_token_type_ids=return_token_type_ids,
        )
    
    tokenized_test = raw_test_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["code1_processed", "code2_processed"],
    )
    
    print(f"✓ 전처리 완료")
    
    # ========================================
    # 추론
    # ========================================
    print(f"\n{'=' * 70}")
    print("추론 시작")
    print(f"{'=' * 70}")
    
    tokenized_test.set_format(type="torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    test_loader = DataLoader(
        tokenized_test,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )
    
    logits_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits_list.append(outputs.logits.cpu())
    
    logits = torch.cat(logits_list, dim=0)
    probs = torch.softmax(logits, dim=-1)[:, 1]
    pred_labels = (probs >= THRESHOLD).int().numpy()
    
    print(f"\n✓ 추론 완료: {len(pred_labels):,}개")
    
    # ========================================
    # 제출 파일 생성
    # ========================================
    print(f"\n{'=' * 70}")
    print("제출 파일 생성")
    print(f"{'=' * 70}")
    
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission["similar"] = pred_labels
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = CHECKPOINT_DIR.name
    output_path = OUTPUT_DIR / f"submission_{checkpoint_name}_{timestamp}.csv"
    submission.to_csv(output_path, index=False)
    
    print(f"\n예측 분포:")
    print(f"  similar=0: {(pred_labels == 0).sum():,}개 ({(pred_labels == 0).sum() / len(pred_labels) * 100:.1f}%)")
    print(f"  similar=1: {(pred_labels == 1).sum():,}개 ({(pred_labels == 1).sum() / len(pred_labels) * 100:.1f}%)")
    
    print(f"\n✓ 제출 파일 저장: {output_path}")
    
    # 미리보기
    print(f"\n미리보기:")
    print(submission.head(10))
    
    print(f"\n{'=' * 70}")
    print("✓ 모든 작업 완료!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
