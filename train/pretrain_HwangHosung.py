"""
RoBERTa-small MLM Pre-training Script
황호성 - Masked Language Modeling 사전학습
- Unigram 토크나이저 사용
- 로컬 학습 또는 허깅페이스 모델 사용 가능

Usage:
    # 로컬에서 처음부터 프리트레이닝
    python train/pretrain_roberta_HwangHosung.py --mode train
    
    # 허깅페이스 사전학습 모델 로드만
    python train/pretrain_roberta_HwangHosung.py --mode load
"""

import os, io, re, math, random, tokenize, time, types, argparse
from pathlib import Path
from typing import Dict, Any

import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModel,
    set_seed,
    TrainerCallback
)

# =============================
# 설정
# =============================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# 마커/리터럴 스페셜 토큰
INDENT_TOK, DEDENT_TOK, NEWLINE_TOK = "<indent>", "<dedent>", "<newline>"
STR_TOK, NUM_TOK = "<str>", "<num>"
BASE_SPECIAL = ["<pad>", "<s>", "</s>", "<mask>", "<unk>", "<nl>", "<py>"]
ALL_SPECIAL = BASE_SPECIAL + [INDENT_TOK, DEDENT_TOK, NEWLINE_TOK, STR_TOK, NUM_TOK]


# =========================================
# 1) 텍스트 전처리: 마커 삽입 & 리터럴 일반화
# =========================================
def code_to_marked_text(code: str,
                        keep_comments: bool=False,
                        normalize_literals: bool=True) -> str:
    """
    - INDENT/DEDENT/NEWLINE -> 스페셜 토큰 삽입
    - 주석 제거
    - 숫자/문자열 리터럴 -> <num> / <str> 로 치환
    - 실패시: 줄바꿈만 <newline>로 치환
    """
    code = code if isinstance(code, str) else str(code)
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
            elif tt == tokenize.COMMENT and not keep_comments:
                continue
            elif normalize_literals and tt == tokenize.NUMBER:
                out.append(f" {NUM_TOK} ")
            elif normalize_literals and tt == tokenize.STRING:
                out.append(f" {STR_TOK} ")
            else:
                out.append(" " + ts + " ")
    except Exception:
        s = code.replace("\r\n","\n").replace("\r","\n").replace("\n", f" {NEWLINE_TOK} ")
        out = [" " + s + " "]
    return " ".join(" ".join(out).split())


def build_dataset_from_df(df, text_col="text", val_ratio=0.1) -> DatasetDict:
    """df[text_col] -> 전처리(markers) -> HF Dataset(train/validation)"""
    print(f"\n데이터셋 구축 중...")
    print(f"  총 샘플 수: {len(df):,}")
    
    texts = [code_to_marked_text(s) for s in tqdm(df[text_col].astype(str), desc="마커 삽입")]
    ds = Dataset.from_dict({"text": texts})
    split = ds.train_test_split(test_size=val_ratio, seed=SEED)
    
    print(f"✓ 데이터셋 분할 완료")
    print(f"  Train: {len(split['train']):,}")
    print(f"  Val: {len(split['test']):,}")
    
    return DatasetDict(train=split["train"], validation=split["test"])


# ===========================================
# 2) 토크나이즈 & 512블록
# ===========================================
def tokenize_and_chunk(ds: DatasetDict,
                       tokenizer: PreTrainedTokenizerFast,
                       max_len: int = 512) -> DatasetDict:
    """
    - add_special_tokens=False: 긴 시퀀스를 이어붙여 고정 길이 블록으로 자름
    - token_type_ids는 사용하지 않음(RoBERTa/Code류)
    """
    print(f"\n토크나이징 & 청킹 중...")
    
    def tok_fn(batch):
        return tokenizer(batch["text"],
                         add_special_tokens=False,
                         return_attention_mask=True,
                         return_token_type_ids=False)
    
    tok = ds.map(tok_fn, batched=True,
                 remove_columns=ds["train"].column_names,
                 desc="Tokenize")

    # 안전: 필요한 컬럼만 유지
    keep = ["input_ids", "attention_mask"]
    drop = [c for c in tok["train"].column_names if c not in keep]
    if drop:
        tok = tok.remove_columns(drop)

    def group_texts(examples):
        concat = {k: sum(examples[k], []) for k in keep}
        total = (len(concat["input_ids"]) // max_len) * max_len
        result = {}
        for k in keep:
            data = concat[k][:total]
            result[k] = [data[i:i+max_len] for i in range(0, total, max_len)]
        return result

    lm_ds = tok.map(group_texts, batched=True, desc=f"Group into {max_len}-blocks")
    
    print(f"✓ 토크나이징 & 청킹 완료")
    print(f"  Train blocks: {len(lm_ds['train']):,}")
    print(f"  Val blocks: {len(lm_ds['validation']):,}")
    
    return lm_ds


# ==================================================
# 3) RoBERTa-small 모델 구성
# ==================================================
def build_roberta_small(tokenizer) -> RobertaForMaskedLM:
    """RoBERTa-small 아키텍처 구성"""
    cfg = RobertaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=514,
        num_hidden_layers=6,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or 0,
        eos_token_id=tokenizer.eos_token_id or 2,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        use_cache=False,
    )
    return RobertaForMaskedLM(cfg)


# ==================================================
# 4) 프리트레이닝 함수
# ==================================================
def pretrain_roberta_mlm(lm_ds: DatasetDict,
                         tokenizer: PreTrainedTokenizerFast,
                         out_dir: str,
                         epochs: int = 2,
                         lr: float = 6e-4,
                         warmup_ratio: float = 0.06,
                         per_device_bs: int = 40,
                         grad_accum: int = 1,
                         scheduler: str = "cosine") -> Dict[str, Any]:
    """
    RoBERTa-small MLM 프리트레이닝
    """
    print(f"\n{'='*70}")
    print("RoBERTa-small 모델 초기화")
    print(f"{'='*70}")
    
    model = build_roberta_small(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"✓ 모델 생성 완료")
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab Size: {len(tokenizer):,}")
    print(f"  Hidden Size: {model.config.hidden_size}")
    print(f"  Num Layers: {model.config.num_hidden_layers}")

    # GPU 설정
    cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    use_bf16 = torch.cuda.is_available() and cap >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # 학습 스텝 계산
    steps_per_epoch = max(1, len(lm_ds["train"]) // max(1, per_device_bs * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(50, int(total_steps * warmup_ratio))
    eval_steps = max(200, total_steps // 10)

    print(f"\n학습 설정:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {per_device_bs}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  FP16: {use_fp16}, BF16: {use_bf16}")

    # TrainingArguments
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=scheduler,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=50,
        save_steps=eval_steps,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch" if not torch.cuda.is_available() else "adamw_torch_fused",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to=[],
        run_name=Path(out_dir).name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Data Collator (MLM)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        data_collator=collator
    )

    print(f"\n{'='*70}")
    print("프리트레이닝 시작")
    print(f"{'='*70}")
    
    # 학습 시작
    trainer.train()
    
    # 평가
    print(f"\n{'='*70}")
    print("최종 평가")
    print(f"{'='*70}")
    metrics = trainer.evaluate()
    
    print(f"\n평가 결과:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 모델 저장
    print(f"\n{'='*70}")
    print("모델 저장")
    print(f"{'='*70}")
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    print(f"✓ 모델 저장 완료: {out_dir}")
    
    return metrics


# ==================================================
# 5) 메인 함수
# ==================================================
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="RoBERTa-small MLM Pre-training")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load"],
        default="train",
        help="train: 로컬에서 프리트레이닝, load: 허깅페이스 모델 로드만"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("RoBERTa-small MLM Pre-training")
    print("황호성 - Unigram Tokenizer")
    print("=" * 70)
    
    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # ========================================
    # Mode: Load (허깅페이스 모델만 로드)
    # ========================================
    if args.mode == "load":
        print(f"\n{'='*70}")
        print("허깅페이스 사전학습 모델 로드")
        print(f"{'='*70}")
        
        HF_MODEL = "hosung1/roberta_small_mlm_from_scratch"
        
        try:
            print(f"\n모델 로드 중: {HF_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
            model = AutoModel.from_pretrained(HF_MODEL)
            
            print(f"✓ 모델 로드 완료!")
            print(f"  Vocab Size: {len(tokenizer):,}")
            print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
            print(f"\n사용법:")
            print(f"  from transformers import AutoModel, AutoTokenizer")
            print(f"  tokenizer = AutoTokenizer.from_pretrained('{HF_MODEL}')")
            print(f"  model = AutoModel.from_pretrained('{HF_MODEL}')")
            
            # 간단한 테스트
            test_code = "def hello():\n    return 'world'"
            marked = code_to_marked_text(test_code)
            inputs = tokenizer(marked, return_tensors="pt")
            
            print(f"\n테스트 인코딩:")
            print(f"  원본: {test_code}")
            print(f"  마커 삽입: {marked}")
            print(f"  토큰 수: {inputs['input_ids'].shape[1]}")
            
        except Exception as e:
            print(f"✗ 모델 로드 실패: {e}")
        
        return
    
    # ========================================
    # Mode: Train (로컬에서 프리트레이닝)
    # ========================================
    print(f"\n{'='*70}")
    print("로컬 프리트레이닝 모드")
    print(f"{'='*70}")
    
    # 설정
    PARQUET_PATH = "./data/code_corpus_processed.parquet"
    TOKENIZER_DIR = "./data/tokenizer_unigram"
    OUTPUT_DIR = "./models/roberta_small_mlm"
    MAX_LEN = 512
    EPOCHS = 2
    BATCH_SIZE = 40
    LEARNING_RATE = 6e-4
    
    print(f"\n설정:")
    print(f"  데이터: {PARQUET_PATH}")
    print(f"  토크나이저: {TOKENIZER_DIR}")
    print(f"  출력: {OUTPUT_DIR}")
    print(f"  Max Length: {MAX_LEN}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ========================================
    # 1. 데이터 로드
    # ========================================
    print(f"\n{'='*70}")
    print("데이터 로드")
    print(f"{'='*70}")
    
    try:
        df = pd.read_parquet(PARQUET_PATH)
        print(f"✓ 데이터 로드 완료: {len(df):,}개 샘플")
        
        if 'text' not in df.columns and 'text_norm' in df.columns:
            df['text'] = df['text_norm']
            print(f"  'text_norm' 컬럼을 'text'로 사용")
    
    except FileNotFoundError:
        print(f"✗ 데이터 파일을 찾을 수 없습니다: {PARQUET_PATH}")
        return
    except Exception as e:
        print(f"✗ 데이터 로드 실패: {e}")
        return
    
    # ========================================
    # 2. 데이터셋 구축 (마커 삽입)
    # ========================================
    print(f"\n{'='*70}")
    print("데이터 전처리 (마커 삽입)")
    print(f"{'='*70}")
    
    ds = build_dataset_from_df(df, text_col="text", val_ratio=0.1)
    
    # ========================================
    # 3. 토크나이저 로드
    # ========================================
    print(f"\n{'='*70}")
    print("토크나이저 로드")
    print(f"{'='*70}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        print(f"✓ 토크나이저 로드 완료: {TOKENIZER_DIR}")
        print(f"  Vocab Size: {len(tokenizer):,}")
    except Exception as e:
        print(f"✗ 토크나이저 로드 실패: {e}")
        print(f"  먼저 tokenizers/unigram_tokenizer_HwangHosung.py를 실행하세요.")
        return
    
    # ========================================
    # 4. 토크나이징 & 청킹
    # ========================================
    print(f"\n{'='*70}")
    print("토크나이징 & 청킹")
    print(f"{'='*70}")
    
    lm_ds = tokenize_and_chunk(ds, tokenizer, max_len=MAX_LEN)
    
    # ========================================
    # 5. 프리트레이닝
    # ========================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    metrics = pretrain_roberta_mlm(
        lm_ds=lm_ds,
        tokenizer=tokenizer,
        out_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        per_device_bs=BATCH_SIZE,
        grad_accum=1,
        scheduler="cosine"
    )
    
    # ========================================
    # 6. 완료
    # ========================================
    print(f"\n{'='*70}")
    print("✓ 모든 작업 완료!")
    print(f"{'='*70}")
    print(f"모델 저장 위치: {OUTPUT_DIR}")
    print(f"\n최종 평가 결과:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n허깅페이스 비교 모델:")
    print(f"  hosung1/roberta_small_mlm_from_scratch")


if __name__ == "__main__":
    main()
