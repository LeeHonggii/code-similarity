"""
Unigram Tokenizer Training Script
이서율 - Unigram Language Model 기반 토크나이저 (마커/리터럴 처리)

Usage:
    python tokenizers/unigram_tokenizer_LeeSeoYul.py
"""

import os, io, re, random, math, tokenize
from pathlib import Path
from typing import Dict, Any, List

import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from tokenizers import Tokenizer
from tokenizers.models import Unigram, BPE
from tokenizers.trainers import UnigramTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast


# =============================
# 설정
# =============================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# 마커 / 리터럴 스페셜 토큰
INDENT_TOK, DEDENT_TOK, NEWLINE_TOK = "<indent>", "<dedent>", "<newline>"
STR_TOK, NUM_TOK = "<str>", "<num>"
BASE_SPECIAL = ["<pad>", "<s>", "</s>", "<mask>", "<unk>", "<nl>", "<py>"]
ALL_SPECIAL = BASE_SPECIAL + [INDENT_TOK, DEDENT_TOK, NEWLINE_TOK, STR_TOK, NUM_TOK]


# =========================================
# 1) 텍스트 전처리: 마커 삽입
# =========================================
def code_to_marked_text(
    code: str,
    keep_comments: bool = False,
    normalize_literals: bool = False
) -> str:
    """
    - INDENT/DEDENT/NEWLINE → 스페셜 토큰 삽입
    - (옵션) 주석 제거
    - (옵션) 숫자/문자열 리터럴을 <num>/<str>로 치환
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
        # 실패 시 전체를 통으로 처리
        s = code.replace("\r\n","\n").replace("\r","\n").replace("\n",f" {NEWLINE_TOK} ")
        out = [" " + s + " "]
    return " ".join("".join(out).split())


def build_dataset_from_df(df, text_col="text", val_ratio=0.1) -> DatasetDict:
    """df[text_col] → 전처리(markers) → HF Dataset(train/validation)"""
    print(f"\n데이터셋 구축 중...")
    print(f"  총 샘플 수: {len(df):,}")
    
    texts = [code_to_marked_text(s) for s in tqdm(df[text_col].astype(str), desc="마커 삽입")]
    ds = Dataset.from_dict({"text": texts})
    split = ds.train_test_split(test_size=val_ratio, seed=SEED)
    
    print(f"✓ 데이터셋 분할 완료")
    print(f"  Train: {len(split['train']):,}")
    print(f"  Val: {len(split['test']):,}")
    
    return DatasetDict(train=split["train"], validation=split["test"])


# =======================================
# 2) 토크나이저 학습 (Unigram / BPE)
# =======================================
def train_tokenizer_from_ds(
    ds: DatasetDict,
    save_dir: str,
    kind: str = "unigram",
    vocab_size: int = 32000,
    max_piece_length: int = 16
) -> PreTrainedTokenizerFast:
    """
    - kind: "unigram" (추천) or "bpe"
    - ds["train"]["text"] 기준으로 학습
    """
    print(f"\n{'='*70}")
    print(f"{kind.upper()} 토크나이저 학습")
    print(f"{'='*70}")
    print(f"설정:")
    print(f"  - Vocab Size: {vocab_size:,}")
    print(f"  - Kind: {kind}")
    print(f"  - Special Tokens: {len(ALL_SPECIAL)}개")
    
    os.makedirs(save_dir, exist_ok=True)
    tmp_corpus = os.path.join(save_dir, "corpus_tmp.txt")
    
    print(f"\n코퍼스 파일 생성 중...")
    with open(tmp_corpus, "w", encoding="utf-8") as f:
        for s in tqdm(ds["train"]["text"], desc="파일 저장"):
            if s: f.write(s.strip() + "\n")
    print(f"✓ 코퍼스 저장: {tmp_corpus}")

    if kind == "unigram":
        tk = Tokenizer(Unigram())
        tk.pre_tokenizer = Whitespace()
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=ALL_SPECIAL,
            unk_token="<unk>",
            max_piece_length=max_piece_length
        )
    elif kind == "bpe":
        tk = Tokenizer(BPE(unk_token="<unk>"))
        tk.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=ALL_SPECIAL,
            min_frequency=2
        )
        tk.decoder = ByteLevelDecoder()
    else:
        raise ValueError("kind must be 'unigram' or 'bpe'")

    print(f"\n토크나이저 학습 중...")
    tk.train([tmp_corpus], trainer)
    
    tok_json = os.path.join(save_dir, f"{kind}.json")
    tk.save(tok_json)
    print(f"✓ 토크나이저 학습 완료: {tok_json}")
    
    # 임시 파일 삭제
    os.remove(tmp_corpus)

    fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_json,
        bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", mask_token="<mask>", unk_token="<unk>"
    )
    fast.add_special_tokens({
        "additional_special_tokens": [
            t for t in ALL_SPECIAL if t not in fast.all_special_tokens
        ]
    })
    fast.save_pretrained(save_dir)
    
    print(f"✓ 토크나이저 저장 완료: {save_dir}")
    print(f"  - {kind}.json")
    print(f"  - tokenizer.json")
    print(f"  - tokenizer_config.json")
    
    return fast


# ===========================================
# 3) 메인 실행
# ===========================================
def main():
    """메인 토크나이저 학습 파이프라인"""
    
    print("=" * 70)
    print("Unigram 토크나이저 학습 (마커 처리)")
    print("이서율")
    print("=" * 70)
    
    # ========================================
    # 설정
    # ========================================
    PARQUET_PATH = "./data/code_corpus_processed.parquet"
    TOKENIZER_KIND = "unigram"  # "unigram" or "bpe"
    VOCAB_SIZE = 32000
    OUTPUT_DIR = f"./data/tokenizer_{TOKENIZER_KIND}_LeeSeoYul"
    
    print(f"\n설정:")
    print(f"  데이터: {PARQUET_PATH}")
    print(f"  토크나이저 종류: {TOKENIZER_KIND}")
    print(f"  Vocab Size: {VOCAB_SIZE:,}")
    print(f"  출력 디렉토리: {OUTPUT_DIR}")
    
    # ========================================
    # 데이터 로드
    # ========================================
    print(f"\n{'='*70}")
    print("데이터 로드")
    print(f"{'='*70}")
    
    try:
        df = pd.read_parquet(PARQUET_PATH)
        print(f"✓ 데이터 로드 완료: {len(df):,}개 샘플")
        
        # 컬럼 확인
        if 'text' not in df.columns:
            if 'text_norm' in df.columns:
                df['text'] = df['text_norm']
                print(f"  'text_norm' 컬럼을 'text'로 사용")
            elif 'text_a' in df.columns:
                df['text'] = df['text_a']
                print(f"  'text_a' 컬럼을 'text'로 사용")
        
        print(f"  컬럼: {df.columns.tolist()}")
        
    except FileNotFoundError:
        print(f"✗ 데이터 파일을 찾을 수 없습니다: {PARQUET_PATH}")
        print(f"  경로를 확인하세요.")
        return
    except Exception as e:
        print(f"✗ 데이터 로드 실패: {e}")
        return
    
    # ========================================
    # 데이터셋 구축 (마커 삽입)
    # ========================================
    print(f"\n{'='*70}")
    print("데이터 전처리 (마커 삽입)")
    print(f"{'='*70}")
    
    ds = build_dataset_from_df(df, text_col="text", val_ratio=0.1)
    
    # 샘플 확인
    print(f"\n전처리 샘플:")
    sample = ds["train"][0]["text"]
    print(f"  {sample[:200]}...")
    
    # ========================================
    # 토크나이저 학습
    # ========================================
    tokenizer = train_tokenizer_from_ds(
        ds, 
        save_dir=OUTPUT_DIR,
        kind=TOKENIZER_KIND,
        vocab_size=VOCAB_SIZE,
        max_piece_length=16
    )
    
    # ========================================
    # 토크나이저 정보 출력
    # ========================================
    print(f"\n{'='*70}")
    print("토크나이저 정보")
    print(f"{'='*70}")
    print(f"  Vocab Size: {len(tokenizer):,}")
    print(f"  Special Tokens: {tokenizer.special_tokens_map}")
    print(f"\n특수 토큰 (마커):")
    for tok in ALL_SPECIAL:
        token_id = tokenizer.convert_tokens_to_ids(tok)
        print(f"  {tok}: {token_id}")
    
    # 테스트 인코딩
    test_code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
    
    print(f"\n{'='*70}")
    print("테스트 인코딩")
    print(f"{'='*70}")
    print(f"원본 코드:")
    print(test_code)
    
    # 마커 삽입
    marked = code_to_marked_text(test_code)
    print(f"\n마커 삽입 후:")
    print(marked)
    
    # 토크나이징
    encoded = tokenizer(marked, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    print(f"\n토큰 수: {encoded['input_ids'].shape[1]}")
    print(f"토큰 IDs: {encoded['input_ids'][0][:20].tolist()}...")
    print(f"토큰: {tokens[:20]}")
    
    # 디코딩
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"\n디코딩 결과:")
    print(decoded[:200])
    
    print(f"\n{'='*70}")
    print("✓ 모든 작업 완료!")
    print(f"{'='*70}")
    print(f"토크나이저 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
