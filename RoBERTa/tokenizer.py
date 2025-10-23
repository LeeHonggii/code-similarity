import pandas as pd
import numpy as np


PARQUET_PATH = ""
df = pd.read_parquet(PARQUET_PATH)

# =============================
# 0) 공통: 라이브러리 & 설정
# =============================
import os, io, re, math, random, tokenize
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

import torch
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

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
    texts = [code_to_marked_text(s) for s in df[text_col].astype(str)]
    ds = Dataset.from_dict({"text": texts})
    split = ds.train_test_split(test_size=val_ratio, seed=SEED)
    return DatasetDict(train=split["train"], validation=split["test"])


# =======================================
# 2) 토크나이저 학습 (Unigram / BPE 선택)
# =======================================
from tokenizers import Tokenizer
from tokenizers.models import Unigram, BPE
from tokenizers.trainers import UnigramTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

def train_tokenizer_from_ds(ds: DatasetDict,
                            save_dir: str,
                            kind: str = "unigram",
                            vocab_size: int = 32000,
                            max_piece_length: int = 16) -> PreTrainedTokenizerFast:
    """
    - kind: "unigram" (권장) | "bpe"
    - ALL_SPECIAL 토큰 포함
    - ds["train"]["text"]로 학습
    """
    os.makedirs(save_dir, exist_ok=True)
    tmp_corpus = os.path.join(save_dir, "corpus_tmp.txt")
    with open(tmp_corpus, "w", encoding="utf-8") as f:
        for s in ds["train"]["text"]:
            if s: f.write(s.strip() + "\n")

    if kind == "unigram":
        tk = Tokenizer(Unigram())
        tk.pre_tokenizer = Whitespace()
        trainer = UnigramTrainer(vocab_size=vocab_size,
                                 special_tokens=ALL_SPECIAL,
                                 unk_token="<unk>",
                                 max_piece_length=max_piece_length)
    elif kind == "bpe":
        tk = Tokenizer(BPE(unk_token="<unk>"))
        tk.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(vocab_size=vocab_size,
                             special_tokens=ALL_SPECIAL,
                             min_frequency=2)
        tk.decoder = ByteLevelDecoder()
    else:
        raise ValueError("kind must be 'unigram' or 'bpe'")

    tk.train([tmp_corpus], trainer)
    tok_json = os.path.join(save_dir, f"{kind}.json")
    tk.save(tok_json)

    fast = PreTrainedTokenizerFast(
        tokenizer_file=tok_json,
        bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", mask_token="<mask>", unk_token="<unk>"
    )
    fast.add_special_tokens({"additional_special_tokens":
        [t for t in ALL_SPECIAL if t not in fast.all_special_tokens]})
    fast.save_pretrained(save_dir)
    return fast


# ===========================================
# 3) 토크나이즈 & 512블록 (Encoder/LM 공통)
# ===========================================
from transformers import DataCollatorForLanguageModeling

def tokenize_and_chunk(ds: DatasetDict,
                       tokenizer: PreTrainedTokenizerFast,
                       max_len: int = 512) -> DatasetDict:
    """
    - add_special_tokens=False: 긴 시퀀스를 이어붙여 고정 길이 블록으로 자름
    - token_type_ids는 사용하지 않음(RoBERTa/Code류)
    """
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
    return lm_ds

# ===========================================
# 4) 토크나이저 학습
# ===========================================
workdir = ''
tokenizer_kind = 'unigram'
max_len = 512

os.makedirs(workdir, exist_ok=True)
# A) 데이터셋
ds = build_dataset_from_df(df, text_col="text", val_ratio=0.1)
# B) 토크나이저 학습
tok_dir = os.path.join(workdir, f"tok_{tokenizer_kind}32k_markers")
tokenizer = train_tokenizer_from_ds(ds, tok_dir, kind=tokenizer_kind, vocab_size=32000)