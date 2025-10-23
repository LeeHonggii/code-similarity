
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


# ===========================================
# 2) 토크나이즈 & 512블록 (Encoder/LM 공통)
# ===========================================
from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast

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

# ================================
# WandB Attach (one-cell, drop-in)
# ================================
import os, math, time, types
from pathlib import Path

# 1) 토글/기본 설정 --------------------------
WANDB_ON      = True          # 끄려면 False
WANDB_PROJECT = "python-mlm"  # 프로젝트 이름
WANDB_ENTITY  = None          # 팀 워크스페이스면 "your-team", 아니면 None
WANDB_GROUP   = "code4sage"   # 실험 묶음 이름
WANDB_TAGS    = ["from-scratch", "python", "pretrain"]

def _safe_get_len(x, default=None):
    try:
        return len(x)
    except Exception:
        return default

def build_wandb_config():
    # 환경에서 가져올 수 있는 정보들을 최대한 긁어서 config로 구성
    cfg = {
        "tokenizer": {
            "vocab_size": _safe_get_len(globals().get("tokenizer")),
            "special": getattr(globals().get("tokenizer"), "all_special_tokens", None),
        },
        "data": {
            "train_blocks": _safe_get_len(globals().get("lm_ds", {}).get("train", [])),
            "val_blocks":   _safe_get_len(globals().get("lm_ds", {}).get("validation", [])),
            "max_len": globals().get("MAX_LEN", 512),
        },
        "env": {
            "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
            "runtime": "colab" if "COLAB_GPU" in os.environ else "local",
        }
    }
    return cfg

# 2) W&B 런 시작/종료 ------------------------
def wandb_start(run_name=None, extra_config:dict=None):
    if not WANDB_ON:
        return None
    import wandb
    base_name = run_name or f"run_{int(time.time())}"
    cfg = build_wandb_config()
    if extra_config:
        cfg.update(extra_config)
    wandb.login()  # 이미 로그인 되어 있으면 noop
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=WANDB_GROUP,
        name=base_name,
        tags=WANDB_TAGS,
        config=cfg
    )
    return run

def wandb_finish():
    if not WANDB_ON:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass

# 3) Trainer용 ETA/지표 콜백 -----------------
from transformers import TrainerCallback

class TimeETA(TrainerCallback):
    """HF Trainer에서 step/ETA/퍼플렉시티를 W&B로 추가 기록"""
    def __init__(self): self.t0=None
    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.perf_counter()
    def on_log(self, args, state, control, logs=None, **kw):
        if not WANDB_ON or self.t0 is None:
            return
        import wandb
        steps = max(1, state.global_step)
        elapsed = time.perf_counter() - self.t0
        sec_per_step = elapsed / steps
        if state.max_steps:
            eta = (state.max_steps - steps) * sec_per_step
        else:
            eta = float("nan")
        wandb.log({
            "time/elapsed_min": elapsed/60,
            "time/eta_min": eta/60 if eta==eta else None,
            "speed/steps_per_sec": 1.0/max(sec_per_step, 1e-9),
        }, step=steps)

    def on_evaluate(self, args, state, control, metrics, **kw):
        if not WANDB_ON:
            return
        import wandb
        log = {}
        for k, v in metrics.items():
            log[f"{k}"] = v
        # ppl 계산(손실 제한)
        if "eval_loss" in metrics and metrics["eval_loss"] is not None:
            log["eval/perplexity"] = math.exp(min(metrics["eval_loss"], 20))
        wandb.log(log, step=state.global_step)

# 4) HF Trainer에 손쉽게 붙이는 도우미 ----------
def enable_wandb_for_hf_trainer_args(training_args, run=None, run_name_hint=None):
    """TrainingArguments를 수정해 W&B 로깅 + best model 선택을 안전하게 설정"""
    # 핵심: report_to / run_name / metric_for_best_model
    training_args.report_to = ["wandb"] if WANDB_ON else []
    if run is not None and hasattr(run, "name"):
        training_args.run_name = run.name
    elif run_name_hint:
        training_args.run_name = run_name_hint
    # best metric 키는 eval_loss 사용(언더스코어!)
    training_args.metric_for_best_model = "eval_loss"
    training_args.greater_is_better = False
    training_args.load_best_model_at_end = True
    return training_args

def attach_eta_callback_to_trainer(trainer):
    """시간/ETA 콜백을 Trainer에 등록"""
    trainer.add_callback(TimeETA())
    return trainer

# 5) 커스텀 ELECTRA 학습기 패치(원본 코드 수정 없음) ----
def enable_wandb_for_electra(electra_trainer, run=None, name_hint="electra-rtd"):
    """
    ElectraRTDTrainer 인스턴스에 W&B 로깅을 주입.
    - _run_epoch를 감싸서 epoch 결과(gen/disc loss) 기록
    """
    if not WANDB_ON:
        return electra_trainer

    import wandb
    if run is None:
        wandb_start(name_hint)

    # 이미 감싼 적 있으면 재감싸지 않음
    if getattr(electra_trainer, "_wandb_wrapped", False):
        return electra_trainer

    old_run_epoch = electra_trainer._run_epoch

    def _run_epoch_logged(self, epoch:int, train:bool=True):
        res = old_run_epoch(epoch, train=train)  # 원래 로직 수행
        phase = "train" if train else "eval"
        step = getattr(self, "_wandb_epoch_step", 0)
        try:
            import wandb
            wandb.log({
                f"{phase}/gen_loss": res.get("gen_loss"),
                f"{phase}/disc_loss": res.get("disc_loss"),
                "epoch": epoch,
            }, step=step)
        except Exception:
            pass
        setattr(self, "_wandb_epoch_step", step+1)
        return res

    electra_trainer._run_epoch = types.MethodType(_run_epoch_logged, electra_trainer)
    electra_trainer._wandb_wrapped = True
    return electra_trainer

run = wandb_start(run_name="roberta-mlm-fromscratch")

# ==================================================
# 4) 모델1: RoBERTa-small (Encoder) + MLM (Trainer)
# ==================================================
from transformers import RobertaConfig, RobertaForMaskedLM, TrainingArguments, Trainer, set_seed

def build_roberta_small(tokenizer) -> RobertaForMaskedLM:
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

def pretrain_roberta_mlm(lm_ds: DatasetDict,
                         tokenizer: PreTrainedTokenizerFast,
                         out_dir: str,
                         epochs:int=2, lr:float=6e-4, warmup_ratio:float=0.06,
                         per_device_bs:int=40, grad_accum:int=1,
                         scheduler:str="cosine") -> Dict[str, Any]:
    model = build_roberta_small(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    use_bf16 = torch.cuda.is_available() and cap >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    steps_per_epoch = max(1, len(lm_ds["train"]) // max(1, per_device_bs * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(50, int(total_steps * warmup_ratio))
    eval_steps = max(200, total_steps // 10)

    args = TrainingArguments(
        output_dir=out_dir, overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr, lr_scheduler_type=scheduler,
        warmup_steps=warmup_steps, weight_decay=0.01,
        eval_strategy="steps", logging_strategy="steps", save_strategy="steps",
        eval_steps=eval_steps, logging_steps=50, save_steps=eval_steps, save_total_limit=1,
        fp16=use_fp16, bf16=use_bf16, optim="adamw_torch_fused",
        dataloader_num_workers=2, dataloader_pin_memory=True,
        report_to=[], run_name=Path(out_dir).name,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args,
                      train_dataset=lm_ds["train"],
                      eval_dataset=lm_ds["validation"],
                      data_collator=collator)


    run = wandb_start(run_name="roberta-mlm-fromscratch")
    args = enable_wandb_for_hf_trainer_args(args, run=run, run_name_hint="roberta-mlm")
    trainer = attach_eta_callback_to_trainer(trainer)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return metrics

import pandas as pd
from transformers import AutoTokenizer

max_len = 512
workdir = ''
tok_dir = 'hosung1/roberta_small_mlm_from_scratch'
PARQUET_PATH = "/content/drive/MyDrive/LikeLion_NLP2/Project_1/Data/code_corpus_processed.parquet"

# A) 데이터셋
df = pd.read_parquet(PARQUET_PATH)
ds = build_dataset_from_df(df, text_col="text", val_ratio=0.1)
# B) 토크나이저
tokenizer = AutoTokenizer.from_pretrained(tok_dir)
# C) 블록화
lm_ds = tokenize_and_chunk(ds, tokenizer, max_len=max_len)

# ---  RoBERTa-small + MLM ---
out_roberta = os.path.join(workdir, "roberta_small_mlm_from_scratch")
print("\n[RoBERTa-small + MLM] start")
roberta_metrics = pretrain_roberta_mlm(lm_ds, tokenizer, out_roberta)
print("[RoBERTa-small + MLM] done:", roberta_metrics)
wandb_finish()