import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import (AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding)

'''
if you want to use custom tok/model, set TOKENIZER_DIR, MODEL_DIR = "hosung1/roberta_small_mlm_from_scratch"
Huggingface pretrained model = "huggingface/CodeBERTa-small-v1"
'''

# train data path
CSV_PATH = "" 
TOKENIZER_DIR = "hosung1/roberta_small_mlm_from_scratch"
MODEL_DIR = "hosung1/roberta_small_mlm_from_scratch"
OUT_DIR = ""

# training data (expects columns: code1, code2, similar)
df = pd.read_csv(CSV_PATH)
# Load custom tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
# Pair classification commonly benefits from using the *end* of code for truncation
tokenizer.truncation_side = "left"

# Split
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['similar'])
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))
raw_ds = DatasetDict(train=train_ds, validation=valid_ds)

MAX_LEN = 512

def preprocess(batch):
    c1 = [x for x in batch['code1']]
    c2 = [x for x in batch['code2']]
    enc = tokenizer(c1, c2, max_length=MAX_LEN, padding="max_length", truncation=True)
    enc['labels'] = batch['similar']
    return enc

tokenized_ds = raw_ds.map(preprocess, batched=True, remove_columns=['code1','code2','similar'])


class SafeCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.inner = DataCollatorWithPadding(tokenizer=tokenizer,
                                             pad_to_multiple_of=pad_to_multiple_of)

    def __call__(self, features):
        batch = self.inner(features)
        # RoBERTa: type ids 사용 안 함 (섞여 있으면 out-of-range 유발)
        batch.pop("token_type_ids", None)

        # 토큰 id 범위 검증
        ids = batch["input_ids"]
        v = self.tokenizer.vocab_size
        if torch.any(ids.ge(v)) or torch.any(ids.lt(0)):
            bad = torch.nonzero((ids>=v) | (ids<0), as_tuple=False)[:10].tolist()
            mx, mn = int(ids.max()), int(ids.min())
            raise RuntimeError(f"out-of-range token id in batch: min={mn}, max={mx}, vocab={v}, bad_positions={bad}")
        return batch

safe_collator = SafeCollator(tokenizer, pad_to_multiple_of=8)



num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=num_labels)

metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    overwrite_output_dir=True,
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,            # common for RoBERTa-style training
    logging_steps=50,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    bf16=False,
    optim="adamw_torch_fused",
    adam_epsilon=1e-6,            # epsilon (stability) — try 1e-6 for RoBERTa-ish
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['validation'],
    tokenizer=tokenizer,
    data_collator=safe_collator,
    compute_metrics=compute_metrics,
)


# a) PAD/특수토큰 점검
print("pad_token_id:", tokenizer.pad_token_id, "| cls/eos/bos:", tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id)

# b) 토큰 인덱스 범위(음수/초과 여부)
def minmax_ids(split):
    arr = tokenized_ds[split]["input_ids"]
    mn = min(min(row) for row in arr)
    mx = max(max(row) for row in arr)
    return mn, mx, len(arr[0])

mn_tr, mx_tr, L_tr = minmax_ids("train")
mn_va, mx_va, L_va = minmax_ids("validation")
print(f"[train] min/max/L = {mn_tr}/{mx_tr}/{L_tr}")
print(f"[valid] min/max/L = {mn_va}/{mx_va}/{L_va}")
print("vocab_size:", tokenizer.vocab_size)

# c) 모델 최대 포지션 확인
print("model.config.max_position_embeddings:", getattr(model.config, "max_position_embeddings", None))

# 훈련 시작
train_result = trainer.train()
metrics = trainer.evaluate()

best_dir = OUT_DIR / "best"
best_dir.mkdir(exist_ok=True, parents=True)
trainer.save_model(best_dir)        # saves model + head
tokenizer.save_pretrained(best_dir) # snapshot tokenizer used for this run

