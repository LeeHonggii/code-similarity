import torch
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BEST_MODEL = "hosung1/code-sim-roberta-small"
TEST_PATH = ""
OUTPUT_PATH = ""

df_test = pd.read_csv(TEST_PATH)


# --- 로드 & 정합성 맞추기 ---
tok = AutoTokenizer.from_pretrained(BEST_MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL).eval()

# RoBERTa 계열: CLS 없으면 BOS를 CLS로 사용
if tok.cls_token_id is None:
    tok.cls_token = tok.bos_token

# special ids 동기화
mdl.config.pad_token_id = tok.pad_token_id
mdl.config.bos_token_id = tok.bos_token_id
mdl.config.eos_token_id = tok.eos_token_id

# vocab 불일치 방지
if mdl.config.vocab_size != tok.vocab_size:
    mdl.resize_token_embeddings(tok.vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl.to(device)


# --- 배치 추론 함수  ---
def predict_batch(df: pd.DataFrame, batch_size: int = 64, max_len: int = 512, thr: float = 0.5, desc: str = "Predict"):
    preds = []
    n = len(df)
    mdl.eval()

    for i in tqdm(range(0, n, batch_size), total=(n + batch_size - 1)//batch_size, desc=desc):
        chunk = df.iloc[i:i+batch_size]
        batch = tok(
            [x for x in chunk["code1"].tolist()],
            [x for x in chunk["code2"].tolist()],
            padding=True, truncation=True, max_length=max_len, return_tensors="pt",
            return_token_type_ids=False,
        )
        batch.pop("token_type_ids", None)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            logits = mdl(**batch).logits
            prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        preds.extend((prob >= thr).astype(int).tolist())

    return preds

# --- 예측 & 저장 ---
preds = predict_batch(df_test, batch_size=64)
submission = pd.DataFrame({"pair_id": df_test["pair_id"].values, "similar": preds})
submission.to_csv(OUTPUT_PATH, index=False)
print("saved: submission.csv")
print(submission.head())