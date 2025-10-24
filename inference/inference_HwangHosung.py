import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ========================================
# 배치 추론 함수
# ========================================
def predict_batch(
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    batch_size: int = 64,
    max_len: int = 512,
    threshold: float = 0.5,
    desc: str = "Predict"
) -> list:
    """
    배치 단위로 코드 유사도 예측
    
    Args:
        df: 테스트 데이터프레임 (code1, code2 컬럼 필요)
        model: 파인튜닝된 모델
        tokenizer: 토크나이저
        device: 디바이스 (cuda/cpu)
        batch_size: 배치 크기
        max_len: 최대 시퀀스 길이
        threshold: 분류 임계값 (확률 >= threshold면 1)
        desc: 진행바 설명
        
    Returns:
        예측 레이블 리스트 (0 or 1)
    """
    preds = []
    n = len(df)
    model.eval()

    for i in tqdm(range(0, n, batch_size), total=(n + batch_size - 1)//batch_size, desc=desc):
        chunk = df.iloc[i:i+batch_size]
        
        # 토크나이징
        batch = tokenizer(
            [str(x) for x in chunk["code1"].tolist()],
            [str(x) for x in chunk["code2"].tolist()],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        
        # RoBERTa는 token_type_ids 사용 안 함
        batch.pop("token_type_ids", None)
        batch = {k: v.to(device) for k, v in batch.items()}

        # 추론
        with torch.no_grad():
            logits = model(**batch).logits
            prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        # 임계값 기반 분류
        preds.extend((prob >= threshold).astype(int).tolist())

    return preds


# ========================================
# 모델 로드 및 설정
# ========================================
def load_model_and_tokenizer(model_path: str, device):
    """
    모델과 토크나이저 로드 및 정합성 맞추기
    
    Args:
        model_path: 모델 경로 (로컬 또는 허깅페이스)
        device: 디바이스
        
    Returns:
        (tokenizer, model)
    """
    print(f"\n{'='*70}")
    print("모델 & 토크나이저 로드")
    print(f"{'='*70}")
    print(f"모델 경로: {model_path}")
    
    # 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
    
    # RoBERTa 계열: CLS 없으면 BOS를 CLS로 사용
    if tokenizer.cls_token_id is None:
        tokenizer.cls_token = tokenizer.bos_token
        print(f"  CLS 토큰이 없어 BOS를 CLS로 사용")
    
    # Special token ids 동기화
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # Vocab 불일치 방지
    if model.config.vocab_size != tokenizer.vocab_size:
        print(f"  Vocab 크기 불일치 감지: {model.config.vocab_size} → {tokenizer.vocab_size}")
        model.resize_token_embeddings(tokenizer.vocab_size)
        print(f"  모델 임베딩 크기 조정 완료")
    
    # 디바이스로 이동
    model.to(device)
    
    print(f"✓ 모델 로드 완료")
    print(f"  Vocab Size: {tokenizer.vocab_size:,}")
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Num Labels: {model.config.num_labels}")
    print(f"  Device: {device}")
    
    return tokenizer, model


# ========================================
# 메인 함수
# ========================================
def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Code Similarity Inference")
    parser.add_argument(
        "--model",
        type=str,
        choices=["local", "huggingface"],
        default="huggingface",
        help="local: 로컬 파인튜닝 모델, huggingface: 허깅페이스 모델"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="./data/test.csv",
        help="테스트 데이터 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./inference/submission_HwangHosung.csv",
        help="제출 파일 저장 경로"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="배치 크기"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="분류 임계값 (0.0~1.0)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Code Similarity Inference")
    print("황호성 - RoBERTa-small")
    print("=" * 70)
    
    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 모델 경로 설정
    if args.model == "local":
        MODEL_PATH = "./models/roberta_small_finetune/best"
        print(f"\n로컬 파인튜닝 모델 사용: {MODEL_PATH}")
    else:  # huggingface
        MODEL_PATH = "hosung1/code-sim-roberta-small"
        print(f"\n허깅페이스 파인튜닝 모델 사용: {MODEL_PATH}")
    
    # 설정 출력
    print(f"\n설정:")
    print(f"  테스트 데이터: {args.test}")
    print(f"  출력 파일: {args.output}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Threshold: {args.threshold}")
    
    # ========================================
    # 1. 테스트 데이터 로드
    # ========================================
    print(f"\n{'='*70}")
    print("테스트 데이터 로드")
    print(f"{'='*70}")
    
    try:
        df_test = pd.read_csv(args.test)
        print(f"✓ 데이터 로드 완료: {len(df_test):,}개 샘플")
        print(f"  컬럼: {df_test.columns.tolist()}")
        
        # 필수 컬럼 확인
        required_cols = ['pair_id', 'code1', 'code2']
        missing = [col for col in required_cols if col not in df_test.columns]
        if missing:
            print(f"✗ 필수 컬럼 누락: {missing}")
            return
        
        print(f"\n데이터 미리보기:")
        print(df_test.head(3))
        
    except FileNotFoundError:
        print(f"✗ 테스트 파일을 찾을 수 없습니다: {args.test}")
        return
    except Exception as e:
        print(f"✗ 데이터 로드 실패: {e}")
        return
    
    # ========================================
    # 2. 모델 & 토크나이저 로드
    # ========================================
    try:
        tokenizer, model = load_model_and_tokenizer(MODEL_PATH, device)
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        return
    
    # ========================================
    # 3. 추론 시작
    # ========================================
    print(f"\n{'='*70}")
    print("추론 시작")
    print(f"{'='*70}")
    
    preds = predict_batch(
        df=df_test,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_len=512,
        threshold=args.threshold,
        desc="Predicting"
    )
    
    print(f"\n✓ 추론 완료!")
    print(f"  총 예측 수: {len(preds):,}개")
    
    # ========================================
    # 4. 제출 파일 생성
    # ========================================
    print(f"\n{'='*70}")
    print("제출 파일 생성")
    print(f"{'='*70}")
    
    submission = pd.DataFrame({
        "pair_id": df_test["pair_id"].values,
        "similar": preds
    })
    
    # 예측 분포
    print(f"\n예측 분포:")
    print(submission['similar'].value_counts())
    print(f"  - similar=0 (다른 문제): {(submission['similar']==0).sum():,}개 ({(submission['similar']==0).sum()/len(submission)*100:.1f}%)")
    print(f"  - similar=1 (같은 문제): {(submission['similar']==1).sum():,}개 ({(submission['similar']==1).sum()/len(submission)*100:.1f}%)")
    
    # 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ 제출 파일 저장 완료: {output_path}")
    
    # 미리보기
    print(f"\n제출 파일 미리보기:")
    print(submission.head(10))
    
    # ========================================
    # 5. 완료
    # ========================================
    print(f"\n{'='*70}")
    print("✓ 모든 작업 완료!")
    print(f"{'='*70}")
    print(f"제출 파일: {output_path}")
    print(f"총 예측 수: {len(submission):,}개")


if __name__ == "__main__":
    main()
