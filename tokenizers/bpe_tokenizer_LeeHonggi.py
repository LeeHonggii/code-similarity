import pandas as pd
import os
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm


def train_bpe_tokenizer(
    data_path: str,
    output_dir: str,
    vocab_size: int = 50265,
    min_frequency: int = 2
):
    """
    BPE 토크나이저 학습
    
    Args:
        data_path: 전처리된 parquet 파일 경로
        output_dir: 토크나이저 저장 디렉토리
        vocab_size: 어휘 크기 (기본값: 50265, RoBERTa와 동일)
        min_frequency: 토큰의 최소 빈도수
    """
    
    # ============================================================================
    # 1. 데이터 로드
    # ============================================================================
    print("=" * 70)
    print("데이터 로딩 중...")
    print("=" * 70)
    df = pd.read_parquet(data_path)
    print(f"✓ 데이터 로드 완료! 총 샘플 수: {len(df):,}")
    
    # ============================================================================
    # 2. 학습용 코퍼스 파일 생성
    # ============================================================================
    print("\n" + "=" * 70)
    print("RoBERTa-style BPE 토크나이저 학습")
    print("=" * 70)
    
    TEMP_CORPUS_PATH = '/tmp/temp_code_corpus.txt'
    
    print("\n1/4: 학습용 코퍼스 파일 생성 중...")
    texts = df['text_norm'].astype(str).tolist()
    print(f"  총 코드 샘플: {len(texts):,}개")
    
    with open(TEMP_CORPUS_PATH, 'w', encoding='utf-8') as f:
        for code in tqdm(texts, desc="파일 저장"):
            f.write(code + '\n')
    
    print(f"✓ 코퍼스 저장 완료: {TEMP_CORPUS_PATH}")
    
    # ============================================================================
    # 3. ByteLevelBPE 토크나이저 학습
    # ============================================================================
    print("\n2/4: ByteLevelBPE 토크나이저 학습 중...")
    
    tokenizer = ByteLevelBPETokenizer()
    
    tokenizer.train(
        files=[TEMP_CORPUS_PATH],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<s>",      # CLS token
            "<pad>",    # PAD token
            "</s>",     # SEP token
            "<unk>",    # UNK token
            "<mask>"    # MASK token
        ],
        show_progress=True
    )
    
    print("✓ 토크나이저 학습 완료!")
    
    # ============================================================================
    # 4. 토크나이저 저장
    # ============================================================================
    print("\n3/4: 토크나이저 저장 중...")
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    
    print(f"✓ 토크나이저 저장 완료: {output_dir}")
    print(f"  - vocab.json")
    print(f"  - merges.txt")
    
    # ============================================================================
    # 5. 특수 토큰 ID 확인
    # ============================================================================
    print("\n4/4: 토크나이저 정보")
    
    vocab = tokenizer.get_vocab()
    special_token_ids = {
        'pad_token_id': vocab.get('<pad>', 0),
        'unk_token_id': vocab.get('<unk>', 1),
        'cls_token_id': vocab.get('<s>', 2),
        'sep_token_id': vocab.get('</s>', 3),
        'mask_token_id': vocab.get('<mask>', 4)
    }
    
    print(f"  Vocab 크기: {tokenizer.get_vocab_size():,}")
    print(f"\n특수 토큰 ID:")
    for name, id in special_token_ids.items():
        print(f"  {name}: {id}")
    
    # 임시 파일 삭제
    if os.path.exists(TEMP_CORPUS_PATH):
        os.remove(TEMP_CORPUS_PATH)
    
    print("\n" + "=" * 70)
    print("✓ 모든 작업 완료!")
    print("=" * 70)
    
    return tokenizer


if __name__ == "__main__":
    # 기본 설정
    DATA_PATH = './data/code_corpus_processed.parquet'
    OUTPUT_DIR = './data/tokenizer'
    
    # 토크나이저 학습 실행
    tokenizer = train_bpe_tokenizer(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        vocab_size=50265,
        min_frequency=2
    )
