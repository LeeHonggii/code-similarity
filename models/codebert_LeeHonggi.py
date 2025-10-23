import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
import os
from transformers import (
    RobertaConfig,
    RobertaModel,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Dataset
# ============================================================================
class CodePairDataset(Dataset):
    """
    코드 페어 유사도 데이터셋
    """
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        code1 = str(row['code1_processed'])
        code2 = str(row['code2_processed'])
        label = int(row['similar'])

        # [CLS] code1 [SEP] code2 [SEP] 형태로 결합
        encoding = self.tokenizer(
            code1,
            code2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Model
# ============================================================================
class CodeSimilarityClassifier(nn.Module):
    """
    CodeBERT + Classification Head
    """
    def __init__(self, encoder, hidden_size=768, num_labels=2, dropout=0.1):
        super(CodeSimilarityClassifier, self).__init__()
        
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # CodeBERT Encoding
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token의 hidden state 추출
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        # Progress bar update
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, criterion, device):
    """평가"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return avg_loss, accuracy, f1


# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    """메인 학습 파이프라인"""
    
    # ========================================================================
    # 1. 환경 설정
    # ========================================================================
    print("=" * 70)
    print("CodeBERT Fine-tuning - 코드 유사도 분류")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # ========================================================================
    # 2. 하이퍼파라미터
    # ========================================================================
    BATCH_SIZE = 64
    MAX_LENGTH = 512
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    WARMUP_STEPS = 100
    
    print(f"\n하이퍼파라미터:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Length: {MAX_LENGTH}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    
    # ========================================================================
    # 3. 토크나이저 로드
    # ========================================================================
    print("\n" + "=" * 70)
    print("토크나이저 로드")
    print("=" * 70)
    
    TOKENIZER_DIR = "./data/tokenizer"
    
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
        print(f"✓ 토크나이저 로드 완료: {TOKENIZER_DIR}")
        print(f"  Vocab Size: {len(tokenizer):,}")
        print(f"  Special Tokens: {tokenizer.special_tokens_map}")
    except Exception as e:
        print(f"✗ 토크나이저 로드 실패: {e}")
        print(f"  경로 확인: {TOKENIZER_DIR}")
        print(f"  먼저 tokenizers/bpe_tokenizer_LeeHonggi.py를 실행하세요.")
        return
    
    # ========================================================================
    # 4. CodeBERT 모델 로드
    # ========================================================================
    print("\n" + "=" * 70)
    print("CodeBERT 모델 로드")
    print("=" * 70)
    
    MODEL_NAME = "microsoft/codebert-base"
    
    config = RobertaConfig.from_pretrained(MODEL_NAME)
    print(f"✓ Config 로드 완료")
    print(f"  Vocab Size: {config.vocab_size:,}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Num Layers: {config.num_hidden_layers}")
    print(f"  Attention Heads: {config.num_attention_heads}")
    
    encoder = RobertaModel.from_pretrained(MODEL_NAME)
    print(f"✓ CodeBERT Encoder 로드 완료")
    print(f"  파라미터 수: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # ========================================================================
    # 5. 데이터 로드
    # ========================================================================
    print("\n" + "=" * 70)
    print("학습 데이터 로드")
    print("=" * 70)
    
    TRAIN_DATA_PATH = "./data/train_pairs.parquet"
    
    try:
        train_df = pd.read_parquet(TRAIN_DATA_PATH)
        print(f"✓ 데이터 로드 완료: {len(train_df):,}개 샘플")
        print(f"\nLabel 분포:")
        print(train_df['similar'].value_counts())
    except Exception as e:
        print(f"✗ 데이터 로드 실패: {e}")
        print(f"  경로 확인: {TRAIN_DATA_PATH}")
        return
    
    # ========================================================================
    # 6. 데이터 분할
    # ========================================================================
    print("\n" + "=" * 70)
    print("데이터 분할 (Train/Val)")
    print("=" * 70)
    
    train_df_split, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42,
        stratify=train_df['similar']
    )
    
    print(f"✓ 데이터 분할 완료")
    print(f"  Train: {len(train_df_split):,}개")
    print(f"  Val: {len(val_df):,}개")
    
    # Dataset & DataLoader 생성
    train_dataset = CodePairDataset(train_df_split, tokenizer, MAX_LENGTH)
    val_dataset = CodePairDataset(val_df, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ 데이터 로더 생성 완료")
    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Val Batches: {len(val_loader)}")
    
    # ========================================================================
    # 7. 모델 초기화
    # ========================================================================
    print("\n" + "=" * 70)
    print("모델 초기화")
    print("=" * 70)
    
    model = CodeSimilarityClassifier(
        encoder=encoder,
        hidden_size=config.hidden_size,
        num_labels=2,
        dropout=0.1
    )
    model.to(device)
    
    print(f"✓ 모델 초기화 완료")
    print(f"  총 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  학습 가능 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    print(f"✓ Optimizer & Scheduler 설정 완료")
    print(f"  Total Steps: {total_steps}")
    print(f"  Warmup Steps: {WARMUP_STEPS}")
    
    # ========================================================================
    # 8. 학습 시작
    # ========================================================================
    print("\n" + "=" * 70)
    print("학습 시작")
    print("=" * 70)
    
    best_val_f1 = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(EPOCHS):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 70}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        
        # Validate
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print results
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_save_path = './models/best_codebert_LeeHonggi.pt'
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✅ Best model saved! (F1: {best_val_f1:.4f})")
    
    # ========================================================================
    # 9. 학습 완료
    # ========================================================================
    print("\n" + "=" * 70)
    print("학습 완료!")
    print("=" * 70)
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"모델 저장 위치: ./models/best_codebert_LeeHonggi.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
