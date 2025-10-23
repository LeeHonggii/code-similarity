# Code Similarity Project

> **Team**: I'm fine tuning  
> **Project Period**: 2025.10.17 ~ 2025.10.23  
> **Status**: ✅ Completed

## 📋 Overview

CodeNet 데이터셋을 활용하여 pretrained 모델을 구축하고, 데이콘 학습 데이터로 fine-tuning을 진행하여 코드 유사도 판별 모델의 성능을 평가하는 프로젝트입니다.

## 🎯 Project Goals

1. **Pretraining**: CodeNet 데이터셋으로 pretrained 모델 구축 ✅
2. **Fine-tuning**: 데이콘 경진대회 데이터로 모델 미세 조정 ✅
3. **Evaluation**: 모델 성능 평가 및 분석 ✅

## 🤖 Models

### 구현된 모델

| 모델명 | 담당자 | 구현 방식 | Public Score | Private Score |
|--------|--------|-----------|--------------|---------------|
| **CodeBERT (HuggingFace)** | 이홍기 | microsoft/codebert-base | **92.84%** | **92.86%** |
| **RoBERTa-small** | 황호성 | Custom Implementation | **91.67%** | **91.64%** |
| **Custom BERT** | 조병률 | From Scratch | **86.33%** | **86.32%** |
| **RoleBERT** | 오정탁 | Role Embedding | - | **49%** |
| **Contrastive Learning** | 이서율 | Dual Encoder + AST | **62.18%** | **62.03%** |

### 모델별 특징

#### 1. CodeBERT (HuggingFace) - 이홍기
- **핵심**: microsoft/codebert-base 사전학습 모델 활용
- **학습 방식**: MLM (Masked Language Modeling) + RTD (Replaced Token Detection)
- **특징**: 
  - 대규모 사전학습 데이터의 효과
  - Transfer Learning 최대 활용

#### 2. RoBERTa-small - 황호성
- **핵심**: RoBERTa 최적화 기법 적용
- **Tokenizer**: Unigram (32k vocab)
- **특징**:
  - 효과적인 전처리 (좌측 절단, 최소화)
  - 구조/리터럴 마커 활용 (`<str>`, `<num>`, `<indent>`)
  - Custom 모델 중 최고 성능 🥈

#### 3. Custom BERT - 조병률
- **핵심**: 처음부터 구축한 BERT 아키텍처
- **학습 방식**: MLM (Masked Language Modeling)
- **특징**:
  - 기본에 충실한 구현
  - Baseline 역할
  - 확장 가능한 구조 

#### 4. RoleBERT - 오정탁
- **핵심**: Role Embedding 추가
- **학습 방식**: MLM + Role Prediction
- **특징**:
  - 구문 정보(KEYWORD, IDENTIFIER 등) 명시적 활용
  - AST 기반 코드 정규화
  - 혁신적 시도

#### 5. Contrastive Learning - 이서율
- **핵심**: MoCo 기반 듀얼 인코더
- **구조**: Text Encoder + AST-GNN
- **특징**:
  - AST 구조 정보 활용
  - Hard Negative Mining
  - 실험적 접근

## 🔧 Preprocessing & Tokenization

### 데이터 전처리 파이프라인 ✅

#### 처리 단계 (순차적 적용)
1. **주석 제거** (`remove_comments`)
2. **식별자 정규화** (`alpha_rename`) - 변수명/함수명/클래스명 익명화
3. **공백 정규화** - 연속 개행을 최대 2개로 제한
4. **코드 포맷팅** (`black`) - 88자 줄 길이 기준
5. **중복 제거** - SHA1 해시 기반

### 토크나이저 구현 ✅

각 모델별로 최적화된 토크나이저를 `/tokenizers` 폴더에 구현:

#### 1. Unigram (황호성 - RoBERTa)
- 희귀 식별자 처리에 안정적
- 소규모 데이터에 빠른 수렴
- 특수 토큰: `<indent>`, `<dedent>`, `<str>`, `<num>` 등

#### 2. BPE (조병률, 오정탁, 이홍기)
- RoBERTa/CodeBERT 표준
- vocab_size: 32k-50k
- 안정적 성능

#### 3. SentencePiece (조병률)
- Custom BERT 구현
- vocab_size: 32k

## 📂 Project Structure

```
code-similarity/
├── README.md                           # 프로젝트 메인 문서
├── meeting-notes/                      # 회의록 모음
│   ├── 2025-10-17.md                  # 킥오프 회의록
│   ├── 2025-10-20.md                  # 전처리 및 토크나이저 결정
│   ├── 2025-10-21.md                  # 전처리 완료 및 학습 시작
│   └── 2025-10-22.md                  # 최종 결과 공유
├── tokenizers/                         # 토크나이저 구현
│   ├── unigram_tokenizer.py           # Unigram (황호성)
│   ├── bpe_tokenizer.py               # BPE (조병률, 오정탁, 이홍기)
│   └── sentencepiece_tokenizer.py     # SentencePiece
├── models/                             # 모델 구현
│   ├── codebert_hf.py                 # CodeBERT HuggingFace (이홍기)
│   ├── roberta_small.py               # RoBERTa-small (황호성)
│   ├── custom_bert.py                 # Custom BERT (조병률)
│   ├── rolebert.py                    # RoleBERT (오정탁)
│   └── contrastive_model.py           # Contrastive Learning (이서율)
├── train/                              # 학습 스크립트
│   ├── pretrain.py                    # Pretraining
│   └── finetune.py                    # Fine-tuning
├── data/                               # 데이터셋
│   ├── codenet/                       # CodeNet 원본
│   └── preprocessed/                  # 전처리 완료 데이터
├── notebooks/                          # 실험 노트북
└── docs/                               # 문서 및 자료
    └── 5명_종합비교분석.md            # 최종 분석 보고서
```

## 📊 Datasets

### 1. CodeNet (Pretraining)
- **출처**: IBM Research
- **설명**: 대규모 프로그래밍 문제 및 솔루션 데이터셋
- **링크**: [CodeNet GitHub](https://github.com/IBM/Project_CodeNet/blob/main/README.md#directory-structure-and-naming-convention)
- **용도**: Pretrained 모델 구축
- **처리 결과**: 604,124개 중복 제거 → 2,639,300개 샘플

### 2. Dacon - 월간 코드 유사성 판단 AI 경진대회 (Fine-tuning)
- **출처**: 데이콘
- **설명**: 코드 유사도 판별 태스크 데이터
- **링크**: [대회 페이지](https://dacon.io/competitions/official/235900/overview/description)
- **용도**: Fine-tuning 및 평가

## 📚 References

### 핵심 논문

1. **CodeBERT**: CodeBERT: A Pre-Trained Model for Programming and Natural Languages (EMNLP 2020)
2. **RoBERTa**: RoBERTa: A Robustly Optimized BERT Pretraining Approach (arXiv 2019)
3. **MoCo**: Momentum Contrast for Unsupervised Visual Representation Learning (CVPR 2020)
4. **ELECTRA**: ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (ICLR 2020)

## ✅ Task List

### Phase 1: 모델 이해 ✅
- [x] CodeBERT 아키텍처 학습
- [x] CodeBERT 사전 학습 방식 이해
- [x] Fine-tuning 방법론 연구
- [x] 각자 학습 내용 정리 및 공유

### Phase 2: 데이터 준비 ✅
- [x] 데이터 전처리 방식 결정 (중복 제거)
- [x] 토크나이저 옵션 검토 (WordPiece vs BPE)
- [x] 데이터 전처리 파이프라인 구축
- [x] 전처리 완료 및 메타데이터 생성

### Phase 3: 모델 학습 ✅
- [x] Fine-tuning 환경 구축
- [x] 각 모델별 학습 진행
  - [x] CodeBERT Fine-tuning (이홍기)
  - [x] RoBERTa-small Fine-tuning (황호성)
  - [x] Custom BERT 학습 (조병률)
  - [x] RoleBERT 학습 (오정탁)
  - [x] Contrastive Learning 학습 (이서율)
- [x] 하이퍼파라미터 튜닝

### Phase 4: 평가 및 분석 ✅
- [x] 모델 성능 평가
- [x] 결과 분석 및 시각화
- [x] 최종 보고서 작성 ([5명_종합비교분석.md](./docs/5명_종합비교분석.md))

## 🏆 Final Results

### 데이콘 리더보드 결과

| 순위 | 팀원 | 모델 | Public Score | Private Score |
|------|------|------|--------------|---------------|
| 🥇 | 이홍기 | CodeBERT (HF) | **92.84%** | **92.86%** |
| 🥈 | 황호성 | RoBERTa-small | **91.67%** | **91.64%** |
| 🥉 | 조병률 | Custom BERT | **86.33%** | **86.32%** |
| 4 | 이서율 | Contrastive | 62.18% | 62.03% |
| 5 | 오정탁 | RoleBERT | - | **49%** |

### 핵심 인사이트

#### 1. 사전학습 모델의 중요성
- **HuggingFace 사전학습 모델** (이홍기) > Custom 최적화 (황호성) > Custom Baseline (조병률)
- 대규모 데이터 학습 효과가 압도적
- Transfer Learning의 힘

#### 2. 데이터 전처리의 중요성
- 황호성의 전처리 방법: **+4.5%p** 성능 향상
- 좌측 절단(Left Truncation) 전략 효과적
- 과도한 정제보다 적절한 최소화가 중요

#### 3. 모델 아키텍처
- 복잡한 구조 < **최적화된 단순 구조**
- 표준 Transformer: 안정적, 효율적
- 구조 정보는 보조적으로 활용

#### 4. 학습 전략
- AdamW + Warmup Scheduler 표준
- Learning Rate: 2e-5 ~ 6e-4
- Full Fine-tuning이 효과적

#### 5. Tokenizer 선택
- **Unigram**: 희귀 식별자 안정적, 소규모 데이터 적합
- **BPE**: 표준적, 대규모 vocab 안정적
- Special Tokens 설계 중요 (`<str>`, `<num>`, `<indent>`)

## 👥 Team Contributions

| 팀원 | 역할 | 주요 기여 |
|------|------|-----------|
| **이홍기** | CodeBERT (HF) | HF 모델 활용, 최고 성능 달성 |
| **황호성** | RoBERTa-small | 전처리 최적화, Custom 중 최고 |
| **조병률** | Custom BERT | Baseline 제공, 기본 구현 |
| **오정탁** | RoleBERT | 혁신적 Role Embedding, 코드 정규화 |
| **이서율** | Contrastive | 대조학습 + AST, 실험적 접근 |

## 📝 Meeting Notes

프로젝트 진행 중 회의 내용:

- [2025-10-17: 킥오프 미팅](./meeting-notes/2025-10-17.md)
- [2025-10-20: 전처리 및 토크나이저 결정](./meeting-notes/2025-10-20.md)
- [2025-10-21: 전처리 완료 및 학습 시작](./meeting-notes/2025-10-21.md)
- [2025-10-22: 최종 결과 공유 및 분석](./meeting-notes/2025-10-22.md)

## 📄 Documentation

### 최종 보고서
프로젝트의 상세한 분석과 인사이트는 다음 문서에서 확인할 수 있습니다:

**[5명_종합비교분석.md](./docs/5명_종합비교분석.md)**

이 보고서는 다음 내용을 포함합니다:
- 각 모델의 아키텍처 상세 분석
- 전처리 및 토크나이저 비교
- 학습 전략 및 하이퍼파라미터 비교
- 성능 분석 및 인사이트
- 향후 개선 방향

## 💡 Lessons Learned

### 성공 요인
1. **대규모 사전학습 모델 활용** - Transfer Learning의 위력
2. **효과적인 전처리** - 좌측 절단, 적절한 최소화
3. **안정적인 학습 전략** - AdamW, Warmup, Scheduler
4. **다양한 접근법 시도** - 5가지 다른 방법론

### 개선 가능 영역
1. **앙상블** - 여러 모델 조합으로 성능 향상 가능
2. **데이터 증강** - 더 많은 학습 데이터 활용
3. **구조 정보 통합** - AST/Role 정보의 경량 통합
4. **하이퍼파라미터 튜닝** - 더 세밀한 최적화

## 📌 Contact

프로젝트 관련 문의사항이 있으시면 팀원에게 연락해주세요.

**Team**: I'm fine tuning

---

**Project Period**: 2025.10.17 ~ 2025.10.23  
**Last Updated**: 2025.10.23  
**Status**: ✅ Completed