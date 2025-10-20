# Code Similarity Project

> **Team**: I'm fine tuning  
> **Start Date**: 2025.10.17

## 📋 Overview

CodeNet 데이터셋을 활용하여 pretrained 모델을 구축하고, 데이콘 학습 데이터로 fine-tuning을 진행하여 코드 유사도 판별 모델의 성능을 평가하는 프로젝트입니다.

## 🎯 Project Goals

1. **Pretraining**: CodeNet 데이터셋으로 pretrained 모델 구축
2. **Fine-tuning**: 데이콘 경진대회 데이터로 모델 미세 조정
3. **Evaluation**: 모델 성능 평가 및 분석

## 🤖 Model

### CodeBERT

CodeBERT는 Microsoft Research에서 개발한 프로그래밍 언어와 자연어를 동시에 이해할 수 있는 사전 학습 모델입니다.

**선정 이유**:
- 코드 이해 및 자연어 처리에 특화된 사전 학습 모델
- 코드 검색, 문서화, 유사도 판별 등 다양한 태스크에서 검증된 성능

## 🔧 Preprocessing & Tokenization

### 데이터 전처리
- **중복 제거만 수행**
- 복잡한 익명화, 정규화 과정은 적용하지 않음

### 토크나이저 (검토 중)

#### 옵션 1: WordPiece
- BERT, CodeBERT에서 사용
- 서브워드 분할 방식
- 기본 스페셜 토큰만 사용

#### 옵션 2: BPE (Byte Pair Encoding)
- GPT, RoBERTa에서 사용
- 빈도 기반 서브워드 병합
- 기본 스페셜 토큰만 사용

**결정 사항**: 각 토크나이저의 기본 제공 토큰만 사용하며, 커스텀 토큰(`<NL>`, `<NUM>`, `<STR>` 등)은 추가하지 않음

## 📂 Project Structure

```
code-similarity-project/
├── README.md                    # 프로젝트 메인 문서
├── meeting-notes/               # 회의록 모음
│   ├── 2025-10-17.md           # 킥오프 회의록
│   └── 2025-10-20.md           # 전처리 및 토크나이저 결정
├── data/                        # 데이터셋 (예정)
├── models/                      # 모델 코드 (예정)
├── notebooks/                   # 실험 노트북 (예정)
└── docs/                        # 문서 및 자료 (예정)
```

## 📊 Datasets

### 1. CodeNet (Pretraining)
- **출처**: IBM Research
- **설명**: 대규모 프로그래밍 문제 및 솔루션 데이터셋
- **링크**: [CodeNet GitHub](https://github.com/IBM/Project_CodeNet/blob/main/README.md#directory-structure-and-naming-convention)
- **용도**: Pretrained 모델 구축

### 2. Dacon - 월간 코드 유사성 판단 AI 경진대회 (Fine-tuning)
- **출처**: 데이콘
- **설명**: 코드 유사도 판별 태스크 데이터
- **링크**: [대회 페이지](https://dacon.io/competitions/official/235900/overview/description)
- **용도**: Fine-tuning 및 평가

## 📚 References

### 핵심 논문

#### 1. CodeBERT (주요 모델)
- **제목**: CodeBERT: A Pre-Trained Model for Programming and Natural Languages
- **저자**: Feng et al. (Microsoft Research)
- **학회**: EMNLP 2020

## ✅ Task List

### Phase 1: 모델 이해 ✓
- [x] CodeBERT 아키텍처 학습
- [x] CodeBERT 사전 학습 방식 이해
- [x] Fine-tuning 방법론 연구
- [x] 각자 학습 내용 정리 및 공유

### Phase 2: 데이터 준비 (진행중)
- [x] 데이터 전처리 방식 결정 (중복 제거)
- [x] 토크나이저 옵션 검토 (WordPiece vs BPE)
- [ ] WordPiece vs BPE 최종 선택
- [ ] CodeNet 데이터 다운로드 및 전처리
- [ ] 데이콘 데이터 분석
- [ ] 데이터 전처리 파이프라인 구축

### Phase 3: 모델 학습 (예정)
- [ ] Pretraining 환경 구축
- [ ] CodeNet으로 모델 사전 학습
- [ ] Fine-tuning 진행
- [ ] 하이퍼파라미터 튜닝

### Phase 4: 평가 및 분석 (예정)
- [ ] 모델 성능 평가
- [ ] 결과 분석 및 시각화
- [ ] 최종 보고서 작성

## 👥 Team Roles

| 담당자 | 역할 |
|--------|------|
| 조병률, 오정탁 | 데이터 구조 |
| 이서율, 황호성 | 모델 개발 |
| 이홍기 | 파인튜닝 |

## 📝 Meeting Notes

프로젝트 진행 중 회의 내용은 [`meeting-notes/`](./meeting-notes/) 폴더에서 확인할 수 있습니다.

- [2025-10-17: 킥오프 미팅](./meeting-notes/2025-10-17.md)
- [2025-10-20: 전처리 및 토크나이저 결정](./meeting-notes/2025-10-20.md)

## 📌 Contact

프로젝트 관련 문의사항이 있으시면 팀원에게 연락해주세요.

---

**Last Updated**: 2025.10.20