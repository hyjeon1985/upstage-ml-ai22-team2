[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UBFhwOwS)

# \[Upstage-AI22-ML-2조\] 아파트 실거래가 예측

## Team

| ![곽은주](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김원재](https://avatars.githubusercontent.com/u/156163982?v=4) | ![왕시훈](https://avatars.githubusercontent.com/u/156163982?v=4) | ![전호열](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                         역할, 담당 역할                          |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview

본 프로젝트는 AI Stages – 아파트 실거래가 예측(House Price Prediction) 경진대회에 참여한
`Upstate AI 22기 ML 2조`의 협업 저장소입니다.

대규모 정형 데이터 기반 회귀 문제를 대상으로 하며,
EDA → 전처리 → 피처 엔지니어링 → 모델링 → 제출까지의 전 과정을 재현 가능하고 일관된 개발 환경에서 수행하는 것을 목표로 합니다.

특히 본 저장소는 다음 원칙을 따릅니다.

- 팀원 전원이 동일한 Python / 패키지 버전 사용

- 실험 코드와 개인 작업을 **공통 패키지 구조(`lab_core`)**로 통합

- `.vscode` + `uv` 기반 워크스페이스 단일화

- 데이터(`data/`)와 결과물(`outputs/`)은 비버전 관리(.gitignore)

### Environment

#### Development Environment

- Python: `3.10.13` (필수 고정)

- 패키지 관리: `uv`

- IDE / Editor: VS Code (권장)

- Notebook: Jupyter / ipykernel

- OS: Windows / macOS / Linux (공통 지원)

모든 팀원은 동일한 개발 환경을 사용하며,
환경 차이로 인한 실험 결과 편차를 최소화합니다.

#### Workspace Convention

본 프로젝트는 아래 요소들을 모든 팀원이 공통으로 사용하는 것을 전제로 합니다.

- `.vscode/settings.json`
  - Python 인터프리터 고정 (`.venv`)

  - `.env` 기반 환경 변수 로딩

  - CSV / Notebook / Formatter 통합 설정

- `uv.lock`
  - 의존성 완전 고정 (requirements.txt 대체)

- `bootstrap.sh`
  - 환경 초기 세팅 및 업데이트 자동화 스크립트

  ```bash
  # 최초 설치 또는 업데이트 시
  bash bootstrap.sh
  ```

  > `git pull` 이후에도 항상 `bootstrap.sh` 실행을 권장합니다.

### Requirements

#### Runtime Dependencies

모델 학습 및 추론에 사용되는 주요 라이브러리입니다.

```
matplotlib==3.7.1
numpy==1.23.5
pandas==1.5.3
lightgbm==4.2.0
scikit-learn==1.2.2
scipy==1.11.3
seaborn==0.12.2
statsmodels==0.14.0
tqdm==4.66.1
```

#### Development Dependencies

EDA 및 실험 환경을 위한 개발 의존성입니다.

```
ipykernel==6.27.1
jupyter==1.0.0
```

> ⚠️ 패키지 버전 임의 변경 금지
> 팀 실험 재현성과 제출 검증을 위해 모든 의존성은 `uv.lock` 기준으로 고정됩니다.

## 1. Competiton Info

### Overview

- 대회명: 아파트 실거래가 예측 (House Price Prediction)

- 주최: AI Stages

- 문제 유형: Regression

- 목표: 서울시 아파트 매매 실거래가 예측

- 평가지표: RMSE (Root Mean Squared Error)

### Timeline

- January 06, 2026 - Start Date
- January 12, 2026 - Final submission deadline

## 2. Components

### Directory

```
├─ data/
│  ├─ raw/        # 대회 제공 원본 데이터 (*.csv)
│  ├─ ext/        # 허용된 외부 공공 데이터 (*.csv)
│  ├─ interim/    # 전처리 중간 산출물 (*.parquet)
│  └─ features/   # 피처 테이블 (*.parquet, *.csv)
│
├─ docs/          # 팀 공유 문서 (*.md, *.pdf)
│
├─ notebooks/     # 공용 노트북 파일
│  ├─ ejk/        # 개인 노트 (팀원 이니셜로 구분)
│  ├─ hyj/
│  ├─ shw/
│  └─ wjk/
│
├─ outputs/       # 학습 모델, 제출 csv 등
│
├─ script/       # CLI 스크립트 (*.py, *.sh)
│
├─ src/
│  └─ lab_core/   # 공통 파이썬 패키지
│     ├─ ejk/     # 개인 모듈 (팀원 이니셜로 구분)
│     ├─ hyj/
│     ├─ shw/
│     ├─ wjk/
│     ├─ styles/  # 공용 시각화 보조 도구 패키지
│     └─ util/    # 공용 유틸 패키지
│
├─ .vscode/       # VSCode 팀 워크스페이스 설정
├─ bootstrap.sh   # 환경 초기 세팅 및 업데이트 자동화 스크립트
├─ pyproject.toml # 파이썬 프로젝트 설정 (의존성 명세 등)
├─ uv.lock
└─ README.md
```

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
