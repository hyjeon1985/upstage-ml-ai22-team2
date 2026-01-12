# 부동산 실거래가 예측 모델링

> 본 노트는 파이프라인과 러너를 구성해 실험을 재현하는 과정을 정리한 문서입니다.  
> 실제 코드 실행 흐름을 먼저 보여주고, 이후 데이터셋 빌더와 각 파이프라인 블록의 역할을 수도코드 형태로 설명합니다.

## Contents

- Overview
- Library Import
- Data Load
- Data Preprocessing
- Feature Engineering
- Model Training
- Inference
- Output File Save

## 1. Overview

- 실험 재현성과 실수 방지를 위해 **데이터셋 빌더 + 러너 구조**로 구성했습니다.
- 전처리/피처 엔지니어링을 한 곳에서 관리해 실험 간 일관성을 확보했습니다.
- 실행 로그와 결과 파일을 자동으로 저장해 실험 추적이 가능하도록 했습니다.

## 2. Library Import

```python
import numpy as np
import pandas as pd

from lab_core.hyj.hpp_dataset_builder import HPPDatasetBuilder
from lab_core.hyj.core.rf_lgbm_runner import ModelParams, RfLgbmRunner
```

## 3. Data Load

데이터 로드는 빌더 내부에서 관리합니다. 실행부에서는 빌더를 생성하고, 러너가 내부적으로 데이터를 불러옵니다.

> 관련 코드: [hpp_dataset_builder.py](../../src/lab_core/hyj/hpp_dataset_builder.py) (load_raw/load_ext)

※ 아래 코드는 전체 흐름을 요약한 수도코드이며, 실제 실행 순서를 설명하기 위한 것입니다.

```text
load_raw():
  train, test = read_csv(...)
  return train, test, target_col

load_ext():
  subway, bus, apt, coord = read_csv(...)
  return {subway, bus, apt, coord}
```

## 4. Data Preprocessing

전처리 및 피처 엔지니어링은 데이터셋 빌더의 파이프라인에서 순차적으로 수행합니다.

## 5. Feature Engineering

전처리와 함께 파이프라인 내부에서 수행하며, 블록 단위로 역할을 분리했습니다.

## 6. Model Training

```python
params = ModelParams(
    lgbm={
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
    },
    rf={
        "n_estimators": 300,
        "max_depth": 24,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
    },
)

builder = HPPDatasetBuilder(target_transform="log1p")
runner = RfLgbmRunner(builder, params=params)
```

## 7. Inference

```python
split_policy = {
    "kind": "time_holdout",
    "ym_col": "계약년월",
    "train_until": 202303,
    "valid_from": 202304,
    "valid_until": 202306,
}

result = runner.run_ensemble(
    weights={"lgbm": 0.7, "rf": 0.3},
    seed=42,
    use_cache=True,
    split_policy=split_policy,
    run_prefix="submission",
)
```

실행 로그에는 다음 정보가 포함됩니다.

- 데이터셋 빌드 진행 상황
- 모델 학습/검증 RMSE
- 앙상블 검증 RMSE

## 8. Output File Save

러너는 결과 파일과 메타 정보를 자동으로 저장합니다.

- 제출 파일: `outputs/subs/<run_id>/*.csv`
- 메타 정보: `outputs/subs/<run_id>/meta.json`

---

# 데이터셋 빌더 및 파이프라인 요약

## 데이터셋 빌더

- 코드: [hpp_dataset_builder.py](../../src/lab_core/hyj/hpp_dataset_builder.py)
- 역할: 데이터 로드 → rename → split → 파이프라인 실행 → 모델 입력 정리

### 수도코드

```
train, test = load_raw()
ext = load_ext()
rename(train, test)
X_train, y_train, X_test = split_target(train, test)
X_train, X_valid, y_train, y_valid = time_holdout_split(X_train, y_train)
X_train = pipeline.fit_transform(X_train)
X_valid = pipeline.transform(X_valid)
X_test = pipeline.transform(X_test)
X_train, X_test = finalize_for_model(X_train, X_test)
return dataset_bundle
```

## 파이프라인 블록

- 주소 파생: `src/lab_core/hyj/hpp_address_block.py`
  - `HppLoadNamePartsBlock`: 도로명 → 전체/앞/뒤 분리
  - `HppGuDongBlock`: 시군구 → 구/동 분리

- 결측/카테고리 정리: `src/lab_core/hyj/core/dataset/category_block.py`, `src/lab_core/hyj/core/dataset/category_keep_block.py`, `src/lab_core/hyj/core/dataset/useless_block.py`
  - `UselessValueToNaBlock`: 의미 없는 값은 결측 처리
  - `CategoryCleanBlock`: 범주형 결측치 통일
  - `CategoryKeepOthersBlock`: 특정 범주 유지 후 나머지 OTHERS

- 좌표 보정: `src/lab_core/hyj/hpp_coord_block.py`
  - k-apt/coord 기반 좌표 보정 및 유사도 보정

- 교통 피처: `src/lab_core/hyj/hpp_transport_block.py`
  - 최근접 거리 및 반경 점수 계산

- 시간/연식: `src/lab_core/hyj/hpp_time_block.py`
  - 계약년/월/분기, 건물나이/건축지연 파생

- 단지 규모/구성: `src/lab_core/hyj/hpp_complex_block.py`
  - 면적 구간 비중, 주차/세대/연면적 비율 파생

- 프리미엄 권역: `src/lab_core/hyj/hpp_premium_block.py`
  - 강남권 여부 이진 변수

- 빈도 인코딩: `src/lab_core/hyj/core/dataset/freq_block.py`
  - 아파트명/시공사/시행사 빈도 인코딩

각 블록의 입력/출력은 코드에 명시되어 있으며, 위 순서대로 실행됩니다.
