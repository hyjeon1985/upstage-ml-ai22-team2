from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_COL_TRAIN = "_is_train"

# -----------------------------
# Data Preprocess
# -----------------------------

def _fe_useless_na_values(x_train: pd.DataFrame) -> pd.DataFrame:
    x_out = x_train.copy()
    t_out = test.copy()

    # 결측치에 해당하는 값들을 np.nan으로 대체
    x_out["등기신청일자"] = x_out["등기신청일자"].replace(" ", np.nan)
    x_out["거래유형"] = x_out["거래유형"].replace("-", np.nan)

    # 무의미하다고 판단되는 컬럼들 삭제
    x_out = x_out.drop("중개사소재지", axis=1)

    # 아파트 좌표 값 복원은 도로명 주소로 대신함.
    x_out = x_out.drop("번지", axis=1)
    x_out = x_out.drop("본번", axis=1)
    x_out = x_out.drop("부번", axis=1)

    return out


def _fe_gu_dong(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # `시군구` 컬럼에서 `구`와 `동` 정보만 추출하여 파생 변수 생성.

    s = out["시군구"].astype(str)

    out["구"] = s.str.extract(r"(\S+구)", expand=False)
    out["동"] = s.str.extract(r"(\S+동)", expand=False)

    out = out.drop(columns=["시군구"])

    return out


def _fe_contract_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    ym = out["계약년월"].astype(str)

    # 계약년월의 모든 값들이 통일된 양식을 가지고 있는지 판단합니다.
    if not ym.str.fullmatch(r"\d{6}").all():
        raise ValueError("계약년월 컬럼에 YYYYMM 형식이 아닌 값이 존재합니다.")

    out["계약년"] = ym.str.slice(0, 4).astype(int)
    out["계약월"] = ym.str.slice(4, 6).astype(int)

    # 알려진 데이터 범위인 2007-01-01 부터 2023-06-30 까지에 속하는지 확인

    if not out["계약년"].between(2007, 2023).all():
        raise ValueError("계약년 컬럼에 2007~2023 범위를 벗어난 값이 존재합니다.")

    if not out["계약월"].between(1, 12).all():
        raise ValueError("계약월 컬럼에 1~12 범위를 벗어난 값이 존재합니다.")

    if not out["계약일"].between(1, 31).all():
        raise ValueError("계약일 컬럼에 1~31 범위를 벗어난 값이 존재합니다.")

    # 데이터 범위까지 검증되었다면, 계약년월 컬럼은 제거합니다.
    out = out.drop("계약년월", axis=1)

    # 계약일자 생성
    out["계약일자"] = pd.to_datetime(
        out["계약년"].astype(str)
        + "-"
        + out["계약월"].astype(str).str.zfill(2)
        + "-"
        + out["계약일"].astype(str).str.zfill(2),
        format="%Y-%m-%d",
        errors="raise",  # 오류 체크 필요.
    )

    return out


def _fe_canceled_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["is_canceled"] = out["해제사유발생일"].notna().astype(int)

    out = out.drop("해제사유발생일", axis=1)

    return out


def _fe_build_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    build_y = out["건축년도"].astype(str).str.strip()

    # 건축년도의 모든 값들이 통일된 양식을 가지고 있는지 판단합니다.
    if not build_y.str.fullmatch(r"\d{4}").all():
        raise ValueError("건축년도 컬럼에 YYYY 형식이 아닌 값이 존재합니다.")

    out["건축년도"] = build_y.astype(int)

    # 건축년도의 범위 체크 (1937년 준공된 충정아파트 기준)
    if not out["건축년도"].between(1937, 2023).all():
        raise ValueError("건축년도 컬럼에 1937~2023 범위를 벗어난 값이 존재합니다.")

    # 건물나이 계산
    build_age = out["계약년"] - out["건축년도"]

    # 1) 계약이 1~2년 선행되는 것은 정상 범주로 본다.
    # 2) 계약이 3~4년 선행되는 것은 특수한 실제 사례로 본다.
    # 3) 계약이 5년 이상 선행되는 것은 데이터 오류로 보고 드랍한다.

    # 비정상 계약 제거 (이상치 제거)
    out = out[build_age >= -4].copy()

    # 이상치 제거 후 다시 계산
    build_age = out["계약년"] - out["건축년도"]

    # 건물나이 파생변수 생성 (연속형, 0 이상)
    out["건물나이"] = build_age.clip(lower=0)

    # 건축지연도 파생 변수 생성 (범주형, 0~4)
    out["건축지연도"] = (-build_age).clip(lower=0)

    return out


def preprocess(df: pd.DataFrame):
    """모델 학습/추론 전 데이터 전처리"""

    # 무의미하거나 컬럼과 유사 결측치 처리
    out = _fe_useless_na_values(df)

    # 시군구 처리
    out = _fe_gu_dong(out)

    # 계약일자 처리
    out = _fe_contract_date(out)

    # 해제사유발생일 처리
    out = _fe_canceled_date(out)

    # 건축년도 처리
    out = _fe_build_year(out)

    return out


def finalize_defaults(df: pd.DataFrame, for_train: bool) -> pd.DataFrame:
    """모델 학습/추론 전 변수 정돈"""

    out = df.copy()

    # 해제사유발생일
    if for_train:
        mk_canceled = out["is_canceled"] == 1
        out = out.loc[out["is_canceled"] != 1].drop(column=["is_canceled"])
        
    out = out.loc[~mk_canceled]
    else:
        out = out.drop(columns=["is_canceled"])

    return out


# 학습용 주요 변수들
_IMP_FEATS_CONT = ["좌표X", "좌표Y", "전용면적", "층", "건물나이"]  # Float32
_IMP_FEATS_DISC = ["계약년", "계약월", "계약일", "건축지연도"]  # Int16 (nullable)
_IMP_FEATS_CATE = ["구", "동", "건축지연도"]  # category


def finalize_for_lgbm(
    df: pd.DataFrame,
    *,
    for_train: bool,
    keep_only_model_cols: bool = False,
    cont_cols: Iterable[str] = _IMP_FEATS_CONT,
    disc_cols: Iterable[str] = _IMP_FEATS_DISC,
    cate_cols: Iterable[str] = _IMP_FEATS_CATE,
) -> tuple[pd.DataFrame, list[str]]:
    """
    LightGBM 입력 직전 dtype 확정 함수.

    - train/test 모두 호출 가능
    - 결측 안전: pandas nullable dtype 사용(Float32/Int16/category)
    - category 컬럼 목록(cat_cols)을 반환하여 train/test 카테고리 정렬에 사용

    Parameters
    ----------
    for_train:
        학습 데이터면 True, 추론(test)이면 False
        (현재 구현은 dtype 확정에는 차이가 없고, 향후 train 전용 로직 확장을 위한 자리)
    keep_only_model_cols:
        True면 최종적으로 모델 입력에 필요한 컬럼만 남김(타겟은 별도 처리 권장)
    """

    out = df.copy()

    # 1) 연속형: Float32 (NaN/NA 안전)
    for c in cont_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float32")

    # 2) 이산/정수형: Int16 (nullable)
    for c in disc_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int16")

    # 3) 범주형: category
    cat_cols = [c for c in cate_cols if c in out.columns]
    for c in cat_cols:
        # 문자열 기반 범주형(구/동 등)은 string으로 한번 정리하면 깔끔
        if out[c].dtype != "category":
            out[c] = out[c].astype("string").astype("category")
        else:
            # 이미 category면 유지
            pass


    return out, cat_cols

def align_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    train의 category schema를 test에 강제 적용.

    - train에서 학습한 category 목록을 그대로 test에 set_categories
    - train에 없던 test의 값은 NaN/NA로 들어감 (안전)
    """

    tr = train.copy()
    te = test.copy()

    for c in cat_cols:
        if c not in tr.columns or c not in te.columns:
            continue

        tr[c] = tr[c].astype("category")
        cats = tr[c].cat.categories

        te[c] = te[c].astype("category").cat.set_categories(cats)

    return tr, te
