from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

__all__ = ["skim_schema"]


@dataclass(frozen=True, slots=True)
class _VCInfo:
    top3: str
    bottom3: str
    sampled: bool
    sample_n: int


def _is_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, CategoricalDtype)


def _infer_kind(s: pd.Series) -> str:
    """최소한의 dtype-based kind 분류."""
    dtype = s.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "time"
    if pd.api.types.is_bool_dtype(dtype):
        return "categorical"
    if (
        _is_categorical(s)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ):
        return "categorical"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    return "other"


def _safe_non_na_head(s: pd.Series, k: int) -> str:
    """결측치 제외 후 앞에서 k개를 repr로 표시."""
    ss = s.dropna()
    if ss.empty:
        return ""
    return ", ".join([repr(x) for x in ss.head(k).tolist()])


def _value_counts_lite(
    s: pd.Series,
    *,
    k: int,
    max_vc_sample: int,
    random_state: int,
) -> _VCInfo:
    """
    값 분포 스키밍용 경량 value_counts.

    - 결측치 제외
    - non-na가 크면 샘플링하여 계산(비용 제한)
    - top k: 빈도 높은 값 k개 (value:count)
    - bottom k: 빈도 낮은 값 k개 (value:count)
    """
    ss = s.dropna()
    if ss.empty:
        return _VCInfo(top3="", bottom3="", sampled=False, sample_n=0)

    sampled = False
    if len(ss) > max_vc_sample:
        sampled = True
        ss = ss.sample(n=max_vc_sample, random_state=random_state)

    vc = ss.value_counts(dropna=True)
    if vc.empty:
        return _VCInfo(top3="", bottom3="", sampled=sampled, sample_n=int(len(ss)))

    top_items = vc.head(k)
    bottom_items = vc.sort_values(ascending=True).head(k)

    top3 = "; ".join([f"{repr(v)}:{int(c)}" for v, c in top_items.items()])
    bottom3 = "; ".join([f"{repr(v)}:{int(c)}" for v, c in bottom_items.items()])

    return _VCInfo(top3=top3, bottom3=bottom3, sampled=sampled, sample_n=int(len(ss)))


def skim_schema(
    df: pd.DataFrame,
    *,
    sort_by: str = "na_rate",  # "na_rate" | "nunique" | "name" | "dtype" | "kind"
    sample_k: int = 3,
    max_vc_sample: int = 20_000,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    단일 DataFrame을 '가볍게 훑기(data skimming)' 위한 컬럼 요약 테이블을 생성합니다.

      - dtype / kind(time|numeric|categorical|other)
      - 결측치 개수/비율
      - 고유값 수(nunique)
      - 결측 제외 샘플
      - (샘플 기반) top/bottom 빈도 값

    Parameters
    ----------
    df:
        입력 데이터프레임
    sort_by:
        정렬 기준: "na_rate", "nunique", "name", "dtype", "kind"
    sample_k:
        samples_non_na, top3/bottom3에 사용할 k
    max_vc_sample:
        value_counts 계산을 위한 최대 샘플 수(비용 제한)
    random_state:
        value_counts 샘플링 시드

    Returns
    -------
    pandas.DataFrame
        컬럼별 요약 리포트 테이블
    """
    n_rows = int(len(df))
    rows: list[dict[str, object]] = []

    for col in df.columns:
        s = df[col]
        na_cnt = int(s.isna().sum())
        na_rate = (na_cnt / n_rows) if n_rows else np.nan
        nunique = int(s.nunique(dropna=True))

        vc = _value_counts_lite(
            s,
            k=sample_k,
            max_vc_sample=max_vc_sample,
            random_state=random_state,
        )

        rows.append(
            {
                "col": col,
                "dtype": str(s.dtype),
                "kind": _infer_kind(s),
                "n_rows": n_rows,
                "na_cnt": na_cnt,
                "na_rate": na_rate,
                "nunique": nunique,
                "samples_non_na": _safe_non_na_head(s, k=sample_k),
                "top3": vc.top3,
                "bottom3": vc.bottom3,
                "vc_sampled": vc.sampled,
                "vc_sample_n": vc.sample_n,
            }
        )

    rep = pd.DataFrame(rows)

    if sort_by == "na_rate":
        rep = rep.sort_values(["na_rate", "col"], ascending=[False, True])
    elif sort_by == "nunique":
        rep = rep.sort_values(["nunique", "col"], ascending=[False, True])
    elif sort_by == "name":
        rep = rep.sort_values(["col"], ascending=[True])
    elif sort_by == "dtype":
        rep = rep.sort_values(["dtype", "col"], ascending=[True, True])
    elif sort_by == "kind":
        rep = rep.sort_values(["kind", "col"], ascending=[True, True])

    return rep.reset_index(drop=True)
