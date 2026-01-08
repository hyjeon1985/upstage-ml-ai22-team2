from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from lab_core.styles.viz import setup_global

from .data_util import float_or_nan

setup_global()

# -------------------------
# Compare Dataset
# -------------------------


def _safe_top_values(s: pd.Series, k: int = 3) -> str:
    if s.empty:
        return ""
    vc = s.value_counts(dropna=True)
    if vc.empty:
        return ""
    items = vc.head(k)
    return "; ".join([f"{repr(idx)}:{int(cnt)}" for idx, cnt in items.items()])


def _safe_samples(s: pd.Series, k: int = 3) -> str:
    # 결측 제외 후 앞에서 k개 샘플
    ss = s.dropna()
    if ss.empty:
        return ""
    return ", ".join([repr(x) for x in ss.head(k).tolist()])


def _series_profile(s: pd.Series) -> dict[str, object]:
    n = len(s)
    na = int(s.isna().sum())
    non_na = n - na
    nunique = int(s.nunique(dropna=True))
    dtype = str(s.dtype)

    return {
        "dtype": dtype,
        "n_rows": n,
        "na_cnt": na,
        "na_rate": (na / n) if n else np.nan,
        "non_na_cnt": non_na,
        "nunique": nunique,
        "top_values": _safe_top_values(s),
        "samples": _safe_samples(s),
    }


def compare_schema(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str = "left",
    right_name: str = "right",
    include: Literal["all", "union", "intersection"] = "union",
    sort_by: Literal["name", "na_diff", "dtype_mismatch", "presence"] = "presence",
) -> pd.DataFrame:
    """
    두 DataFrame의 스키마/결측/기초 통계를 비교하는 EDA 리포트를 만든다.

    Parameters
    ----------
    left, right:
        비교 대상 DataFrame.
    left_name, right_name:
        리포트 컬럼명 접두어로 사용할 이름.
    include:
        - "union": 양쪽 컬럼 합집합(기본)
        - "intersection": 교집합만
        - "all": union과 동일(호환용)
    sort_by:
        - "presence": 한쪽에만 있는 컬럼 우선
        - "dtype_mismatch": dtype 불일치 우선
        - "na_diff": 결측률 차이 큰 컬럼 우선
        - "name": 컬럼명 정렬

    Returns
    -------
    pd.DataFrame
        컬럼별 비교 리포트.
    """
    left_cols = list(left.columns)
    right_cols = list(right.columns)

    if include in ("union", "all"):
        cols = sorted(set(left_cols).union(right_cols))
    elif include == "intersection":
        cols = sorted(set(left_cols).intersection(right_cols))
    else:
        raise ValueError(f"include={include} is invalid")

    rows: list[dict[str, object]] = []
    for c in cols:
        in_left = c in left.columns
        in_right = c in right.columns

        lp = _series_profile(left[c]) if in_left else None
        rp = _series_profile(right[c]) if in_right else None

        row: dict[str, object] = {
            "col": c,
            "in_left": in_left,
            "in_right": in_right,
            "presence": (
                "both"
                if in_left and in_right
                else f"only_{left_name}"
                if in_left
                else f"only_{right_name}"
            ),
        }

        # 왼쪽/오른쪽 프로필 펼치기
        if lp:
            for k, v in lp.items():
                row[f"{left_name}_{k}"] = v
        else:
            for k in [
                "dtype",
                "n_rows",
                "na_cnt",
                "na_rate",
                "non_na_cnt",
                "nunique",
                "top_values",
                "samples",
            ]:
                row[f"{left_name}_{k}"] = np.nan if k in ("na_rate",) else None

        if rp:
            for k, v in rp.items():
                row[f"{right_name}_{k}"] = v
        else:
            for k in [
                "dtype",
                "n_rows",
                "na_cnt",
                "na_rate",
                "non_na_cnt",
                "nunique",
                "top_values",
                "samples",
            ]:
                row[f"{right_name}_{k}"] = np.nan if k in ("na_rate",) else None

        # 비교 지표
        dtype_mismatch = (in_left and in_right) and (
            row[f"{left_name}_dtype"] != row[f"{right_name}_dtype"]
        )
        na_diff = (
            float_or_nan(row[f"{left_name}_na_rate"])
            - float_or_nan(row[f"{right_name}_na_rate"])
            if (in_left and in_right)
            else np.nan
        )

        row["dtype_mismatch"] = bool(dtype_mismatch)
        row["na_rate_diff"] = float(na_diff) if pd.notna(na_diff) else np.nan

        rows.append(row)

    rep = pd.DataFrame(rows)

    # 정렬
    if sort_by == "presence":
        # 한쪽만 있는 컬럼 -> dtype mismatch -> 나머지
        rep["_presence_rank"] = rep["presence"].apply(lambda x: 0 if x != "both" else 1)
        rep["_dtype_rank"] = rep["dtype_mismatch"].astype(int).map({1: 0, 0: 1})
        rep = rep.sort_values(["_presence_rank", "_dtype_rank", "col"]).drop(
            columns=["_presence_rank", "_dtype_rank"]
        )
    elif sort_by == "dtype_mismatch":
        rep = rep.sort_values(["dtype_mismatch", "col"], ascending=[False, True])
    elif sort_by == "na_diff":
        rep = rep.sort_values(["na_rate_diff", "col"], ascending=[False, True])
    elif sort_by == "name":
        rep = rep.sort_values("col")

    return rep.reset_index(drop=True)
