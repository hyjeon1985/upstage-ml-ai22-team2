from __future__ import annotations

from typing import Any

import pandas as pd


def assert_same_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    if train_cols != test_cols:
        only_train = sorted(train_cols - test_cols)
        only_test = sorted(test_cols - train_cols)
        raise ValueError(
            "train/test 컬럼이 일치하지 않습니다.\n"
            f" - train only: {only_train}\n"
            f" - test only : {only_test}"
        )


def assert_same_dtypes(train: pd.DataFrame, test: pd.DataFrame) -> None:
    # 공통 컬럼만 비교
    diffs = []
    for c in train.columns:
        if train[c].dtype != test[c].dtype:
            diffs.append((c, str(train[c].dtype), str(test[c].dtype)))
    if diffs:
        msg = "\n".join([f"- {c}: train={dt1}, test={dt2}" for c, dt1, dt2 in diffs])
        raise ValueError("train/test dtype 불일치:\n" + msg)


def assert_allowed_values(s: pd.Series, allowed: set[Any]) -> None:
    actual = set(s.dropna().unique().tolist())
    extra = actual - allowed
    if extra:
        raise ValueError(
            f"'{s.name}'에 허용되지 않은 값이 있습니다: {sorted(extra)} "
            f"(allowed={sorted(allowed)})"
        )
