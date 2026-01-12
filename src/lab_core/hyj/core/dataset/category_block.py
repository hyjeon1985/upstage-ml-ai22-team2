from __future__ import annotations

import json

import pandas as pd

from .pipeline import BaseBlock


class CategoryCleanBlock(BaseBlock):
    """
    범주형 컬럼 정리:
    - 공백/빈 문자열 제거
    - 결측치 통일 토큰으로 치환
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        cols: list[str],
        fill_value: str = "__NA__",
    ):
        super().__init__(meta_cols=meta_cols, name=name)
        self.cols = cols
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame) -> None:
        return None

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        for col in self.cols:
            if col not in out.columns:
                continue
            s = out[col].astype("string").str.strip()
            s = s.replace("", pd.NA)
            out[col] = s.fillna(self.fill_value)
        return out

    def describe(self) -> dict[str, str]:
        return {
            "cols": json.dumps(self.cols, ensure_ascii=False),
            "fill_value": self.fill_value,
        }


class FitCategoriesBlock(BaseBlock):
    """
    범주형 카테고리 정렬/고정 예시.

    - fit에서 train 카테고리를 저장하고
    - transform에서 동일 카테고리를 적용한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        cat_cols: list[str],
    ):
        super().__init__(meta_cols, name)
        self.cat_cols = cat_cols
        self.categories_: dict[str, pd.Index] = {}

    def fit(self, X: pd.DataFrame) -> None:
        for c in self.cat_cols:
            if c in X.columns:
                self.categories_[c] = pd.Series(X[c].astype("category")).cat.categories

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        for c, cats in self.categories_.items():
            if c in out.columns:
                out[c] = pd.Categorical(out[c], categories=cats)
        return out

    def describe(self) -> dict[str, str]:
        return {
            "cat_cols": json.dumps(self.cat_cols, ensure_ascii=False),
            "categories": json.dumps(
                {k: list(v) for k, v in self.categories_.items()}, ensure_ascii=False
            ),
        }
