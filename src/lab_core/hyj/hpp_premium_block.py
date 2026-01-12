from __future__ import annotations

import json

import pandas as pd

from .core.dataset.pipeline import BaseBlock

DEFAULT_GU_COL = "구"
DEFAULT_PREMIUM_AREA_OUT_COL = "강남권여부"
DEFAULT_PREMIUM_AREA_SET_COLS = {
    "강서구",
    "영등포구",
    "동작구",
    "서초구",
    "강남구",
    "송파구",
    "강동구",
}


class HppPremiumAreaBlock(BaseBlock):
    """
    프리미엄 권역 여부를 파생한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        gu_col: str = DEFAULT_GU_COL,
        out_col: str = DEFAULT_PREMIUM_AREA_OUT_COL,
        premium: set[str] = DEFAULT_PREMIUM_AREA_SET_COLS,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.gu_col = gu_col
        self.out_col = out_col
        self.premium = premium

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        out[self.out_col] = (
            out[self.gu_col].astype(str).isin(self.premium).astype("int8")
        )
        return out

    def describe(self) -> dict[str, str]:
        return {
            "gu_col": self.gu_col,
            "out_col": self.out_col,
            "premium": json.dumps(sorted(self.premium), ensure_ascii=False),
        }
