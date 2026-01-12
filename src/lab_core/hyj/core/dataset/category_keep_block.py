from __future__ import annotations

import json

import pandas as pd

from .pipeline import BaseBlock


class CategoryKeepOthersBlock(BaseBlock):
    """
    지정한 값만 유지하고 나머지는 OTHERS로 묶는다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        src_col: str,
        out_col: str,
        keep_values: set[str],
        other_value: str = "OTHERS",
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.src_col = src_col
        self.out_col = out_col
        self.keep_values = keep_values
        self.other_value = other_value

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        s = out[self.src_col].astype("string").str.strip()
        s = s.replace("", pd.NA).fillna(self.other_value)
        out[self.out_col] = s.where(s.isin(self.keep_values), self.other_value)
        return out

    def describe(self) -> dict[str, str]:
        return {
            "src_col": self.src_col,
            "out_col": self.out_col,
            "keep_values": json.dumps(sorted(self.keep_values), ensure_ascii=False),
            "other_value": self.other_value,
        }
