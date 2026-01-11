import json

import numpy as np
import pandas as pd

from .pipeline import BaseBlock


class UselessValueToNaBlock(BaseBlock):
    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        rules: dict[str, list[str]],
    ):
        super().__init__(meta_cols, name)
        self.rules = rules

    def fit(self, X: pd.DataFrame) -> None:
        pass  # 학습할 것 없음

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()
        for col, values in self.rules.items():
            if col in X.columns:
                X[col] = X[col].replace(values, np.nan)
        return X

    def describe(self) -> dict[str, str]:
        return {"rules": json.dumps(self.rules, ensure_ascii=False)}
