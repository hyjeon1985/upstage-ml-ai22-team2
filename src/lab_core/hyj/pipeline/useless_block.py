import numpy as np
import pandas as pd

from .pipeline import BaseBlock


class UselessValueToNaBlock(BaseBlock):
    def __init__(self, rules: dict[str, list[str]]):
        """
        rules 예:
        {
            "등기신청일자": [" "],
            "거래유형": ["-"],
        }
        """
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


class DropUselessColumnsBlock(BaseBlock):
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame) -> None:
        pass  # 학습 없음

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()
        drop_cols = [c for c in self.cols if c in X.columns]
        return X.drop(columns=drop_cols)
