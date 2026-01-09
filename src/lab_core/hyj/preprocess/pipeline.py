from __future__ import annotations

import pandas as pd


class BaseBlock:
    def fit(self, X: pd.DataFrame) -> None:
        return None

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        raise NotImplementedError


class Pipeline:
    def __init__(self, blocks: list[BaseBlock]):
        self.blocks = blocks

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for b in self.blocks:
            b.fit(X)  # train only
            X = b.transform(X, is_train=True)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for b in self.blocks:
            X = b.transform(X, is_train=False)
        return X
