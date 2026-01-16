from __future__ import annotations

import pandas as pd

from .base_block import BaseBlock


class Pipeline:
    """
    BaseBlock 여러 개를 순서대로 적용하는 간단한 파이프라인.
    """

    def __init__(self, blocks: list[BaseBlock]):
        self.blocks = blocks

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for b in self.blocks:
            b.fit(X)
            X = b.transform(X, is_train=True)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for b in self.blocks:
            X = b.transform(X, is_train=False)
        return X

    def summarize(self) -> list[dict[str, dict[str, str]]]:
        return [{b.name(): b.describe()} for b in self.blocks]
