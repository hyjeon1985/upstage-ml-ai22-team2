from __future__ import annotations

import json

import pandas as pd

from .pipeline import BaseBlock


class FrequencyEncodeBlock(BaseBlock):
    """
    범주형 컬럼의 빈도(비율)를 인코딩한다.

    - fit: train의 빈도 맵 생성
    - transform: 빈도 맵을 적용해 신규 컬럼 생성
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        col_map: dict[str, str],
        na_token: str = "__NA__",
        default_value: float = 0.0,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.col_map = col_map
        self.na_token = na_token
        self.default_value = default_value
        self._freq_maps: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame) -> None:
        self._freq_maps = {}
        for src in self.col_map.keys():
            if src not in X.columns:
                continue
            s = X[src].astype("string").str.strip()
            s = s.replace("", pd.NA).fillna(self.na_token)
            freq = s.value_counts(normalize=True)
            self._freq_maps[src] = freq.to_dict()

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        for src, out_col in self.col_map.items():
            if src not in out.columns:
                continue
            s = out[src].astype("string").str.strip()
            s = s.replace("", pd.NA).fillna(self.na_token)
            freq_map = self._freq_maps.get(src, {})
            out[out_col] = s.map(freq_map).fillna(self.default_value)
        return out

    def describe(self) -> dict[str, str]:
        return {
            "col_map": json.dumps(self.col_map, ensure_ascii=False),
            "na_token": self.na_token,
            "default_value": str(self.default_value),
        }
