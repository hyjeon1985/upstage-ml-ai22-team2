from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseBlock(ABC):
    """
    전처리/피처 엔지니어링 블록의 최소 인터페이스.

    - fit: 학습 데이터 기반 상태 저장
    - transform: train/valid/test 변환
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        use_cache: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self._meta_cols = meta_cols
        self._name = name
        self._use_cache = use_cache
        self._cache_dir = cache_dir

    def fit(self, X: pd.DataFrame) -> None:
        """train 데이터에서만 상태를 학습한다. (기본 no-op)"""
        return None

    @abstractmethod
    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        """데이터프레임을 변환하여 반환한다."""
        raise NotImplementedError

    def name(self) -> str:
        if self._name:
            return self._name
        return self._to_snake(self.__class__.__name__)

    @abstractmethod
    def describe(self) -> dict[str, str]:
        """블록 설정을 요약해 JSON 문자열 dict로 반환한다."""
        raise NotImplementedError

    @staticmethod
    def _to_snake(value: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()
