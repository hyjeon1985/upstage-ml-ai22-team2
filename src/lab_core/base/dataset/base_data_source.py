from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data_cache import CacheStore


@dataclass(frozen=True)
class DataSpec:
    target_col: str
    feature_cols: list[str] | None = None


class BaseDataSource(ABC):
    """
    데이터 원천을 제공하는 최소한의 데이터 소스.

    - train/test 데이터 제공
    - 데이터 스펙 제공
    - 디스크 캐시는 CacheStore에 위임
    """

    def __init__(self, *, cache: CacheStore | None = None) -> None:
        self._cache = cache or CacheStore()

    @abstractmethod
    def load_train(self) -> pd.DataFrame:
        """학습 데이터 로드."""
        raise NotImplementedError

    @abstractmethod
    def load_test(self) -> pd.DataFrame:
        """테스트 데이터 로드."""
        raise NotImplementedError

    @abstractmethod
    def spec(self) -> DataSpec:
        """데이터 스펙(타겟, 컬럼 목록 등)을 반환."""
        raise NotImplementedError

    def build_refs(
        self,
        *,
        train: pd.DataFrame,
        test: pd.DataFrame,
        spec: DataSpec,
    ) -> dict[str, Any]:
        """참조 테이블을 생성한다. 기본은 빈 dict."""
        _ = train
        _ = test
        _ = spec
        return {}

    def get_train(self, *, use_cache: bool = True) -> pd.DataFrame:
        if not use_cache:
            return self.load_train()
        cached = self._cache.get("train")
        if cached is None:
            cached = self.load_train()
            self._cache.set("train", cached)
        return cached

    def get_test(self, *, use_cache: bool = True) -> pd.DataFrame:
        if not use_cache:
            return self.load_test()
        cached = self._cache.get("test")
        if cached is None:
            cached = self.load_test()
            self._cache.set("test", cached)
        return cached

    def get_spec(self, *, use_cache: bool = True) -> DataSpec:
        if not use_cache:
            return self.spec()
        cached = self._cache.get_obj("spec")
        if cached is None:
            cached = self.spec()
            self._cache.set_obj("spec", cached)
        return cached

    def get_refs(self, *, use_cache: bool = True) -> dict[str, Any]:
        if not use_cache:
            train = self.load_train()
            test = self.load_test()
            spec = self.spec()
            return self.build_refs(train=train, test=test, spec=spec)
        cached = self._cache.get_obj("refs")
        if cached is None:
            train = self.get_train(use_cache=True)
            test = self.get_test(use_cache=True)
            spec = self.get_spec(use_cache=True)
            cached = self.build_refs(train=train, test=test, spec=spec)
            self._cache.set_obj("refs", cached)
        return cached
