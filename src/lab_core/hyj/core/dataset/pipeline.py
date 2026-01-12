from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseBlock(ABC):
    """
    Pipeline을 구성하는 '전처리/피처 엔지니어링 블록'의 최소 인터페이스.

    설계 의도
    ----------
    - sklearn의 Transformer(=fit/transform) 느낌을 최소한으로 흉내내되,
      학습(train)과 추론(test) 시점의 처리가 달라질 수 있으므로
      transform에 `is_train: bool`을 추가했다.

    구현 규칙(중요)
    --------------
    1) fit(X)
       - train 데이터에 대해서만 호출된다. (Pipeline.fit_transform에서만 호출)
       - 범주형 카테고리 목록 저장, 스케일 파라미터 계산 등
         '학습 데이터에만 기반해야 하는 상태(state)'를 여기서 준비한다.
       - 반환값은 없으며(self에 상태를 저장), X를 변경하지 않는 것을 권장한다.

    2) transform(X, is_train)
       - train/valid/test 어디에서든 호출될 수 있다.
       - `is_train=True`: 학습용 전처리
         예) 타깃 누수 방지 위해 학습에서만 가능한 필터링/이상치 제거 적용
       - `is_train=False`: 추론용 전처리
         예) 행 drop 같은 데이터 손실은 일반적으로 피하고(정책적으로 결정),
             학습 시에 결정한 카테고리/통계치를 그대로 적용한다.
       - 반드시 DataFrame을 반환해야 한다.

    3) X.copy() 정책
       - Pipeline이 각 단계 시작에 X.copy()를 수행한다.
       - 그럼에도 블록 내부에서 안전을 위해 out = X.copy()를 추가하는 것은 허용된다.
         (특히 초보 단계에서는 안전지향이 낫다)

    4) 메타 컬럼
       - 블록 생성 시 메타 컬럼 목록을 전달받고 내부 로직에서만 참조한다.
       - 메타 컬럼 제거는 "모델 입력 직전" 정리 단계에서 처리한다.
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
        """train 데이터에서만 상태(state)를 학습한다. (기본은 no-op)"""
        return None

    @abstractmethod
    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        """
        DataFrame을 변환하여 반환한다.

        Parameters
        ----------
        X:
            입력 데이터프레임.
        is_train:
            True면 학습용 transform, False면 추론용 transform.

        Returns
        -------
        pd.DataFrame:
            변환된 데이터프레임.

        Notes
        -----
        - 이 메서드는 반드시 오버라이드해야 한다.
        """
        raise NotImplementedError

    def name(self) -> str:
        """
        블록을 대표하는 짧은 이름을 반환한다.
        """
        if self._name:
            return self._name
        return self._to_snake(self.__class__.__name__)

    @abstractmethod
    def describe(self) -> dict[str, str]:
        """
        블록의 핵심 설정값을 요약한다.

        - 값은 JSON으로 기록 가능한 형태여야 한다.
        """
        raise NotImplementedError

    @staticmethod
    def _to_snake(value: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()


class Pipeline:
    """
    BaseBlock 여러 개를 순서대로 적용하는 간단한 파이프라인.

    사용 패턴
    --------
    1) 학습(Train) 시:
        pipe = Pipeline([
            BlockA(...),
            BlockB(...),
            ...
            FinalizeBlock(...),
        ])
        X_train_fe = pipe.fit_transform(X_train)

    2) 추론(Test) 시:
        X_test_fe = pipe.transform(X_test)

    호출 순서 보장
    -------------
    - fit_transform(X_train):
        각 블록에 대해
            b.fit(X_train)           # train only
            X_train = b.transform(X_train, is_train=True)

    - transform(X_test):
        각 블록에 대해
            X_test = b.transform(X_test, is_train=False)

    주의(중요)
    ----------
    - 이 Pipeline은 y를 다루지 않는다. (X 전처리 전용)
      y 필터링이 필요하면:
        1) X에 메타 컬럼으로 기준을 남기고(예: drop_mask),
        2) 메인 로직에서 y를 동일 기준으로 정렬/필터링
      혹은 별도의 데이터 컨테이너 설계를 고려한다.

    - train에서만 행 drop을 하는 블록이 있으면,
      train의 y도 같은 행을 제거해야 한다.
      이 경우 `_origin_index`를 유지하면 y 정합성 유지가 쉬워진다.
    """

    def __init__(self, blocks: list[BaseBlock]):
        self.blocks = blocks

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        학습 데이터 전처리.

        - 각 블록의 fit을 먼저 호출하여 train 기반 상태를 학습하고
        - 이어서 transform(is_train=True)로 train 변환을 수행한다.
        """
        X = X.copy()
        for b in self.blocks:
            print(f"[pipeline] fit: {b.name()}")
            b.fit(X)  # train only
            print(f"[pipeline] transform(train): {b.name()}")
            X = b.transform(X, is_train=True)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        추론 데이터 전처리.

        - fit은 호출하지 않는다. (train에서 학습한 상태를 재사용)
        - transform(is_train=False)만 수행한다.
        """
        X = X.copy()
        for b in self.blocks:
            print(f"[pipeline] transform: {b.name()}")
            X = b.transform(X, is_train=False)
        return X

    def summarize(self) -> list[dict[str, dict[str, str]]]:
        """
        파이프라인 구성 요약을 만든다.

        - blocks 속성이 있으면 name/describe 기반으로 기록한다.
        """
        return [{b.name(): b.describe()} for b in self.blocks]
