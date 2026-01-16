from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRunner(ABC):
    """
    단일 모델 실행을 담당하는 러너 베이스.
    """

    def __init__(self, *, name: str | None = None) -> None:
        self._name = name

    def name(self) -> str:
        return self._name or self.__class__.__name__

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """모델 학습/예측을 수행하고 결과를 반환한다."""
        raise NotImplementedError
