from __future__ import annotations

from typing import Any

from .base_runner import BaseRunner


class RunOrchestrator:
    """
    여러 러너를 조합해 실행하는 오케스트레이터.

    - 앙상블/벤치마크/스태킹 등 다양한 조합을 담당한다.
    """

    def __init__(self, runners: list[BaseRunner]) -> None:
        self.runners = runners

    def run_all(self, **kwargs: Any) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for runner in self.runners:
            results[runner.name()] = runner.run(**kwargs)
        return results
