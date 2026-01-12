from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from lab_core.util.path import out_dir
from lab_core.util.time_ids import make_run_id

from .dataset.builder import BaseDatasetBuilder, DatasetBundle, ModelKind


@dataclass(frozen=True)
class RunResult:
    pred: pd.Series
    meta: dict[str, Any]
    artifacts: dict[str, Any]
    run_id: str
    run_dir: Path
    pred_path: Path
    meta_path: Path


class BaseRunner(ABC):
    """
    모델 학습/추론을 담당하는 기본 러너.

    - 데이터셋 빌더는 전처리/분할까지 책임진다.
    - 러너는 모델 학습/추론/저장을 책임진다.
    """

    def __init__(self, builder: BaseDatasetBuilder) -> None:
        self.builder = builder

    def build_dataset(
        self,
        *,
        model: Optional[ModelKind] = None,
        use_cache: bool = True,
        split_policy: dict[str, Any] | None = None,
    ) -> DatasetBundle:
        return self.builder.build(
            model=model, use_cache=use_cache, split_policy=split_policy
        )

    def run(
        self,
        *,
        model: Optional[ModelKind] = None,
        seed: int = 42,
        use_cache: bool = True,
        split_policy: dict[str, Any] | None = None,
        run_prefix: str = "submission",
        out_subdir: str = "subs",
    ) -> RunResult:
        print(f"[runner] start: model={model}, seed={seed}")
        bundle = self.build_dataset(
            model=model, use_cache=use_cache, split_policy=split_policy
        )
        print("[runner] dataset ready")
        pred, artifacts = self.train_predict(bundle, model=model, seed=seed)
        print("[runner] predict done")

        model_tag = artifacts.get("model_tag", model or "model")
        run_id = make_run_id(run_prefix, mid_id=self.__class__.__name__)
        run_dir = out_dir(out_subdir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        pred_path = run_dir / f"{run_id}_{model_tag}.csv"
        meta_path = run_dir / "meta.json"

        pd.DataFrame({"target": BaseRunner._to_int_pred(pred)}).to_csv(
            pred_path, index=False
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(bundle.meta, f, ensure_ascii=False, indent=2)
        print(f"[runner] saved: {pred_path}")

        return RunResult(
            pred=pred,
            meta=bundle.meta,
            artifacts=artifacts,
            run_id=run_id,
            run_dir=run_dir,
            pred_path=pred_path,
            meta_path=meta_path,
        )

    @abstractmethod
    def train_predict(
        self,
        bundle: DatasetBundle,
        *,
        model: Optional[ModelKind],
        seed: int,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """모델 학습/예측을 수행하고 결과와 부가 정보를 반환한다."""
        raise NotImplementedError

    @staticmethod
    def _to_int_pred(pred: pd.Series) -> pd.Series:
        return pred.round().astype(int)


class BasicRunner(BaseRunner):
    """
    도메인 독립 러너 베이스.
    """

    def __init__(self, builder: BaseDatasetBuilder) -> None:
        super().__init__(builder)
