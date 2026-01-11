from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

ModelKind = Literal["lgbm", "rf", "cat", "linear"]
SplitKind = Literal["contract_ym", "time_holdout", "rolling_window", "kfold", "group_kfold"]


class PipelineLike(Protocol):
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def summarize(self) -> list[dict[str, dict[str, str]]]: ...


@dataclass(frozen=True)
class DatasetBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    X_valid: pd.DataFrame | None
    y_valid: pd.Series | None
    meta: dict[str, Any]


class BaseDatasetBuilder(ABC):
    """
    DatasetBuilder는 '데이터셋의 소유자'다.

    핵심 원칙
    --------
    - train/test concat 금지.
    - fit은 train에만 적용.
    - test는 transform만 적용.
    - _org_idx(원본 인덱스)를 내부 고정 컬럼명으로 사용하여 y 정렬을 안정화한다.
    - rename 정책은 Builder 내부에서 통제한다(외부 주입 지양).

    확장 포인트
    ----------
    - load_raw(): raw train/test 로드 (필수)
    - load_ext(): 외부 데이터 로드/캐시 (선택)
    - rename_map(): 컬럼명 정리 정책 (선택)
    - make_pipeline(ext): 파이프라인 구성 (필수)
    - transform_target(y): 타겟 변환 정책 (선택)
    - finalize_for_model(model): 모델별 입력 포맷 정리 (선택)
    - feature_cols(model): 최종 입력 컬럼 선택 (선택)
    """

    _ORG_IDX_COL = "_org_idx"
    _META_COLS: tuple[str, ...] = (_ORG_IDX_COL,)

    def __init__(self) -> None:
        self._raw: tuple[pd.DataFrame, pd.DataFrame, str] | None = None
        self._ext: dict[str, Any] | None = None
        self._pipeline: PipelineLike | None = None
        self._rm: dict[str, str] | None = None
        self._cache_dir: Path | None = None

    # -----------------
    # required overrides
    # -----------------
    @abstractmethod
    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        train/test 로드 및 target_col 반환.

        Builder가 파일 경로/포맷과 target_col을 알고 있어야 한다.
        """
        raise NotImplementedError

    @abstractmethod
    def make_pipeline(
        self,
        *,
        meta_cols: tuple[str, ...],
        ext: dict[str, Any],
        use_cache: bool,
    ) -> PipelineLike:
        """외부데이터(ext)를 포함하여 블록을 구성하고 Pipeline을 반환."""
        raise NotImplementedError

    # -----------------
    # optional overrides
    # -----------------
    def load_ext(self) -> dict[str, Any]:
        """교통/좌표 등 외부 데이터를 로드해 반환. 기본은 빈 dict."""
        return {}

    def rename_map(self) -> dict[str, str]:
        """(원본→정리된 이름) 리네임 맵. 기본은 비움."""
        return {}

    def cache_dir(self) -> Path:
        """디스크 캐시 경로. 구체 Builder에서 정의한다."""
        raise NotImplementedError

    def transform_target(self, y: pd.Series) -> tuple[pd.Series, str, dict[str, Any]]:
        """
        타겟 변환 정책.

        - 기본은 변환 없음.
        - 반환값은 (변환된 y, 변환 이름, 변환 파라미터) 형태다.
        """
        return y, "none", {}

    def apply_target_transform(
        self, y: pd.Series, *, transform: str, params: dict[str, Any]
    ) -> pd.Series:
        """학습 시 산출된 변환 정보를 기반으로 y를 변환한다."""
        _ = params
        if transform == "none":
            return y
        if transform == "log1p":
            return pd.Series(np.log1p(y), index=y.index)
        raise ValueError(f"unknown target transform: {transform}")

    def finalize_for_model(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, *, model: Optional[ModelKind]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """모델별 dtype/인코딩 정리. 기본 no-op."""
        _ = model
        return X_train, X_test

    def feature_cols(self, *, model: Optional[ModelKind]) -> Optional[Sequence[str]]:
        """최종 입력 컬럼. None이면 meta만 제거하고 전부 사용."""
        _ = model
        return None

    def feature_meta(self, *, model: Optional[ModelKind]) -> Optional[dict[str, list[str]]]:
        """피처 메타 정보(수치/범주)를 반환한다."""
        _ = model
        return None

    def split_train_valid(
        self, X: pd.DataFrame, y: pd.Series, *, split_policy: dict[str, Any]
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """분할 정책에 따라 train/valid를 나눈다."""
        raise NotImplementedError

    @staticmethod
    def inverse_target(
        y: pd.Series, *, transform: str, params: dict[str, Any]
    ) -> pd.Series:
        """
        타겟 복원 정책.

        - transform 이름으로 역변환 규칙을 찾는다.
        - 기본은 'none'만 지원한다.
        """
        _ = params
        if transform == "none":
            return y
        if transform == "log1p":
            return pd.Series(np.expm1(y), index=y.index)
        raise ValueError(f"unknown target transform: {transform}")

    # -----------------
    # public entrypoint
    # -----------------
    def build(
        self,
        *,
        model: Optional[ModelKind] = None,
        use_cache: bool = True,
        split_policy: dict[str, Any] | None = None,
    ) -> DatasetBundle:
        """
        use_cache=True이면 raw/ext/pipeline을 메모리에 캐시한다.
        """
        print("[builder] start")
        # 0) raw + ext 로드(Builder 내부에서 수행)
        print("[builder] load_raw/load_ext")
        train, test, target_col = self._get_or_load_raw(use_cache=use_cache)
        ext = self._get_or_load_ext(use_cache=use_cache)

        # 1) rename (train/test 동일 적용)
        print("[builder] rename_map")
        rm = self._get_rename_map(use_cache=use_cache)
        train, test = self._rename_train_test(train, test, rm, target_col)

        # 2) split X/y + attach _org_idx
        print("[builder] split X/y + _org_idx")
        X_train, y_train, X_test = self._split_and_attach_org_idx(
            train, test, target_col
        )

        # 3) split (필요 시) - 파이프라인 전에 분할
        print("[builder] time_holdout split" if split_policy else "[builder] no split")
        X_train_base = X_train
        y_train_base = y_train
        X_valid = None
        y_valid = None
        if split_policy is not None:
            X_train_base, y_train_base, X_valid, y_valid = self.split_train_valid(
                X_train, y_train, split_policy=split_policy
            )

        # 4) pipeline fit/transform
        print("[builder] make pipeline")
        pipe = self._pipeline if use_cache else None
        if pipe is None:
            pipe = self.make_pipeline(
                meta_cols=self._META_COLS,
                ext=ext,
                use_cache=use_cache,
            )
            self._pipeline = pipe

        print("[builder] fit_transform train")
        X_train_fe = pipe.fit_transform(X_train_base)
        print("[builder] transform test")
        X_test_fe = pipe.transform(X_test)
        print("[builder] transform valid" if X_valid is not None else "[builder] skip valid")
        X_valid_fe = pipe.transform(X_valid) if X_valid is not None else None

        # 5) y align (row-drop 대응)
        print("[builder] align y")
        y_aligned = self._align_y(X_train_fe, y_train_base)
        y_valid_aligned = (
            self._align_y(X_valid_fe, y_valid)
            if X_valid_fe is not None and y_valid is not None
            else None
        )

        # 6) 타겟 스케일링/정규화(구체 Builder에서 override)
        print("[builder] target transform")
        y_scaled, target_transform, target_params = self.transform_target(y_aligned)
        y_valid_scaled = (
            self.apply_target_transform(
                y_valid_aligned, transform=target_transform, params=target_params
            )
            if y_valid_aligned is not None
            else None
        )

        # 7) model finalize
        print("[builder] finalize for model")
        X_train_fin, X_test_fin = self.finalize_for_model(
            X_train_fe, X_test_fe, model=model
        )
        X_valid_fin = None
        if X_valid_fe is not None:
            _, X_valid_fin = self.finalize_for_model(
                X_train_fe, X_valid_fe, model=model
            )

        # 8) feature select (meta drop 포함)
        print("[builder] feature select")
        feat_cols = self.feature_cols(model=model)
        X_train_sel, X_test_sel = self._select_features(
            X_train_fin, X_test_fin, feature_cols=feat_cols
        )
        if X_valid_fin is not None:
            X_valid_fin, _ = self._select_features(
                X_valid_fin, X_valid_fin, feature_cols=feat_cols
            )

        print("[builder] done")
        return DatasetBundle(
            X_train=X_train_sel,
            y_train=y_scaled,
            X_test=X_test_sel,
            X_valid=X_valid_fin,
            y_valid=y_valid_scaled,
            meta={
                "model": model,
                "target_col": target_col,
                "rename_map": rm,
                "ext_datas": list(ext.keys()),
                # "pipeline": pipe.summarize(),
                "target_transform": target_transform,
                "target_trans_params": target_params,
                # "feature_cols": feat_cols,
                "feature_meta": self.feature_meta(model=model),
                "split_policy": split_policy,
            },
        )

    # -----------------
    # internal utilities
    # -----------------
    def _get_or_load_raw(
        self, *, use_cache: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not use_cache:
            return self.load_raw()

        if self._raw is None:
            self._raw = self.load_raw()
        return self._raw

    def _get_or_load_ext(self, *, use_cache: bool) -> dict[str, Any]:
        if not use_cache:
            return self.load_ext()

        if self._ext is None:
            self._ext = self.load_ext()
        return self._ext

    def _get_rename_map(self, *, use_cache: bool) -> dict[str, str]:
        if not use_cache:
            return self.rename_map()

        if self._rm is None:
            self._rm = self.rename_map()
        return self._rm

    def _get_cache_dir(self, *, use_cache: bool) -> Path:
        if not use_cache:
            return self.cache_dir()
        if self._cache_dir is None:
            self._cache_dir = self.cache_dir()
        return self._cache_dir

    def _rename_train_test(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        rm: dict[str, str],
        target_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not rm:
            return train, test

        missing_train = [
            k for k in rm.keys() if k not in train.columns and k != target_col
        ]
        missing_test = [
            k for k in rm.keys() if k not in test.columns and k != target_col
        ]
        # test에는 target이 없을 수 있으니 target_col은 예외로 둔다.
        if missing_train or missing_test:
            raise KeyError(
                f"rename_map keys not found. train={missing_train}, test={missing_test}"
            )

        return train.rename(columns=rm).copy(), test.rename(columns=rm).copy()

    def _split_and_attach_org_idx(
        self, train: pd.DataFrame, test: pd.DataFrame, target_col: str
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        if target_col not in train.columns:
            raise KeyError(f"target col not found: {target_col}")

        y = train[target_col].copy()
        X_train = train.drop(columns=[target_col]).copy()
        X_test = test.copy()

        if self._ORG_IDX_COL not in X_train.columns:
            X_train[self._ORG_IDX_COL] = X_train.index
        if self._ORG_IDX_COL not in X_test.columns:
            X_test[self._ORG_IDX_COL] = X_test.index

        return X_train, y, X_test

    def _align_y(self, X_train_after: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        if self._ORG_IDX_COL not in X_train_after.columns:
            raise KeyError(
                f"{self._ORG_IDX_COL} missing after pipeline. blocks must preserve it."
            )

        idx = X_train_after[self._ORG_IDX_COL]
        if idx.duplicated().any():
            raise ValueError(
                f"{self._ORG_IDX_COL} has duplicates. cannot align y safely."
            )

        # y_train.index가 _org_idx로 이미 고정되어 있다는 전제
        try:
            return y_train.loc[idx]
        except KeyError as e:
            missing = pd.Index(idx).difference(y_train.index)[:10].tolist()
            raise KeyError(
                f"y_train is not indexed by {self._ORG_IDX_COL}. missing examples: {missing}"
            ) from e

    def _select_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        *,
        feature_cols: Optional[Sequence[str]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols = feature_cols
        drop_meta = self._META_COLS

        if cols is None:
            return (
                X_train.drop(columns=drop_meta, errors="ignore"),
                X_test.drop(columns=drop_meta, errors="ignore"),
            )

        missing_tr = [c for c in cols if c not in X_train.columns]
        missing_te = [c for c in cols if c not in X_test.columns]
        if missing_tr or missing_te:
            raise KeyError(
                f"missing feature cols. train={missing_tr}, test={missing_te}"
            )

        return X_train[list(cols)].copy(), X_test[list(cols)].copy()
