from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


class CacheStore:
    """
    메모리 + 디스크 캐시 저장소.

    - 메모리 캐시 우선
    - 디스크는 parquet/csv/ pickle 순서로 조회
    """

    def __init__(self, *, base_dir: Path | None = None) -> None:
        self._store: dict[str, Any] = {}
        self._base_dir = base_dir

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get_df(self, key: str) -> pd.DataFrame | None:
        cached = self.get(key)
        if isinstance(cached, pd.DataFrame):
            return cached
        if self._base_dir is None:
            return None
        parquet_path, csv_path = self._df_paths(key)
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception:
                pass
        if csv_path.exists():
            return pd.read_csv(csv_path, low_memory=False)
        return None

    def set_df(self, key: str, df: pd.DataFrame) -> None:
        self.set(key, df)
        if self._base_dir is None:
            return
        parquet_path, csv_path = self._df_paths(key)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
        df.to_csv(csv_path, index=False)

    def get_obj(self, key: str) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        if self._base_dir is None:
            return None
        pkl_path = self._obj_path(key)
        if not pkl_path.exists():
            return None
        with pkl_path.open("rb") as f:
            return pickle.load(f)

    def set_obj(self, key: str, value: Any) -> None:
        self.set(key, value)
        if self._base_dir is None:
            return
        pkl_path = self._obj_path(key)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with pkl_path.open("wb") as f:
            pickle.dump(value, f)

    def clear(self) -> None:
        self._store.clear()

    def _df_paths(self, key: str) -> tuple[Path, Path]:
        assert self._base_dir is not None
        safe = key.replace("/", "_")
        return (
            self._base_dir / f"{safe}.parquet",
            self._base_dir / f"{safe}.csv",
        )

    def _obj_path(self, key: str) -> Path:
        assert self._base_dir is not None
        safe = key.replace("/", "_")
        return self._base_dir / f"{safe}.pkl"
