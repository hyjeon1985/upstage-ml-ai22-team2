from __future__ import annotations

import hashlib
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from lab_core.util.geo import haversine_distance, latlon_to_km

from .core.dataset.pipeline import BaseBlock

DEFAULT_APT_YX = ("좌표Y", "좌표X")
DEFAULT_SUBWAY_YX = ("위도", "경도")
DEFAULT_BUS_YX = ("Y좌표", "X좌표")
DEFAULT_SUBWAY_NAME = "역사명"
DEFAULT_SUBWAY_LINE = "호선"
DEFAULT_BUS_NAME = "정류소명"
DEFAULT_BUS_NO = "정류소번호"

DEFAULT_TRANSPORT_OUT_COLS = {
    "subway_dist": "지하철거리",
    "bus_dist": "버스거리",
    "subway_pt": "지하철500m내점수",
    "bus_pt": "버스300m내점수",
}


class HppTransportBlock(BaseBlock):
    """
    교통 접근성 피처를 생성한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        subway_df: pd.DataFrame,
        bus_df: pd.DataFrame,
        apt_yx: tuple[str, str] = DEFAULT_APT_YX,
        subway_yx: tuple[str, str] = DEFAULT_SUBWAY_YX,
        bus_yx: tuple[str, str] = DEFAULT_BUS_YX,
        subway_name_col: str = DEFAULT_SUBWAY_NAME,
        subway_line_col: str = DEFAULT_SUBWAY_LINE,
        bus_name_col: str = DEFAULT_BUS_NAME,
        bus_no_col: str = DEFAULT_BUS_NO,
        use_cache: bool = False,
        cache_dir: Path | None = None,
        subway_r_km: float = 0.5,
        bus_r_km: float = 0.3,
        out_cols: dict[str, str] = DEFAULT_TRANSPORT_OUT_COLS,
    ) -> None:
        super().__init__(
            meta_cols=meta_cols, name=name, use_cache=use_cache, cache_dir=cache_dir
        )
        self.subway_df = subway_df
        self.bus_df = bus_df
        self.apt_yx = apt_yx
        self.subway_yx = subway_yx
        self.bus_yx = bus_yx
        self.subway_name_col = subway_name_col
        self.subway_line_col = subway_line_col
        self.bus_name_col = bus_name_col
        self.bus_no_col = bus_no_col
        self.subway_r_km = subway_r_km
        self.bus_r_km = bus_r_km
        self.out_cols = out_cols
        self._subway_meta: pd.DataFrame | None = None
        self._bus_meta: pd.DataFrame | None = None

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        t0 = time.perf_counter()

        y_col, x_col = self.apt_yx
        sub_y_col, sub_x_col = self.subway_yx
        bus_y_col, bus_x_col = self.bus_yx

        valid = out[x_col].notna() & out[y_col].notna()
        out[self.out_cols["subway_dist"]] = np.nan
        out[self.out_cols["bus_dist"]] = np.nan
        out[self.out_cols["subway_pt"]] = np.nan
        out[self.out_cols["bus_pt"]] = np.nan

        if not valid.any():
            return out

        t_meta = time.perf_counter()
        subway_meta = self._get_subway_meta(
            y_col=sub_y_col,
            x_col=sub_x_col,
            name_col=self.subway_name_col,
            line_col=self.subway_line_col,
        )
        print(f"[transport] subway meta: {time.perf_counter() - t_meta:.2f}s")
        subway_coords = subway_meta[[sub_y_col, sub_x_col]].values
        subway_names = subway_meta[self.subway_name_col].to_numpy()
        subway_weight_map = self._get_subway_weight_map()
        t_meta = time.perf_counter()
        bus_meta = self._get_bus_meta(
            y_col=bus_y_col,
            x_col=bus_x_col,
            name_col=self.bus_name_col,
            no_col=self.bus_no_col,
        )
        print(f"[transport] bus meta: {time.perf_counter() - t_meta:.2f}s")
        bus_coords = bus_meta[[bus_y_col, bus_x_col]].values
        bus_names = bus_meta[self.bus_name_col].to_numpy()
        bus_weight_map = self._get_bus_weight_map()
        apt_coords = out.loc[valid, [y_col, x_col]].values

        if len(apt_coords) == 0:
            return out

        t_lat = time.perf_counter()
        lat0 = float(np.nanmean(apt_coords[:, 0]))
        apt_yx = latlon_to_km(apt_coords[:, 0], apt_coords[:, 1], lat0=lat0)
        print(f"[transport] apt to_km: {time.perf_counter() - t_lat:.2f}s")

        if len(subway_coords):
            t_sub = time.perf_counter()
            subway_yx = latlon_to_km(
                subway_coords[:, 0], subway_coords[:, 1], lat0=lat0
            )
            subway_tree = cKDTree(subway_yx)
            _, subway_idx = subway_tree.query(apt_yx, k=1)
            print(f"[transport] subway tree/query: {time.perf_counter() - t_sub:.2f}s")
            t_sub = time.perf_counter()
            out.loc[valid, self.out_cols["subway_dist"]] = [
                haversine_distance(
                    apt_coords[i, 0],
                    apt_coords[i, 1],
                    subway_coords[subway_idx[i], 0],
                    subway_coords[subway_idx[i], 1],
                )
                for i in range(len(apt_coords))
            ]
            print(f"[transport] subway dist: {time.perf_counter() - t_sub:.2f}s")
            t_sub = time.perf_counter()
            subway_500m = subway_tree.query_ball_point(
                apt_yx, r=self.subway_r_km, p=2.0
            )
            out.loc[valid, self.out_cols["subway_pt"]] = [
                float(sum(subway_weight_map.get(n, 0.0) for n in set(subway_names[idxs])))
                if idxs
                else 0.0
                for idxs in subway_500m
            ]
            print(f"[transport] subway score: {time.perf_counter() - t_sub:.2f}s")

        if len(bus_coords):
            t_bus = time.perf_counter()
            bus_yx = latlon_to_km(bus_coords[:, 0], bus_coords[:, 1], lat0=lat0)
            bus_tree = cKDTree(bus_yx)
            _, bus_idx = bus_tree.query(apt_yx, k=1)
            print(f"[transport] bus tree/query: {time.perf_counter() - t_bus:.2f}s")
            t_bus = time.perf_counter()
            out.loc[valid, self.out_cols["bus_dist"]] = [
                haversine_distance(
                    apt_coords[i, 0],
                    apt_coords[i, 1],
                    bus_coords[bus_idx[i], 0],
                    bus_coords[bus_idx[i], 1],
                )
                for i in range(len(apt_coords))
            ]
            print(f"[transport] bus dist: {time.perf_counter() - t_bus:.2f}s")
            t_bus = time.perf_counter()
            bus_300m = bus_tree.query_ball_point(apt_yx, r=self.bus_r_km, p=2.0)
            out.loc[valid, self.out_cols["bus_pt"]] = [
                float(sum(bus_weight_map.get(n, 0.0) for n in set(bus_names[idxs])))
                if idxs
                else 0.0
                for idxs in bus_300m
            ]
            print(f"[transport] bus score: {time.perf_counter() - t_bus:.2f}s")

        print(f"[transport] total: {time.perf_counter() - t0:.2f}s")
        return out

    def describe(self) -> dict[str, str]:
        return {
            "apt_yx": json.dumps(self.apt_yx, ensure_ascii=False),
            "subway_yx": json.dumps(self.subway_yx, ensure_ascii=False),
            "bus_yx": json.dumps(self.bus_yx, ensure_ascii=False),
            "subway_name_col": self.subway_name_col,
            "subway_line_col": self.subway_line_col,
            "bus_name_col": self.bus_name_col,
            "bus_no_col": self.bus_no_col,
            "use_cache": str(self._use_cache),
            "cache_dir": str(self._cache_dir) if self._cache_dir else "",
            "subway_r_km": str(self.subway_r_km),
            "bus_r_km": str(self.bus_r_km),
            "out_cols": json.dumps(self.out_cols, ensure_ascii=False),
        }

    def _get_subway_meta(
        self, *, y_col: str, x_col: str, name_col: str, line_col: str
    ) -> pd.DataFrame:
        if self._subway_meta is not None:
            return self._subway_meta
        if self._use_cache:
            cache_path = self._subway_cache_path()
            if cache_path is not None and cache_path.exists():
                with cache_path.open("rb") as f:
                    meta = pickle.load(f)
                self._subway_meta = meta
                return meta

        meta = self.subway_df[[y_col, x_col, name_col, line_col]].dropna().copy()
        meta[name_col] = meta[name_col].astype(str).str.strip()
        meta[line_col] = meta[line_col].astype(str).str.strip()
        line_count_map = meta.groupby(name_col)[line_col].nunique().to_dict()
        # 역사명별 호선 수를 파생 컬럼으로 보관한다.
        meta["호선수"] = meta[name_col].map(line_count_map).fillna(0).astype(int)
        meta["호선수_가중치"] = meta["호선수"].astype(float) ** 0.7
        self._subway_meta = meta
        if self._use_cache:
            cache_path = self._subway_cache_path()
            if cache_path is not None:
                with cache_path.open("wb") as f:
                    pickle.dump(meta, f)
        return meta

    def _get_bus_meta(
        self, *, y_col: str, x_col: str, name_col: str, no_col: str
    ) -> pd.DataFrame:
        if self._bus_meta is not None:
            return self._bus_meta
        if self._use_cache:
            cache_path = self._bus_cache_path()
            if cache_path is not None and cache_path.exists():
                with cache_path.open("rb") as f:
                    meta = pickle.load(f)
                self._bus_meta = meta
                return meta

        meta = self.bus_df[[y_col, x_col, name_col, no_col]].dropna().copy()
        meta[name_col] = meta[name_col].astype(str).str.strip()
        meta[no_col] = meta[no_col].astype(str).str.strip()
        line_count_map = meta.groupby(name_col)[no_col].nunique().to_dict()
        meta["노선수"] = meta[name_col].map(line_count_map).fillna(0).astype(int)
        meta["노선수_가중치"] = meta["노선수"].astype(float) ** 0.7
        self._bus_meta = meta
        if self._use_cache:
            cache_path = self._bus_cache_path()
            if cache_path is not None:
                with cache_path.open("wb") as f:
                    pickle.dump(meta, f)
        return meta

    def _subway_cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "apt_yx": self.apt_yx,
            "subway_yx": self.subway_yx,
            "subway_name_col": self.subway_name_col,
            "subway_line_col": self.subway_line_col,
            "subway_r_km": self.subway_r_km,
        }
        sig = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        key = hashlib.md5(sig).hexdigest()[:12]
        return self._cache_dir / f"{self.name()}_{key}_subway_meta.pkl"

    def _bus_cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "apt_yx": self.apt_yx,
            "bus_yx": self.bus_yx,
            "bus_name_col": self.bus_name_col,
            "bus_no_col": self.bus_no_col,
            "bus_r_km": self.bus_r_km,
        }
        sig = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        key = hashlib.md5(sig).hexdigest()[:12]
        return self._cache_dir / f"{self.name()}_{key}_bus_meta.pkl"

    def _get_subway_weight_map(self) -> dict[str, float]:
        meta = self._get_subway_meta(
            y_col=self.subway_yx[0],
            x_col=self.subway_yx[1],
            name_col=self.subway_name_col,
            line_col=self.subway_line_col,
        )
        return meta.groupby(self.subway_name_col)["호선수_가중치"].max().to_dict()

    def _get_bus_weight_map(self) -> dict[str, float]:
        meta = self._get_bus_meta(
            y_col=self.bus_yx[0],
            x_col=self.bus_yx[1],
            name_col=self.bus_name_col,
            no_col=self.bus_no_col,
        )
        return meta.groupby(self.bus_name_col)["노선수_가중치"].max().to_dict()
