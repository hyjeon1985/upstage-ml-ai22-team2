from __future__ import annotations

import difflib
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from .core.dataset.pipeline import BaseBlock

DEFAULT_TARGET_COLS = {"gu": "구", "dong": "동", "y": "좌표Y", "x": "좌표X"}
DEFAULT_ROAD_COLS = {"full": "도로명_전체", "main": "도로명_앞", "sub": "도로명_뒤"}
DEFAULT_APT_COLS = {
    "gu": "주소(시군구)",
    "dong": "주소(읍면동)",
    "road_main": "주소(도로명)",
    "road_sub": "주소(도로상세주소)",
    "y": "좌표Y",
    "x": "좌표X",
}
DEFAULT_COORD_COLS = {"key": "매핑키", "y": "좌표Y", "x": "좌표X"}


class HppCoordFillBlock(BaseBlock):
    """
    k-apt/coord 데이터를 사용해 좌표 결측을 보정한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        use_cache: bool = False,
        cache_dir: Path | None = None,
        apt_df: pd.DataFrame,
        coord_df: pd.DataFrame,
        target_cols: dict[str, str] = DEFAULT_TARGET_COLS,
        road_cols: dict[str, str] = DEFAULT_ROAD_COLS,
        apt_cols: dict[str, str] = DEFAULT_APT_COLS,
        coord_cols: dict[str, str] = DEFAULT_COORD_COLS,
    ) -> None:
        super().__init__(
            meta_cols=meta_cols, name=name, use_cache=use_cache, cache_dir=cache_dir
        )
        self.apt_df = apt_df
        self.coord_df = coord_df
        self.target_cols = target_cols
        self.road_cols = road_cols
        self.apt_cols = apt_cols
        self.coord_cols = coord_cols
        self._apt_coord_dict: dict[str, tuple[float, float]] | None = None
        self._coord_dict: dict[str, tuple[float, float]] | None = None
        self._apt_index: (
            dict[tuple[str, str, str], list[tuple[str, tuple[float, float]]]] | None
        ) = None

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()

        apt_coord_dict, coord_dict, apt_index = self._get_or_build_coord_refs()

        out = self._fill_coords(
            out,
            coord_dict=apt_coord_dict,
            gu_col=self.target_cols["gu"],
            dong_col=self.target_cols["dong"],
            road_col=self.road_cols["full"],
            y_col=self.target_cols["y"],
            x_col=self.target_cols["x"],
        )
        out = self._fill_coords(
            out,
            coord_dict=coord_dict,
            gu_col=self.target_cols["gu"],
            dong_col=self.target_cols["dong"],
            road_col=self.road_cols["full"],
            y_col=self.target_cols["y"],
            x_col=self.target_cols["x"],
        )
        out = self._fill_coords_by_apt_similarity(
            out,
            apt_index=apt_index,
            gu_col=self.target_cols["gu"],
            dong_col=self.target_cols["dong"],
            main_col=self.road_cols["main"],
            sub_col=self.road_cols["sub"],
            y_col=self.target_cols["y"],
            x_col=self.target_cols["x"],
        )

        # 디버깅용 좌표 스냅샷 저장
        # self._dump_coords_snapshot(out, is_train=is_train)

        return out

    def describe(self) -> dict[str, str]:
        return {
            "target_cols": json.dumps(self.target_cols, ensure_ascii=False),
            "road_cols": json.dumps(self.road_cols, ensure_ascii=False),
            "apt_cols": json.dumps(self.apt_cols, ensure_ascii=False),
            "coord_cols": json.dumps(self.coord_cols, ensure_ascii=False),
            "use_cache": str(self._use_cache),
            "cache_dir": str(self._cache_dir) if self._cache_dir else "",
        }

    def _cache_paths(self) -> tuple[Path, Path, Path] | None:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_cols": self.target_cols,
            "road_cols": self.road_cols,
            "apt_cols": self.apt_cols,
            "coord_cols": self.coord_cols,
        }
        sig = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        key = hashlib.md5(sig).hexdigest()[:12]
        prefix = f"{self.name()}_{key}"
        return (
            self._cache_dir / f"{prefix}_apt_coord.pkl",
            self._cache_dir / f"{prefix}_coord.pkl",
            self._cache_dir / f"{prefix}_apt_index.pkl",
        )

    def _get_or_build_coord_refs(
        self,
    ) -> tuple[
        dict[str, tuple[float, float]],
        dict[str, tuple[float, float]],
        dict[tuple[str, str, str], list[tuple[str, tuple[float, float]]]],
    ]:
        if (
            self._use_cache
            and self._apt_coord_dict is not None
            and self._coord_dict is not None
            and self._apt_index is not None
        ):
            return (
                self._apt_coord_dict,
                self._coord_dict,
                self._apt_index,
            )
        if self._use_cache:
            cache_paths = self._cache_paths()
            if cache_paths is not None and all(p.exists() for p in cache_paths):
                apt_path, coord_path, index_path = cache_paths
                with apt_path.open("rb") as f:
                    apt_coord_dict = pickle.load(f)
                with coord_path.open("rb") as f:
                    coord_dict = pickle.load(f)
                with index_path.open("rb") as f:
                    apt_index = pickle.load(f)
                self._apt_coord_dict = apt_coord_dict
                self._coord_dict = coord_dict
                self._apt_index = apt_index
                return apt_coord_dict, coord_dict, apt_index

        apt = self.apt_df.copy()
        coord = self.coord_df.copy()

        apt = self._build_load_name_full(
            apt,
            main_col=self.apt_cols["road_main"],
            sub_col=self.apt_cols["road_sub"],
            out_full=self.road_cols["full"],
        )

        apt_coord_dict = self._build_coord_dict(
            apt,
            key_col=self.coord_cols["key"],
            gu_col=self.apt_cols["gu"],
            road_col=self.road_cols["full"],
            y_col=self.apt_cols["y"],
            x_col=self.apt_cols["x"],
        )
        coord_dict = self._build_coord_dict(
            coord,
            key_col=self.coord_cols["key"],
            y_col=self.coord_cols["y"],
            x_col=self.coord_cols["x"],
        )
        apt_index = self._build_apt_similarity_index(
            apt,
            gu_col=self.apt_cols["gu"],
            dong_col=self.apt_cols["dong"],
            main_col=self.apt_cols["road_main"],
            sub_col=self.apt_cols["road_sub"],
            y_col=self.apt_cols["y"],
            x_col=self.apt_cols["x"],
        )
        if self._use_cache:
            self._apt_coord_dict = apt_coord_dict
            self._coord_dict = coord_dict
            self._apt_index = apt_index
            cache_paths = self._cache_paths()
            if cache_paths is not None:
                apt_path, coord_path, index_path = cache_paths
                with apt_path.open("wb") as f:
                    pickle.dump(apt_coord_dict, f)
                with coord_path.open("wb") as f:
                    pickle.dump(coord_dict, f)
                with index_path.open("wb") as f:
                    pickle.dump(apt_index, f)

        return apt_coord_dict, coord_dict, apt_index

    @staticmethod
    def _build_load_name_full(
        df: pd.DataFrame,
        *,
        main_col: str,
        sub_col: str,
        out_full: str,
    ) -> pd.DataFrame:
        # 도로명 본문/상세주소를 합쳐 전체 도로명 컬럼을 만든다.
        out = df.copy()
        main = out[main_col].fillna("").astype(str).str.strip()
        sub = out[sub_col].fillna("").astype(str).str.strip()
        out[out_full] = (main + " " + sub).str.strip()
        out[out_full] = out[out_full].where(out[out_full] != "", main)
        return out

    @staticmethod
    def _make_coord_key(df: pd.DataFrame, *, gu_col: str, road_col: str) -> pd.Series:
        # 구와 도로명을 조합해 매핑 키를 만든다.
        gu = df[gu_col].fillna("").astype(str).str.strip()
        road = df[road_col].fillna("").astype(str).str.strip()
        return (gu + " " + road).str.strip()

    @staticmethod
    def _make_gudong_key(df: pd.DataFrame, *, gu_col: str, dong_col: str) -> pd.Series:
        # 구와 동을 조합해 매핑 키를 만든다.
        gu = df[gu_col].fillna("").astype(str).str.strip()
        dong = df[dong_col].fillna("").astype(str).str.strip()
        return (gu + " " + dong).str.strip()

    @staticmethod
    def _build_coord_dict(
        df: pd.DataFrame,
        *,
        key_col: str,
        gu_col: str | None = None,
        road_col: str | None = None,
        y_col: str,
        x_col: str,
    ) -> dict[str, tuple[float, float]]:
        # 좌표 결측 보정을 위한 매핑 사전을 만든다.
        c = df.copy()
        if key_col in c.columns:
            key = c[key_col].fillna("").astype(str).str.strip()
        else:
            if gu_col is None or road_col is None:
                raise KeyError("gu_col/road_col이 필요합니다.")
            # key_col이 없을 때는 구/도로명으로 매핑키를 만든다.
            key = HppCoordFillBlock._make_coord_key(c, gu_col=gu_col, road_col=road_col)
        # 유효 키와 y/x가 모두 있는 행만 대상으로 좌표 사전을 만든다.
        ok = (key != "") & c[y_col].notna() & c[x_col].notna()
        c = c.loc[ok].copy()
        key = key.loc[ok]
        # 반환값은 y,x 순서로 통일한다.
        return {str(k): (float(y), float(x)) for k, y, x in zip(key, c[y_col], c[x_col])}

    @staticmethod
    def _build_apt_similarity_index(
        apt_df: pd.DataFrame,
        *,
        gu_col: str,
        dong_col: str,
        main_col: str,
        sub_col: str,
        y_col: str,
        x_col: str,
    ) -> dict[tuple[str, str, str], list[tuple[str, tuple[float, float]]]]:
        # 유사도 보정에 쓸 주소 후보 인덱스를 만든다.
        cols = [gu_col, dong_col, main_col, sub_col, y_col, x_col]
        tmp = apt_df[cols].copy()
        # 후보군 비교용이므로 좌표가 없는 행은 제외한다.
        tmp = tmp.dropna(subset=[y_col, x_col])
        tmp[gu_col] = tmp[gu_col].astype(str).str.strip()
        tmp[dong_col] = tmp[dong_col].astype(str).str.strip()
        tmp[main_col] = tmp[main_col].astype(str).str.strip()
        tmp[sub_col] = tmp[sub_col].astype(str).str.strip()

        index: dict[tuple[str, str, str], list[tuple[str, tuple[float, float]]]] = {}
        for _, row in tmp.iterrows():
            key = (row[gu_col], row[dong_col], row[main_col])
            # y,x 순서 유지
            yx = (float(row[y_col]), float(row[x_col]))
            index.setdefault(key, []).append((row[sub_col], yx))
        return index

    @staticmethod
    def _fill_coords(
        df: pd.DataFrame,
        *,
        coord_dict: dict[str, tuple[float, float]],
        gu_col: str,
        dong_col: str,
        road_col: str,
        y_col: str,
        x_col: str,
    ) -> pd.DataFrame:
        # 매핑 사전으로 좌표 결측을 채운다.
        out = df.copy()
        key = HppCoordFillBlock._make_coord_key(out, gu_col=gu_col, road_col=road_col)

        na = out[y_col].isna() | out[x_col].isna()
        if na.any():
            # key 기반으로 좌표 사전을 매핑한 뒤, 결측에만 채운다.
            mapped = key.map(coord_dict)
            got = mapped.notna() & na
            if got.any():
                mapped_yx = mapped.loc[got]
                out.loc[got, y_col] = [v[0] for v in mapped_yx]
                out.loc[got, x_col] = [v[1] for v in mapped_yx]
        return out

    @staticmethod
    def _best_yx_by_similarity(
        target_sub: str, candidates: list[tuple[str, tuple[float, float]]]
    ) -> tuple[float, float] | None:
        # 도로명 상세주소 유사도 기준 최적 좌표를 고른다.
        if not candidates:
            return None
        if not target_sub:
            ys = [yx[0] for _, yx in candidates]
            xs = [yx[1] for _, yx in candidates]
            return (sum(ys) / len(ys), sum(xs) / len(xs))
        # 유사도 최대 후보의 좌표를 반환한다.
        best_yx = None
        best_score = -1.0
        for cand_sub, yx in candidates:
            score = difflib.SequenceMatcher(None, target_sub, cand_sub).ratio()
            if score > best_score:
                best_score = score
                best_yx = yx
        return best_yx

    @staticmethod
    def _fill_coords_by_apt_similarity(
        df: pd.DataFrame,
        *,
        apt_index: dict[tuple[str, str, str], list[tuple[str, tuple[float, float]]]],
        gu_col: str,
        dong_col: str,
        main_col: str,
        sub_col: str,
        y_col: str,
        x_col: str,
    ) -> pd.DataFrame:
        # 유사도 후보 인덱스로 좌표 결측을 보정한다.
        out = df.copy()
        na = out[y_col].isna() | out[x_col].isna()
        if not na.any():
            return out

        for idx in out.index[na]:
            gu = str(out.at[idx, gu_col]).strip()
            dong = str(out.at[idx, dong_col]).strip()
            main = str(out.at[idx, main_col]).strip()
            sub = str(out.at[idx, sub_col]).strip()
            key = (gu, dong, main)
            candidates = apt_index.get(key)
            if candidates:
                # 도로명 상세 주소가 가장 유사한 후보의 좌표로 보정한다.
                yx = HppCoordFillBlock._best_yx_by_similarity(sub, candidates)
            else:
                yx = None
            if yx is None:
                continue
            out.at[idx, y_col] = yx[0]
            out.at[idx, x_col] = yx[1]
        return out

    def _dump_coords_snapshot(self, df: pd.DataFrame, *, is_train: bool) -> None:
        if not (self._use_cache and self._cache_dir is not None):
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cols = [
            self.target_cols["gu"],
            self.target_cols["dong"],
            self.road_cols["full"],
            self.road_cols["main"],
            self.road_cols["sub"],
            self.target_cols["y"],
            self.target_cols["x"],
        ]
        out = df.loc[:, cols].copy()
        suffix = "train" if is_train else "test"
        out_path = self._cache_dir / f"coords_filled_{suffix}.csv"
        if out_path.exists():
            return
        out.to_csv(out_path, index=False)
