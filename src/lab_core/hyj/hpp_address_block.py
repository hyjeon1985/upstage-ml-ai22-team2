from __future__ import annotations

import json

import pandas as pd

from .core.dataset.pipeline import BaseBlock

DEFAULT_GU_DONG_SRC_COL = "시군구"
DEFAULT_GU_DONG_OUT_COLS = {"gu": "구", "dong": "동"}

DEFAULT_ROAD_SRC_COL = "도로명"
DEFAULT_ROAD_OUT_COLS = {"full": "도로명_전체", "main": "도로명_앞", "sub": "도로명_뒤"}


class HppGuDongBlock(BaseBlock):
    """
    시군구 컬럼에서 구/동을 파생한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        src_col: str = DEFAULT_GU_DONG_SRC_COL,
        out_cols: dict[str, str] = DEFAULT_GU_DONG_OUT_COLS,
        drop_src: bool = False,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.src_col = src_col
        self.out_cols = out_cols
        self.drop_src = drop_src

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        s = out[self.src_col].astype(str)
        out[self.out_cols["gu"]] = s.str.extract(r"(\S+구)", expand=False)
        out[self.out_cols["dong"]] = s.str.extract(r"(\S+동)", expand=False)
        if self.drop_src:
            out = out.drop(columns=[self.src_col])
        return out

    def describe(self) -> dict[str, str]:
        return {
            "src_col": self.src_col,
            "out_cols": json.dumps(self.out_cols, ensure_ascii=False),
        }


class HppLoadNamePartsBlock(BaseBlock):
    """
    도로명 전체/메인/서브를 분리한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        src_col: str = DEFAULT_ROAD_SRC_COL,
        out_cols: dict[str, str] = DEFAULT_ROAD_OUT_COLS,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.src_col = src_col
        self.out_cols = out_cols

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        s = out[self.src_col].fillna("").astype(str).str.strip()
        out[self.out_cols["full"]] = s
        split = s.str.split(n=1, expand=True)
        out[self.out_cols["main"]] = split[0].fillna("").astype(str)
        out[self.out_cols["sub"]] = (
            split[1].fillna("").astype(str) if split.shape[1] > 1 else ""
        )
        return out

    def describe(self) -> dict[str, str]:
        return {
            "src_col": self.src_col,
            "out_cols": json.dumps(self.out_cols, ensure_ascii=False),
        }


class HppLoadNameFullBlock(BaseBlock):
    """
    도로명 메인/서브를 결합해 전체 도로명을 만든다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        in_cols: dict[str, str],
        out_col: str,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.in_cols = in_cols
        self.out_col = out_col

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()
        main = out[self.in_cols["main"]].fillna("").astype(str).str.strip()
        sub = out[self.in_cols["sub"]].fillna("").astype(str).str.strip()
        out[self.out_col] = (main + " " + sub).str.strip()
        out[self.out_col] = out[self.out_col].where(out[self.out_col] != "", main)
        return out

    def describe(self) -> dict[str, str]:
        return {
            "in_cols": json.dumps(self.in_cols, ensure_ascii=False),
            "out_col": self.out_col,
        }
