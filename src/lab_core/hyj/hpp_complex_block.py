from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .core.dataset.pipeline import BaseBlock

DEFAULT_COMPLEX_SRC_COL = "전용면적"
DEFAULT_COMPLEX_UNIT_COLS = {
    "le_60": "전용면적세대수_60",
    "60_85": "전용면적세대수_60_85",
    "85_135": "전용면적세대수_85_135",
    "gt_135": "전용면적세대수_135",
}
DEFAULT_COMPLEX_OUT_COLS = {"share": "단지면적비중", "missing": "단지면적정보결측"}
DEFAULT_COMPLEX_INPUT_COLS = {
    "total_buildings": "전체동수",
    "total_units": "전체세대수",
    "total_area": "연면적",
    "net_area": "주거전용면적",
    "building_area": "건축면적",
    "parking": "주차대수",
}
DEFAULT_COMPLEX_SCALE_OUT_COLS = {
    "parking_per_unit": "주차대수_세대당",
    "units_per_building": "세대수_동수당",
    "gross_to_net_ratio": "연면적_전용면적비",
}


class HppComplexAreaShareBlock(BaseBlock):
    """
    단지 내 면적구간 세대수 비중을 계산한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        area_col: str = DEFAULT_COMPLEX_SRC_COL,
        unit_cols: dict[str, str] = DEFAULT_COMPLEX_UNIT_COLS,
        out_cols: dict[str, str] = DEFAULT_COMPLEX_OUT_COLS,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.area_col = area_col
        self.unit_cols = unit_cols
        self.out_cols = out_cols

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()

        cols = [
            self.unit_cols["le_60"],
            self.unit_cols["60_85"],
            self.unit_cols["85_135"],
            self.unit_cols["gt_135"],
        ]
        for col in cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            else:
                out[col] = np.nan

        total = out[cols].sum(axis=1, min_count=1)
        out[self.out_cols["missing"]] = total.isna().astype("int8")

        area_val = pd.to_numeric(out[self.area_col], errors="coerce")
        chosen = np.select(
            [
                area_val <= 60,
                (area_val > 60) & (area_val <= 85),
                (area_val > 85) & (area_val <= 135),
                area_val > 135,
            ],
            [
                out[self.unit_cols["le_60"]],
                out[self.unit_cols["60_85"]],
                out[self.unit_cols["85_135"]],
                out[self.unit_cols["gt_135"]],
            ],
            default=np.nan,
        )
        out[self.out_cols["share"]] = np.where(
            total.notna() & (total > 0), chosen / total, np.nan
        )
        return out

    def describe(self) -> dict[str, str]:
        return {
            "area_col": self.area_col,
            "unit_cols": json.dumps(self.unit_cols, ensure_ascii=False),
            "out_cols": json.dumps(self.out_cols, ensure_ascii=False),
        }


class HppComplexScaleBlock(BaseBlock):
    """
    단지 규모 관련 파생 변수를 생성한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        input_cols: dict[str, str] = DEFAULT_COMPLEX_INPUT_COLS,
        out_cols: dict[str, str] = DEFAULT_COMPLEX_SCALE_OUT_COLS,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.input_cols = input_cols
        self.out_cols = out_cols

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()

        numeric_cols = [
            self.input_cols["total_buildings"],
            self.input_cols["total_units"],
            self.input_cols["total_area"],
            self.input_cols["net_area"],
            self.input_cols["building_area"],
            self.input_cols["parking"],
        ]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out[self.out_cols["parking_per_unit"]] = np.where(
            out[self.input_cols["total_units"]] > 0,
            out[self.input_cols["parking"]] / out[self.input_cols["total_units"]],
            np.nan,
        )
        out[self.out_cols["units_per_building"]] = np.where(
            out[self.input_cols["total_buildings"]] > 0,
            out[self.input_cols["total_units"]]
            / out[self.input_cols["total_buildings"]],
            np.nan,
        )
        out[self.out_cols["gross_to_net_ratio"]] = np.where(
            out[self.input_cols["net_area"]] > 0,
            out[self.input_cols["total_area"]] / out[self.input_cols["net_area"]],
            np.nan,
        )
        return out

    def describe(self) -> dict[str, str]:
        return {
            "input_cols": json.dumps(self.input_cols, ensure_ascii=False),
            "out_cols": json.dumps(self.out_cols, ensure_ascii=False),
        }
