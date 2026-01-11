from __future__ import annotations

import json

import pandas as pd

from .core.dataset.pipeline import BaseBlock

DEFAULT_CONTRACT_YM_COL = "계약년월"
DEFAULT_CONTRACT_Y_COL = "계약년"
DEFAULT_CONTRACT_OUT_COLS = {
    "year": DEFAULT_CONTRACT_Y_COL,
    "month": "계약월",
    "quarter": "계약분기",
}

DEFAULT_BUILD_YEAR_COL = "건축년도"
DEFAULT_BUILD_OUT_COLS = {"age": "건물나이", "delay": "건축지연"}


class HppContractDateBlock(BaseBlock):
    """
    계약 시점 관련 파생 변수를 생성한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        ym_col: str = DEFAULT_CONTRACT_YM_COL,
        out_cols: dict[str, str] = DEFAULT_CONTRACT_OUT_COLS,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.ym_col = ym_col
        self.out_cols = out_cols

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()

        ym = out[self.ym_col].astype(str).str.strip()
        if not ym.str.fullmatch(r"\d{6}").all():
            raise ValueError("계약년월이 YYYYMM 형식이 아닌 값이 존재합니다.")

        out[self.out_cols["year"]] = ym.str.slice(0, 4).astype(int)
        out[self.out_cols["month"]] = ym.str.slice(4, 6).astype(int)
        out[self.out_cols["quarter"]] = (
            (out[self.out_cols["month"]] - 1) // 3 + 1
        ).astype("int8")
        return out

    def describe(self) -> dict[str, str]:
        return {
            "ym_col": self.ym_col,
            "out_cols": json.dumps(self.out_cols, ensure_ascii=False),
        }


class HppBuildYearBlock(BaseBlock):
    """
    건축 시점 관련 파생 변수를 생성한다.
    """

    def __init__(
        self,
        meta_cols: tuple[str, ...],
        name: str | None = None,
        *,
        build_year_col: str = DEFAULT_BUILD_YEAR_COL,
        contract_year_col: str = DEFAULT_CONTRACT_Y_COL,
        out_cols: dict[str, str] = DEFAULT_BUILD_OUT_COLS,
    ) -> None:
        super().__init__(meta_cols=meta_cols, name=name)
        self.build_year_col = build_year_col
        self.contract_year_col = contract_year_col
        self.out_cols = out_cols

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        out = X.copy()

        y = out[self.build_year_col].astype(str).str.strip()
        if not y.str.fullmatch(r"\d{4}").all():
            raise ValueError("건축년도에 YYYY 형식이 아닌 값이 존재합니다.")
        out[self.build_year_col] = y.astype(int)

        build_age = out[self.contract_year_col] - out[self.build_year_col]
        out[self.out_cols["age"]] = build_age.clip(lower=0)
        out[self.out_cols["delay"]] = (-build_age).clip(lower=0)
        return out

    def describe(self) -> dict[str, str]:
        return {
            "build_year_col": self.build_year_col,
            "contract_year_col": self.contract_year_col,
            "out_cols": json.dumps(self.out_cols, ensure_ascii=False),
        }
