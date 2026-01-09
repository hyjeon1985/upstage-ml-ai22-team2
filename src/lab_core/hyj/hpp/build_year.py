from __future__ import annotations

import pandas as pd

from ..preprocess.pipeline import BaseBlock


class BuildYearFeaturesBlock(BaseBlock):
    """
    - 건축년도 정리/검증
    - 건물나이, 건축지연도 파생
    - 행 제거는 하지 않음(안전)
    """

    def __init__(
        self,
        *,
        build_year_col: str = "건축년도",
        contract_year_col: str = "계약년",
        min_year: int = 1937,
        max_year: int = 2023,
        validate: bool = True,
    ):
        self.build_year_col = build_year_col
        self.contract_year_col = contract_year_col
        self.min_year = min_year
        self.max_year = max_year
        self.max_lead_year: 
        self.validate = validate

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        X = X.copy()
        del is_train

        if self.build_year_col not in X.columns:
            raise KeyError(f"필수 컬럼 누락: {self.build_year_col}")
        if self.contract_year_col not in X.columns:
            raise KeyError(f"필수 컬럼 누락: {self.contract_year_col}")

        build_y = X[self.build_year_col].astype(str).str.strip()

        if self.validate:
            if not build_y.str.fullmatch(r"\d{4}").all():
                bad = build_y[~build_y.str.fullmatch(r"\d{4}")].head(5).tolist()
                raise ValueError(f"건축년도에 YYYY 형식이 아닌 값 존재. 예시: {bad}")

        X[self.build_year_col] = build_y.astype(int)

        if self.validate:
            if not X[self.build_year_col].between(self.min_year, self.max_year).all():
                bad = (
                    X.loc[
                        ~X[self.build_year_col].between(self.min_year, self.max_year),
                        self.build_year_col,
                    ]
                    .head(5)
                    .tolist()
                )
                raise ValueError(
                    f"건축년도 범위({self.min_year}~{self.max_year}) 밖 값 존재. 예시: {bad}"
                )

        build_age = X[self.contract_year_col] - X[self.build_year_col]

        # 연속형(0 이상)
        X["건물나이"] = build_age.clip(lower=0)

        # 범주형 성격(0~) : 계약이 건축보다 선행한 정도
        X["건축지연도"] = (-build_age).clip(lower=0, upper=4)

        return X


class DropInvalidBuildAgeBlock(BaseBlock):
    """
    build_age = 계약년 - 건축년도
    - train: build_age < -max_lead_year 드랍
    - test : 절대 드랍하지 않음 (제출 row 깨짐 방지)
    """

    def __init__(
        self,
        *,
        build_year_col: str = "건축년도",
        contract_year_col: str = "계약년",
        max_lead_year: int = 4,
    ):
        self.build_year_col = build_year_col
        self.contract_year_col = contract_year_col
        self.max_lead_year = max_lead_year

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        X = X.copy()

        if not is_train:
            return X  # test에서는 행 제거 금지

        build_age = X[self.contract_year_col] - X[self.build_year_col]
        return X.loc[build_age >= -self.max_lead_year]
