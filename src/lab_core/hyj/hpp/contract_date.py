from __future__ import annotations

import pandas as pd

from ..preprocess.pipeline import BaseBlock


class ContractDateBlock(BaseBlock):
    """
    계약년월(YYYYMM) -> 계약년, 계약월 생성 + 계약일자 생성 + 계약년월 drop

    - fit(): 없음(규칙 기반)
    - transform(): 검증 + 파생
    """

    def __init__(
        self,
        *,
        validate: bool = True,
        year_min: int = 2007,
        year_max: int = 2023,  # 필요하면 느슨하게(예: 2024) 조정
        drop_yyyymm: bool = True,
        create_date: bool = True,
        date_col: str = "계약일자",
    ):
        self.validate = validate
        self.year_min = year_min
        self.year_max = year_max
        self.drop_yyyymm = drop_yyyymm
        self.create_date = create_date
        self.date_col = date_col

    def fit(self, X: pd.DataFrame) -> None:
        pass

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()

        # 필수 컬럼 체크
        required = ["계약년월", "계약일"]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise KeyError(f"필수 컬럼 누락: {missing}")

        ym = X["계약년월"].astype(str)

        if self.validate:
            if not ym.str.fullmatch(r"\d{6}").all():
                bad = ym[~ym.str.fullmatch(r"\d{6}")].head(5).tolist()
                raise ValueError(f"계약년월에 YYYYMM 형식이 아닌 값 존재. 예시: {bad}")

        X["계약년"] = ym.str.slice(0, 4).astype(int)
        X["계약월"] = ym.str.slice(4, 6).astype(int)

        if self.validate:
            if not X["계약년"].between(self.year_min, self.year_max).all():
                bad = (
                    X.loc[~X["계약년"].between(self.year_min, self.year_max), "계약년"]
                    .head(5)
                    .tolist()
                )
                raise ValueError(
                    f"계약년 범위({self.year_min}~{self.year_max}) 밖 값 존재. 예시: {bad}"
                )

            if not X["계약월"].between(1, 12).all():
                bad = X.loc[~X["계약월"].between(1, 12), "계약월"].head(5).tolist()
                raise ValueError(f"계약월 1~12 밖 값 존재. 예시: {bad}")

            if not X["계약일"].between(1, 31).all():
                bad = X.loc[~X["계약일"].between(1, 31), "계약일"].head(5).tolist()
                raise ValueError(f"계약일 1~31 밖 값 존재. 예시: {bad}")

        if self.drop_yyyymm:
            X = X.drop(columns=["계약년월"])

        if self.create_date:
            X[self.date_col] = pd.to_datetime(
                X["계약년"].astype(str)
                + "-"
                + X["계약월"].astype(str).str.zfill(2)
                + "-"
                + X["계약일"].astype(str).str.zfill(2),
                format="%Y-%m-%d",
                errors="raise",
            )

        return X
