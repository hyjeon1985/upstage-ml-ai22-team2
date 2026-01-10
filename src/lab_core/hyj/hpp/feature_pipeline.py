from __future__ import annotations

import pandas as pd

from ..pipeline.base import BaseBlock, Pipeline
from ..pipeline.category_block import CategoryCleanBlock
from ..pipeline.useless_block import UselessValueToNaBlock


class RenameColumnsBlock(BaseBlock):
    """컬럼 리네임 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class LoadNamePartsBlock(BaseBlock):
    """도로명 파생 컬럼 생성 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class LeaseTypeBlock(BaseBlock):
    """임대구분 정리 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class SplitGuDongBlock(BaseBlock):
    """시군구에서 구/동 파생 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class CoordFillBlock(BaseBlock):
    """좌표 결측치 보정 블록 (TODO: 구현)"""

    def fit(self, X: pd.DataFrame) -> None:
        return None

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class TransportFeaturesBlock(BaseBlock):
    """교통 접근성 피처 생성 블록 (TODO: 구현)"""

    def fit(self, X: pd.DataFrame) -> None:
        return None

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class ContractDateBlock(BaseBlock):
    """계약일 파생 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class BuildYearBlock(BaseBlock):
    """건축년도 파생 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class PremiumAreaBlock(BaseBlock):
    """강남 여부 파생 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class ComplexAreaShareBlock(BaseBlock):
    """단지 내 면적구간 비중 파생 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


class ComplexScaleBlock(BaseBlock):
    """단지 규모 파생 블록 (TODO: 구현)"""

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        return X


def build_fe_pipeline() -> Pipeline:
    """
    FE 파이프라인 스켈레톤.
    구현은 추후 블록별로 채운다.
    """
    return Pipeline(
        blocks=[
            RenameColumnsBlock(),
            LoadNamePartsBlock(),
            UselessValueToNaBlock(
                rules={
                    "trade_type": ["-"],
                }
            ),
            CategoryCleanBlock(
                cols=[
                    "corridor_type",
                    "heating_type",
                    "manage_type",
                    "electric_contract_type",
                    "cleaning_type",
                    "complex_type",
                    "trade_type",
                    "sale_type",
                ]
            ),
            LeaseTypeBlock(),
            SplitGuDongBlock(),
            CoordFillBlock(),
            TransportFeaturesBlock(),
            ContractDateBlock(),
            BuildYearBlock(),
            PremiumAreaBlock(),
            ComplexAreaShareBlock(),
            ComplexScaleBlock(),
        ]
    )
