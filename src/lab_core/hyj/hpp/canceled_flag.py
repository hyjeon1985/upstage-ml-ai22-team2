from __future__ import annotations

import pandas as pd

from ..preprocess.pipeline import BaseBlock


class CanceledFlagBlock(BaseBlock):
    """해제사유발생일 -> is_canceled 생성 후, 해제사유발생일 drop"""

    def __init__(self, *, col: str, flag: str):
        self.src_col = col
        self.flag_col = flag

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        X = X.copy()
        del is_train

        # src_col이 없으면: 이미 처리됐거나 데이터에 없는 케이스를 안전하게 통과
        if self.src_col not in X.columns:
            if self.flag_col not in X.columns:
                X[self.flag_col] = 0
            return X

        X[self.flag_col] = X[self.src_col].notna().astype("int8")
        return X.drop(columns=[self.src_col])


class CanceledFinalizeBlock(BaseBlock):
    """
    - train: is_canceled==1 행 제거(선택) + 컬럼 제거(선택)
    - test : 행 제거 절대 안 함 + 컬럼 제거(선택)
    """

    def __init__(
        self,
        *,
        col: str,
        drop_canceled_rows_in_train: bool = True,
        drop_col_after: bool = True,
    ):
        self.col = col
        self.drop_canceled_rows_in_train = drop_canceled_rows_in_train
        self.drop_col_after = drop_col_after

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        X = X.copy()

        if self.col not in X.columns:
            return X

        if is_train and self.drop_canceled_rows_in_train:
            X = X.loc[X[self.col] != 1]

        # test에서는 행 제거 금지(명시적으로 보장)
        # (is_train==False일 때는 필터링하지 않음)

        if self.drop_col_after:
            X = X.drop(columns=[self.col])

        return X
