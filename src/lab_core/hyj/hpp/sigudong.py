import pandas as pd

from ..preprocess.pipeline import BaseBlock

# -----------------------------
# 시구동 컬럼 분리
# -----------------------------


class SplitGuDongBlock(BaseBlock):
    def fit(self, X: pd.DataFrame) -> None:
        pass  # 학습할 내용 없음

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()

        if "시군구" not in X.columns:
            return X

        s = X["시군구"].astype(str)

        X["구"] = s.str.extract(r"(\S+구)", expand=False)
        X["동"] = s.str.extract(r"(\S+동)", expand=False)

        X = X.drop(columns=["시군구"])
        return X


# -----------------------------
# 시구동 컬럼 분리
# -----------------------------
