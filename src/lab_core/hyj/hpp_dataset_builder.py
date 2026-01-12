from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab_core.util.path import cache_data_dir, ext_data_dir, raw_data_dir

from .core.dataset.builder import BaseDatasetBuilder, PipelineLike
from .core.dataset.category_block import CategoryCleanBlock
from .core.dataset.category_keep_block import CategoryKeepOthersBlock
from .core.dataset.freq_block import FrequencyEncodeBlock
from .core.dataset.pipeline import Pipeline
from .core.dataset.useless_block import UselessValueToNaBlock
from .hpp_address_block import HppGuDongBlock, HppLoadNamePartsBlock
from .hpp_complex_block import HppComplexAreaShareBlock, HppComplexScaleBlock
from .hpp_coord_block import HppCoordFillBlock
from .hpp_premium_block import HppPremiumAreaBlock
from .hpp_time_block import HppBuildYearBlock, HppContractDateBlock
from .hpp_transport_block import HppTransportBlock


class HPPDatasetBuilder(BaseDatasetBuilder):
    FEATURE_COLS = [
        "전용면적",
        "층",
        "건축년도",
        "좌표X",
        "좌표Y",
        "계약년",
        "계약월",
        "건물나이",
        "건축지연",
        "강남권여부",
        "계약분기",
        "지하철거리",
        "버스거리",
        "지하철500m내점수",
        "버스300m내점수",
        "단지면적비중",
        "단지면적정보결측",
        "전체동수",
        "전체세대수",
        "연면적",
        "주거전용면적",
        "관리비부과면적",
        "건축면적",
        "주차대수",
        "주차대수_세대당",
        "세대수_동수당",
        "연면적_전용면적비",
        "건설시공사_빈도",
        "시행사_빈도",
        "아파트명_빈도",
        "구",
        "동",
        "복도유형",
        "난방방식",
        "관리방식",
        "세대전기계약방법",
        "청소비관리형태",
        "단지분류",
        "거래유형",
        "임대구분",
        "세대분양형태",
    ]
    CATEGORICAL_COLS = [
        "구",
        "동",
        "복도유형",
        "난방방식",
        "관리방식",
        "세대전기계약방법",
        "청소비관리형태",
        "단지분류",
        "거래유형",
        "임대구분",
        "세대분양형태",
    ]
    INT_COLS = {"계약년", "계약월", "계약분기", "층", "건축년도"}
    FEATURE_COLS_RF = FEATURE_COLS
    RAW_FILES = {
        "TRAIN": "train.csv",
        "TEST": "test.csv",
    }

    EXT_FILES = {
        "APT": "k-apt.csv",
        "COORD": "coord.csv",
        "SUBWAY": "subway_feature.csv",
        "BUS": "bus_feature.csv",
    }

    def __init__(self, *, target_transform: str = "log1p") -> None:
        super().__init__()
        self.target_transform = target_transform

    def _load_raw_datas(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_path = raw_data_dir(self.RAW_FILES["TRAIN"])
        test_path = raw_data_dir(self.RAW_FILES["TEST"])

        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        return train, test

    def _load_ext_datas(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        apt_path = ext_data_dir(self.EXT_FILES["APT"])
        coord_path = ext_data_dir(self.EXT_FILES["COORD"])
        subway_path = ext_data_dir(self.EXT_FILES["SUBWAY"])
        bus_path = ext_data_dir(self.EXT_FILES["BUS"])

        apt = pd.read_csv(apt_path, low_memory=False)
        coord = pd.read_csv(coord_path, low_memory=False)
        subway = pd.read_csv(subway_path, low_memory=False)
        bus = pd.read_csv(bus_path, low_memory=False)
        return apt, coord, subway, bus

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        train, test = self._load_raw_datas()
        return train, test, "target"

    def load_ext(self) -> dict[str, Any]:
        # 여기서 한 번 로드해서 캐시 → make_pipeline에서 주입
        apt, coord, subway, bus = self._load_ext_datas()
        return {"subway": subway, "bus": bus, "apt": apt, "coord": coord}

    def rename_map(self) -> dict[str, str]:
        # 외부 주입 X: 구현체 내부에서 통제
        return {
            "전용면적(㎡)": "전용면적",
            "k-건설사(시공사)": "건설시공사",
            "k-단지분류(아파트,주상복합등등)": "단지분류",
            "k-세대타입(분양형태)": "세대분양형태",
            "k-관리방식": "관리방식",
            "k-복도유형": "복도유형",
            "k-난방방식": "난방방식",
            "k-전체동수": "전체동수",
            "k-전체세대수": "전체세대수",
            "k-시행사": "시행사",
            "k-연면적": "연면적",
            "k-주거전용면적": "주거전용면적",
            "k-관리비부과면적": "관리비부과면적",
            "k-전용면적별세대현황(60㎡이하)": "전용면적세대수_60",
            "k-전용면적별세대현황(60㎡~85㎡이하)": "전용면적세대수_60_85",
            "k-85㎡~135㎡이하": "전용면적세대수_85_135",
            "k-135㎡초과": "전용면적세대수_135",
            "세대전기계약방법": "세대전기계약방법",
            "청소비관리형태": "청소비관리형태",
            "건축면적": "건축면적",
            "주차대수": "주차대수",
            "기타/의무/임대/임의=1/2/3/4": "의무_임대_임의_기타",
        }

    def cache_dir(self) -> Path:
        return cache_data_dir()

    def transform_target(self, y: pd.Series) -> tuple[pd.Series, str, dict[str, Any]]:
        if self.target_transform == "log1p":
            y_out = pd.Series(np.log1p(y), index=y.index)
            return y_out, "log1p", {}
        if self.target_transform == "none":
            return y, "none", {}
        raise ValueError(f"unknown target_transform: {self.target_transform}")

    def split_train_valid(
        self, X: pd.DataFrame, y: pd.Series, *, split_policy: dict[str, Any]
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if split_policy.get("kind") != "time_holdout":
            raise ValueError(f"unknown split kind: {split_policy.get('kind')}")
        ym_col = split_policy.get("ym_col", "계약년월")
        train_until = int(split_policy.get("train_until", 202303))
        valid_from = int(split_policy.get("valid_from", 202304))
        valid_until = int(split_policy.get("valid_until", 202306))

        ym = pd.to_numeric(X[ym_col], errors="coerce")
        train_mask = ym <= train_until
        valid_mask = (ym >= valid_from) & (ym <= valid_until)

        X_train = X.loc[train_mask].copy()
        y_train = y.loc[X_train.index].copy()
        X_valid = X.loc[valid_mask].copy()
        y_valid = y.loc[X_valid.index].copy()
        return X_train, y_train, X_valid, y_valid

    def finalize_for_model(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, *, model: str | None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if model is None:
            return X_train, X_test

        X_train = X_train.copy()
        X_test = X_test.copy()
        cols = self.FEATURE_COLS if model == "lgbm" else self.FEATURE_COLS_RF

        for col in cols:
            if col not in X_train.columns:
                continue
            if col in self.CATEGORICAL_COLS:
                if model == "lgbm":
                    tr = X_train[col].astype("string").fillna("__NA__")
                    te = X_test[col].astype("string").fillna("__NA__")
                    X_train[col] = tr.astype("category")
                    cats = X_train[col].cat.categories
                    X_test[col] = pd.Categorical(te, categories=cats)
                else:
                    tr = X_train[col].astype("string").fillna("UNKNOWN")
                    te = X_test[col].astype("string").fillna("UNKNOWN")
                    cats = pd.Index(tr.unique())
                    mapping = {k: i for i, k in enumerate(cats)}
                    X_train[col] = tr.map(mapping).astype("Int16")
                    X_test[col] = te.map(mapping).fillna(-1).astype("Int16")
                continue

            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            if model == "rf":
                med = X_train[col].median()
                X_train[col] = X_train[col].fillna(med)
                X_test[col] = X_test[col].fillna(med)

            if col in self.INT_COLS:
                X_train[col] = X_train[col].astype("Int16")
                X_test[col] = X_test[col].astype("Int16")
            else:
                X_train[col] = X_train[col].astype("Float32")
                X_test[col] = X_test[col].astype("Float32")

        return X_train, X_test

    def feature_cols(self, *, model: str | None) -> list[str] | None:
        if model is None:
            return None
        return self.FEATURE_COLS if model == "lgbm" else self.FEATURE_COLS_RF

    def feature_meta(self, *, model: str | None) -> dict[str, list[str]] | None:
        if model is None:
            return None
        cols = self.FEATURE_COLS if model == "lgbm" else self.FEATURE_COLS_RF
        cat_cols = [c for c in self.CATEGORICAL_COLS if c in cols]
        num_cols = [c for c in cols if c not in self.CATEGORICAL_COLS]
        return {"numeric": num_cols, "categorical": cat_cols}

    def make_pipeline(
        self, *, meta_cols: tuple[str, ...], ext: dict[str, Any], use_cache: bool
    ) -> PipelineLike:
        subway = ext["subway"]
        bus = ext["bus"]
        apt = ext["apt"]
        coord = ext["coord"]
        cache_dir = self._get_cache_dir(use_cache=use_cache)
        coord_cache_dir = cache_dir
        transport_cache_dir = cache_dir

        blocks = [
            UselessValueToNaBlock(
                meta_cols,
                rules={
                    "거래유형": ["-"],
                },
            ),
            CategoryCleanBlock(
                meta_cols,
                cols=[
                    "복도유형",
                    "난방방식",
                    "관리방식",
                    "세대전기계약방법",
                    "청소비관리형태",
                    "단지분류",
                    "거래유형",
                    "세대분양형태",
                ],
                fill_value="__NA__",
            ),
            CategoryKeepOthersBlock(
                meta_cols,
                src_col="의무_임대_임의_기타",
                out_col="임대구분",
                keep_values={"의무", "임대", "임의"},
            ),
            HppGuDongBlock(meta_cols),
            HppLoadNamePartsBlock(meta_cols),
            HppCoordFillBlock(
                meta_cols,
                use_cache=use_cache,
                cache_dir=coord_cache_dir,
                apt_df=apt,
                coord_df=coord,
            ),
            HppTransportBlock(
                meta_cols,
                subway_df=subway,
                bus_df=bus,
                use_cache=use_cache,
                cache_dir=transport_cache_dir,
                subway_r_km=0.5,
                bus_r_km=0.3,
            ),
            HppContractDateBlock(meta_cols),
            HppBuildYearBlock(meta_cols),
            HppPremiumAreaBlock(meta_cols),
            HppComplexAreaShareBlock(meta_cols),
            HppComplexScaleBlock(meta_cols),
            FrequencyEncodeBlock(
                meta_cols,
                col_map={
                    "건설시공사": "건설시공사_빈도",
                    "시행사": "시행사_빈도",
                    "아파트명": "아파트명_빈도",
                },
            ),
        ]
        return Pipeline(blocks)
