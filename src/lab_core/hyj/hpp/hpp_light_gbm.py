import pandas as pd

from ...util.path import ext_data_dir, raw_data_dir
from ..preprocess.pipeline import Pipeline
from ..preprocess.useless_block import DropUselessColumnsBlock, UselessValueToNaBlock
from .build_year import BuildYearFeaturesBlock, DropInvalidBuildAgeBlock
from .canceled_flag import CanceledFinalizeBlock, CanceledFlagBlock
from .contract_date import ContractDateBlock
from .sigudong import SplitGuDongBlock


def hpp_light_gbm():
    # 데이터 로드
    train_path = raw_data_dir("train.csv")
    test_path = raw_data_dir("test.csv")
    subway_path = raw_data_dir("subway_feature.csv")
    bus_path = raw_data_dir("bus_feature.csv")
    coord_path = ext_data_dir("coord.csv")  # 카카오 API로 생성한 좌표 데이터

    # 데이터 로드
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    subway_df = pd.read_csv(subway_path)
    bus_df = pd.read_csv(bus_path)
    coord_df = pd.read_csv(coord_path)

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"지하철역: {len(subway_df)}개")
    print(f"버스정류장: {len(bus_df):,}개")
    print(f"좌표 데이터: {len(coord_df):,}개")

    pipeline = Pipeline(
        blocks=[
            UselessValueToNaBlock(
                {
                    "등기신청일자": [" "],
                    "거래유형": ["-"],
                }
            ),
            DropUselessColumnsBlock(
                [
                    "중개사소재지",
                    "번지",
                    "본번",
                    "부번",
                ]
            ),
            SplitGuDongBlock(),
            ContractDateBlock(),
            CanceledFlagBlock(col="해제사유발생일", flag="_is_canceled"),
            BuildYearFeaturesBlock(),
            CanceledFinalizeBlock(col="_is_canceled"),
            DropInvalidBuildAgeBlock(),
            # ParseAddressBlock(),
            # MedianImputer(cols=["전용면적(㎡)", "층"]),
            # AreaOutlierFilter(col="전용면적(㎡)"),
        ]
    )

    X_train = pipeline.fit_transform(train_df.drop(columns="target"))
    y_train = train_df.loc[X_train.index, "target"]

    X_test = pipeline.transform(test_df)

    # X_train, X_test = finalize_for_lgbm(
    #     X_train, X_test, cat_cols=["구", "동", "아파트명"]
    # )


# def finalize_for_lgbm(train: pd.DataFrame, test: pd.DataFrame, *, cat_cols: Iterable[str]): ->

# )
