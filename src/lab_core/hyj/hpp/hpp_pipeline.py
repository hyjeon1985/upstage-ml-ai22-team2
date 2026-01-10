from __future__ import annotations

import pandas as pd

from ...util.path import ext_data_dir, raw_data_dir
from ..pipeline.base import Pipeline
from ..pipeline.useless_block import DropUselessColumnsBlock, UselessValueToNaBlock


def hpp_pipeline():
    """
    HPP 파이프라인 스켈레톤.
    - 데이터 로드
    - 블록 조합
    - train/test 변환
    """
    # 데이터 로드
    train_path = raw_data_dir("train.csv")
    test_path = raw_data_dir("test.csv")
    subway_path = raw_data_dir("subway_feature.csv")
    bus_path = raw_data_dir("bus_feature.csv")
    coord_path = ext_data_dir("coord.csv")
    apt_path = ext_data_dir("k-apt.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    subway_df = pd.read_csv(subway_path)
    bus_df = pd.read_csv(bus_path)
    coord_df = pd.read_csv(coord_path)
    apt_df = pd.read_csv(apt_path)

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"지하철역: {len(subway_df)}개")
    print(f"버스정류장: {len(bus_df):,}개")
    print(f"좌표 데이터: {len(coord_df):,}개")
    print(f"k-apt 데이터: {len(apt_df):,}개")

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
            # TODO: 아래에 필요한 블록들을 순서대로 추가
        ]
    )

    X_train = pipeline.fit_transform(train_df.drop(columns="target"))
    y_train = train_df.loc[X_train.index, "target"]
    X_test = pipeline.transform(test_df)

    return X_train, y_train, X_test
