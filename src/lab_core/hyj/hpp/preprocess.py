from __future__ import annotations

import numpy as np
import pandas as pd

_COL_TRAIN = "_is_train"

# -----------------------------
# Data Preprocess
# -----------------------------


def fill_missing_coords(train_df, test_df, coord_df):
    """좌표 결측치 채우기 (coord.csv 활용)

    카카오 API로 미리 수집한 좌표 데이터(coord.csv)를 사용하여
    Train/Test의 좌표 결측치를 채웁니다.

    매핑 키: '구 + 도로명' (예: '강남구 테헤란로 123')

    Parameters:
        train_df: 학습 데이터프레임
        test_df: 테스트 데이터프레임
        coord_df: 좌표 데이터프레임 (매핑키, 좌표X, 좌표Y 컬럼 필요)

    Returns:
        좌표가 채워진 (train, test) 튜플
    """
    train = train_df.copy()
    test = test_df.copy()

    # 매핑 키 생성 (구 + 도로명)
    train["구"] = train["시군구"].str.split().str[1]
    test["구"] = test["시군구"].str.split().str[1]
    train["매핑키"] = train["구"] + " " + train["도로명"].fillna("")
    test["매핑키"] = test["구"] + " " + test["도로명"].fillna("")

    # 좌표 딕셔너리 생성 (NaN 제외)
    coord_valid = coord_df[coord_df["좌표X"].notna()]
    coord_dict = dict(
        zip(coord_valid["매핑키"], zip(coord_valid["좌표X"], coord_valid["좌표Y"]))
    )
    print(f"좌표 딕셔너리 크기: {len(coord_dict):,}개")

    # Train 좌표 결측치 채우기
    train_before = train["좌표X"].isna().sum()
    mask = train["좌표X"].isna()
    train.loc[mask, "좌표X"] = train.loc[mask, "매핑키"].map(
        lambda x: coord_dict.get(x, (np.nan, np.nan))[0]
    )
    train.loc[mask, "좌표Y"] = train.loc[mask, "매핑키"].map(
        lambda x: coord_dict.get(x, (np.nan, np.nan))[1]
    )
    train_after = train["좌표X"].isna().sum()

    # Test 좌표 결측치 채우기
    test_before = test["좌표X"].isna().sum()
    mask = test["좌표X"].isna()
    test.loc[mask, "좌표X"] = test.loc[mask, "매핑키"].map(
        lambda x: coord_dict.get(x, (np.nan, np.nan))[0]
    )
    test.loc[mask, "좌표Y"] = test.loc[mask, "매핑키"].map(
        lambda x: coord_dict.get(x, (np.nan, np.nan))[1]
    )
    test_after = test["좌표X"].isna().sum()

    # 임시 컬럼 제거
    train = train.drop(columns=["매핑키"], errors="ignore")
    test = test.drop(columns=["매핑키"], errors="ignore")

    # 결과 출력
    print("\n[좌표 결측치 채우기 결과]")
    print(
        f"  Train: {train_before:,} → {train_after:,} (채움: {train_before - train_after:,})"
    )
    print(
        f"  Test: {test_before:,} → {test_after:,} (채움: {test_before - test_after:,})"
    )
    print(f"\n  Train 커버율: {train['좌표X'].notna().mean() * 100:.1f}%")
    print(f"  Test 커버율: {test['좌표X'].notna().mean() * 100:.1f}%")

    # 남은 결측치는 구 평균 좌표로 채우기
    combined = pd.concat([train, test], ignore_index=True)
    gu_mean = (
        combined[combined["좌표X"].notna()]
        .groupby("구")[["좌표X", "좌표Y"]]
        .mean()
        .to_dict("index")
    )

    for df in [train, test]:
        mask = df["좌표X"].isna()
        for idx in df[mask].index:
            gu = df.loc[idx, "구"]
            if gu in gu_mean:
                df.loc[idx, "좌표X"] = gu_mean[gu]["좌표X"]
                df.loc[idx, "좌표Y"] = gu_mean[gu]["좌표Y"]

    return train, test
