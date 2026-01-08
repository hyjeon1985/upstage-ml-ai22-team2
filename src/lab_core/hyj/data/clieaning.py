from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitByFlagResult:
    train: pd.DataFrame
    test: pd.DataFrame
    flag_col: str
    train_value: int
    test_value: int


def split_by_flag(
    df: pd.DataFrame,
    *,
    flag_col: str = "is_test",
    train_value: int = 0,
    test_value: int = 1,
    drop_flag: bool = True,
    validate: bool = True,
) -> SplitByFlagResult:
    """
    flag_col 값으로 결합된 데이터프레임을 train/test로 다시 분리한다.

    Parameters
    ----------
    df:
        결합된 DataFrame.
    flag_col:
        출처 구분 컬럼명.
    train_value, test_value:
        train/test로 간주할 flag 값.
    drop_flag:
        True면 반환되는 train/test에서 flag_col을 제거한다.
    validate:
        True면 아래 조건을 검증한다.
        - flag_col이 존재해야 함
        - flag_col에 train_value/test_value 외 값이 있으면 에러

    Returns
    -------
    SplitByFlagResult
        train, test: 분리된 DataFrame
    """
    if flag_col not in df.columns:
        raise KeyError(
            f"'{flag_col}' 컬럼이 없습니다. 결합 단계에서 flag를 추가했는지 확인하세요."
        )

    if validate:
        allowed = {train_value, test_value}
        actual = set(df[flag_col].dropna().unique().tolist())
        extra = actual - allowed
        if extra:
            raise ValueError(
                f"'{flag_col}'에 허용되지 않은 값이 있습니다: {sorted(extra)} "
                f"(allowed={sorted(allowed)})"
            )

    train = df[df[flag_col] == train_value].copy()
    test = df[df[flag_col] == test_value].copy()

    if drop_flag:
        train = train.drop(columns=[flag_col])
        test = test.drop(columns=[flag_col])

    return SplitByFlagResult(
        train=train,
        test=test,
        flag_col=flag_col,
        train_value=train_value,
        test_value=test_value,
    )


def rename_columns(
    df: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    컬럼명을 명시적 매핑(mapping)을 이용해 변경한다.

    Parameters
    ----------
    df : pd.DataFrame
        컬럼명을 변경할 대상 데이터프레임.
    mapping : Mapping[str, str]
        {기존 컬럼명: 변경할 컬럼명} 형태의 매핑.
        매핑에 포함된 컬럼이 df에 존재하지 않으면 KeyError를 발생시킨다.
    inplace : bool, default False
        True이면 원본 DataFrame을 직접 수정한다.
        False이면 컬럼명이 변경된 새로운 DataFrame을 반환한다.

    Returns
    -------
    pd.DataFrame
        컬럼명이 변경된 DataFrame.
        inplace=True인 경우에도 수정된 df 객체를 반환한다.

    Raises
    ------
    KeyError
        mapping에 포함된 컬럼 중 df에 존재하지 않는 컬럼이 있을 경우.

    Notes
    -----
    - pandas의 `DataFrame.rename`과 달리, 이 함수는 존재하지 않는
      컬럼명을 허용하지 않는다.
    - EDA 초반이나 데이터 스키마가 중요한 파이프라인 단계에서
      조용한 실패(silent failure)를 방지하기 위한 용도로 설계되었다.
    """

    missing = set(mapping) - set(df.columns)
    if missing:
        raise KeyError(f"존재하지 않는 컬럼: {missing}")

    if inplace:
        df.rename(columns=mapping, inplace=True)
        return df

    return df.rename(columns=mapping)


def drop_high_missing(
    df: pd.DataFrame, threshold: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    결측치 개수가 threshold 이상인 컬럼을 제거한다.

    Returns
    -------
    df_clean : pd.DataFrame
        결측치 많은 컬럼이 제거된 데이터프레임
    dropped_summary : pd.Series
        제거된 컬럼들의 결측치 개수
    """
    sr_na = df.isna().sum()
    cols = sr_na[sr_na >= threshold].index

    if len(cols) == 0:
        return df, sr_na.iloc[0:0]

    return df.drop(columns=cols), sr_na.loc[cols]
