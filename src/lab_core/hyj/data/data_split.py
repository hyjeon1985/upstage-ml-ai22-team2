from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_validate import assert_allowed_values, assert_same_columns

# -------------------------
# Data Spec.
# -------------------------


@dataclass(frozen=True)
class ConcatSplitResult:
    data: pd.DataFrame
    flag_col: str
    train_value: int
    test_value: int


@dataclass(frozen=True)
class SplitByFlagResult:
    train: pd.DataFrame
    test: pd.DataFrame
    flag_col: str
    train_value: int
    test_value: int


# -------------------------
# Functions
# -------------------------


def concat_with_split_flag(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    flag_col: str = "is_test",
    train_value: int = 0,
    test_value: int = 1,
    ignore_index: bool = True,
    sort: bool = False,
    validate: bool = True,
) -> ConcatSplitResult:
    """
    train/test를 결합하되, 출처를 구분하는 플래그 컬럼(flag_col)을 추가한다.

    Parameters
    ----------
    train, test:
        결합할 DataFrame.
    flag_col:
        출처 구분용 컬럼명.
    train_value, test_value:
        flag_col에 넣을 값. (예: 0/1, "train"/"test" 등)
    axis:
        pd.concat 축. 일반적으로 row 결합이므로 0을 사용.
    ignore_index:
        True면 결합 후 인덱스를 0..N-1로 재생성.
    sort:
        컬럼 정렬 여부 (pd.concat 옵션).
    validate:
        True면 아래 조건을 검증한다.
        - flag_col이 기존 컬럼에 존재하지 않아야 함
        - train/test 컬럼 집합이 동일해야 함

    Returns
    -------
    ConcatSplitResult
        data: flag_col이 추가된 결합 데이터
        flag_col/train_value/test_value: 메타 정보

    Raises
    ------
    KeyError
        flag_col이 이미 존재할 때.
    ValueError
        validate=True인데 train/test 컬럼이 다를 때.
    """
    if flag_col in train.columns or flag_col in test.columns:
        raise KeyError(f"'{flag_col}' 컬럼이 이미 존재합니다. 다른 이름을 사용하세요.")

    if validate:
        assert_same_columns(train, test)

    train_ = train.assign(**{flag_col: train_value})
    test_ = test.assign(**{flag_col: test_value})

    all_data = pd.concat([train_, test_], axis=0, ignore_index=ignore_index, sort=sort)

    return ConcatSplitResult(
        data=all_data,
        flag_col=flag_col,
        train_value=train_value,
        test_value=test_value,
    )


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
        assert_allowed_values(df[flag_col], allowed)

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


def split_from_concat_result(
    df: pd.DataFrame,
    concat_result,
    *,
    drop_flag: bool = True,
    validate: bool = True,
):
    return split_by_flag(
        df,
        flag_col=concat_result.flag_col,
        train_value=concat_result.train_value,
        test_value=concat_result.test_value,
        drop_flag=drop_flag,
        validate=validate,
    )
