"""
House Price Prediction - Pipeline Skeleton (for Codex)

목표:
- 팀 동료 상위 점수 노트북 로직을 "Block 기반 파이프라인"으로 재현 가능한 형태로 뼈대 제공
- 여기서부터 세부 로직/파라미터 튜닝/LightGBM 전환은 Codex에게 확장 구현 지시

주의:
- 이 코드는 "스켈레톤"이다. 핵심 로직은 TODO로 남겨두고, 구조/인터페이스/순서만 고정한다.
- train/test 누수 방지를 원칙으로 하되, 노트북 재현 목적으로 일부 union-fit이 필요하면 별도 블록으로 분리한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


# =========================
# 1) Core Interfaces
# =========================

class Block(Protocol):
    """파이프라인 블록 인터페이스"""

    def fit(self, X: pd.DataFrame) -> None: ...
    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame: ...


class BaseBlock:
    """기본 블록: stateless 블록은 fit을 구현하지 않아도 됨"""

    def fit(self, X: pd.DataFrame) -> None:
        return None

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class Pipeline:
    """Block 리스트를 순서대로 적용"""

    blocks: Sequence[Block]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for b in self.blocks:
            b.fit(X)  # train only
            X = b.transform(X, is_train=True)
        return X

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        X = X.copy()
        for b in self.blocks:
            X = b.transform(X, is_train=is_train)
        return X


# =========================
# 2) Utility Functions
# =========================

def time_split_by_ym(
    df: pd.DataFrame,
    *,
    ym_col: str = "계약년월",
    train_until: int = 202303,
    valid_from: int = 202304,
    valid_until: int = 202306,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """노트북 스타일 시계열 홀드아웃 분할"""
    ym = df[ym_col].astype(int)
    tr = df.loc[ym <= train_until].copy()
    va = df.loc[(ym >= valid_from) & (ym <= valid_until)].copy()
    return tr, va


# =========================
# 3) Blocks (Skeleton)
# =========================

class GuDongBlock(BaseBlock):
    """
    정책:
    - '시군구'에서 '구', '동' 파생
    - 원본 '시군구'는 제거(선택)
    """

    def __init__(self, *, src_col: str = "시군구", drop_src: bool = True):
        self.src_col = src_col
        self.drop_src = drop_src

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()

        # TODO: 당신이 이미 만든 _fe_gu_dong 로직으로 대체 가능
        s = X[self.src_col].astype(str)
        X["구"] = s.str.extract(r"(\S+구)", expand=False)
        X["동"] = s.str.extract(r"(\S+동)", expand=False)

        if self.drop_src:
            X = X.drop(columns=[self.src_col])
        return X


class CoordFillBlock(BaseBlock):
    """
    노트북 핵심 로직 재현 블록

    정책:
    - 외부 coord_df(카카오 API 등으로 수집된 좌표 테이블)로 결측 좌표 복원
    - 키: '{구} {도로명}' 형태로 매핑
    - 복원 실패 시: '구'별 평균 좌표로 채움 (train 기준 권장)

    입력 가정:
    - X에 '구', '도로명', '좌표X', '좌표Y' 존재
    - coord_df에 키 컬럼(또는 구/도로명)과 좌표 컬럼 존재
    """

    def __init__(
        self,
        coord_df: pd.DataFrame,
        *,
        key_cols: tuple[str, str] = ("구", "도로명"),
        x_col: str = "좌표X",
        y_col: str = "좌표Y",
        fill_by_gu_mean: bool = True,
    ):
        self.coord_df = coord_df.copy()
        self.key_cols = key_cols
        self.x_col = x_col
        self.y_col = y_col
        self.fill_by_gu_mean = fill_by_gu_mean

        self._coord_dict: dict[str, tuple[float, float]] = {}
        self._gu_mean: dict[str, tuple[float, float]] = {}

    @staticmethod
    def _mk_key(df: pd.DataFrame, *, key_cols: tuple[str, str]) -> pd.Series:
        a, b = key_cols
        # 도로명 결측/공백 방어
        return (df[a].astype(str).str.strip() + " " + df[b].astype(str).str.strip()).str.strip()

    def fit(self, X: pd.DataFrame) -> None:
        # 1) coord_df로부터 dict 구축
        # TODO: coord_df의 실제 컬럼명에 맞춰 보정 필요
        c = self.coord_df.copy()
        key = self._mk_key(c, key_cols=self.key_cols)

        # 좌표가 유효한 행만
        ok = c[self.x_col].notna() & c[self.y_col].notna()
        c = c.loc[ok].copy()
        key = key.loc[ok]

        self._coord_dict = {k: (float(x), float(y)) for k, x, y in zip(key, c[self.x_col], c[self.y_col])}

        # 2) train 기준 구 평균 좌표 구축 (권장)
        if self.fill_by_gu_mean:
            # X의 현재 좌표(훈련 데이터 기준)로 구 평균
            tmp = X[[self.key_cols[0], self.x_col, self.y_col]].copy()
            tmp = tmp.dropna(subset=[self.x_col, self.y_col])
            g = tmp.groupby(self.key_cols[0])[[self.x_col, self.y_col]].mean()
            self._gu_mean = {gu: (float(row[self.x_col]), float(row[self.y_col])) for gu, row in g.iterrows()}

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()

        key = self._mk_key(X, key_cols=self.key_cols)

        # 1) dict 매핑으로 좌표 복원
        na = X[self.x_col].isna() | X[self.y_col].isna()
        if na.any():
            mapped = key.map(self._coord_dict)
            got = mapped.notna() & na

            # mapped 값은 (x,y) 튜플
            if got.any():
                xy = mapped.loc[got]
                X.loc[got, self.x_col] = [v[0] for v in xy]
                X.loc[got, self.y_col] = [v[1] for v in xy]

        # 2) 남은 결측은 구 평균으로
        if self.fill_by_gu_mean:
            na2 = X[self.x_col].isna() | X[self.y_col].isna()
            if na2.any():
                gu = X.loc[na2, self.key_cols[0]].astype(str)
                mean_xy = gu.map(self._gu_mean)
                got2 = mean_xy.notna()
                if got2.any():
                    xy = mean_xy.loc[got2]
                    idx = mean_xy.loc[got2].index
                    X.loc[idx, self.x_col] = [v[0] for v in xy]
                    X.loc[idx, self.y_col] = [v[1] for v in xy]

        return X


class ContractDateBlock(BaseBlock):
    """
    정책:
    - 계약년월(YYYYMM) -> 계약년, 계약월, 계약분기 파생
    """

    def __init__(self, *, ym_col: str = "계약년월"):
        self.ym_col = ym_col

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()

        ym = X[self.ym_col].astype(str).str.strip()
        if not ym.str.fullmatch(r"\d{6}").all():
            raise ValueError("계약년월이 YYYYMM 형식이 아닌 값이 있습니다.")

        X["계약년"] = ym.str.slice(0, 4).astype(int)
        X["계약월"] = ym.str.slice(4, 6).astype(int)
        X["계약분기"] = ((X["계약월"] - 1) // 3 + 1).astype(int)
        return X


class BuildYearBlock(BaseBlock):
    """
    정책(Policy):
    - build_age = 계약년 - 건축년도
    - build_age < -max_lead_year 는 데이터 오류로 간주하여 제거 (주의: test에서는 제거 금지)
      -> 이 스켈레톤에서는 train only drop 권장. (test drop은 제출 row 깨짐)
    - 건물나이: 연속형 (0 이상)
    - 건축지연도: 범주형 (0~max_lead_year)
    - 신축여부: (건물나이 <= new_cutoff_year) 같은 단순 규칙 (파라미터화)

    구현 노트:
    - row drop은 기본적으로 train에서만 수행한다.
    """

    def __init__(
        self,
        *,
        build_year_col: str = "건축년도",
        contract_year_col: str = "계약년",
        max_lead_year: int = 4,
        new_cutoff_year: int = 1,
        validate: bool = True,
        drop_invalid_in_train: bool = True,
    ):
        self.build_year_col = build_year_col
        self.contract_year_col = contract_year_col
        self.max_lead_year = max_lead_year
        self.new_cutoff_year = new_cutoff_year
        self.validate = validate
        self.drop_invalid_in_train = drop_invalid_in_train

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        X = X.copy()

        y = X[self.build_year_col].astype(str).str.strip()
        if self.validate:
            if not y.str.fullmatch(r"\d{4}").all():
                raise ValueError("건축년도에 YYYY 형식이 아닌 값이 있습니다.")
        X[self.build_year_col] = y.astype(int)

        build_age = X[self.contract_year_col] - X[self.build_year_col]

        # train only drop (제출 안전)
        if is_train and self.drop_invalid_in_train:
            X = X.loc[build_age >= -self.max_lead_year].copy()
            build_age = X[self.contract_year_col] - X[self.build_year_col]

        X["건물나이"] = build_age.clip(lower=0)
        X["건축지연도"] = (-build_age).clip(lower=0, upper=self.max_lead_year)
        X["신축여부"] = (X["건물나이"] <= self.new_cutoff_year).astype("int8")
        return X


class PremiumAreaBlock(BaseBlock):
    """정책: 강남3구 플래그 파생"""

    def __init__(self, *, gu_col: str = "구", out_col: str = "강남여부"):
        self.gu_col = gu_col
        self.out_col = out_col
        self._premium = {"강남구", "서초구", "송파구"}

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()
        X[self.out_col] = X[self.gu_col].astype(str).isin(self._premium).astype("int8")
        return X


class BinningBlock(BaseBlock):
    """
    정책:
    - 전용면적/층 구간화 (노트북 스타일)
    """

    def __init__(
        self,
        *,
        area_col: str = "전용면적(㎡)",
        floor_col: str = "층",
        area_bins: list[float] | None = None,
        area_labels: list[str] | None = None,
        floor_bins: list[float] | None = None,
        floor_labels: list[str] | None = None,
        out_area: str = "면적구간",
        out_floor: str = "층구간",
    ):
        self.area_col = area_col
        self.floor_col = floor_col
        self.out_area = out_area
        self.out_floor = out_floor

        self.area_bins = area_bins or [0, 40, 60, 85, 135, np.inf]
        self.area_labels = area_labels or ["초소형", "소형", "중형", "중대형", "대형"]

        self.floor_bins = floor_bins or [0, 5, 10, 20, np.inf]
        self.floor_labels = floor_labels or ["저층", "중저층", "중층", "고층"]

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()

        X[self.out_area] = pd.cut(
            X[self.area_col].astype(float),
            bins=self.area_bins,
            labels=self.area_labels,
            right=True,
            include_lowest=True,
        ).astype(str)

        X[self.out_floor] = pd.cut(
            X[self.floor_col].astype(float),
            bins=self.floor_bins,
            labels=self.floor_labels,
            right=True,
            include_lowest=True,
        ).astype(str)

        return X


class TransportBlock(BaseBlock):
    """
    노트북 핵심 로직 재현 블록(스켈레톤)

    정책:
    - 좌표X/좌표Y 기반으로:
      - 지하철/버스 최근접 거리
      - 반경 내 개수
    - KDTree 사용을 권장 (scipy.spatial.cKDTree)

    구현 전략:
    - fit에서 KDTree 구축
    - transform에서 query/query_ball_point 결과로 컬럼 생성

    TODO:
    - 거리 계산(좌표계에 따라 haversine or 단순 euclidean)을 확정해야 한다.
      노트북은 "좌표 단위 반경"으로 대략 처리한 것으로 보인다.
    """

    def __init__(
        self,
        subway_df: pd.DataFrame,
        bus_df: pd.DataFrame,
        *,
        x_col: str = "좌표X",
        y_col: str = "좌표Y",
        subway_r: float = 0.005,  # 노트북 유사
        bus_r: float = 0.003,     # 노트북 유사
    ):
        self.subway_df = subway_df.copy()
        self.bus_df = bus_df.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.subway_r = subway_r
        self.bus_r = bus_r

        self._subway_tree = None
        self._bus_tree = None
        self._subway_xy = None
        self._bus_xy = None

    def fit(self, X: pd.DataFrame) -> None:
        # TODO: scipy가 필요하다 (scipy.spatial.cKDTree)
        # from scipy.spatial import cKDTree
        # self._subway_xy = self.subway_df[[self.x_col, self.y_col]].to_numpy()
        # self._bus_xy = self.bus_df[[self.x_col, self.y_col]].to_numpy()
        # self._subway_tree = cKDTree(self._subway_xy)
        # self._bus_tree = cKDTree(self._bus_xy)
        return None

    def transform(self, X: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
        _ = is_train
        X = X.copy()

        # TODO: KDTree query/query_ball_point로 아래 컬럼 생성
        # - 지하철_최근접_거리, 버스_최근접_거리
        # - 지하철_500m_개수, 버스_300m_개수
        #
        # 방어:
        # - 좌표 결측이 남아있으면 계산 불가하므로 assert/na 처리 필요
        return X


# =========================
# 4) Feature Selection / Model I/O
# =========================

FEATURE_COLS: list[str] = [
    # 수치형
    "전용면적(㎡)", "층", "건축년도", "좌표X", "좌표Y",
    "계약년", "계약월", "계약분기",
    "건물나이", "신축여부", "강남여부",
    "지하철_최근접_거리", "버스_최근접_거리",
    "지하철_500m_개수", "버스_300m_개수",
    # 범주형
    "구", "동", "아파트명", "도로명", "면적구간", "층구간",
]


TARGET_COL = "target"  # TODO: 실제 타겟 컬럼명으로 교체


# =========================
# 5) Example: Wiring
# =========================

def build_pipeline(
    *,
    coord_df: pd.DataFrame,
    subway_df: pd.DataFrame,
    bus_df: pd.DataFrame,
) -> Pipeline:
    """노트북 유사 로직 파이프라인 구성(순서 중요)"""
    return Pipeline(
        blocks=[
            GuDongBlock(),
            CoordFillBlock(coord_df),
            ContractDateBlock(),
            BuildYearBlock(max_lead_year=4, new_cutoff_year=1),
            PremiumAreaBlock(),
            BinningBlock(),
            TransportBlock(subway_df, bus_df),  # TODO 구현 필요
            # TODO: 카테고리 인코딩/캐스팅 블록(LGBM이면 category dtype 권장)
        ]
    )


def run_train_valid_demo(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    coord_df: pd.DataFrame,
    subway_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    ym_col: str = "계약년월",
):
    """
    스켈레톤 실행 예시:
    - 시계열 split
    - pipeline fit/transform
    - feature 선택
    - 모델 학습은 TODO (RF 재현 -> LGBM 전환)
    """
    pipe = build_pipeline(coord_df=coord_df, subway_df=subway_df, bus_df=bus_df)

    tr, va = time_split_by_ym(train_df, ym_col=ym_col)

    y_tr = tr[TARGET_COL].copy()
    y_va = va[TARGET_COL].copy()
    X_tr = tr.drop(columns=[TARGET_COL])
    X_va = va.drop(columns=[TARGET_COL])

    X_tr_fe = pipe.fit_transform(X_tr)
    X_va_fe = pipe.transform(X_va, is_train=True)  # valid도 train으로 취급(동일 전처리)

    X_test_fe = pipe.transform(test_df, is_train=False)

    # TODO: FEATURE_COLS 존재 여부 검증 + 누락 시 로그
    X_tr_f = X_tr_fe[FEATURE_COLS].copy()
    X_va_f = X_va_fe[FEATURE_COLS].copy()
    X_te_f = X_test_fe[FEATURE_COLS].copy()

    # TODO: 모델 학습/평가/추론
    # - (재현용) RandomForestRegressor 먼저
    # - (대회용) LightGBMRegressor로 교체
    return X_tr_f, y_tr, X_va_f, y_va, X_te_f
