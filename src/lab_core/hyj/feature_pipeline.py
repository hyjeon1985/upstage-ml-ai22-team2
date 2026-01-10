import difflib

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from lab_core.hyj.feature_data import load_ext_datas, load_raw_datas
from lab_core.util.geo import haversine_distance, to_xy_km
from lab_core.util.path import out_dir
from lab_core.util.time_ids import make_run_id

# 데이터 셋 컬럼 정보
# - Train columns: ['시군구', '번지', '본번', '부번', '아파트명', '전용면적(㎡)', '계약년월', '계약일', '층', '건축년도', '도로명', '해제사유발생일', '등기신청일자', '거래유형', '중개사소재지', 'k-단지분류(아파트,주상복합등등)', 'k-전화번호', 'k-팩스번호', '단지소개기존clob', 'k-세대타입(분양형태)', 'k-관리방식', 'k-복도유형', 'k-난방방식', 'k-전체동수', 'k-전체세대수', 'k-건설사(시공사)', 'k-시행사', 'k-사용검사일-사용승인일', 'k-연면적', 'k-주거전용면적', 'k-관리비부과면적', 'k-전용면적별세대현황(60㎡이하)', 'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-85㎡~135㎡이하', 'k-135㎡초과', 'k-홈페이지', 'k-등록일자', 'k-수정일자', '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태', '건축면적', '주차대수', '기타/의무/임대/임의=1/2/3/4', '단지승인일', '사용허가여부', '관리비 업로드', '좌표X', '좌표Y', '단지신청일', 'target']
# - Test columns: ['시군구', '번지', '본번', '부번', '아파트명', '전용면적(㎡)', '계약년월', '계약일', '층', '건축년도', '도로명', '해제사유발생일', '등기신청일자', '거래유형', '중개사소재지', 'k-단지분류(아파트,주상복합등등)', 'k-전화번호', 'k-팩스번호', '단지소개기존clob', 'k-세대타입(분양형태)', 'k-관리방식', 'k-복도유형', 'k-난방방식', 'k-전체동수', 'k-전체세대수', 'k-건설사(시공사)', 'k-시행사', 'k-사용검사일-사용승인일', 'k-연면적', 'k-주거전용면적', 'k-관리비부과면적', 'k-전용면적별세대현황(60㎡이하)', 'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-85㎡~135㎡이하', 'k-135㎡초과', 'k-홈페이지', 'k-등록일자', 'k-수정일자', '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태', '건축면적', '주차대수', '기타/의무/임대/임의=1/2/3/4', '단지승인일', '사용허가여부', '관리비 업로드', '좌표X', '좌표Y', '단지신청일']
# - Subway columns: ['역사_ID', '역사명', '호선', '위도', '경도']
# - Bus columns: ['노드 ID', '정류소번호', '정류소명', 'X좌표', 'Y좌표', '정류소 타입']
# - Apt columns: ['번호', 'k-아파트코드', 'k-아파트명', 'k-단지분류(아파트,주상복합등등)', 'kapt도로명주소', '주소(시도)k-apt주소split', '주소(시군구)', '주소(읍면동)', '나머지주소', '주소(도로명)', '주소(도로상세주소)', 'k-전화번호', 'k-팩스번호', '단지소개기존clob', '단지첨부파일', 'k-세대타입(분양형태)', 'k-관리방식', 'k-복도유형', 'k-난방방식', 'k-전체동수', 'k-전체세대수', 'k-건설사(시공사)', 'k-시행사', 'k-사용검사일-사용승인일', 'k-연면적', 'k-주거전용면적', 'k-관리비부과면적', 'k-전용면적별세대현황(60㎡이하)', 'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-85㎡~135㎡이하', 'k-135㎡초과', 'k-홈페이지', 'k-등록일자', 'k-수정일자', '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태', '건축면적', '주차대수', '기타/의무/임대/임의=1/2/3/4', '단지승인일', '사용허가여부', '관리비 업로드', '좌표X', '좌표Y', '단지신청일']
# - Coord columns: ['매핑키', '좌표X', '좌표Y']


PARAMS = {
    # build-year policy
    "MAX_LEAD_YEAR": 4,  # 계약이 건축보다 선행 가능한 최대 연수
    # transport radii (km)
    "SUBWAY_R": 0.5,
    "BUS_R": 0.3,
}

# Columns
TARGET_COL = "target"


# Feature columns (shared across models)
FEATURE_COLS = [
    "area_m2",
    "floor",
    "build_year",
    "x",
    "y",
    "contract_year",
    "contract_month",
    "build_age",
    "build_delay",
    "is_gangnam",
    "contract_quarter",
    "subway_dist",
    "bus_dist",
    "subway_cnt_500m",
    "bus_cnt_300m",
    "complex_area_share",
    "complex_area_missing",
    "complex_buildings",
    "complex_units",
    "gross_floor_area",
    "net_area",
    "manage_fee_area",
    "building_area",
    "parking_spaces",
    "parking_per_unit",
    "units_per_building",
    "gross_to_net_ratio",
    "builder_freq",
    "developer_freq",
    "apt_name_freq",
    "gu",
    "dong",
    "apt_name",
    "corridor_type",
    "heating_type",
    "manage_type",
    "electric_contract_type",
    "cleaning_type",
    "complex_type",
    "trade_type",
    "lease_type",
    "sale_type",
]

CATEGORICAL_COLS = [
    "gu",
    "dong",
    "apt_name",
    "corridor_type",
    "heating_type",
    "manage_type",
    "electric_contract_type",
    "cleaning_type",
    "complex_type",
    "trade_type",
    "lease_type",
    "sale_type",
]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

# RF에서는 고카디널리티 컬럼을 제외한다.
FEATURE_COLS_RF = [c for c in FEATURE_COLS if c not in {"apt_name"}]

# Int-like columns (use nullable Int16 where safe)
INT_COLS = [
    "contract_year",
    "contract_month",
    "contract_quarter",
    "floor",
    "build_year",
]

# =========================
# Model Params
# =========================

LGBM_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 24,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
}

ENSEMBLE_WEIGHTS = {
    "lgbm": 0.7,
    "rf": 0.3,
}


def _lr_tag(value: float) -> str:
    """
    learning_rate를 파일명용 문자열로 변환한다. (예: 0.05 -> "05")
    """
    return f"{value:.2f}".replace("0.", "")


def _model_tag(model_type: str) -> str:
    """
    파일명에 들어갈 파라미터 태그를 생성한다.
    """
    if model_type == "lgbm":
        return (
            f"lgbm_n{LGBM_PARAMS['n_estimators']}"
            f"_lr{_lr_tag(LGBM_PARAMS['learning_rate'])}"
            f"_l{LGBM_PARAMS['num_leaves']}"
            f"_m{LGBM_PARAMS['min_data_in_leaf']}"
        )
    if model_type == "rf":
        depth = RF_PARAMS.get("max_depth")
        depth_tag = "0" if depth is None else str(depth)
        return (
            f"rf_n{RF_PARAMS['n_estimators']}"
            f"_d{depth_tag}"
            f"_m{RF_PARAMS['min_samples_leaf']}"
        )
    if model_type == "ensemble":
        w_lgbm = int(round(ENSEMBLE_WEIGHTS["lgbm"] * 100))
        w_rf = int(round(ENSEMBLE_WEIGHTS["rf"] * 100))
        return f"ens_{_model_tag('lgbm')}_{_model_tag('rf')}_w{w_lgbm}-{w_rf}"
    return model_type


# Meta Column Names
_IS_TRAIN = "_is_train"
_ORG_IDX = "_ORG_IDX"

DROP_META_COLS = [_IS_TRAIN, _ORG_IDX]


# =========================
# Column Rename
# =========================

RAW_COL_RENAME = {
    "시군구": "sigungu",
    "아파트명": "apt_name",
    "전용면적(㎡)": "area_m2",
    "계약년월": "contract_ym",
    "층": "floor",
    "건축년도": "build_year",
    "도로명": "road_name",
    "거래유형": "trade_type",
    "k-단지분류(아파트,주상복합등등)": "complex_type",
    "k-세대타입(분양형태)": "sale_type",
    "k-관리방식": "manage_type",
    "k-복도유형": "corridor_type",
    "k-난방방식": "heating_type",
    "k-전체동수": "complex_buildings",
    "k-전체세대수": "complex_units",
    "k-건설사(시공사)": "builder",
    "k-시행사": "developer",
    "k-연면적": "gross_floor_area",
    "k-주거전용면적": "net_area",
    "k-관리비부과면적": "manage_fee_area",
    "k-전용면적별세대현황(60㎡이하)": "units_area_le_60",
    "k-전용면적별세대현황(60㎡~85㎡이하)": "units_area_60_85",
    "k-85㎡~135㎡이하": "units_area_85_135",
    "k-135㎡초과": "units_area_gt_135",
    "세대전기계약방법": "electric_contract_type",
    "청소비관리형태": "cleaning_type",
    "건축면적": "building_area",
    "주차대수": "parking_spaces",
    "기타/의무/임대/임의=1/2/3/4": "lease_type_raw",
    "좌표X": "x",
    "좌표Y": "y",
}

APT_COL_RENAME = {
    "주소(시군구)": "gu",
    "주소(읍면동)": "dong",
    "주소(도로명)": "load_name_main",
    "주소(도로상세주소)": "load_name_sub",
    "좌표X": "x",
    "좌표Y": "y",
}

SUBWAY_COL_RENAME = {
    "역사_ID": "station_id",
    "역사명": "station_name",
    "호선": "line",
    "위도": "lat",
    "경도": "lon",
}

BUS_COL_RENAME = {
    "노드 ID": "node_id",
    "정류소번호": "stop_no",
    "정류소명": "stop_name",
    "X좌표": "x",
    "Y좌표": "y",
    "정류소 타입": "stop_type",
}

COORD_COL_RENAME = {
    "매핑키": "map_key",
    "좌표X": "x",
    "좌표Y": "y",
}


def rename_train_test(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RAW_COL_RENAME)


def rename_subway(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=SUBWAY_COL_RENAME)


def rename_bus(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=BUS_COL_RENAME)


def rename_coord(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COORD_COL_RENAME)


def rename_apt(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=APT_COL_RENAME)


def build_load_name_parts(
    df: pd.DataFrame,
    *,
    src_col: str = "road_name",
    out_full: str = "load_name_full",
    out_main: str = "load_name_main",
    out_sub: str = "load_name_sub",
) -> pd.DataFrame:
    """
    도로명 전체/메인/서브를 분리해서 저장한다.
    """
    out = df.copy()
    s = out[src_col].fillna("").astype(str).str.strip()
    out[out_full] = s
    split = s.str.split(n=1, expand=True)
    out[out_main] = split[0].fillna("").astype(str)
    out[out_sub] = split[1].fillna("").astype(str) if split.shape[1] > 1 else ""
    return out


def build_load_name_full(
    df: pd.DataFrame,
    *,
    main_col: str = "load_name_main",
    sub_col: str = "load_name_sub",
    out_full: str = "load_name_full",
) -> pd.DataFrame:
    """
    도로명 메인/서브를 합쳐 전체 도로명 컬럼을 만든다.
    """
    out = df.copy()
    main = out[main_col].fillna("").astype(str).str.strip()
    sub = out[sub_col].fillna("").astype(str).str.strip()
    out[out_full] = (main + " " + sub).str.strip()
    out[out_full] = out[out_full].where(out[out_full] != "", main)
    return out


def build_apt_similarity_index(
    apt_df: pd.DataFrame,
    *,
    gu_col: str = "gu",
    dong_col: str = "dong",
    main_col: str = "load_name_main",
    sub_col: str = "load_name_sub",
    x_col: str = "x",
    y_col: str = "y",
) -> dict[tuple[str, str, str], list[tuple[str, tuple[float, float]]]]:
    """
    k-apt 좌표를 유사도 매칭하기 위한 인덱스를 만든다.
    """
    tmp = apt_df[[gu_col, dong_col, main_col, sub_col, x_col, y_col]].copy()
    tmp = tmp.dropna(subset=[gu_col, dong_col, main_col, x_col, y_col])
    tmp[gu_col] = tmp[gu_col].astype(str).str.strip()
    tmp[dong_col] = tmp[dong_col].astype(str).str.strip()
    tmp[main_col] = tmp[main_col].astype(str).str.strip()
    tmp[sub_col] = tmp[sub_col].fillna("").astype(str).str.strip()

    index: dict[tuple[str, str, str], list[tuple[str, tuple[float, float]]]] = {}
    for _, row in tmp.iterrows():
        key = (row[gu_col], row[dong_col], row[main_col])
        index.setdefault(key, []).append(
            (row[sub_col], (float(row[x_col]), float(row[y_col])))
        )
    return index


def fill_coords_by_apt_similarity(
    df: pd.DataFrame,
    *,
    apt_index: dict[tuple[str, str, str], list[tuple[str, tuple[float, float]]]],
    gu_col: str = "gu",
    dong_col: str = "dong",
    main_col: str = "load_name_main",
    sub_col: str = "load_name_sub",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    k-apt 후보군 내에서 도로상세주소 유사도가 가장 높은 좌표로 보정한다.
    """
    out = df.copy()
    na = out[x_col].isna() | out[y_col].isna()
    if not na.any():
        return out

    def _best_xy(
        target_sub: str, candidates: list[tuple[str, tuple[float, float]]]
    ) -> tuple[float, float] | None:
        if not candidates:
            return None
        if not target_sub:
            return candidates[0][1]
        best_xy = None
        best_score = -1.0
        for cand_sub, xy in candidates:
            score = difflib.SequenceMatcher(None, target_sub, cand_sub).ratio()
            if score > best_score:
                best_score = score
                best_xy = xy
        return best_xy

    for idx in out.index[na]:
        gu = str(out.at[idx, gu_col]).strip()
        dong = str(out.at[idx, dong_col]).strip()
        main = str(out.at[idx, main_col]).strip()
        sub = str(out.at[idx, sub_col]).strip()
        key = (gu, dong, main)
        candidates = apt_index.get(key)
        if not candidates:
            continue
        xy = _best_xy(sub, candidates)
        if xy is None:
            continue
        out.at[idx, x_col] = xy[0]
        out.at[idx, y_col] = xy[1]
    return out


# =========================
# FE Functions
# =========================


# -------------------------
# Basic FE
# -------------------------
def fe_useless_to_na(
    df: pd.DataFrame,
    *,
    rules: dict[str, list[str]],
) -> pd.DataFrame:
    """
    유의미하지 않은 문자열 값을 결측치로 변환한다.
    """
    out = df.copy()
    for col, values in rules.items():
        if col in out.columns:
            out[col] = out[col].replace(values, np.nan)
    return out


def fe_category_clean(
    df: pd.DataFrame,
    *,
    cols: list[str],
    fill_value: str = "__NA__",
) -> pd.DataFrame:
    """
    범주형 컬럼을 정리하고 결측을 동일한 토큰으로 통일한다.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        s = out[col].astype("string").str.strip()
        s = s.replace("", pd.NA)
        out[col] = s.fillna(fill_value)
    return out


def fe_lease_type(
    df: pd.DataFrame,
    *,
    src_col: str = "lease_type_raw",
    out_col: str = "lease_type",
    keep_values: set[str] | None = None,
    fill_value: str = "__NA__",
) -> pd.DataFrame:
    """
    임대 구분을 정리하고 의미 없는 값은 OTHERS로 묶는다.
    """
    out = df.copy()
    if keep_values is None:
        keep_values = {"의무", "임대", "임의", "기타"}
    s = out[src_col].astype("string").str.strip()
    s = s.replace("", pd.NA).fillna(fill_value)
    out[out_col] = s.where(s.isin(keep_values) | (s == fill_value), "OTHERS")
    return out


def fe_gu_dong(
    df: pd.DataFrame, *, src_col: str = "sigungu", drop_src: bool = True
) -> pd.DataFrame:
    out = df.copy()
    s = out[src_col].astype(str)
    out["gu"] = s.str.extract(r"(\S+구)", expand=False)
    out["dong"] = s.str.extract(r"(\S+동)", expand=False)
    if drop_src:
        out = out.drop(columns=[src_col])
    return out


# -------------------------
# Coord Helpers
# -------------------------
def _make_coord_key(df: pd.DataFrame, *, gu_col: str, road_col: str) -> pd.Series:
    gu = df[gu_col].fillna("").astype(str).str.strip()
    road = df[road_col].fillna("").astype(str).str.strip()
    key = (gu + " " + road).str.strip()
    return key


def _road_prefix(road: pd.Series) -> pd.Series:
    return road.astype(str).str.strip().str.split().str[0].fillna("")


def _make_gudong_key(df: pd.DataFrame, *, gu_col: str, dong_col: str) -> pd.Series:
    gu = df[gu_col].fillna("").astype(str).str.strip()
    dong = df[dong_col].fillna("").astype(str).str.strip()
    return (gu + " " + dong).str.strip()


def _make_gudong_road_prefix_key(
    df: pd.DataFrame, *, gu_col: str, dong_col: str, road_col: str
) -> pd.Series:
    gu = df[gu_col].fillna("").astype(str).str.strip()
    dong = df[dong_col].fillna("").astype(str).str.strip()
    road_prefix = _road_prefix(df[road_col])
    return (gu + " " + dong + " " + road_prefix).str.strip()


def build_coord_dict(
    coord_df: pd.DataFrame,
    *,
    key_col: str = "map_key",
    gu_col="gu",
    road_col="road_name",
    x_col="x",
    y_col="y",
) -> dict[str, tuple[float, float]]:
    c = coord_df.copy()
    if key_col in c.columns:
        key = c[key_col].fillna("").astype(str).str.strip()
    else:
        key = _make_coord_key(c, gu_col=gu_col, road_col=road_col)
    ok = (key != "") & c[x_col].notna() & c[y_col].notna()
    c = c.loc[ok].copy()
    key = key.loc[ok]
    return {str(k): (float(x), float(y)) for k, x, y in zip(key, c[x_col], c[y_col])}


def build_gudong_mean_xy(
    df: pd.DataFrame, *, gu_col="gu", dong_col="dong", x_col="x", y_col="y"
) -> dict[str, tuple[float, float]]:
    tmp = df[[gu_col, dong_col, x_col, y_col]].dropna(subset=[x_col, y_col]).copy()
    key = _make_gudong_key(tmp, gu_col=gu_col, dong_col=dong_col)
    tmp = tmp.assign(_key=key)
    g = tmp.groupby("_key")[[x_col, y_col]].mean()
    return {str(k): (float(row[x_col]), float(row[y_col])) for k, row in g.iterrows()}


def build_gudong_road_prefix_mean_xy(
    df: pd.DataFrame,
    *,
    gu_col="gu",
    dong_col="dong",
    road_col="road_name",
    x_col="x",
    y_col="y",
) -> dict[str, tuple[float, float]]:
    tmp = df[[gu_col, dong_col, road_col, x_col, y_col]].dropna(subset=[x_col, y_col])
    key = _make_gudong_road_prefix_key(
        tmp, gu_col=gu_col, dong_col=dong_col, road_col=road_col
    )
    tmp = tmp.assign(_key=key)
    ok = tmp["_key"] != ""
    tmp = tmp.loc[ok]
    g = tmp.groupby("_key")[[x_col, y_col]].mean()
    return {str(k): (float(row[x_col]), float(row[y_col])) for k, row in g.iterrows()}


def fill_coords(
    df: pd.DataFrame,
    *,
    coord_dict: dict[str, tuple[float, float]],
    gudong_prefix_mean_xy: dict[str, tuple[float, float]] | None = None,
    gudong_mean_xy: dict[str, tuple[float, float]] | None = None,
    gu_col="gu",
    dong_col="dong",
    road_col="road_name",
    x_col="x",
    y_col="y",
) -> pd.DataFrame:
    out = df.copy()
    key = _make_coord_key(out, gu_col=gu_col, road_col=road_col)

    na = out[x_col].isna() | out[y_col].isna()
    if na.any():
        mapped = key.map(coord_dict)
        got = mapped.notna() & na
        if got.any():
            xy = mapped.loc[got]
            out.loc[got, x_col] = [v[0] for v in xy]
            out.loc[got, y_col] = [v[1] for v in xy]

    if gudong_prefix_mean_xy is not None:
        na2 = out[x_col].isna() | out[y_col].isna()
        if na2.any():
            prefix_key = _make_gudong_road_prefix_key(
                out.loc[na2],
                gu_col=gu_col,
                dong_col=dong_col,
                road_col=road_col,
            )
            mean_xy = prefix_key.map(gudong_prefix_mean_xy)
            got2 = mean_xy.notna()
            if got2.any():
                xy = mean_xy.loc[got2]
                idx = mean_xy.loc[got2].index
                out.loc[idx, x_col] = [v[0] for v in xy]
                out.loc[idx, y_col] = [v[1] for v in xy]

    if gudong_mean_xy is not None:
        na2 = out[x_col].isna() | out[y_col].isna()
        if na2.any():
            key = _make_gudong_key(out.loc[na2], gu_col=gu_col, dong_col=dong_col)
            mean_xy = key.map(gudong_mean_xy)
            got2 = mean_xy.notna()
            if got2.any():
                xy = mean_xy.loc[got2]
                idx = mean_xy.loc[got2].index
                out.loc[idx, x_col] = [v[0] for v in xy]
                out.loc[idx, y_col] = [v[1] for v in xy]

    return out


# -------------------------
# Transport FE
# -------------------------
def fe_transport_features(
    df: pd.DataFrame,
    subway_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    subway_lat_col: str = "lat",
    subway_lon_col: str = "lon",
    bus_y_col: str = "y",
    bus_x_col: str = "x",
    subway_r_km: float = PARAMS["SUBWAY_R"],
    bus_r_km: float = PARAMS["BUS_R"],
) -> pd.DataFrame:
    out = df.copy()

    valid = out[x_col].notna() & out[y_col].notna()
    out["subway_dist"] = np.nan
    out["bus_dist"] = np.nan
    out["subway_cnt_500m"] = np.nan
    out["bus_cnt_300m"] = np.nan

    if not valid.any():
        return out

    subway_coords = subway_df[[subway_lat_col, subway_lon_col]].dropna().values
    bus_coords = bus_df[[bus_y_col, bus_x_col]].dropna().values
    apt_coords = out.loc[valid, [y_col, x_col]].values

    if len(apt_coords) == 0:
        return out

    lat0 = float(np.nanmean(apt_coords[:, 0]))
    apt_xy = to_xy_km(apt_coords[:, 0], apt_coords[:, 1], lat0=lat0)

    if len(subway_coords):
        subway_xy = to_xy_km(subway_coords[:, 0], subway_coords[:, 1], lat0=lat0)
        subway_tree = cKDTree(subway_xy)
        _, subway_idx = subway_tree.query(apt_xy, k=1)
        out.loc[valid, "subway_dist"] = [
            haversine_distance(
                apt_coords[i, 0],
                apt_coords[i, 1],
                subway_coords[subway_idx[i], 0],
                subway_coords[subway_idx[i], 1],
            )
            for i in range(len(apt_coords))
        ]
        subway_500m = subway_tree.query_ball_point(apt_xy, r=subway_r_km, p=2.0)
        out.loc[valid, "subway_cnt_500m"] = [len(x) for x in subway_500m]

    if len(bus_coords):
        bus_xy = to_xy_km(bus_coords[:, 0], bus_coords[:, 1], lat0=lat0)
        bus_tree = cKDTree(bus_xy)
        _, bus_idx = bus_tree.query(apt_xy, k=1)
        out.loc[valid, "bus_dist"] = [
            haversine_distance(
                apt_coords[i, 0],
                apt_coords[i, 1],
                bus_coords[bus_idx[i], 0],
                bus_coords[bus_idx[i], 1],
            )
            for i in range(len(apt_coords))
        ]
        bus_300m = bus_tree.query_ball_point(apt_xy, r=bus_r_km, p=2.0)
        out.loc[valid, "bus_cnt_300m"] = [len(x) for x in bus_300m]

    return out


# -------------------------
# Time / Build FE
# -------------------------
def fe_contract_date(
    df: pd.DataFrame,
    *,
    ym_col: str = "contract_ym",
) -> pd.DataFrame:
    """
    계약 시점 관련 파생 변수 생성
    - 계약년, 계약월
    - 계약분기 (1~4)
    """
    out = df.copy()

    ym = out[ym_col].astype(str).str.strip()
    if not ym.str.fullmatch(r"\d{6}").all():
        raise ValueError("contract_ym이 YYYYMM 형식이 아닌 값이 존재합니다.")

    out["contract_year"] = ym.str.slice(0, 4).astype(int)
    out["contract_month"] = ym.str.slice(4, 6).astype(int)

    # 분기: 외부 이벤트(금리, 정책, 규제 등)와의 결합 가능성
    out["contract_quarter"] = ((out["contract_month"] - 1) // 3 + 1).astype("int8")

    return out


def fe_build_year(
    df: pd.DataFrame,
    *,
    build_year_col: str = "build_year",
    contract_year_col: str = "contract_year",
) -> pd.DataFrame:
    """
    건축 시점 관련 파생 변수
    - 건물나이: max(계약년 - 건축년도, 0)
    - 건축지연시간: max(건축년도 - 계약년, 0)
    """
    out = df.copy()

    y = out[build_year_col].astype(str).str.strip()
    if not y.str.fullmatch(r"\d{4}").all():
        raise ValueError("build_year에 YYYY 형식이 아닌 값이 존재합니다.")

    out[build_year_col] = y.astype(int)

    # 계약 기준 건물 나이 (>= 0)
    build_age = out[contract_year_col] - out[build_year_col]
    out["build_age"] = build_age.clip(lower=0)

    # 계약이 건축보다 선행된 시간 (>= 0)
    out["build_delay"] = (-build_age).clip(lower=0)

    return out


# -------------------------
# Area / Category FE
# -------------------------
def fe_premium_area(
    df: pd.DataFrame, *, gu_col: str = "gu", out_col: str = "is_gangnam"
) -> pd.DataFrame:
    out = df.copy()
    premium = {"강남구", "서초구", "송파구"}
    out[out_col] = out[gu_col].astype(str).isin(premium).astype("int8")
    return out


def fe_complex_area_share(
    df: pd.DataFrame,
    *,
    out_share_col: str = "complex_area_share",
    out_missing_col: str = "complex_area_missing",
) -> pd.DataFrame:
    """
    단지 내 면적구간 세대수 비중과 세대수 정보 유무 플래그를 생성한다.
    """
    out = df.copy()

    cols = {
        "60이하": "units_area_le_60",
        "60_85": "units_area_60_85",
        "85_135": "units_area_85_135",
        "135초과": "units_area_gt_135",
    }

    for key, col in cols.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    total = out[list(cols.values())].sum(axis=1, min_count=1)
    out[out_missing_col] = total.isna().astype("int8")

    area_val = pd.to_numeric(out["area_m2"], errors="coerce")
    chosen = np.select(
        [
            area_val <= 60,
            (area_val > 60) & (area_val <= 85),
            (area_val > 85) & (area_val <= 135),
            area_val > 135,
        ],
        [
            out[cols["60이하"]],
            out[cols["60_85"]],
            out[cols["85_135"]],
            out[cols["135초과"]],
        ],
        default=np.nan,
    )

    out[out_share_col] = np.where(total.notna() & (total > 0), chosen / total, np.nan)

    return out


def fe_complex_scale(
    df: pd.DataFrame,
    *,
    total_buildings_col: str = "complex_buildings",
    total_units_col: str = "complex_units",
    total_area_col: str = "gross_floor_area",
    net_area_col: str = "net_area",
    building_area_col: str = "building_area",
    parking_col: str = "parking_spaces",
) -> pd.DataFrame:
    """
    단지 규모 관련 수치를 정리하고 비율 파생 변수를 생성한다.
    """
    out = df.copy()

    numeric_cols = [
        total_buildings_col,
        total_units_col,
        total_area_col,
        net_area_col,
        building_area_col,
        parking_col,
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["parking_per_unit"] = np.where(
        out[total_units_col] > 0,
        out[parking_col] / out[total_units_col],
        np.nan,
    )
    out["units_per_building"] = np.where(
        out[total_buildings_col] > 0,
        out[total_units_col] / out[total_buildings_col],
        np.nan,
    )
    out["gross_to_net_ratio"] = np.where(
        out[net_area_col] > 0,
        out[total_area_col] / out[net_area_col],
        np.nan,
    )

    return out


def build_freq_map(series: pd.Series) -> dict[str, float]:
    s = series.astype("string").str.strip()
    s = s.replace("", pd.NA).fillna("__NA__")
    freq = s.value_counts(normalize=True)
    return freq.to_dict()


def apply_freq_map(
    df: pd.DataFrame,
    *,
    src_col: str,
    out_col: str,
    freq_map: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    s = out[src_col].astype("string").str.strip()
    s = s.replace("", pd.NA).fillna("__NA__")
    out[out_col] = s.map(freq_map).fillna(0.0)
    return out


# -------------------------
# Merge / Split Helpers
# -------------------------
def concat_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    is_train_col: str = _IS_TRAIN,
    ori_index_col: str = _ORG_IDX,
) -> tuple[pd.DataFrame, pd.Series]:
    y = train[target_col].copy()
    X_tr = train.drop(columns=[target_col]).copy()
    X_tr[is_train_col] = 1
    X_tr[ori_index_col] = X_tr.index

    X_te = test.copy()
    X_te[is_train_col] = 0
    X_te[ori_index_col] = X_te.index

    X_all = pd.concat([X_tr, X_te], axis=0, ignore_index=True)

    return (X_all, y)


# -------------------------
# Merge / Split Helpers
# -------------------------
def split_train_test(
    all: pd.DataFrame,
    y: pd.Series,
    *,
    is_train_col: str = _IS_TRAIN,
    ori_index_col: str = _ORG_IDX,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """병합된 FE 결과를 train/test로 되돌리고 y를 정렬한다.

    `ori_index_col`로 원본 순서를 복원하고, 메타 컬럼 제거 후 인덱스를 리셋한다.
    """
    X_train = all.loc[all[is_train_col] == 1].copy()
    X_test = all.loc[all[is_train_col] == 0].copy()

    # 원본 인덱스 기준으로 정렬
    X_train = X_train.sort_values(ori_index_col)
    X_test = X_test.sort_values(ori_index_col)

    # train에서 row drop이 있었을 수 있으므로 원본 인덱스 기준으로 y 정렬
    y_train = y.loc[X_train[ori_index_col]].copy()

    # 불필요한 메타 컬럼 제거
    keep_cols = ~X_train.columns.isin(DROP_META_COLS)
    X_train = X_train.loc[:, keep_cols]
    X_test = X_test.loc[:, keep_cols]

    # 각 데이터 프레임의 인덱스 재설정
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    return X_train, y_train, X_test


# -------------------------
# Model Prep
# -------------------------
def prepare_for_lgbm(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLS,
    cat_cols: list[str] = CATEGORICAL_COLS,
    int_cols: list[str] = INT_COLS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    LightGBM 입력용 전처리
    - feature_cols만 유지
    - 범주형 컬럼은 category dtype으로 변환
    """
    out = df.copy()
    out = out.loc[:, [c for c in feature_cols if c in out.columns]]

    cat_cols = [c for c in cat_cols if c in out.columns]
    for c in cat_cols:
        if out[c].dtype != "category":
            out[c] = out[c].astype("string").astype("category")

    for c in out.columns:
        if c in cat_cols:
            continue
        if c in int_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int16")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float32")

    return out, cat_cols


def prepare_for_rf(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLS,
    cat_cols: list[str] = CATEGORICAL_COLS,
    int_cols: list[str] = INT_COLS,
) -> pd.DataFrame:
    """
    RandomForest 입력용 전처리
    - feature_cols만 유지
    - 범주형 컬럼은 라벨 인코딩
    - 연속형 결측치는 중앙값으로 보간
    """
    out = df.copy()
    out = out.loc[:, [c for c in feature_cols if c in out.columns]]

    for c in cat_cols:
        if c not in out.columns:
            continue
        out[c] = out[c].astype("string").fillna("UNKNOWN")
        codes, _ = pd.factorize(out[c])
        out[c] = codes.astype("int32")

    for c in out.columns:
        if c in cat_cols:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(out[c].median())
        if c in int_cols:
            out[c] = out[c].astype("Int16")
        else:
            out[c] = out[c].astype("Float32")

    return out


def prepare_for_lgbm_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLS,
    cat_cols: list[str] = CATEGORICAL_COLS,
    int_cols: list[str] = INT_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    LightGBM용 train/test 동시 전처리 (train 기준 category 정렬).
    """
    X_train, cat_cols = prepare_for_lgbm(
        train, feature_cols=feature_cols, cat_cols=cat_cols, int_cols=int_cols
    )
    X_test, _ = prepare_for_lgbm(
        test, feature_cols=feature_cols, cat_cols=cat_cols, int_cols=int_cols
    )
    for c in cat_cols:
        cats = X_train[c].cat.categories
        X_test[c] = X_test[c].cat.set_categories(cats)
    X_test = X_test.reindex(columns=X_train.columns)
    return X_train, X_test, cat_cols


def prepare_for_rf_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLS,
    cat_cols: list[str] = CATEGORICAL_COLS,
    int_cols: list[str] = INT_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    RandomForest용 train/test 동시 전처리 (train 기준 라벨 인코딩).
    """
    X_train = train.loc[:, [c for c in feature_cols if c in train.columns]].copy()
    X_test = test.loc[:, [c for c in feature_cols if c in test.columns]].copy()

    for c in cat_cols:
        if c not in X_train.columns or c not in X_test.columns:
            continue
        tr = X_train[c].astype("string").fillna("UNKNOWN")
        te = X_test[c].astype("string").fillna("UNKNOWN")
        cats = pd.Index(tr.unique())
        mapping = {k: i for i, k in enumerate(cats)}
        X_train[c] = tr.map(mapping).astype("Int16")
        X_test[c] = te.map(mapping).fillna(-1).astype("Int16")

    for c in X_train.columns:
        if c in cat_cols:
            continue
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)
        if c in int_cols:
            X_train[c] = X_train[c].astype("Int16")
            X_test[c] = X_test[c].astype("Int16")
        else:
            X_train[c] = X_train[c].astype("Float32")
            X_test[c] = X_test[c].astype("Float32")

    X_test = X_test.reindex(columns=X_train.columns)
    return X_train, X_test


def prepare_for_lgbm_train_valid_test(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLS,
    cat_cols: list[str] = CATEGORICAL_COLS,
    int_cols: list[str] = INT_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    X_train, cat_cols = prepare_for_lgbm(
        train, feature_cols=feature_cols, cat_cols=cat_cols, int_cols=int_cols
    )
    X_valid, _ = prepare_for_lgbm(
        valid, feature_cols=feature_cols, cat_cols=cat_cols, int_cols=int_cols
    )
    X_test, _ = prepare_for_lgbm(
        test, feature_cols=feature_cols, cat_cols=cat_cols, int_cols=int_cols
    )
    for c in cat_cols:
        cats = X_train[c].cat.categories
        X_valid[c] = X_valid[c].cat.set_categories(cats)
        X_test[c] = X_test[c].cat.set_categories(cats)
    X_valid = X_valid.reindex(columns=X_train.columns)
    X_test = X_test.reindex(columns=X_train.columns)
    return X_train, X_valid, X_test, cat_cols


def prepare_for_rf_train_valid_test(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str] = FEATURE_COLS,
    cat_cols: list[str] = CATEGORICAL_COLS,
    int_cols: list[str] = INT_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train.loc[:, [c for c in feature_cols if c in train.columns]].copy()
    X_valid = valid.loc[:, [c for c in feature_cols if c in valid.columns]].copy()
    X_test = test.loc[:, [c for c in feature_cols if c in test.columns]].copy()

    for c in cat_cols:
        if c not in X_train.columns:
            continue
        tr = X_train[c].astype("string").fillna("UNKNOWN")
        va = X_valid[c].astype("string").fillna("UNKNOWN")
        te = X_test[c].astype("string").fillna("UNKNOWN")
        cats = pd.Index(tr.unique())
        mapping = {k: i for i, k in enumerate(cats)}
        X_train[c] = tr.map(mapping).astype("Int16")
        X_valid[c] = va.map(mapping).fillna(-1).astype("Int16")
        X_test[c] = te.map(mapping).fillna(-1).astype("Int16")

    for c in X_train.columns:
        if c in cat_cols:
            continue
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_valid[c] = pd.to_numeric(X_valid[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_valid[c] = X_valid[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)
        if c in int_cols:
            X_train[c] = X_train[c].astype("Int16")
            X_valid[c] = X_valid[c].astype("Int16")
            X_test[c] = X_test[c].astype("Int16")
        else:
            X_train[c] = X_train[c].astype("Float32")
            X_valid[c] = X_valid[c].astype("Float32")
            X_test[c] = X_test[c].astype("Float32")

    return X_train, X_valid, X_test


def split_by_contract_ym(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    ym_col: str = "contract_ym",
    train_until: int = 202303,
    valid_from: int = 202304,
    valid_until: int = 202306,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    ym = pd.to_numeric(X[ym_col], errors="coerce")
    train_mask = ym <= train_until
    valid_mask = (ym >= valid_from) & (ym <= valid_until)
    X_train = X.loc[train_mask].copy()
    y_train = y.loc[X_train.index].copy()
    X_valid = X.loc[valid_mask].copy()
    y_valid = y.loc[X_valid.index].copy()
    return X_train, y_train, X_valid, y_valid


def apply_freq_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    builder_col: str = "builder",
    developer_col: str = "developer",
    apt_name_col: str = "apt_name",
    builder_out: str = "builder_freq",
    developer_out: str = "developer_freq",
    apt_name_out: str = "apt_name_freq",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    builder_map = build_freq_map(train[builder_col])
    developer_map = build_freq_map(train[developer_col])
    apt_name_map = build_freq_map(train[apt_name_col])
    train = apply_freq_map(
        train, src_col=builder_col, out_col=builder_out, freq_map=builder_map
    )
    valid = apply_freq_map(
        valid, src_col=developer_col, out_col=developer_out, freq_map=developer_map
    )
    valid = apply_freq_map(
        valid, src_col=builder_col, out_col=builder_out, freq_map=builder_map
    )
    valid = apply_freq_map(
        valid, src_col=apt_name_col, out_col=apt_name_out, freq_map=apt_name_map
    )
    test = apply_freq_map(
        test, src_col=builder_col, out_col=builder_out, freq_map=builder_map
    )
    test = apply_freq_map(
        test, src_col=developer_col, out_col=developer_out, freq_map=developer_map
    )
    test = apply_freq_map(
        test, src_col=apt_name_col, out_col=apt_name_out, freq_map=apt_name_map
    )
    train = apply_freq_map(
        train, src_col=developer_col, out_col=developer_out, freq_map=developer_map
    )
    train = apply_freq_map(
        train, src_col=apt_name_col, out_col=apt_name_out, freq_map=apt_name_map
    )
    return train, valid, test


def _apply_target_transform(y: pd.Series, *, target_transform: str) -> pd.Series:
    if target_transform == "log1p":
        return pd.Series(np.log1p(y), index=y.index)
    return y


def _invert_target_transform(
    values: np.ndarray, *, target_transform: str
) -> np.ndarray:
    if target_transform == "log1p":
        return np.expm1(values)
    return values


def train_predict_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    target_transform: str = "log1p",
    seed: int = 42,
    ym_col: str = "contract_ym",
    train_until: int = 202303,
    valid_from: int = 202304,
    valid_until: int = 202306,
) -> pd.Series:
    X_tr, y_tr, X_val, y_val = split_by_contract_ym(
        X_train,
        y_train,
        ym_col=ym_col,
        train_until=train_until,
        valid_from=valid_from,
        valid_until=valid_until,
    )

    X_tr, X_val, X_test = apply_freq_features(X_tr, X_val, X_test)
    y_tr_t = _apply_target_transform(y_tr, target_transform=target_transform)
    y_val_t = _apply_target_transform(y_val, target_transform=target_transform)
    y_full_t = _apply_target_transform(y_train, target_transform=target_transform)

    X_tr_p, X_val_p, X_test_p, _ = prepare_for_lgbm_train_valid_test(
        X_tr, X_val, X_test
    )
    model = LGBMRegressor(**LGBM_PARAMS, random_state=seed)
    model.fit(
        X_tr_p,
        y_tr_t,
        eval_set=[(X_val_p, y_val_t)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=100),
        ],
    )
    val_pred = np.asarray(model.predict(X_val_p))
    val_pred = _invert_target_transform(val_pred, target_transform=target_transform)
    y_val_eval = _invert_target_transform(
        np.asarray(y_val_t), target_transform=target_transform
    )
    rmse = np.sqrt(mean_squared_error(y_val_eval, val_pred))
    print(f"[LGBM] valid RMSE: {rmse:,.0f}")

    X_full_p, X_test_p, _ = prepare_for_lgbm_train_test(X_train, X_test)
    model.fit(X_full_p, y_full_t)
    pred = np.asarray(model.predict(X_test_p))
    pred = _invert_target_transform(pred, target_transform=target_transform)
    return pd.Series(pred)


def train_predict_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    target_transform: str = "log1p",
    seed: int = 42,
    ym_col: str = "contract_ym",
    train_until: int = 202303,
    valid_from: int = 202304,
    valid_until: int = 202306,
) -> pd.Series:
    X_tr, y_tr, X_val, y_val = split_by_contract_ym(
        X_train,
        y_train,
        ym_col=ym_col,
        train_until=train_until,
        valid_from=valid_from,
        valid_until=valid_until,
    )

    X_tr, X_val, X_test = apply_freq_features(X_tr, X_val, X_test)
    y_tr_t = _apply_target_transform(y_tr, target_transform=target_transform)
    y_val_t = _apply_target_transform(y_val, target_transform=target_transform)
    y_full_t = _apply_target_transform(y_train, target_transform=target_transform)

    X_tr_p, X_val_p, X_test_p = prepare_for_rf_train_valid_test(
        X_tr, X_val, X_test, feature_cols=FEATURE_COLS_RF
    )
    model = RandomForestRegressor(**RF_PARAMS, random_state=seed, n_jobs=-1)
    model.fit(X_tr_p, y_tr_t)
    val_pred = np.asarray(model.predict(X_val_p))
    val_pred = _invert_target_transform(val_pred, target_transform=target_transform)
    y_val_eval = _invert_target_transform(
        np.asarray(y_val_t), target_transform=target_transform
    )
    rmse = np.sqrt(mean_squared_error(y_val_eval, val_pred))
    print(f"[RF] valid RMSE: {rmse:,.0f}")

    X_full_p, X_test_p = prepare_for_rf_train_test(
        X_train, X_test, feature_cols=FEATURE_COLS_RF
    )
    model.fit(X_full_p, y_full_t)
    pred = np.asarray(model.predict(X_test_p))
    pred = _invert_target_transform(pred, target_transform=target_transform)
    return pd.Series(pred)


MODEL_RUNNERS = {
    "lgbm": train_predict_lgbm,
    "rf": train_predict_rf,
}


def train_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    model_type: str = "lgbm",
    target_transform: str = "log1p",
    seed: int = 42,
    ym_col: str = "contract_ym",
    train_until: int = 202303,
    valid_from: int = 202304,
    valid_until: int = 202306,
) -> pd.Series:
    if model_type not in MODEL_RUNNERS:
        raise ValueError(f"지원하지 않는 model_type: {model_type}")
    return MODEL_RUNNERS[model_type](
        X_train,
        y_train,
        X_test,
        target_transform=target_transform,
        seed=seed,
        ym_col=ym_col,
        train_until=train_until,
        valid_from=valid_from,
        valid_until=valid_until,
    )


def run_fe_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    subway: pd.DataFrame,
    bus: pd.DataFrame,
    apt: pd.DataFrame,
    coord: pd.DataFrame,
    params: dict,
):
    """
    Feature Engineering 로직을
    train / test 동시 처리 형태로 재현한다.

    반환:
        X_train_fe, y_train_aligned, X_test_fe
    """

    # ---------------------
    # 0. target 분리 + 결합
    # ---------------------
    train = rename_train_test(train)
    test = rename_train_test(test)
    subway = rename_subway(subway)
    bus = rename_bus(bus)
    apt = rename_apt(apt)
    coord = rename_coord(coord)

    X_all, y = concat_train_test(train, test)
    X_all = build_load_name_parts(X_all, src_col="road_name")

    # ---------------------
    # 0-1. 유의미하지 않은 값 처리
    # ---------------------
    X_all = fe_useless_to_na(
        X_all,
        rules={
            "trade_type": ["-"],
        },
    )

    # ---------------------
    # 0-2. 범주형 정리
    # ---------------------
    X_all = fe_category_clean(
        X_all,
        cols=[
            "corridor_type",
            "heating_type",
            "manage_type",
            "electric_contract_type",
            "cleaning_type",
            "complex_type",
            "trade_type",
            "sale_type",
        ],
    )
    X_all = fe_lease_type(X_all)

    # 1. 구 / 동 파생
    # ---------------------
    X_all = fe_gu_dong(X_all, src_col="sigungu", drop_src=True)

    # ---------------------
    # 2. 좌표 보정 준비 (train 기준)
    # ---------------------
    apt = build_load_name_full(apt)
    apt_coord_dict = build_coord_dict(
        apt,
        key_col="map_key",
        gu_col="gu",
        road_col="load_name_full",
        x_col="x",
        y_col="y",
    )
    coord_dict = build_coord_dict(
        coord,
        key_col="map_key",
        x_col="x",
        y_col="y",
    )
    apt_index = build_apt_similarity_index(apt)

    # ---------------------
    # 3. 좌표 채우기
    # ---------------------
    X_all = fill_coords(
        X_all,
        coord_dict=apt_coord_dict,
        gudong_prefix_mean_xy=None,
        gudong_mean_xy=None,
        gu_col="gu",
        dong_col="dong",
        road_col="load_name_full",
        x_col="x",
        y_col="y",
    )
    X_all = fill_coords(
        X_all,
        coord_dict=coord_dict,
        gudong_prefix_mean_xy=None,
        gudong_mean_xy=None,
        gu_col="gu",
        dong_col="dong",
        road_col="load_name_full",
        x_col="x",
        y_col="y",
    )
    X_all = fill_coords_by_apt_similarity(
        X_all,
        apt_index=apt_index,
        gu_col="gu",
        dong_col="dong",
        main_col="load_name_main",
        sub_col="load_name_sub",
        x_col="x",
        y_col="y",
    )

    # ---------------------
    # 3-1. 교통 접근성 피처
    # ---------------------
    X_all = fe_transport_features(
        X_all,
        subway,
        bus,
        x_col="x",
        y_col="y",
        subway_lat_col="lat",
        subway_lon_col="lon",
        bus_y_col="y",
        bus_x_col="x",
        subway_r_km=params["SUBWAY_R"],
        bus_r_km=params["BUS_R"],
    )

    # ---------------------
    # 4. 계약년/월/분기
    # ---------------------
    X_all = fe_contract_date(X_all, ym_col="contract_ym")

    # ---------------------
    # 5. 건축년도 / 건물나이
    # ---------------------
    X_all = fe_build_year(
        X_all,
        build_year_col="build_year",
        contract_year_col="contract_year",
    )

    # ---------------------
    # 6. 강남 여부
    # ---------------------
    X_all = fe_premium_area(X_all, gu_col="gu", out_col="is_gangnam")

    # ---------------------
    # 7. 단지 내 면적구간 비중
    # ---------------------
    X_all = fe_complex_area_share(X_all)

    # ---------------------
    # 7-1. 단지 규모 파생
    # ---------------------
    X_all = fe_complex_scale(X_all)

    # 8. train / test 분리
    # ---------------------
    X_train, y_train, X_test = split_train_test(X_all, y)

    return X_train, y_train, X_test


def main(
    model_type: str = "lgbm",
    target_transform: str = "log1p",
    seed: int = 42,
):
    train, test, subway, bus = load_raw_datas()
    apt, coord = load_ext_datas()

    X_train, y_train, X_test = run_fe_train_test(
        train,
        test,
        subway=subway,
        bus=bus,
        apt=apt,
        coord=coord,
        params=PARAMS,
    )

    out_dir_path = out_dir("subs")
    out_dir_path.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("submission")

    if model_type == "both":
        pred_lgbm = train_predict(
            X_train,
            y_train,
            X_test,
            model_type="lgbm",
            target_transform=target_transform,
            seed=seed,
        )
        pred_rf = train_predict(
            X_train,
            y_train,
            X_test,
            model_type="rf",
            target_transform=target_transform,
            seed=seed,
        )
        sub_lgbm = pd.DataFrame({"target": pred_lgbm.astype(int)})
        sub_rf = pd.DataFrame({"target": pred_rf.astype(int)})
        out_lgbm = out_dir_path / f"{run_id}_{_model_tag('lgbm')}.csv"
        out_rf = out_dir_path / f"{run_id}_{_model_tag('rf')}.csv"
        sub_lgbm.to_csv(out_lgbm, index=False)
        sub_rf.to_csv(out_rf, index=False)
        print(f"제출 파일 저장: {out_lgbm}")
        print(f"제출 파일 저장: {out_rf}")

        ensemble = (
            ENSEMBLE_WEIGHTS["lgbm"] * pred_lgbm + ENSEMBLE_WEIGHTS["rf"] * pred_rf
        )
        sub_ens = pd.DataFrame({"target": ensemble.round().astype(int)})
        out_ens = out_dir_path / f"{run_id}_{_model_tag('ensemble')}.csv"
        sub_ens.to_csv(out_ens, index=False)
        print(f"제출 파일 저장: {out_ens}")
        return

    pred = train_predict(
        X_train,
        y_train,
        X_test,
        model_type=model_type,
        target_transform=target_transform,
        seed=seed,
    )
    submission = pd.DataFrame({"target": pred.astype(int)})
    out_path = out_dir_path / f"{run_id}_{_model_tag(model_type)}.csv"
    submission.to_csv(out_path, index=False)
    print(f"제출 파일 저장: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FE + 모델 학습/예측 실행")
    parser.add_argument(
        "--model",
        choices=["lgbm", "rf", "both"],
        default="lgbm",
        help="학습할 모델 종류",
    )
    parser.add_argument(
        "--target-transform",
        choices=["log1p", "none"],
        default="log1p",
        help="타깃 변환 방식",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    main(
        model_type=args.model,
        target_transform=args.target_transform,
        seed=args.seed,
    )
