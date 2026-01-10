from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from lab_core.util.path import eda_out_dir, ext_data_dir, raw_data_dir
from lab_core.util.time_ids import make_run_id

RAW_FILES = {
    # Train columns: ['시군구', '번지', '본번', '부번', '아파트명', '전용면적(㎡)', '계약년월', '계약일', '층', '건축년도', '도로명', '해제사유발생일', '등기신청일자', '거래유형', '중개사소재지', 'k-단지분류(아파트,주상복합등등)', 'k-전화번호', 'k-팩스번호', '단지소개기존clob', 'k-세대타입(분양형태)', 'k-관리방식', 'k-복도유형', 'k-난방방식', 'k-전체동수', 'k-전체세대수', 'k-건설사(시공사)', 'k-시행사', 'k-사용검사일-사용승인일', 'k-연면적', 'k-주거전용면적', 'k-관리비부과면적', 'k-전용면적별세대현황(60㎡이하)', 'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-85㎡~135㎡이하', 'k-135㎡초과', 'k-홈페이지', 'k-등록일자', 'k-수정일자', '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태', '건축면적', '주차대수', '기타/의무/임대/임의=1/2/3/4', '단지승인일', '사용허가여부', '관리비 업로드', '좌표X', '좌표Y', '단지신청일', 'target']
    "TRAIN": "train.csv",
    # Test columns: ['시군구', '번지', '본번', '부번', '아파트명', '전용면적(㎡)', '계약년월', '계약일', '층', '건축년도', '도로명', '해제사유발생일', '등기신청일자', '거래유형', '중개사소재지', 'k-단지분류(아파트,주상복합등등)', 'k-전화번호', 'k-팩스번호', '단지소개기존clob', 'k-세대타입(분양형태)', 'k-관리방식', 'k-복도유형', 'k-난방방식', 'k-전체동수', 'k-전체세대수', 'k-건설사(시공사)', 'k-시행사', 'k-사용검사일-사용승인일', 'k-연면적', 'k-주거전용면적', 'k-관리비부과면적', 'k-전용면적별세대현황(60㎡이하)', 'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-85㎡~135㎡이하', 'k-135㎡초과', 'k-홈페이지', 'k-등록일자', 'k-수정일자', '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태', '건축면적', '주차대수', '기타/의무/임대/임의=1/2/3/4', '단지승인일', '사용허가여부', '관리비 업로드', '좌표X', '좌표Y', '단지신청일']
    "TEST": "test.csv",
    # Subway columns: ['역사_ID', '역사명', '호선', '위도', '경도']
    "SUBWAY": "subway_feature.csv",
    # Bus columns: ['노드 ID', '정류소번호', '정류소명', 'X좌표', 'Y좌표', '정류소 타입']
    "BUS": "bus_feature.csv",
}


EXT_FILES = {
    # Apt columns: ['번호', 'k-아파트코드', 'k-아파트명', 'k-단지분류(아파트,주상복합등등)', 'kapt도로명주소', '주소(시도)k-apt주소split', '주소(시군구)', '주소(읍면동)', '나머지주소', '주소(도로명)', '주소(도로상세주소)', 'k-전화번호', 'k-팩스번호', '단지소개기존clob', '단지첨부파일', 'k-세대타입(분양형태)', 'k-관리방식', 'k-복도유형', 'k-난방방식', 'k-전체동수', 'k-전체세대수', 'k-건설사(시공사)', 'k-시행사', 'k-사용검사일-사용승인일', 'k-연면적', 'k-주거전용면적', 'k-관리비부과면적', 'k-전용면적별세대현황(60㎡이하)', 'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-85㎡~135㎡이하', 'k-135㎡초과', 'k-홈페이지', 'k-등록일자', 'k-수정일자', '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태', '건축면적', '주차대수', '기타/의무/임대/임의=1/2/3/4', '단지승인일', '사용허가여부', '관리비 업로드', '좌표X', '좌표Y', '단지신청일']
    "APT": "k-apt.csv",
    # Coord columns: ['매핑키', '좌표X', '좌표Y']
    "COORD": "coord.csv",
}


def load_train_data() -> pd.DataFrame:
    train_path = raw_data_dir(RAW_FILES["TRAIN"])

    train = pd.read_csv(train_path)

    return train


def load_raw_datas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 경로 설정
    train_path = raw_data_dir(RAW_FILES["TRAIN"])
    test_path = raw_data_dir(RAW_FILES["TEST"])
    subway_path = raw_data_dir(RAW_FILES["SUBWAY"])
    bus_path = raw_data_dir(RAW_FILES["BUS"])

    # Train, Test 데이터셋 로드
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 지하철 정보 데이터셋 로드
    subway = pd.read_csv(subway_path)

    # 버스 정보 데이터셋 로드
    bus = pd.read_csv(bus_path)

    return (train, test, subway, bus)


def load_ext_datas() -> tuple[pd.DataFrame, pd.DataFrame]:
    # 경로 설정
    apt_path = ext_data_dir(EXT_FILES["APT"])
    coord_path = ext_data_dir(EXT_FILES["COORD"])

    # 서울 아파트 단지 정보 데이터셋 로드
    apt = pd.read_csv(apt_path)

    # 아파트 좌표정보 데이터셋 로드
    coord = pd.read_csv(coord_path)

    return (apt, coord)


def get_eda_out_dir(title: str) -> Path:
    id = make_run_id(title)
    dir = eda_out_dir(id)
    dir.mkdir(parents=True, exist_ok=True)
    return dir
