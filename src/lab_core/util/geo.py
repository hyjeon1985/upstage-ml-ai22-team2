import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """위경도 기반 거리 계산 (km)

    Haversine 공식을 사용하여 두 지점 간의 대원거리(great-circle distance)를 계산합니다.
    지구를 완전한 구로 가정합니다.

    Parameters:
        lat1, lon1: 첫 번째 지점의 위도, 경도
        lat2, lon2: 두 번째 지점의 위도, 경도

    Returns:
        두 지점 간의 거리 (km)
    """
    R = 6371  # 지구 반지름 (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
