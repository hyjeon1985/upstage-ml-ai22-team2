import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """위경도 기반 대원거리(km)를 계산한다."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    earth_radius_km = 6371.0
    return earth_radius_km * c


def latlon_to_km(lat: np.ndarray, lon: np.ndarray, *, lat0: float) -> np.ndarray:
    """위경도를 기준 위도 기준의 평면 km 좌표로 근사 변환한다."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat0_r = np.radians(lat0)
    y_km = lat_r * 6371.0
    x_km = lon_r * np.cos(lat0_r) * 6371.0
    return np.column_stack([y_km, x_km])
