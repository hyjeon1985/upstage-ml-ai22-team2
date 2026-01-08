import numpy as np


def float_or_nan(x) -> float:
    """
    숫자로 변환 가능하면 float로, 아니면 np.nan 반환
    """
    try:
        if x is None:
            return np.nan
        return float(x)
    except (TypeError, ValueError):
        return np.nan
