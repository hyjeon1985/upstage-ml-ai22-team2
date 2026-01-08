from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


class Preset(str, Enum):
    LIGHT = "light"
    DARK = "dark"


_LIGHT_PALETTE = {
    # 배경
    "figure_bg": "#ffffff",
    "axes_bg": "#ffffff",
    # 텍스트
    "text": "#222222",
    "title": "#111111",
    "label": "#222222",
    "tick": "#444444",
    # 그리드 / 스파인
    "grid": "#dddddd",
    "spine": "#cccccc",
    # 범례
    "legend_bg": "#f7f7f7",
    "legend_edge": "#cccccc",
    # 기본 선 색상 (Tableau 계열, 보편적)
    "line_colors": [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ],
}

_DARK_PALETTE = {
    # 배경
    "figure_bg": "#1e1e1e",
    "axes_bg": "#1e1e1e",
    # 텍스트
    "text": "#e6e6e6",
    "title": "#ffffff",
    "label": "#e6e6e6",
    "tick": "#cfcfcf",
    # 그리드 / 스파인
    "grid": "#3a3a3a",
    "spine": "#555555",
    # 범례
    "legend_bg": "#2a2a2a",
    "legend_edge": "#555555",
    # 선 색상 (채도/명도 균형)
    "line_colors": [
        "#4e79a7",  # muted blue
        "#f28e2b",  # muted orange
        "#59a14f",  # muted green
        "#e15759",  # muted red
        "#b07aa1",  # muted purple
        "#9c755f",  # muted brown
        "#edc949",  # muted yellow
        "#76b7b2",  # muted cyan
        "#bab0ac",  # muted gray
        "#ff9da7",  # soft pink
    ],
}


@dataclass(frozen=True)
class VizConfig:
    # 한국어 표시를 위한 최소 설정(필수)
    font_candidates: tuple[str, ...] = (
        "Pretendard",
        "NanumBarunGothic",
        "Malgun Gothic",  # Windows fallback
        "AppleGothic",  # macOS fallback
        "DejaVu Sans",  # 최후 fallback
    )

    base_font_size: float = 11.0
    font_weight: str = "medium"

    # 한글 환경에서 '-' 깨짐 방지: 보통 False 권장
    unicode_minus: bool = False


def _pick_font(candidates: tuple[str, ...]) -> Optional[str]:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return None


# ---- 전역(안전) 설정: 폰트/마이너스/기본 폰트 크기 정도만 ----
def setup_global(
    *,
    cfg: VizConfig = VizConfig(),
) -> None:
    """
    커널 전역에 적용해도 안전한 한글 관련 설정을 수행합니다.

    Args:
        cfg:
            시각화 설정 객체
    """
    _apply_font_base(cfg)


def _apply_font_base(cfg: VizConfig) -> None:
    font = _pick_font(cfg.font_candidates)
    if font:
        mpl.rcParams["font.family"] = font
    mpl.rcParams.update(
        {
            "font.weight": cfg.font_weight,
            "axes.unicode_minus": cfg.unicode_minus,
            "font.size": cfg.base_font_size,
        }
    )


def _apply_font_detail(cfg: VizConfig) -> None:
    mpl.rcParams.update(
        {
            "axes.titlesize": cfg.base_font_size + 3,
            "axes.labelsize": cfg.base_font_size + 1,
            "xtick.labelsize": cfg.base_font_size - 1,
            "ytick.labelsize": cfg.base_font_size - 1,
            "legend.fontsize": cfg.base_font_size - 1,
        }
    )


def _apply_palette(preset: Preset) -> None:
    is_dark = preset == Preset.DARK
    p = _DARK_PALETTE if is_dark else _LIGHT_PALETTE

    mpl.rcParams.update(
        {
            "figure.facecolor": p["figure_bg"],
            "axes.facecolor": p["axes_bg"],
            "savefig.facecolor": p["figure_bg"],
            "savefig.edgecolor": p["figure_bg"],
            "text.color": p["text"],
            "axes.labelcolor": p["label"],
            "axes.titlecolor": p["title"],
            "xtick.color": p["tick"],
            "ytick.color": p["tick"],
            "axes.edgecolor": p["spine"],
            "axes.grid": True,
            "axes.spines.top": is_dark,
            "axes.spines.right": is_dark,
            "grid.color": p["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.6,
            "legend.facecolor": p["legend_bg"],
            "legend.edgecolor": p["legend_edge"],
            "legend.framealpha": 0.95,
            "axes.prop_cycle": mpl.cycler(color=p["line_colors"]),
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
        }
    )


def _apply_seaborn_style(preset: Preset) -> None:
    try:
        import seaborn as sns
    except ModuleNotFoundError:
        return

    style = "darkgrid" if preset == Preset.DARK else "whitegrid"
    sns.set_style(style=style)


# ---- 임시(안전) 적용: 프리셋 스타일은 컨텍스트로만 ----
@contextmanager
def use_style(
    *, preset: Preset = Preset.LIGHT, cfg: VizConfig = VizConfig()
) -> Iterator[None]:
    """
    프리셋 스타일을 '블록 내부에서만' 적용합니다.
    블록 종료 시 rcParams는 자동 원복되어 다른 노트북에 영향이 거의 없습니다.
    """

    with mpl.rc_context():
        # 'default'로 초기화는 컨텍스트 내부에서만 수행 -> 외부 오염 없음
        plt.style.use("default")

        _apply_seaborn_style(preset)
        _apply_palette(preset)
        _apply_font_base(cfg)
        _apply_font_detail(cfg)

        yield
