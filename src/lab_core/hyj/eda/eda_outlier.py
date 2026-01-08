from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab_core.styles.viz import setup_global, use_style
from lab_core.util.time_ids import utc_iso

setup_global()

# -------------------------
# Outlier report - Types
# -------------------------


@dataclass(frozen=True)
class _ColumnResult:
    summary: dict[str, Any]  # col 단위 요약 row
    outliers: pd.DataFrame  # _REPORT_COLS 스키마 (+ 필요시 value만)


@dataclass(frozen=True)
class OutlierReportResult:
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    outliers: pd.DataFrame


_REPORT_COLS = ["row_index", "value", "rule", "side", "detail"]

RareOrder = Literal["first_seen", "count_asc", "count_desc", "value"]
OutlierSide = Literal["low", "high", "rare"]


# -------------------------
# Outlier report - Statistics
# -------------------------


def _numeric_outliers(
    s: pd.Series,
    *,
    k: float = 1.5,
) -> _ColumnResult:
    x = pd.to_numeric(s, errors="coerce").dropna()

    empty_out = pd.DataFrame(columns=_REPORT_COLS)
    base_summary: dict[str, Any] = {
        "dtype": str(s.dtype),
        "n_non_na": int(len(x)),
        "rule": "IQR",
        "k": float(k),
        "outlier_cnt": 0,
        "outlier_rate": 0.0,
        "outlier_low_cnt": 0,
        "outlier_high_cnt": 0,
        "iqr": np.nan,
        "iqr_lo": np.nan,
        "iqr_hi": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
    }

    if x.empty:
        return _ColumnResult(outliers=empty_out, summary=base_summary)

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        # 분포가 상수(또는 너무 단순) → outlier 의미 없음
        base_summary.update(
            {
                "iqr": float(iqr) if pd.notna(iqr) else np.nan,
                "min": float(x.min()),
                "median": float(x.median()),
                "max": float(x.max()),
            }
        )
        return _ColumnResult(outliers=empty_out, summary=base_summary)

    lo = q1 - k * iqr
    hi = q3 + k * iqr

    mask = (x < lo) | (x > hi)
    if not mask.any():
        base_summary.update(
            {
                "iqr": float(iqr),
                "iqr_lo": float(lo),
                "iqr_hi": float(hi),
                "min": float(x.min()),
                "median": float(x.median()),
                "max": float(x.max()),
            }
        )
        return _ColumnResult(outliers=empty_out, summary=base_summary)

    vals = x[mask]
    side = np.where(vals < lo, "low", "high")

    outliers = pd.DataFrame(
        {
            "row_index": vals.index,
            "value": vals.values,
            "rule": "IQR",
            "side": side,
            "detail": f"k={k:g}, iqr={float(iqr):.4g}, lo={float(lo):.4g}, hi={float(hi):.4g}",
        }
    )

    low_cnt = int((outliers["side"] == "low").sum())
    high_cnt = int((outliers["side"] == "high").sum())
    out_cnt = int(len(outliers))
    n = int(len(x))

    base_summary.update(
        {
            "outlier_cnt": out_cnt,
            "outlier_rate": (out_cnt / n) if n else 0.0,
            "outlier_low_cnt": low_cnt,
            "outlier_high_cnt": high_cnt,
            "iqr": float(iqr),
            "iqr_lo": float(lo),
            "iqr_hi": float(hi),
            "min": float(x.min()),
            "median": float(x.median()),
            "max": float(x.max()),
        }
    )
    return _ColumnResult(outliers=outliers, summary=base_summary)


def _categorical_outliers(
    s: pd.Series,
    *,
    min_count: int = 3,
    max_examples: int | None = 30,
    order: RareOrder = "count_asc",
) -> _ColumnResult:
    ss = s.dropna()

    empty_out = pd.DataFrame(columns=_REPORT_COLS)
    base_summary: dict[str, Any] = {
        "dtype": str(s.dtype),
        "n_non_na": int(len(ss)),
        "rule": "RARE",
        "min_count": int(min_count),
        "outlier_cnt": 0,
        "outlier_rate": 0.0,
        "outlier_low_cnt": 0,
        "outlier_high_cnt": 0,
    }

    if ss.empty:
        return _ColumnResult(outliers=empty_out, summary=base_summary)

    vc = ss.value_counts(dropna=False)
    rare_counts = vc[vc < min_count]
    if rare_counts.empty:
        return _ColumnResult(outliers=empty_out, summary=base_summary)

    rare_ss = ss[ss.isin(rare_counts.index)]
    reps = rare_ss.groupby(rare_ss, sort=False).head(1).to_frame(name="value")
    reps.index.name = "row_index"
    reps = reps.reset_index()
    reps["_count"] = reps["value"].map(lambda v: int(rare_counts.loc[v]))

    if order == "count_asc":
        reps = reps.sort_values(["_count", "row_index"], ascending=[True, True])
    elif order == "count_desc":
        reps = reps.sort_values(["_count", "row_index"], ascending=[False, True])
    elif order == "value":
        reps = reps.sort_values(["value", "row_index"], ascending=[True, True])
    elif order == "first_seen":
        pass
    else:
        raise ValueError(f"Unknown order: {order}")

    if max_examples is not None:
        reps = reps.head(max_examples)

    outliers = pd.DataFrame(
        {
            "row_index": reps["row_index"].values,
            "value": reps["value"].values,
            "rule": "RARE",
            "side": "rare",
            "detail": reps["_count"]
            .map(lambda c: f"count={int(c)}, min_count={min_count}")
            .values,
        }
    )

    n = int(len(ss))
    out_cnt = int(len(outliers))
    base_summary.update(
        {
            "outlier_cnt": out_cnt,
            "outlier_rate": (out_cnt / n) if n else 0.0,
        }
    )
    return _ColumnResult(outliers=outliers, summary=base_summary)


# -------------------------
# Outlier report - Plots
# -------------------------


def _is_plot_worthy_numeric(
    row: pd.Series,
    *,
    n_rows: int,
    min_non_na: int = 500,
    min_non_na_rate: float = 0.05,
    min_outlier_cnt: int = 10,
    min_outlier_rate: float = 0.001,
) -> bool:
    n_non_na = int(row.get("n_non_na", 0) or 0)
    if n_non_na < min_non_na:
        return False
    if (n_non_na / max(n_rows, 1)) < min_non_na_rate:
        return False

    iqr = row.get("iqr", np.nan)
    if pd.isna(iqr) or float(iqr) <= 0:
        return False

    # nunique가 summary에 없으면 생략 가능(있으면 이진 제외에 도움)
    nunique = row.get("nunique", None)
    if nunique is not None and int(nunique) <= 2:
        return False

    out_cnt = int(row.get("outlier_cnt", 0) or 0)
    out_rate = float(row.get("outlier_rate", 0.0) or 0.0)

    return (out_cnt >= min_outlier_cnt) or (out_rate >= min_outlier_rate)


def _plot_numeric_panels(
    s: pd.Series,
    *,
    col: str,
    mode: Literal["show", "save"] = "save",
    save_file: Path | None = None,
    dpi: int = 150,
):
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return

    n = x.size
    n_neg = int((x < 0).sum())
    n_zero = int((x == 0).sum())

    with use_style():
        if mode == "show":
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

            # 1) box
            axes[0].boxplot(x, vert=False, showfliers=True)
            axes[0].set_title(f"Boxplot: {col}")
            axes[0].text(
                0.99,
                0.05,
                f"n={n}\nneg={n_neg}\nzero={n_zero}",
                ha="right",
                va="bottom",
                transform=axes[0].transAxes,
                fontsize=9,
                alpha=0.8,
            )

            # 2) “의미 있는” 히스토그램 1개만
            # - 양수/0만 충분히 많고 스케일이 크면 log1p
            pos = x[x >= 0]
            use_log1p = False
            if pos.size >= 50:
                pos_pos = pos[pos > 0]
                if pos_pos.size >= 10:
                    lo = float(pos_pos.min())
                    hi = float(pos_pos.max())
                    if (hi / max(lo, 1e-12)) > 1e3:
                        use_log1p = True

            if use_log1p:
                axes[1].hist(np.log1p(pos), bins=30)
                axes[1].set_title("Histogram (log1p, x>=0)")
                axes[1].set_xlabel(f"log1p({col})")
            else:
                axes[1].hist(x, bins=30)
                axes[1].set_title("Histogram (linear)")
                axes[1].set_xlabel(col)

            fig.suptitle(f"Outlier Check: {col}")
        else:
            # save: 2x2 (4 panels)
            fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
            ax = axes.ravel()

            # 1) box
            ax[0].boxplot(x, vert=False, showfliers=True)
            ax[0].set_title("Boxplot")
            ax[0].text(
                0.99,
                0.05,
                f"n={n}\nneg={n_neg}\nzero={n_zero}",
                ha="right",
                va="bottom",
                transform=ax[0].transAxes,
                fontsize=9,
                alpha=0.8,
            )

            # 2) linear hist
            ax[1].hist(x, bins=30)
            ax[1].set_title("Histogram (linear)")
            ax[1].set_xlabel(col)
            ax[1].set_ylabel("count")

            # 3) log1p hist (x>=0 only)
            pos = x[x >= 0]
            if pos.size > 0:
                ax[2].hist(np.log1p(pos), bins=30)
                ax[2].set_title("Histogram (log1p, x>=0)")
                ax[2].set_xlabel(f"log1p({col})")
                ax[2].set_ylabel("count")
            else:
                ax[2].axis("off")
                ax[2].text(
                    0.5, 0.5, "no x>=0 values", ha="center", va="center", alpha=0.6
                )

            # 4) ECDF (정규성/꼬리 판단 보조용: Q-Q보다 구현 간단/견고)
            xs = np.sort(x)
            p = (np.arange(1, xs.size + 1) - 0.5) / xs.size
            ax[3].plot(xs, p)
            ax[3].set_title("ECDF")
            ax[3].set_xlabel(col)
            ax[3].set_ylabel("p")

            fig.suptitle(f"Outlier & Distribution (saved): {col}")

        if save_file is not None:
            fig.savefig(save_file, dpi=dpi)
        else:
            plt.show()
        plt.close(fig)


# -------------------------
# Outlier report
# -------------------------


def outlier_eda_report(
    df: pd.DataFrame,
    *,
    rare_min_count: int = 3,
    max_rows_per_col: int = 50,
    plot: bool = True,
    plot_top_k: int = 30,
    save_path: Path | None = None,
    dpi: int = 150,
) -> OutlierReportResult:
    """
    이상치 EDA 리포트를 생성한다.

    생성물
    ------
    1) summary: 이상치 리포트 요약
    3) outliers: 숫자형(IQR) 극단값 + 범주형(희귀값) 탐지 결과

    Notes
    -----
    - outlier 탐지는 EDA용 휴리스틱이며, 정답 판정이 아니라 후보를 뽑아주는 기능이다.
    """
    plot_path: Path | None = None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

        plot_path = save_path / "plots"
        plot_path.mkdir(exist_ok=True)

    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []
    out_rows = []

    for c in df.columns:
        s = df[c]

        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            res = _numeric_outliers(s)
            row = res.summary.copy()
            row["col"] = c
            numeric_rows.append(row)
        else:
            res = _categorical_outliers(s, min_count=rare_min_count)
            row = res.summary.copy()
            row["col"] = c
            categorical_rows.append(row)

        # outliers
        out = res.outliers
        if not out.empty:
            out = out.head(max_rows_per_col).copy()
            out.insert(0, "col", c)
            out_rows.append(out)

    numeric_summary = (
        pd.DataFrame(numeric_rows).set_index("col")
        if numeric_rows
        else pd.DataFrame().set_index(pd.Index([], name="col"))
    )

    categorical_summary = (
        pd.DataFrame(categorical_rows).set_index("col")
        if categorical_rows
        else pd.DataFrame().set_index(pd.Index([], name="col"))
    )

    if not numeric_summary.empty:
        numeric_summary = numeric_summary.sort_values(
            ["outlier_cnt", "outlier_rate", "n_non_na"],
            ascending=[False, False, False],
        )

    if not categorical_summary.empty:
        categorical_summary = categorical_summary.sort_values(
            ["outlier_cnt", "outlier_rate", "n_non_na"],
            ascending=[False, False, False],
        )

    outliers = (
        pd.concat(out_rows, ignore_index=True)
        if out_rows
        else pd.DataFrame(columns=["col", *_REPORT_COLS])
    )

    if not plot and not save_path:
        return OutlierReportResult(
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            outliers=outliers,
        )

    # ---- plot
    n_rows = len(df)

    candidates = [
        c
        for c, row in numeric_summary.iterrows()
        if _is_plot_worthy_numeric(row, n_rows=n_rows)
    ]

    # show: 1~2장씩, 최대 show_max개
    show_max = 5
    show_cols = candidates[:show_max]

    # save: candidates 전체
    for c in candidates:
        s = df[c]
        str_c = str(c)
        save_file = (
            (plot_path / f"outlier_{str_c.replace('/', '_').replace(' ', '_')}.png")
            if plot_path
            else None
        )

        # 화면 출력은 show_cols만
        if plot and (c in show_cols):
            _plot_numeric_panels(s, col=str_c, mode="show", dpi=dpi, save_file=None)

        # 저장은 후보 전부
        if save_file is not None:
            _plot_numeric_panels(
                s, col=str_c, mode="save", dpi=dpi, save_file=save_file
            )

    # Save DataFrames
    if save_path is not None:
        # ---- tables
        numeric_summary.to_csv(save_path / "outlier_numeric_summary.csv", index=True)
        categorical_summary.to_csv(
            save_path / "outlier_categorical_summary.csv", index=True
        )
        outliers.to_csv(save_path / "outliers.csv", index=False)

        # ---- meta
        meta = {
            "created_at": utc_iso(),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "rare_min_count": rare_min_count,
            "max_rows_per_col": max_rows_per_col,
            "plot": plot,
            "plot_top_k": plot_top_k,
        }

        with open(save_path / "outlier_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return OutlierReportResult(
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        outliers=outliers,
    )
