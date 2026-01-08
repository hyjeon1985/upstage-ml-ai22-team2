from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import pandas as pd

from lab_core.styles.viz import setup_global, use_style
from lab_core.util.time_ids import utc_iso

setup_global()

# -------------------------
# Missing report
# -------------------------

TokenMatchMode = Literal["exact", "startswith", "endswith"]


@dataclass(frozen=True)
class MissingReportResult:
    summary: pd.DataFrame
    token_hits: pd.DataFrame


def _norm_text(x: object) -> str:
    # 비교 목적 정규화: 공백 정리 + 소문자화
    if x is None:
        return ""
    s = str(x)
    s = " ".join(s.split())
    return s.casefold()


def _is_like_missing(
    x: object,
    tokens_norm: list[str],
    *,
    mode: TokenMatchMode = "startswith",
) -> bool:
    """
    정규화된 텍스트가 user_missing_tokens와
    exact / startswith / endswith 중 지정된 방식으로 매칭되면 True.
    """
    t = _norm_text(x)
    if not t:
        return False

    if mode == "exact":
        return t in tokens_norm

    if mode == "startswith":
        return any(t.startswith(tok) for tok in tokens_norm)

    if mode == "endswith":
        return any(t.endswith(tok) for tok in tokens_norm)

    raise ValueError(f"Unknown token match mode: {mode}")


def _match_tokens(
    v_norm: str,
    tokens_norm: list[str],
    *,
    mode: TokenMatchMode,
) -> list[str]:
    if mode == "exact":
        return [tok for tok in tokens_norm if v_norm == tok]

    if mode == "startswith":
        return [tok for tok in tokens_norm if v_norm.startswith(tok)]

    if mode == "endswith":
        return [tok for tok in tokens_norm if v_norm.endswith(tok)]

    return []


def _series_top_values(s: pd.Series, k: int = 5) -> str:
    vc = s.value_counts(dropna=True)
    if vc.empty:
        return ""
    return "; ".join(f"{repr(idx)}:{int(cnt)}" for idx, cnt in vc.head(k).items())


def missing_eda_report(
    df: pd.DataFrame,
    *,
    user_missing_tokens: Iterable[str] = (
        "없음",
        "모름",
        "해당없음",
        "없",
        "미상",
        "unknown",
        "n/a",
        "na",
        "none",
        "null",
    ),
    top_n: int = 30,
    token_match_mode: TokenMatchMode = "startswith",
    plot: bool = True,
    plot_top_k: int = 30,
    save_path: Path | None = None,
    dpi: int = 150,
) -> MissingReportResult:
    """
    결측치 EDA 리포트를 생성한다.

    생성물
    ------
    1) summary: 컬럼별 결측치 수/비율/유니크/상위값 요약
    2) token_hits: 사용자 정의 결측 토큰(부분 포함)과 유사한 문자열 값 탐지 결과

    Notes
    -----
    - user_missing_tokens는 "포함" 기반 매칭이며, 정규화(casefold/공백정리) 후 비교한다.
    """
    plot_path: Path | None = None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

        plot_path = save_path / "plots"
        plot_path.mkdir(exist_ok=True)

    n = len(df)

    # ---- summary
    na_cnt = df.isna().sum().astype(int)
    na_rate = (na_cnt / n).astype(float) if n else na_cnt.astype(float)

    nunique = df.nunique(dropna=True)
    dtypes = df.dtypes.astype(str)

    top_values = df.apply(lambda s: _series_top_values(s, k=5))

    summary = pd.DataFrame(
        {
            "dtype": dtypes,
            "n_rows": n,
            "na_cnt": na_cnt,
            "na_rate": na_rate,
            "non_na_cnt": (n - na_cnt).astype(int),
            "nunique": nunique.astype(int),
            "top_values": top_values,
        }
    ).sort_values(["na_cnt", "nunique"], ascending=[False, False])

    # ---- token_hits
    tokens_norm = [_norm_text(t) for t in user_missing_tokens if _norm_text(t)]
    hits = []
    if tokens_norm:
        obj_cols = [
            c
            for c in df.columns
            if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])
        ]
        for c in obj_cols:
            s = df[c].dropna()
            mask = s.map(
                lambda x: _is_like_missing(
                    x,
                    tokens_norm,
                    mode=token_match_mode,
                )
            )
            if not mask.any():
                continue

            # 어떤 토큰에 걸렸는지(대표)도 남김
            hit_vals = s[mask]
            # 대표값 상위 몇 개만 기록
            vc = hit_vals.value_counts(dropna=False).head(top_n)
            for v, cnt in vc.items():
                v_norm = _norm_text(v)
                matched = _match_tokens(v_norm, tokens_norm, mode=token_match_mode)
                hits.append(
                    {
                        "col": c,
                        "value": v,
                        "count": int(cnt),
                        "matched_tokens": ", ".join(matched[:5]),
                    }
                )
    token_hits = (
        pd.DataFrame(hits).sort_values(["count", "col"], ascending=[False, True])
        if hits
        else pd.DataFrame(columns=["col", "value", "count", "matched_tokens"])
    )

    # ---- plot
    with use_style():
        # ---- missing plots

        # 결측치 많은 컬럼 top-k bar
        topk = summary[summary["na_cnt"] > 0].head(plot_top_k)
        if not topk.empty:
            fig = plt.figure(figsize=(max(10, int(0.35 * len(topk))), 4))
            plt.bar(topk.index.astype(str), topk["na_cnt"].to_numpy())
            plt.xticks(rotation=75, ha="right")
            plt.title("Missing Count (Top-K columns)")
            plt.tight_layout()
            if plot_path is not None:
                fig.savefig(plot_path / "missing_count_topk.png", dpi=dpi)
            if plot:
                plt.show()
            plt.close(fig)

        # 결측률 히스토그램
        fig = plt.figure(figsize=(8, 4))
        plt.hist(summary["na_rate"].dropna().to_numpy(), bins=30)
        plt.title("Missing Rate Distribution")
        plt.tight_layout()
        if plot_path is not None:
            fig.savefig(plot_path / "missing_rate_hist.png", dpi=dpi)
        if plot:
            plt.show()
        plt.close(fig)

    # Save DataFrames
    if save_path is not None:
        # ---- tables
        summary.to_csv(save_path / "missing_summary.csv", index=True)
        token_hits.to_csv(save_path / "missing_token_hits.csv", index=False)
        # ---- meta
        meta = {
            "created_at": utc_iso(),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "user_missing_tokens": list(user_missing_tokens),
            "plot": plot,
            "plot_top_k": plot_top_k,
        }

        with open(save_path / "missing_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return MissingReportResult(summary=summary, token_hits=token_hits)
