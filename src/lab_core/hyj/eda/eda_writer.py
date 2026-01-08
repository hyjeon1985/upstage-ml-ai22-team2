from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EdaRunArtifacts:
    run_dir: Path
    meta: dict[str, Any]
    summary: pd.DataFrame
    token_hits: pd.DataFrame
    outliers: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_eda_run(run_dir: str | Path) -> EdaRunArtifacts:
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "meta.json")

    # summary는 index=True로 저장했으므로 첫 컬럼이 index(원래 컬럼명)가 됩니다.
    summary = pd.read_csv(run_dir / "summary.csv")
    # summary.csv 저장 방식을 "index=True"로 했으면 보통 첫 컬럼명이 "Unnamed: 0"
    # 이를 col로 정리
    if "Unnamed: 0" in summary.columns:
        summary = summary.rename(columns={"Unnamed: 0": "col"})
    elif "index" in summary.columns:
        summary = summary.rename(columns={"index": "col"})
    else:
        # 이미 col이 있을 수도 있음
        pass

    token_hits = pd.read_csv(run_dir / "token_hits.csv")
    outliers = pd.read_csv(run_dir / "outliers.csv")

    return EdaRunArtifacts(
        run_dir=run_dir,
        meta=meta,
        summary=summary,
        token_hits=token_hits,
        outliers=outliers,
    )


def _df_to_md_table(df: pd.DataFrame, *, max_rows: int = 20) -> str:
    """tabulate 없이도 동작하는 단순 Markdown table."""
    if df.empty:
        return "_(empty)_"

    view = df.head(max_rows).copy()
    # 값이 너무 길면 잘라서 문서 가독성 확보
    for c in view.columns:
        view[c] = (
            view[c].astype(str).map(lambda x: x if len(x) <= 80 else x[:77] + "...")
        )

    cols = list(view.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _img_md(rel_path: str, *, alt: str) -> str:
    return f"![{alt}]({rel_path})"


def write_eda_markdown(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    dataset_name: str = "all_data",
    top_missing_cols: int = 20,
    top_token_hits: int = 30,
    top_outliers: int = 50,
) -> Path:
    """
    eda_outputs/<run>/ 안의 meta/summary/token_hits/outliers/plots 를 기반으로
    배포 가능한 Markdown 리포트를 생성한다.
    """
    art = load_eda_run(run_dir)
    run_dir = art.run_dir

    if output_path is None:
        output_path = run_dir / "EDA_Report.md"
    output_path = Path(output_path)

    meta = art.meta
    summary = art.summary.copy()
    token_hits = art.token_hits.copy()
    outliers = art.outliers.copy()

    # 요약 뽑기
    n_rows = int(meta.get("n_rows", 0))
    n_cols = int(meta.get("n_cols", summary.shape[0]))

    # 결측 상위
    missing_top = (
        summary.loc[summary["na_cnt"] > 0]
        .sort_values(["na_cnt", "nunique"], ascending=[False, False])
        .head(top_missing_cols)
    )

    # 토큰 히트 상위
    if not token_hits.empty:
        token_top = token_hits.sort_values(
            ["count", "col"], ascending=[False, True]
        ).head(top_token_hits)
    else:
        token_top = token_hits

    # 이상치 상위
    if not outliers.empty:
        out_top = outliers.head(top_outliers)
        # rule별 집계
        out_rule_cnt = outliers["rule"].value_counts().reset_index()
        out_rule_cnt.columns = ["rule", "count"]
    else:
        out_top = outliers
        out_rule_cnt = pd.DataFrame(columns=["rule", "count"])

    # plots 상대경로
    plots_dir = run_dir / "plots"
    has_missing_count = (plots_dir / "missing_count_topk.png").exists()
    has_missing_rate = (plots_dir / "missing_rate_hist.png").exists()

    # outlier 이미지: outlier_<col>.png 형태를 일부만 링크(존재하는 것만)
    # 문서에서는 outliers top에서 col 몇 개 뽑아 대표로 붙이기
    outlier_cols = []
    if not outliers.empty and "col" in outliers.columns:
        outlier_cols = (
            outliers.loc[outliers["rule"] == "IQR", "col"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()[:10]
        )

    outlier_imgs_md: list[str] = []
    for c in outlier_cols:
        safe = c.replace("/", "_").replace("\\", "_").replace(" ", "_")
        p = plots_dir / f"outlier_{safe}.png"
        if p.exists():
            outlier_imgs_md.append(_img_md(f"plots/{p.name}", alt=f"outlier_{c}"))

    tokens = meta.get("user_missing_tokens", [])

    md_parts: list[str] = []
    md_parts.append("# EDA Report\n")
    md_parts.append("## 1. Overview\n")
    md_parts.append(
        "\n".join(
            [
                f"- Dataset: `{dataset_name}`",
                f"- Run dir: `{run_dir.name}`",
                f"- Rows: `{n_rows:,}`",
                f"- Columns: `{n_cols:,}`",
                f"- Generated at: `{meta.get('created_at', '')}`",
                f"- Missing tokens: `{tokens}`",
            ]
        )
    )
    md_parts.append("\n---\n")

    md_parts.append("## 2. Missing Value Summary\n")
    md_parts.append(f"### 2.1 High-missing columns (Top {top_missing_cols})\n")
    md_parts.append(
        _df_to_md_table(
            missing_top[["col", "dtype", "na_cnt", "na_rate", "nunique"]],
            max_rows=top_missing_cols,
        )
    )
    md_parts.append("\n")
    md_parts.append("> Full table: `summary.csv`\n")

    if has_missing_count:
        md_parts.append(
            _img_md("plots/missing_count_topk.png", alt="missing_count_topk")
        )
        md_parts.append("\n")
    if has_missing_rate:
        md_parts.append(_img_md("plots/missing_rate_hist.png", alt="missing_rate_hist"))
        md_parts.append("\n")

    md_parts.append("\n---\n")
    md_parts.append("## 3. User-defined Missing Token Detection\n")
    md_parts.append(f"### 3.1 Token hit examples (Top {top_token_hits})\n")
    # token_hits 컬럼이 표준 형태(col,value,count,matched_tokens)라고 가정
    cols = [
        c for c in ["col", "value", "count", "matched_tokens"] if c in token_top.columns
    ]
    md_parts.append(_df_to_md_table(token_top[cols], max_rows=top_token_hits))
    md_parts.append("\n")
    md_parts.append("> Full list: `token_hits.csv`\n")

    md_parts.append("\n---\n")
    md_parts.append("## 4. Outlier / Rare-value Analysis\n")
    md_parts.append("### 4.1 Rules summary\n")
    md_parts.append(_df_to_md_table(out_rule_cnt, max_rows=20))
    md_parts.append("\n")
    md_parts.append(f"### 4.2 Outlier examples (Top {top_outliers})\n")
    cols2 = [
        c for c in ["col", "index", "value", "rule", "detail"] if c in out_top.columns
    ]
    md_parts.append(_df_to_md_table(out_top[cols2], max_rows=top_outliers))
    md_parts.append("\n")
    md_parts.append("> Full list: `outliers.csv`\n")

    if outlier_imgs_md:
        md_parts.append("\n### 4.3 Outlier plots (selected)\n")
        md_parts.append("\n\n".join(outlier_imgs_md))
        md_parts.append("\n")

    md_parts.append("\n---\n")
    md_parts.append("## 5. Notes & Recommendations\n")
    md_parts.append(
        "\n".join(
            [
                "- This report is intended for EDA and data-quality auditing.",
                "- Decisions (drop/impute/clip) should be made after checking business/domain constraints.",
                "- Token hits should generally be normalized to NaN before modeling.",
            ]
        )
    )
    md_parts.append("\n")

    output_path.write_text("\n".join(md_parts), encoding="utf-8")
    return output_path
