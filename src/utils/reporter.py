"""
Reporting module: HTML, CSV, plots, ranking tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_csv(df: pd.DataFrame, path: Path, description: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info(f"CSV saved: {path} ({description})")


def plot_equity_curve(
    equity_curve: pd.Series,
    drawdown_curve: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(equity_curve.index, equity_curve.values, color="steelblue", linewidth=1)
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel("Equity (USDT)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax1.grid(True, alpha=0.3)

    dd_pct = drawdown_curve * 100
    ax2.fill_between(dd_pct.index, dd_pct.values, 0, color="tomato", alpha=0.5)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    data: pd.DataFrame,
    title: str,
    output_path: Path,
    fmt: str = ".2f",
    cmap: str = "RdYlGn",
) -> None:
    """Plot a DataFrame as a color heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(data.columns) * 1.2),
                                    max(4, len(data) * 0.5)))

    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        plt.close(fig)
        return

    im = ax.imshow(numeric_data.values, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(numeric_data.columns)))
    ax.set_xticklabels(numeric_data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(numeric_data.index)))
    ax.set_yticklabels(numeric_data.index, fontsize=8)

    for i in range(len(numeric_data.index)):
        for j in range(len(numeric_data.columns)):
            val = numeric_data.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                       fontsize=6, color="black")

    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def build_html_report(
    ranking_df: pd.DataFrame,
    title: str,
    output_path: Path,
    extra_sections: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Build a simple HTML report from a ranking DataFrame.
    extra_sections: list of {"heading": str, "html": str}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table_html = ranking_df.to_html(
        classes="ranking-table",
        border=0,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        na_rep="N/A",
    )

    extra_html = ""
    if extra_sections:
        for sec in extra_sections:
            extra_html += f"<h2>{sec['heading']}</h2>\n{sec['html']}\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .ranking-table {{ border-collapse: collapse; width: 100%; background: white; }}
  .ranking-table th {{ background: #2c3e50; color: white; padding: 8px; text-align: left; }}
  .ranking-table td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; font-size: 12px; }}
  .ranking-table tr:nth-child(even) {{ background: #f9f9f9; }}
  .ranking-table tr:hover {{ background: #eaf3fb; }}
</style>
</head>
<body>
<h1>{title}</h1>
{table_html}
{extra_html}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML report saved: {output_path}")


def compile_rankings(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten a list of result dicts into a sorted ranking DataFrame."""
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)

    # Sort by robustness score descending, then by sharpe
    sort_cols = []
    if "robustness_score" in df.columns:
        sort_cols.append("robustness_score")
    if "sharpe_ratio" in df.columns:
        sort_cols.append("sharpe_ratio")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    return df.reset_index(drop=True)