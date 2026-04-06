"""
reporter.py
───────────
Generates all output artefacts:
    - CSV ranking tables
    - HTML summary reports
    - Equity curve plots
    - Heatmap plots
    - Parameter sensitivity surfaces
    - Year-by-year breakdown tables
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.utils.config_loader import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BTC Research Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e0e0e0;
          margin: 0; padding: 20px; }}
  h1   {{ color: #f7c94b; border-bottom: 2px solid #f7c94b; padding-bottom: 8px; }}
  h2   {{ color: #7ec8e3; margin-top: 30px; }}
  h3   {{ color: #aaa; }}
  table{{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }}
  th   {{ background: #1e2235; color: #f7c94b; padding: 8px 12px; text-align: left; }}
  td   {{ padding: 6px 12px; border-bottom: 1px solid #2a2d3e; }}
  tr:hover td {{ background: #1a1d2e; }}
  .good  {{ color: #4ade80; }}
  .bad   {{ color: #f87171; }}
  .warn  {{ color: #fbbf24; }}
  .meta  {{ color: #888; font-size: 12px; }}
  img    {{ max-width: 100%; border-radius: 8px; margin: 10px 0; }}
  .section {{ background: #161925; border-radius: 8px; padding: 16px; margin: 16px 0; }}
</style>
</head>
<body>
<h1>🔬 BTC Research Report</h1>
<p class="meta">Generated: {timestamp}</p>

<div class="section">
<h2>📊 Top Strategies by Robustness</h2>
{top_table}
</div>

<div class="section">
<h2>🗓️ Best Strategy by Timeframe</h2>
{tf_table}
</div>

<div class="section">
<h2>🔁 Walk-Forward Summary</h2>
{wf_table}
</div>

<div class="section">
<h2>📅 Year-by-Year Breakdown (Best Strategy)</h2>
{yearly_table}
</div>

<div class="section">
<h2>📈 Equity Curves</h2>
{equity_imgs}
</div>

<div class="section">
<h2>🗺️ Performance Heatmap</h2>
{heatmap_imgs}
</div>

<div class="section">
<h2>❌ Rejected Strategies</h2>
{rejected_table}
</div>

</body>
</html>"""


def _fmt(val: Any, pct: bool = False, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if pct:
        return f"{val:.{decimals}%}"
    return f"{val:.{decimals}f}"


def _color_td(val: Any, good_above: float = 0, pct: bool = False) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "<td>—</td>"
    cls = "good" if val > good_above else "bad"
    return f'<td class="{cls}">{_fmt(val, pct=pct)}</td>'


def _df_to_html_table(
    df: pd.DataFrame,
    color_cols: Optional[Dict[str, float]] = None,
    pct_cols: Optional[List[str]] = None,
    max_rows: int = 50,
) -> str:
    """Convert a DataFrame to a styled HTML table."""
    if df is None or df.empty:
        return "<p class='meta'>No data.</p>"

    color_cols = color_cols or {}
    pct_cols   = pct_cols   or []
    df         = df.head(max_rows)

    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows   = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col in color_cols:
                cells.append(_color_td(val, good_above=color_cols[col], pct=col in pct_cols))
            elif col in pct_cols and isinstance(val, (int, float)):
                cells.append(f"<td>{_fmt(val, pct=True)}</td>")
            else:
                if isinstance(val, float) and not np.isnan(val):
                    cells.append(f"<td>{val:.4f}</td>")
                else:
                    cells.append(f"<td>{val}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


class Reporter:
    """
    Generates all output artefacts for the research run.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg        = cfg
        self.out        = cfg.output_dir
        self.reports    = self.out / "reports"
        self.rankings   = self.out / "rankings"
        self.plots_dir  = self.out / "plots"
        for d in [self.reports, self.rankings, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── CSV outputs ────────────────────────────────────────────────────────────

    def save_rankings(
        self,
        all_results: pd.DataFrame,
        filename: str = "all_results.csv",
    ) -> Path:
        path = self.rankings / filename
        all_results.to_csv(path, index=False)
        logger.info(f"Rankings saved → {path}")
        return path

    def save_wf_summary(self, wf_rows: List[dict], filename: str = "wf_summary.csv") -> Path:
        df   = pd.DataFrame(wf_rows)
        path = self.rankings / filename
        df.to_csv(path, index=False)
        logger.info(f"WF summary saved → {path}")
        return path

    def save_yearly_breakdown(self, yearly: pd.DataFrame, name: str) -> Path:
        path = self.rankings / f"yearly_{name}.csv"
        yearly.to_csv(path)
        logger.info(f"Yearly breakdown saved → {path}")
        return path

    # ── Equity curve plots ─────────────────────────────────────────────────────

    def plot_equity_curve(
        self,
        equity:    pd.Series,
        drawdown:  pd.Series,
        name:      str,
        timestamps: Optional[pd.Series] = None,
    ) -> Path:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]},
            facecolor="#0f1117"
        )
        for ax in [ax1, ax2]:
            ax.set_facecolor("#161925")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values():
                spine.set_color("#2a2d3e")

        x = timestamps.values if timestamps is not None else np.arange(len(equity))

        ax1.plot(x, equity.values, color="#f7c94b", linewidth=1.2, label="Equity")
        ax1.set_ylabel("Portfolio Value ($)", color="#aaa")
        ax1.set_title(f"Equity Curve — {name}", color="#e0e0e0", fontsize=13)
        ax1.grid(True, color="#2a2d3e", alpha=0.5)
        ax1.legend(facecolor="#1e2235", labelcolor="#e0e0e0")

        ax2.fill_between(x, drawdown.values * 100, 0, color="#f87171", alpha=0.6)
        ax2.set_ylabel("Drawdown (%)", color="#aaa")
        ax2.set_xlabel("Date", color="#aaa")
        ax2.grid(True, color="#2a2d3e", alpha=0.5)

        plt.tight_layout()
        path = self.plots_dir / f"equity_{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        logger.debug(f"Equity plot saved → {path}")
        return path

    # ── Heatmap ────────────────────────────────────────────────────────────────

    def plot_heatmap(
        self,
        df:       pd.DataFrame,
        row_col:  str,
        col_col:  str,
        val_col:  str,
        title:    str,
        filename: str,
    ) -> Path:
        """
        Plot a 2-D heatmap of val_col as a function of row_col × col_col.
        """
        try:
            pivot = df.pivot_table(
                index=row_col, columns=col_col, values=val_col, aggfunc="mean"
            )
        except Exception as exc:
            logger.warning(f"Cannot create heatmap ({exc})")
            return Path()

        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0f1117")
        ax.set_facecolor("#161925")

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", color="#aaa", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, color="#aaa", fontsize=9)
        ax.set_title(title, color="#e0e0e0", fontsize=13)
        ax.set_xlabel(col_col, color="#aaa")
        ax.set_ylabel(row_col, color="#aaa")

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="#aaa")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaa")

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="white")

        plt.tight_layout()
        path = self.plots_dir / filename
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        logger.debug(f"Heatmap saved → {path}")
        return path

    # ── Timeframe × strategy table ─────────────────────────────────────────────

    def plot_tf_strategy_heatmap(self, df: pd.DataFrame, metric: str = "oos_sharpe") -> Path:
        if "strategy" not in df.columns or "timeframe" not in df.columns:
            return Path()
        best = (
            df.groupby(["strategy", "timeframe"])[metric]
            .max()
            .reset_index()
        )
        return self.plot_heatmap(
            df=best,
            row_col="strategy",
            col_col="timeframe",
            val_col=metric,
            title=f"Strategy × Timeframe — {metric}",
            filename=f"heatmap_tf_strategy_{metric}.png",
        )

    # ── HTML report ────────────────────────────────────────────────────────────

    def generate_html_report(
        self,
        all_results:  pd.DataFrame,
        wf_rows:      List[dict],
        best_yearly:  pd.DataFrame,
        rejected:     pd.DataFrame,
        equity_plots: List[Path],
        heatmap_plots: List[Path],
        top_n: int = 20,
    ) -> Path:
        """Generate the main HTML summary report."""

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # ── Top strategies table ───────────────────────────────────────────────
        display_cols = [
            "strategy", "timeframe", "oos_return", "oos_sharpe",
            "oos_pf", "oos_drawdown", "oos_trades", "robustness",
        ]
        top_df = all_results[[c for c in display_cols if c in all_results.columns]].head(top_n)
        top_table = _df_to_html_table(
            top_df,
            color_cols={"oos_return": 0, "oos_sharpe": 0, "oos_pf": 1.0, "robustness": 0},
            pct_cols=["oos_return", "oos_drawdown"],
        )

        # ── Best per timeframe ─────────────────────────────────────────────────
        if not all_results.empty and "timeframe" in all_results.columns:
            tf_best = (
                all_results
                .sort_values("robustness", ascending=False)
                .groupby("timeframe")
                .first()
                .reset_index()
            )
            tf_cols = [c for c in display_cols if c in tf_best.columns]
            tf_table = _df_to_html_table(
                tf_best[tf_cols],
                color_cols={"oos_return": 0, "oos_sharpe": 0, "oos_pf": 1.0},
                pct_cols=["oos_return", "oos_drawdown"],
            )
        else:
            tf_table = "<p class='meta'>No data.</p>"

        # ── WF table ───────────────────────────────────────────────────────────
        wf_df    = pd.DataFrame(wf_rows) if wf_rows else pd.DataFrame()
        wf_table = _df_to_html_table(
            wf_df,
            color_cols={"oos_sharpe": 0, "oos_pf": 1.0},
        )

        # ── Yearly table ───────────────────────────────────────────────────────
        yearly_table = _df_to_html_table(
            best_yearly,
            color_cols={"return": 0, "sharpe": 0, "profit_factor": 1.0},
            pct_cols=["return", "max_drawdown"],
        )

        # ── Rejected table ─────────────────────────────────────────────────────
        rej_table = _df_to_html_table(rejected.head(30) if not rejected.empty else pd.DataFrame())

        # ── Equity images ──────────────────────────────────────────────────────
        def _img_tag(p: Path) -> str:
            rel = p.relative_to(self.out) if p.is_absolute() else p
            return f'<img src="../{rel}" alt="{p.stem}" loading="lazy">'

        equity_imgs  = "\n".join(_img_tag(p) for p in equity_plots  if p.exists())
        heatmap_imgs = "\n".join(_img_tag(p) for p in heatmap_plots if p.exists())

        html = HTML_TEMPLATE.format(
            timestamp    = timestamp,
            top_table    = top_table,
            tf_table     = tf_table,
            wf_table     = wf_table,
            yearly_table = yearly_table,
            equity_imgs  = equity_imgs or "<p class='meta'>No equity plots generated.</p>",
            heatmap_imgs = heatmap_imgs or "<p class='meta'>No heatmaps generated.</p>",
            rejected_table = rej_table,
        )

        path = self.reports / "research_report.html"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info(f"HTML report saved → {path}")
        return path

    # ── Stress test summary ────────────────────────────────────────────────────

    def save_stress_summary(self, rows: List[dict]) -> Path:
        df   = pd.DataFrame(rows)
        path = self.rankings / "stress_test.csv"
        df.to_csv(path, index=False)
        return path
