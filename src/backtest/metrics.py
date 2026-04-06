"""
metrics.py
──────────
Computes all performance metrics from a trade log and equity curve.

Input:
    trades    : pd.DataFrame with one row per closed trade
    equity    : pd.Series of portfolio equity values (one per bar)
    bars_per_year : number of bars in one year (for annualisation)

Output:
    dict with all metrics
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Return drawdown as a fraction (0 to -1) at each point."""
    peak = equity.cummax()
    return (equity - peak) / peak.replace(0, np.nan)


def compute_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    bars_per_year: float,
    initial_capital: float = 10_000.0,
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.

    Parameters
    ----------
    trades          : DataFrame with columns:
                      entry_price, exit_price, side (1=long/-1=short),
                      pnl, pnl_pct, bars_held
    equity          : equity curve (starting from initial_capital)
    bars_per_year   : approximate number of bars in one calendar year
    initial_capital : starting capital

    Returns
    -------
    dict of metric_name → value
    """
    m: Dict[str, float] = {}

    if trades.empty or equity.empty:
        return _empty_metrics()

    n_trades = len(trades)
    m["trade_count"] = n_trades

    # ── Returns ────────────────────────────────────────────────────────────────
    final_equity = equity.iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    m["total_return"] = total_return

    # CAGR
    n_years = len(equity) / bars_per_year
    if n_years > 0 and final_equity > 0:
        m["cagr"] = (final_equity / initial_capital) ** (1.0 / n_years) - 1
    else:
        m["cagr"] = np.nan

    # ── Drawdown ───────────────────────────────────────────────────────────────
    dd = _drawdown_series(equity)
    m["max_drawdown"] = dd.min()  # negative number

    # Calmar ratio
    if m["max_drawdown"] != 0 and not np.isnan(m.get("cagr", np.nan)):
        m["calmar"] = m["cagr"] / abs(m["max_drawdown"])
    else:
        m["calmar"] = np.nan

    # ── Sharpe / Sortino ───────────────────────────────────────────────────────
    bar_returns = equity.pct_change().dropna()
    if bar_returns.std() > 0:
        m["sharpe"] = float(
            (bar_returns.mean() / bar_returns.std()) * np.sqrt(bars_per_year)
        )
    else:
        m["sharpe"] = np.nan

    negative_returns = bar_returns[bar_returns < 0]
    if len(negative_returns) > 0 and negative_returns.std() > 0:
        m["sortino"] = float(
            (bar_returns.mean() / negative_returns.std()) * np.sqrt(bars_per_year)
        )
    else:
        m["sortino"] = np.nan

    # ── Trade-level metrics ────────────────────────────────────────────────────
    pnl = trades["pnl"].values
    wins  = pnl[pnl > 0]
    loses = pnl[pnl < 0]

    m["win_rate"]    = len(wins) / n_trades if n_trades > 0 else np.nan
    m["avg_win"]     = float(wins.mean())  if len(wins)  > 0 else 0.0
    m["avg_loss"]    = float(loses.mean()) if len(loses) > 0 else 0.0
    m["expectancy"]  = m["avg_win"] * m["win_rate"] + m["avg_loss"] * (1 - m["win_rate"])

    # Profit factor
    gross_profit = wins.sum()  if len(wins)  > 0 else 0.0
    gross_loss   = abs(loses.sum()) if len(loses) > 0 else 0.0
    m["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Average R multiple (using pnl_pct if available)
    if "r_multiple" in trades.columns:
        m["avg_r"] = float(trades["r_multiple"].mean())
    else:
        m["avg_r"] = np.nan

    # ── Streaks ────────────────────────────────────────────────────────────────
    win_flags = (pnl > 0).astype(int)
    m["max_win_streak"]  = _max_streak(win_flags, 1)
    m["max_loss_streak"] = _max_streak(win_flags, 0)

    # ── Exposure ───────────────────────────────────────────────────────────────
    if "bars_held" in trades.columns:
        total_bars_in_market = trades["bars_held"].sum()
        total_bars = len(equity)
        m["exposure"] = total_bars_in_market / total_bars if total_bars > 0 else np.nan
    else:
        m["exposure"] = np.nan

    # ── Average bars held ─────────────────────────────────────────────────────
    if "bars_held" in trades.columns:
        m["avg_bars_held"] = float(trades["bars_held"].mean())
    else:
        m["avg_bars_held"] = np.nan

    return m


def _max_streak(flags: np.ndarray, value: int) -> int:
    """Find the longest consecutive run of `value` in `flags`."""
    max_s, cur_s = 0, 0
    for f in flags:
        if f == value:
            cur_s += 1
            max_s = max(max_s, cur_s)
        else:
            cur_s = 0
    return max_s


def _empty_metrics() -> Dict[str, float]:
    return {
        "trade_count": 0, "total_return": 0.0, "cagr": np.nan,
        "max_drawdown": 0.0, "calmar": np.nan, "sharpe": np.nan,
        "sortino": np.nan, "win_rate": np.nan, "avg_win": 0.0,
        "avg_loss": 0.0, "expectancy": 0.0, "profit_factor": np.nan,
        "avg_r": np.nan, "max_win_streak": 0, "max_loss_streak": 0,
        "exposure": np.nan, "avg_bars_held": np.nan,
    }


def yearly_breakdown(
    trades: pd.DataFrame,
    equity: pd.Series,
    timestamps: pd.Series,
    bars_per_year: float,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Compute per-year performance metrics.

    Parameters
    ----------
    trades     : trade log
    equity     : equity curve (aligned with timestamps)
    timestamps : DatetimeIndex or Series of bar timestamps
    bars_per_year : bars in one year

    Returns
    -------
    DataFrame with one row per year
    """
    if trades.empty:
        return pd.DataFrame()

    rows = []
    years = sorted(timestamps.dt.year.unique())

    for year in years:
        mask_bar    = timestamps.dt.year == year
        eq_year     = equity[mask_bar]

        if "entry_time" in trades.columns:
            mask_trade  = pd.to_datetime(trades["entry_time"]).dt.year == year
        else:
            mask_trade  = pd.Series([True] * len(trades))

        tr_year = trades[mask_trade]

        if eq_year.empty:
            continue

        start_eq = eq_year.iloc[0]
        end_eq   = eq_year.iloc[-1]
        yr_return = (end_eq - start_eq) / start_eq

        dd_yr   = _drawdown_series(eq_year)
        bpr     = max(1, bars_per_year)

        row = {
            "year":         year,
            "return":       yr_return,
            "max_drawdown": dd_yr.min(),
            "trade_count":  len(tr_year),
        }

        bar_ret  = eq_year.pct_change().dropna()
        if bar_ret.std() > 0:
            row["sharpe"] = float((bar_ret.mean() / bar_ret.std()) * np.sqrt(bpr))
        else:
            row["sharpe"] = np.nan

        pnl_yr = tr_year["pnl"].values if len(tr_year) > 0 else np.array([])
        wins   = pnl_yr[pnl_yr > 0]
        loses  = pnl_yr[pnl_yr < 0]
        gp = wins.sum()   if len(wins) > 0 else 0.0
        gl = abs(loses.sum()) if len(loses) > 0 else 0.0
        row["profit_factor"] = gp / gl if gl > 0 else np.nan
        row["win_rate"]      = len(wins) / len(pnl_yr) if len(pnl_yr) > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("year")
