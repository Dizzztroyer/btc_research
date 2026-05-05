#!/usr/bin/env python3
"""
run_portfolio_study.py
───────────────────────
Portfolio study: combine swing_breakout on 8h + 12h + 6h.

Each strategy runs independently with its confirmed configuration:
  8h:  ML_ATR + TP_L1.2_S0.8 + partial 50% at 1R
  12h: ML_SIZE + PART_L50_S70
  6h:  ML_SIZE (no ATR, no exit mods yet)

Portfolio = all three running simultaneously, capital shared.
Risk scaling: each strategy uses risk_per_trade independently
(they run on different candles, not shared position sizing).

Tests:
  STRAT_8h   — 8h alone
  STRAT_12h  — 12h alone
  STRAT_6h   — 6h alone
  PORT_EQ    — equal weight (risk ×1.0 each)
  PORT_W8    — 8h heavy (8h×1.5, 12h×1.0, 6h×0.7)
  PORT_W12   — 12h heavy (8h×0.7, 12h×1.5, 6h×1.0)
  PORT_NOATR — 8h without ATR (baseline comparison)

Key questions:
  1. Does combining increase Sharpe?
  2. Does MDD decrease (diversification)?
  3. Which weighting is best?
  4. What is the inter-strategy correlation?

Usage
─────
    python run_portfolio_study.py
    python run_portfolio_study.py --risk 0.01  # 1% per strategy
    python run_portfolio_study.py --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.features.feature_engine import FeatureEngine
from src.backtest.engine import SimConfig
from src.backtest.metrics import compute_metrics, yearly_breakdown
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig
from src.strategies.structure import SwingBreakoutStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/portfolio_study.log"))

BASELINE_WF = {
    "8h":  {"avg_sharpe": 1.800, "avg_pf": 2.361, "avg_mdd": -0.0184},
    "12h": {"avg_sharpe": 1.530, "avg_pf": 2.241, "avg_mdd": -0.0111},
    "6h":  {"avg_sharpe": 1.635, "avg_pf": 1.903, "avg_mdd": -0.0153},
}


# ── ATR regime ────────────────────────────────────────────────────────────────

def _atr_mults(df: pd.DataFrame, low_m=1.3, high_m=0.7,
               lookback=200, lp=33, hp=66) -> pd.Series:
    col = next((c for c in ["atr_14_pct","atr_14"] if c in df.columns), None)
    if col:
        atr = df[col].shift(1)
    else:
        tr  = pd.concat([df["high"]-df["low"],
                         (df["high"]-df["close"].shift(1)).abs(),
                         (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
        atr = (tr.ewm(span=14,adjust=False).mean()/df["close"]).shift(1)
    rl = atr.rolling(lookback, min_periods=50).quantile(lp/100)
    rh = atr.rolling(lookback, min_periods=50).quantile(hp/100)
    m  = pd.Series(1.0, index=df.index)
    m[atr < rl] = low_m
    m[atr > rh] = high_m
    return m


# ── Per-strategy simulation ───────────────────────────────────────────────────

def _simulate_strategy(
    df:           pd.DataFrame,
    sim_cfg:      SimConfig,
    size_mults:   pd.Series,
    risk_scale:   float = 1.0,
    long_tp_mult: float = 1.0,
    short_tp_mult:float = 1.0,
    long_partial: Optional[float] = None,
    short_partial:Optional[float] = None,
    partial_r:    float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Simulate a single strategy. Returns (trades_df, pnl_series aligned to df index).
    pnl_series: daily/bar P&L in dollar terms, indexed to df.
    """
    df   = df.sort_values("timestamp").reset_index(drop=True)
    mults= size_mults.reset_index(drop=True).to_numpy(np.float64)
    n    = len(df)

    opens   = df["open"].to_numpy(np.float64)
    highs   = df["high"].to_numpy(np.float64)
    lows    = df["low"].to_numpy(np.float64)
    closes  = df["close"].to_numpy(np.float64)
    signals = df["signal"].fillna(0).to_numpy(np.int64)
    sl_arr  = df["sl_price"].to_numpy(np.float64) if "sl_price" in df.columns else np.full(n, np.nan)
    tp_arr  = df["tp_price"].to_numpy(np.float64) if "tp_price" in df.columns else np.full(n, np.nan)
    atr_col = next((c for c in ["atr_14","atr_7","atr_21"] if c in df.columns), None)
    atrs    = df[atr_col].to_numpy(np.float64) if atr_col else (highs-lows)

    dir_int = {"both":0,"long":1,"short":-1}.get(sim_cfg.direction, 0)
    cap0    = sim_cfg.initial_capital
    pnl_bar = np.zeros(n)  # P&L realised on each bar
    rows:   List[dict] = []

    in_pos = False
    side   = pos_size = ep = sl = tp = 0.0
    ebar   = 0; be_hit = False; pd_done = False; risk_amt = 0.0

    for i in range(1, n):
        o,h,l,c = opens[i],highs[i],lows[i],closes[i]
        atr = atrs[i]; atr = max(atr, c*0.005) if np.isnan(atr) or atr<=0 else atr

        if in_pos:
            ex = -1.0; er = -1

            # SL
            if side==1  and l <= sl: ex,er = sl,0
            elif side==-1 and h >= sl: ex,er = sl,0

            # TP
            if ex<0:
                if side==1  and h >= tp: ex,er = tp,1
                elif side==-1 and l <= tp: ex,er = tp,1

            # Signal reversal
            if ex<0:
                sig = int(signals[i-1])
                if (side==1 and sig==-1) or (side==-1 and sig==1): ex,er = o,2

            # Partial close
            pp = long_partial if side==1 else short_partial
            if ex<0 and pp is not None and not pd_done:
                move = (o-ep)*side
                if risk_amt>0 and pos_size>0 and move >= partial_r*risk_amt/pos_size:
                    pc   = pos_size * pp
                    fill = o*(1+(-side)*sim_cfg.slippage)
                    net  = side*(fill-ep)/ep*pc - pc*sim_cfg.fees
                    pnl_bar[i] += net
                    pos_size -= pc; pd_done = True
                    rows.append(_trow(df,ebar,i,side,ep,fill,pc,net,risk_amt,i-ebar,"partial"))

            # Full exit
            if ex >= 0:
                fill = ex*(1+(-side)*sim_cfg.slippage)
                net  = side*(fill-ep)/ep*pos_size - pos_size*sim_cfg.fees
                pnl_bar[i] += net
                rows.append(_trow(df,ebar,i,side,ep,fill,pos_size,net,risk_amt,i-ebar,
                                   {0:"sl",1:"tp",2:"sig",3:"end"}.get(er,"?")))
                in_pos=False; side=0

        # Entry
        if not in_pos:
            sig = int(signals[i-1])
            if sig==0: continue
            if dir_int==1  and sig!= 1: continue
            if dir_int==-1 and sig!=-1: continue

            sd = sig
            raw_sl = float(sl_arr[i-1])
            sl_p = raw_sl if not np.isnan(raw_sl) and raw_sl>0 else o*(1-sd*sim_cfg.default_sl_pct)
            raw_tp = float(tp_arr[i-1])
            tp_base= raw_tp if not np.isnan(raw_tp) and raw_tp>0 else o*(1+sd*sim_cfg.default_tp_pct)

            tp_m = long_tp_mult if sd==1 else short_tp_mult
            if tp_m != 1.0:
                tp_p = o + sd * abs(tp_base-o) * tp_m
            else:
                tp_p = tp_base

            if sd==1  and sl_p>=o: continue
            if sd==-1 and sl_p<=o: continue

            fill    = o*(1+sd*sim_cfg.slippage)
            rsk_pct = abs(fill-sl_p)/fill
            if rsk_pct<=0: continue

            mult     = max(0.5, min(2.0, float(mults[i-1])))
            eff_risk = sim_cfg.risk_per_trade * mult * risk_scale
            # Size based on initial capital (strategies run in parallel on same capital)
            sz       = min((cap0*eff_risk)/rsk_pct, cap0*sim_cfg.leverage)
            pnl_bar[i] -= sz * sim_cfg.fees

            in_pos=True; side=sd; ep=fill; ebar=i; sl=sl_p; tp=tp_p
            pos_size=sz; risk_amt=abs(fill-sl_p)/fill*sz; be_hit=False; pd_done=False

    if in_pos:
        fill = closes[n-1]*(1-side*sim_cfg.slippage)
        net  = side*(fill-ep)/ep*pos_size - pos_size*sim_cfg.fees
        pnl_bar[n-1] += net
        rows.append(_trow(df,ebar,n-1,side,ep,fill,pos_size,net,risk_amt,n-1-ebar,"end"))

    trades = pd.DataFrame(rows) if rows else pd.DataFrame()
    pnl_s  = pd.Series(pnl_bar, index=df.index, name="pnl")
    return trades, pnl_s


def _trow(df,eb,xb,side,ep,fp,sz,net,ra,bh,reason):
    ra2 = ra if ra>0 else sz*0.02
    return {"entry_time":df["timestamp"].iloc[eb],"exit_time":df["timestamp"].iloc[xb],
            "side":side,"entry_price":ep,"exit_price":fp,"size":sz,"pnl":net,
            "pnl_pct":net/sz if sz>0 else 0,"r_multiple":net/ra2 if ra2>0 else 0,
            "bars_held":bh,"exit_reason":reason}


# ── Portfolio combination ─────────────────────────────────────────────────────

def combine_portfolio(
    pnl_dict:  Dict[str, pd.Series],
    ts_common: pd.DatetimeIndex,
    initial_capital: float = 10000.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Combine P&L series from multiple strategies into a portfolio equity curve.
    Strategies run in parallel — P&L is additive.
    Returns: equity, drawdown, daily_pnl (all aligned to ts_common).
    """
    # Align all series to common daily timestamp
    combined_pnl = pd.Series(0.0, index=ts_common)
    for name, pnl in pnl_dict.items():
        # Resample P&L to daily and align
        pnl_reindexed = pnl.copy()
        pnl_reindexed.index = ts_common[:len(pnl)]  # align by position
        if len(pnl_reindexed) < len(combined_pnl):
            pnl_reindexed = pnl_reindexed.reindex(ts_common, fill_value=0.0)
        combined_pnl = combined_pnl.add(pnl_reindexed[:len(combined_pnl)], fill_value=0.0)

    equity   = initial_capital + combined_pnl.cumsum()
    drawdown = (equity - equity.cummax()) / equity.cummax().replace(0, np.nan)
    return equity, drawdown, combined_pnl


def portfolio_metrics(
    pnl_dict:  Dict[str, pd.Series],
    ts_common: pd.DatetimeIndex,
    initial_capital: float,
    timestamps: pd.Series,
) -> dict:
    """Compute portfolio metrics from combined P&L."""
    equity, drawdown, daily_pnl = combine_portfolio(pnl_dict, ts_common, initial_capital)

    total_ret = (equity.iloc[-1] - initial_capital) / initial_capital
    mdd       = drawdown.min()

    # Sharpe from daily P&L
    tot_secs = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    n_bars   = len(timestamps) - 1
    bpy      = 365.25 * 86400 / (tot_secs / max(n_bars, 1)) if tot_secs > 0 else 365.0

    pnl_arr  = daily_pnl.values
    mean_pnl = pnl_arr.mean()
    std_pnl  = pnl_arr.std()
    sharpe   = (mean_pnl / std_pnl) * np.sqrt(bpy) if std_pnl > 0 else 0.0

    all_trades_pnl = [p for p in pnl_arr if p != 0]
    gp = sum(p for p in all_trades_pnl if p > 0)
    gl = abs(sum(p for p in all_trades_pnl if p < 0))
    pf = gp / gl if gl > 0 else np.nan

    return {
        "total_return": total_ret,
        "sharpe":       sharpe,
        "pf":           pf,
        "mdd":          mdd,
        "equity":       equity,
        "drawdown":     drawdown,
        "daily_pnl":    daily_pnl,
    }


# ── Correlation analysis ──────────────────────────────────────────────────────

def compute_correlation(pnl_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute pairwise correlation between strategy P&L series."""
    # Align all to shortest common length
    min_len = min(len(s) for s in pnl_dict.values())
    aligned = {k: v.values[:min_len] for k,v in pnl_dict.items()}
    df = pd.DataFrame(aligned)
    return df.corr()


# ── Load params ───────────────────────────────────────────────────────────────

def _load_params(csv, strategy, tf):
    if not Path(csv).exists(): return None
    df   = pd.read_csv(csv)
    mask = (df["strategy"]==strategy)&(df["timeframe"]==tf)
    sub  = df[mask]
    if sub.empty: return None
    best   = sub.sort_values("robustness",ascending=False).iloc[0]
    p_cols = [c for c in best.index if c.startswith("p_") and not pd.isna(best[c])]
    out    = {}
    for c in p_cols:
        v=best[c]; k=c[2:]
        out[k] = int(v) if isinstance(v,float) and v==int(v) else v
    return out


def _m(name, metrics, trades_count):
    return {
        "version":     name,
        "trades":      trades_count,
        "pf":          metrics.get("pf",        np.nan),
        "sharpe":      metrics.get("sharpe",     np.nan),
        "total_return":metrics.get("total_return",np.nan),
        "mdd":         metrics.get("mdd",        np.nan),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_equity(equity_dict: Dict[str, pd.Series],
                 dd_dict: Dict[str, pd.Series],
                 timestamps: pd.Series, out_dir: Path) -> None:
    colors = {
        "STRAT_8h":   "#378ADD",
        "STRAT_12h":  "#7F77DD",
        "STRAT_6h":   "#BA7517",
        "PORT_EQ":    "#1D9E75",
        "PORT_W8":    "#f7c94b",
        "PORT_W12":   "#D85A30",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="#0f1117",
                              gridspec_kw={"height_ratios":[3,1.5]})
    ts = pd.to_datetime(timestamps).reset_index(drop=True)

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa", labelsize=8)
        for s in ax.spines.values(): s.set_color("#2a2d3e")

    for name, eq in equity_dict.items():
        col = colors.get(name, "white")
        lw  = 2.5 if "PORT" in name else 1.2
        al  = 0.95 if "PORT" in name else 0.55
        axes[0].plot(ts[:len(eq)], eq.values, label=name, color=col, linewidth=lw, alpha=al)

    axes[0].set_title("Portfolio Study — Equity Curves", color="#e0e0e0", fontsize=12)
    axes[0].set_ylabel("Value ($)", color="#aaa")
    axes[0].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8, ncol=3)
    axes[0].grid(True, color="#2a2d3e", alpha=0.4)

    for name, dd in dd_dict.items():
        col = colors.get(name, "white")
        lw  = 2.0 if "PORT" in name else 0.9
        al  = 0.9 if "PORT" in name else 0.5
        axes[1].plot(ts[:len(dd)], dd.values*100, color=col, linewidth=lw, alpha=al, label=name)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_title("Drawdown (%)", color="#e0e0e0", fontsize=10)
    axes[1].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=7, ncol=3)
    axes[1].grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    path = out_dir / "portfolio_equity.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Equity plot → {path}")


def _plot_correlation(corr_dict: Dict[str, pd.Series], out_dir: Path) -> None:
    corr = compute_correlation(corr_dict)
    n = len(corr)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f1117")
    ax.set_facecolor("#161925")
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, color="#aaa", fontsize=9, rotation=30)
    ax.set_yticklabels(corr.index,   color="#aaa", fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.values[i,j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if abs(v)<0.6 else "white", fontsize=10)
    ax.set_title("P&L Correlation Matrix", color="#e0e0e0")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = out_dir / "portfolio_correlation.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Correlation plot → {path}")


def _plot_yearly(yearly_dict: Dict[str, Dict[int, float]], out_dir: Path) -> None:
    years = sorted(set(y for d in yearly_dict.values() for y in d.keys()))
    if not years: return
    x = np.arange(len(years))
    w = 0.8 / max(len(yearly_dict), 1)
    colors = {"STRAT_8h":"#378ADD","STRAT_12h":"#7F77DD","STRAT_6h":"#BA7517",
              "PORT_EQ":"#1D9E75","PORT_W8":"#f7c94b"}

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f1117")
    ax.set_facecolor("#161925")
    ax.tick_params(colors="#aaa", labelsize=9)
    for s in ax.spines.values(): s.set_color("#2a2d3e")

    for idx, (name, yr_data) in enumerate(yearly_dict.items()):
        vals = [yr_data.get(y, 0) * 100 for y in years]
        offset = (idx - len(yearly_dict)/2 + 0.5) * w
        ax.bar(x + offset, vals, w*0.9, label=name,
               color=colors.get(name, "white"), alpha=0.85)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(years, color="#aaa")
    ax.set_ylabel("Return %", color="#aaa")
    ax.set_title("Yearly Returns by Strategy", color="#e0e0e0")
    ax.legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
    ax.grid(True, color="#2a2d3e", alpha=0.3, axis="y")
    plt.tight_layout()
    path = out_dir / "portfolio_yearly.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Yearly plot → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Portfolio study: 8h + 12h + 6h")
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--results-csv", default="outputs/rankings/all_results.csv")
    p.add_argument("--risk",        type=float, default=0.01, help="Risk per trade per strategy")
    p.add_argument("--n-estimators",type=int,   default=400)
    p.add_argument("--no-plots",    action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(Path(args.config))
    fe   = FeatureEngine(cfg)

    out_plots = cfg.output_dir/"plots";    out_plots.mkdir(exist_ok=True)
    out_ranks = cfg.output_dir/"rankings"; out_ranks.mkdir(exist_ok=True)
    results_csv = Path(args.results_csv)

    sizer_cfg = SizerConfig(
        n_estimators=args.n_estimators,
        min_train_trades=150, min_val_trades=30, min_mult_std=0.05,
    )

    logger.info("=== Portfolio Study ===")
    logger.info(f"Risk per strategy: {args.risk:.1%}")
    logger.info("Strategies: swing/8h (ML_ATR+TP_asym+P50), swing/12h (ML+PART_L50_S70), swing/6h (ML)")

    strategy   = SwingBreakoutStrategy()
    initial_cap= cfg.validation.get("initial_capital", 10000.0) \
                  if hasattr(cfg.validation,"get") else 10000.0
    try:
        initial_cap = cfg.initial_capital
    except AttributeError:
        initial_cap = 10000.0

    # ── Load data and generate signals for all TFs ─────────────────────────────
    strat_data: Dict[str, dict] = {}  # tf → {df_is, df_oos, df_oos_sig, params, size_mults}

    TF_CONFIG = {
        "8h": {
            "long_tp_mult": 1.2, "short_tp_mult": 0.8,
            "long_partial": 0.5, "short_partial": 0.5,
            "use_atr": True,
        },
        "12h": {
            "long_tp_mult": 1.0, "short_tp_mult": 1.0,
            "long_partial": 0.5, "short_partial": 0.7,
            "use_atr": False,
        },
        "6h": {
            "long_tp_mult": 1.0, "short_tp_mult": 1.0,
            "long_partial": None, "short_partial": None,
            "use_atr": False,
        },
    }

    for tf in ["8h","12h","6h"]:
        logger.info(f"\n  Loading {tf}...")
        try:
            df = fe.load(tf)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except Exception as exc:
            logger.error(f"  {tf} load failed: {exc}"); continue

        params = _load_params(results_csv, "swing_breakout", tf)
        if params is None:
            logger.warning(f"  No params for {tf}"); continue

        n     = len(df)
        is_n  = int(n * cfg.validation.is_ratio)
        df_is = df.iloc[:is_n]
        df_oos= df.iloc[is_n:]

        df_oos_sig = strategy.generate_signals(df_oos.copy(), params)
        df_oos_sig = df_oos_sig.reset_index(drop=True)

        # ML sizing
        sim_base = SimConfig(fees=cfg.fees, slippage=cfg.slippage,
                             leverage=cfg.leverage, risk_per_trade=args.risk,
                             direction="both")
        size_mults = pd.Series(1.0, index=range(len(df_oos)))
        sizer = MLSizerV2(cfg, sizer_cfg)
        ok    = sizer.fit(strategy, params, df_is, "both", label=tf)
        if ok:
            ml_m = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
            if TF_CONFIG[tf]["use_atr"]:
                atr_m = _atr_mults(df_oos.reset_index(drop=True)).reset_index(drop=True)
                size_mults = (ml_m * atr_m).clip(0.4, 2.0)
            else:
                size_mults = ml_m
            logger.info(f"  {tf} ML enabled: AUC={sizer.auc:.3f}")
        else:
            logger.warning(f"  {tf} ML disabled")

        strat_data[tf] = {
            "df_oos":      df_oos,
            "df_oos_sig":  df_oos_sig,
            "params":      params,
            "size_mults":  size_mults,
            "sim_cfg":     sim_base,
            "tf_cfg":      TF_CONFIG[tf],
        }
        logger.info(f"  {tf}: {len(df_oos)} OOS bars | {(df_oos_sig['signal']!=0).sum()} signals")

    if len(strat_data) < 2:
        logger.error("Need at least 2 strategies"); return

    # ── Run individual strategies ──────────────────────────────────────────────
    logger.info("\n  Running individual strategies...")
    indiv_trades: Dict[str, pd.DataFrame] = {}
    indiv_pnl:    Dict[str, pd.Series]    = {}
    indiv_equity: Dict[str, pd.Series]    = {}
    indiv_dd:     Dict[str, pd.Series]    = {}
    indiv_metrics:Dict[str, dict]         = {}

    # Find shortest common OOS window (use 8h timestamps as reference)
    ref_tf = min(strat_data.keys(), key=lambda k: len(strat_data[k]["df_oos_sig"]))
    ref_len= len(strat_data[ref_tf]["df_oos_sig"])
    ref_ts = strat_data[ref_tf]["df_oos"]["timestamp"].reset_index(drop=True)

    for tf, sd in strat_data.items():
        cfg_tf = sd["tf_cfg"]
        trades, pnl = _simulate_strategy(
            df           = sd["df_oos_sig"],
            sim_cfg      = sd["sim_cfg"],
            size_mults   = sd["size_mults"],
            risk_scale   = 1.0,
            long_tp_mult = cfg_tf["long_tp_mult"],
            short_tp_mult= cfg_tf["short_tp_mult"],
            long_partial = cfg_tf["long_partial"],
            short_partial= cfg_tf["short_partial"],
        )
        indiv_trades[tf] = trades
        indiv_pnl[tf]    = pnl

        # Equity and metrics for this strategy
        eq = initial_cap + pnl.cumsum()
        dd = (eq - eq.cummax()) / eq.cummax().replace(0, np.nan)
        indiv_equity[tf] = eq
        indiv_dd[tf]     = dd

        tot  = (sd["df_oos"]["timestamp"].iloc[-1] - sd["df_oos"]["timestamp"].iloc[0]).total_seconds()
        bpy  = 365.25*86400 / (tot/max(len(sd["df_oos"])-1,1))
        met  = compute_metrics(trades, eq, bpy, initial_cap)
        indiv_metrics[tf] = met

        n_t  = int(met.get("trade_count",0))
        logger.info(f"  {tf}: trades={n_t}  PF={met.get('profit_factor',np.nan):.3f}  "
                    f"Sh={met.get('sharpe',np.nan):.3f}  MDD={met.get('max_drawdown',np.nan):.2%}")

    # ── Portfolio combinations ─────────────────────────────────────────────────
    logger.info("\n  Computing portfolio combinations...")

    def run_portfolio(name: str, scales: Dict[str, float]) -> dict:
        """Combine strategies with given risk scaling."""
        port_pnl: Dict[str, pd.Series] = {}
        total_trades = 0
        for tf, scale in scales.items():
            if tf not in strat_data: continue
            sd  = strat_data[tf]
            cfg_tf = sd["tf_cfg"]
            t, pnl = _simulate_strategy(
                df           = sd["df_oos_sig"],
                sim_cfg      = sd["sim_cfg"],
                size_mults   = sd["size_mults"],
                risk_scale   = scale,
                long_tp_mult = cfg_tf["long_tp_mult"],
                short_tp_mult= cfg_tf["short_tp_mult"],
                long_partial = cfg_tf["long_partial"],
                short_partial= cfg_tf["short_partial"],
            )
            port_pnl[tf]   = pnl
            total_trades   += int(len(t))

        # Align to reference timestamps and combine
        min_len   = min(len(s) for s in port_pnl.values())
        combined  = sum(s.values[:min_len] for s in port_pnl.values())
        eq        = pd.Series(initial_cap + np.cumsum(combined))
        dd        = pd.Series((eq - eq.cummax()) / eq.cummax().replace(0, np.nan).values)

        tot_ret   = (eq.iloc[-1] - initial_cap) / initial_cap
        pnl_arr   = combined
        mean_p    = pnl_arr.mean()
        std_p     = pnl_arr.std()
        # Use 8h bpy as reference
        bpy = 365.25 * 3  # approximate for 8h bars
        sh  = (mean_p/std_p)*np.sqrt(bpy) if std_p>0 else 0
        gp  = pnl_arr[pnl_arr>0].sum()
        gl  = abs(pnl_arr[pnl_arr<0].sum())
        pf  = gp/gl if gl>0 else np.nan
        mdd = dd.min()

        return {
            "equity": eq, "drawdown": dd, "pnl": pd.Series(combined),
            "sharpe": sh, "pf": pf, "mdd": mdd, "total_return": tot_ret,
            "total_trades": total_trades, "scales": scales,
            "port_pnl": port_pnl,
        }

    portfolios = {
        "PORT_EQ":  run_portfolio("PORT_EQ",  {"8h":1.0,"12h":1.0,"6h":1.0}),
        "PORT_W8":  run_portfolio("PORT_W8",  {"8h":1.5,"12h":1.0,"6h":0.7}),
        "PORT_W12": run_portfolio("PORT_W12", {"8h":0.7,"12h":1.5,"6h":1.0}),
        "PORT_W8_12":run_portfolio("PORT_W8_12",{"8h":1.2,"12h":1.2,"6h":0.6}),
    }

    # ── Correlation ────────────────────────────────────────────────────────────
    min_corr_len = min(len(s) for s in indiv_pnl.values())
    corr_pnl = {f"swing/{tf}": indiv_pnl[tf].values[:min_corr_len] for tf in indiv_pnl}
    corr_df  = pd.DataFrame(corr_pnl).corr()
    logger.info("\n  P&L Correlation Matrix:")
    for row in corr_df.to_string().split("\n"):
        logger.info(f"    {row}")

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("PORTFOLIO STUDY — RESULTS")
    print(f"{'='*72}")

    print(f"\n{'─'*72}")
    print("INDIVIDUAL STRATEGIES:")
    print(f"  {'TF':<8} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9} {'Return':>9}")
    print(f"  {'─'*58}")
    for tf in ["8h","12h","6h"]:
        if tf not in indiv_metrics: continue
        m  = indiv_metrics[tf]
        t  = int(m.get("trade_count",0))
        bm = BASELINE_WF.get(tf,{})
        print(f"  {tf:<8} {t:>7} {m.get('profit_factor',np.nan):>8.3f} "
              f"{m.get('sharpe',np.nan):>8.3f} "
              f"{m.get('max_drawdown',np.nan):>8.2%} "
              f"{m.get('total_return',np.nan):>8.1%}")

    print(f"\n{'─'*72}")
    print("PORTFOLIO COMBINATIONS:")
    print(f"  {'Name':<14} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9} {'Return':>9} {'Weights'}")
    print(f"  {'─'*70}")

    all_sharpes = {tf: indiv_metrics[tf].get("sharpe",0) for tf in indiv_metrics}
    best_indiv  = max(all_sharpes.values())

    for name, port in portfolios.items():
        weights_str = " ".join(f"{tf}×{s}" for tf,s in port["scales"].items())
        beat = " ★" if port["sharpe"] > best_indiv else ""
        print(f"  {name:<14} {port['total_trades']:>7} {port['pf']:>8.3f} "
              f"{port['sharpe']:>8.3f} {port['mdd']:>8.2%} "
              f"{port['total_return']:>8.1%}  {weights_str}{beat}")

    print(f"\n{'─'*72}")
    print("P&L CORRELATION MATRIX:")
    print(corr_df.round(3).to_string())

    print(f"\n{'─'*72}")
    print("DIVERSIFICATION ANALYSIS:")
    best_port = max(portfolios.items(), key=lambda x: x[1]["sharpe"])
    print(f"  Best portfolio: {best_port[0]}  Sharpe={best_port[1]['sharpe']:.3f}")
    print(f"  Best individual: max Sharpe={best_indiv:.3f}")
    port_lift = best_port[1]["sharpe"] - best_indiv
    if port_lift > 0.05:
        print(f"  ★ Portfolio improves Sharpe by {port_lift:+.3f}")
    elif port_lift > -0.05:
        print(f"  ~ Portfolio is comparable ({port_lift:+.3f})")
    else:
        print(f"  ✗ Portfolio underperforms best individual ({port_lift:+.3f})")

    # MDD comparison
    indiv_mdds  = [indiv_metrics[tf].get("max_drawdown",0) for tf in indiv_metrics]
    best_indiv_mdd = min(indiv_mdds)
    port_mdd = best_port[1]["mdd"]
    if port_mdd > best_indiv_mdd:
        print(f"  ★ Portfolio MDD={port_mdd:.2%} better than best individual MDD={best_indiv_mdd:.2%}")
    else:
        print(f"  MDD: portfolio={port_mdd:.2%}  best individual={best_indiv_mdd:.2%}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    rows = []
    for tf, met in indiv_metrics.items():
        rows.append({
            "version": f"STRAT_{tf}", "type": "individual", "trades": met.get("trade_count",0),
            "pf": met.get("profit_factor",np.nan), "sharpe": met.get("sharpe",np.nan),
            "mdd": met.get("max_drawdown",np.nan), "total_return": met.get("total_return",np.nan),
        })
    for name, port in portfolios.items():
        rows.append({
            "version": name, "type": "portfolio", "trades": port["total_trades"],
            "pf": port["pf"], "sharpe": port["sharpe"],
            "mdd": port["mdd"], "total_return": port["total_return"],
        })

    sdf = pd.DataFrame(rows)
    path= out_ranks/"portfolio_results.csv"
    sdf.to_csv(path, index=False)
    logger.info(f"\nResults → {path}")

    corr_df.to_csv(out_ranks/"portfolio_correlation.csv")
    logger.info(f"Correlation → {out_ranks}/portfolio_correlation.csv")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        equity_all = {f"STRAT_{tf}": indiv_equity[tf] for tf in indiv_equity}
        dd_all     = {f"STRAT_{tf}": indiv_dd[tf]     for tf in indiv_dd}
        for name, port in portfolios.items():
            equity_all[name] = port["equity"]
            dd_all[name]     = port["drawdown"]

        # Plots — trim all series to shortest equity for alignment
        min_eq_len = min(len(eq) for eq in equity_all.values())
        equity_trimmed = {k: pd.Series(v.values[:min_eq_len]) for k,v in equity_all.items()}
        dd_trimmed     = {k: pd.Series(v.values[:min_eq_len]) for k,v in dd_all.items()}
        ref_ts_plot = (strat_data[min(strat_data, key=lambda k: len(strat_data[k]["df_oos"]))]
                       ["df_oos"]["timestamp"].reset_index(drop=True).iloc[:min_eq_len])
        _plot_equity(equity_trimmed, dd_trimmed, ref_ts_plot, out_plots)
        _plot_correlation({f"swing/{tf}": indiv_pnl[tf] for tf in indiv_pnl}, out_plots)

    logger.info("Done.")


if __name__ == "__main__":
    main()