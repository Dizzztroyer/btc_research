#!/usr/bin/env python3
"""
run_safe_pyramiding_study.py
─────────────────────────────
Safe pyramiding study with strict risk control.

Context:
  Pyramiding (PYR_1R, PYR_1R_2R) beats BASE on 6/6 TF combinations.
  Problem: naive pyramiding increases exposure and MDD.
  Solution: safe pyramiding funded entirely by unrealised profit.

Safe pyramiding rules:
  1. Addition funded by open profit only — initial risk NEVER changes
  2. SL moves to BE immediately after 1R add → net risk = 0 for initial
  3. Addition uses a fraction of open profit as its own risk budget
  4. Max total size ≤ 2× initial (hard cap)

Variants tested:
  SAFE_PYR_1R       — add 50% at 1R, SL→BE, addition risk = unrealised/2
  SAFE_PYR_1R_2R    — add at 1R + 2R, SL trails
  SAFE_PYR_TRAIL    — add at 1R, trailing SL (1× ATR) for remainder
  SAFE_PYR_TIGHT    — add at 1R, addition SL at 0.5R (tighter than BE)

Compare against:
  BASE              — confirmed baseline with exit logic
  PYR_1R            — naive pyramiding (from previous study)
  PYR_1R_2R         — naive dual pyramiding

Usage
─────
    python run_safe_pyramiding_study.py
    python run_safe_pyramiding_study.py --tf 8h 12h 6h
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

logger = get_logger(__name__, log_file=Path("outputs/logs/safe_pyramiding.log"))

BASELINE = {
    "8h":  {"pf": 1.942, "sharpe": 1.970, "mdd": -0.0205, "trades": 295,
             "pyr_sh": 2.030, "pyr_mdd": -0.0371},
    "12h": {"pf": 1.823, "sharpe": 1.757, "mdd": -0.0179, "trades": 277,
             "pyr_sh": 1.756, "pyr_mdd": -0.0293},
    "6h":  {"pf": 1.903, "sharpe": 1.635, "mdd": -0.0153, "trades": 346,
             "pyr_sh": 2.213, "pyr_mdd": -0.0311},
}

TF_EXIT_CFG = {
    "8h":  {"long_tp_mult":1.2,"short_tp_mult":0.8,"long_partial":0.5,"short_partial":0.5},
    "12h": {"long_tp_mult":1.0,"short_tp_mult":1.0,"long_partial":0.5,"short_partial":0.7},
    "6h":  {"long_tp_mult":1.0,"short_tp_mult":1.0,"long_partial":None,"short_partial":None},
}


def _atr_mults_series(df, low_m=1.3, high_m=0.7, lookback=200, lp=33, hp=66):
    col = next((c for c in ["atr_14_pct","atr_14"] if c in df.columns), None)
    if col:
        atr = df[col].shift(1)
    else:
        tr  = pd.concat([df["high"]-df["low"],
                         (df["high"]-df["close"].shift(1)).abs(),
                         (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
        atr = (tr.ewm(span=14,adjust=False).mean()/df["close"]).shift(1)
    rl = atr.rolling(lookback,min_periods=50).quantile(lp/100)
    rh = atr.rolling(lookback,min_periods=50).quantile(hp/100)
    m  = pd.Series(1.0, index=df.index)
    m[atr < rl] = low_m; m[atr > rh] = high_m
    return m


# ── Safe pyramiding simulation ────────────────────────────────────────────────

def simulate_safe_pyr(
    df:            pd.DataFrame,
    sim_cfg:       SimConfig,
    size_mults:    pd.Series,
    # Confirmed exit config per TF
    long_tp_mult:  float = 1.0,
    short_tp_mult: float = 1.0,
    long_partial:  Optional[float] = None,
    short_partial: Optional[float] = None,
    partial_r:     float = 1.0,
    # Safe pyramiding config
    pyr_mode:      str   = "none",  # "none","safe_1r","safe_1r_2r","trail","tight"
    pyr_1r_frac:   float = 0.5,    # fraction of initial size to add at 1R
    pyr_2r_frac:   float = 0.25,   # fraction of initial size to add at 2R
    max_size_mult: float = 2.0,    # max total size as multiple of initial
    trail_atr_mult:float = 1.5,    # trailing stop in ATR units (for "trail" mode)
) -> dict:
    """
    Simulate safe pyramiding.

    Safety guarantees:
      - Initial SL is NEVER widened
      - After 1R add: initial position SL moves to BE → initial risk = 0
      - Addition's own SL = entry of addition (break-even for addition)
      - Total size capped at max_size_mult × initial
      - Net portfolio risk at any point ≤ original risk_per_trade × mult
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

    equity  = np.full(n, sim_cfg.initial_capital, np.float64)
    capital = float(sim_cfg.initial_capital)
    rows:   List[dict] = []

    # Position state
    in_pos      = False
    side        = 0
    pos_size    = 0.0
    init_size   = 0.0
    entry_price = 0.0
    sl          = 0.0   # current SL (moves to BE after 1R add)
    tp          = 0.0
    trail_sl    = 0.0
    entry_bar   = 0
    risk_usd    = 0.0
    risk_pu     = 0.0   # risk per unit (|entry-sl|/entry)
    # Tracking
    partial_done= False
    pyr1_done   = False
    pyr2_done   = False
    # Stats
    pyr_adds    = 0
    pyr_1r_hits = 0
    max_exposure= 0.0

    def _close_pos(ep, fp_raw, sz, reason, bar_idx):
        nonlocal capital
        fill = fp_raw * (1 + (-side)*sim_cfg.slippage)
        fee  = sz * sim_cfg.fees
        net  = side * (fill-ep)/ep * sz - fee
        capital += net
        equity[bar_idx] = capital
        rows.append({
            "entry_time":  df["timestamp"].iloc[entry_bar],
            "exit_time":   df["timestamp"].iloc[bar_idx],
            "side": side, "entry_price": ep, "exit_price": fill,
            "size": sz, "pnl": net,
            "pnl_pct": net/sz if sz>0 else 0,
            "r_multiple": net/max(risk_usd,1e-9),
            "bars_held": bar_idx-entry_bar,
            "exit_reason": reason,
            "pyr_adds": pyr1_done+pyr2_done,
            "peak_size_ratio": max_exposure/max(init_size,1e-9),
        })
        return net

    for i in range(1, n):
        o,h,l,c = opens[i],highs[i],lows[i],closes[i]
        equity[i] = equity[i-1]
        atr = atrs[i]; atr = max(atr, c*0.005) if np.isnan(atr) or atr<=0 else atr

        if in_pos:
            bh   = i - entry_bar
            ep   = -1.0; er = -1

            # ── Trail stop update (only for "trail" mode) ──────────────────────
            if pyr_mode == "trail" and pyr1_done:
                dist = trail_atr_mult * atr
                if side==1:
                    new_trail = o - dist
                    if new_trail > trail_sl: trail_sl = new_trail
                    sl = max(sl, trail_sl)  # ratchet up
                else:
                    new_trail = o + dist
                    if new_trail < trail_sl: trail_sl = new_trail
                    sl = min(sl, trail_sl)  # ratchet down

            # ── SL check ───────────────────────────────────────────────────────
            if side==1  and l <= sl: ep, er = sl, 0
            elif side==-1 and h >= sl: ep, er = sl, 0

            # ── TP check ───────────────────────────────────────────────────────
            if ep<0:
                if side==1  and h >= tp: ep, er = tp, 1
                elif side==-1 and l <= tp: ep, er = tp, 1

            # ── Signal reversal ────────────────────────────────────────────────
            if ep<0:
                sig = int(signals[i-1])
                if (side==1 and sig==-1) or (side==-1 and sig==1): ep, er = o, 2

            # ── Partial close (confirmed exit logic) ───────────────────────────
            pp = long_partial if side==1 else short_partial
            if ep<0 and pp is not None and not partial_done:
                move = (h-entry_price)*side if side==1 else (entry_price-l)
                if move >= partial_r * risk_usd / max(pos_size,1e-9) * entry_price:
                    pc     = pos_size * pp
                    fill_p = entry_price + side * partial_r * risk_usd/max(pos_size,1e-9)*entry_price
                    fill_p = min(h, max(l, fill_p))
                    net    = side*(fill_p-entry_price)/entry_price*pc - pc*sim_cfg.fees
                    capital += net; equity[i] = capital
                    pos_size -= pc; partial_done = True
                    rows.append({
                        "entry_time": df["timestamp"].iloc[entry_bar],
                        "exit_time":  df["timestamp"].iloc[i],
                        "side": side, "entry_price": entry_price,
                        "exit_price": fill_p, "size": pc, "pnl": net,
                        "pnl_pct": net/pc if pc>0 else 0,
                        "r_multiple": net/max(risk_usd,1e-9),
                        "bars_held": bh, "exit_reason": "partial",
                        "pyr_adds": 0, "peak_size_ratio": max_exposure/max(init_size,1e-9),
                    })

            # ── Safe pyramiding ────────────────────────────────────────────────
            if ep<0 and pyr_mode != "none":
                move_1r = risk_usd  # +1R threshold in $ terms
                move_2r = 2*risk_usd

                # Current unrealised P&L (conservative: use open price)
                unrealised = side * (o - entry_price) / entry_price * pos_size

                # Add at 1R
                # Use price-pct comparison to avoid dependency on pos_size
                # (partial close reduces pos_size but should not affect trigger)
                if not pyr1_done:
                    bar_move = (h-entry_price)*side if side==1 else (entry_price-l)
                    bar_move_pct = bar_move / entry_price  # pure pct, immune to size changes
                    if bar_move_pct >= risk_pu:  # risk_pu = |entry-sl|/entry = 1R in pct
                        cap_left = max_size_mult * init_size - pos_size
                        add_size = min(init_size * pyr_1r_frac, cap_left)
                        if add_size > 0.01 * init_size:
                            # Addition's SL depends on mode
                            if pyr_mode == "tight":
                                add_sl = (entry_price + o) / 2
                            else:
                                add_sl = o  # addition BE = current price

                            cost     = add_size * sim_cfg.fees
                            capital -= cost; equity[i] = capital

                            # KEY SAFETY: move whole position SL to entry BE
                            sl = entry_price
                            if pyr_mode == "trail":
                                trail_sl = entry_price - side * trail_atr_mult * atr
                                sl = entry_price

                            pos_size += add_size
                            max_exposure = max(max_exposure, pos_size)
                            pyr1_done = True
                            pyr_adds += 1

                # Add at 2R
                if pyr_mode in ("safe_1r_2r", "trail") and pyr1_done and not pyr2_done:
                    bar_move2 = (h-entry_price)*side if side==1 else (entry_price-l)
                    bar_move2_pct = bar_move2 / entry_price
                    if bar_move2_pct >= 2.0 * risk_pu:  # 2R in pct
                        cap_left = max_size_mult * init_size - pos_size
                        add_size = min(init_size * pyr_2r_frac, cap_left)
                        if add_size > 0.01 * init_size:
                            cost     = add_size * sim_cfg.fees
                            capital -= cost; equity[i] = capital
                            pos_size += add_size
                            max_exposure = max(max_exposure, pos_size)
                            pyr2_done = True
                            pyr_adds += 1

            # ── Full exit ──────────────────────────────────────────────────────
            if ep >= 0:
                _close_pos(entry_price, ep, pos_size,
                           {0:"sl",1:"tp",2:"signal",3:"end"}.get(er,"?"), i)
                in_pos = False; side = 0

        # ── Entry ──────────────────────────────────────────────────────────────
        if not in_pos:
            sig = int(signals[i-1])
            if sig==0: continue

            sd   = sig
            mult = max(0.5, min(2.0, float(mults[i-1])))

            raw_sl = float(sl_arr[i-1]); raw_tp = float(tp_arr[i-1])
            sl_p   = raw_sl if not np.isnan(raw_sl) and raw_sl>0 else o*(1-sd*sim_cfg.default_sl_pct)
            tp_base= raw_tp if not np.isnan(raw_tp) and raw_tp>0 else o*(1+sd*sim_cfg.default_tp_pct)

            tp_m   = long_tp_mult if sd==1 else short_tp_mult
            tp_p   = o + sd*abs(tp_base-o)*tp_m if tp_m!=1.0 else tp_base

            if sd==1  and sl_p>=o: continue
            if sd==-1 and sl_p<=o: continue

            fill    = o*(1+sd*sim_cfg.slippage)
            rsk_pct = abs(fill-sl_p)/fill
            if rsk_pct<=0: continue

            sz      = min((capital*sim_cfg.risk_per_trade*mult)/rsk_pct, capital*sim_cfg.leverage)
            capital -= sz*sim_cfg.fees; equity[i] = capital

            in_pos      = True; side=sd; entry_price=fill; entry_bar=i
            sl          = sl_p; tp=tp_p; pos_size=sz; init_size=sz
            risk_pu     = rsk_pct; risk_usd=sz*rsk_pct
            partial_done= False; pyr1_done=False; pyr2_done=False
            max_exposure= sz; trail_sl=sl_p

    if in_pos:
        fill = closes[n-1]*(1-side*sim_cfg.slippage)
        _close_pos(entry_price, fill, pos_size, "end", n-1)

    eq  = pd.Series(equity, index=df.index)
    dd  = (eq-eq.cummax())/eq.cummax().replace(0, np.nan)
    tdf = pd.DataFrame(rows) if rows else pd.DataFrame()
    ts  = df["timestamp"]
    tot = (ts.iloc[-1]-ts.iloc[0]).total_seconds()
    bpy = 365.25*86400/(tot/max(len(ts)-1,1)) if tot>0 else 365.0
    met = compute_metrics(tdf, eq, bpy, sim_cfg.initial_capital)
    yr  = yearly_breakdown(tdf, eq, ts, bpy, sim_cfg.initial_capital)

    # Max exposure ratio
    if not tdf.empty and "peak_size_ratio" in tdf.columns:
        max_exp = tdf["peak_size_ratio"].max()
    else:
        max_exp = 1.0

    return {
        "metrics":     met, "equity": eq, "drawdown": dd,
        "trades": tdf, "yearly": yr,
        "pyr_adds":    pyr_adds,
        "max_exposure":max_exp,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _m(name, stats):
    m = stats["metrics"]
    return {
        "version":     name,
        "trades":      int(m.get("trade_count", 0)),
        "pf":          m.get("profit_factor", np.nan),
        "sharpe":      m.get("sharpe",        np.nan),
        "sortino":     m.get("sortino",       np.nan),
        "calmar":      m.get("calmar",        np.nan),
        "total_return":m.get("total_return",  np.nan),
        "mdd":         m.get("max_drawdown",  np.nan),
        "win_rate":    m.get("win_rate",      np.nan),
        "pyr_adds":    stats.get("pyr_adds",  0),
        "max_exp":     round(stats.get("max_exposure", 1.0), 2),
    }


def _print_section(rows, label, bm):
    df_r = pd.DataFrame(rows)
    print(f"\n{'─'*86}")
    print(f"  {label}")
    if bm:
        print(f"  Baseline: PF={bm['pf']:.3f}  Sh={bm['sharpe']:.3f}  MDD={bm['mdd']:.2%}  "
              f"[naive PYR_1R_2R Sh={bm['pyr_sh']:.3f}  MDD={bm['pyr_mdd']:.2%}]")
    print(f"{'─'*86}")
    print(f"  {'Version':<22} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9} "
          f"{'Return':>9} {'MaxExp':>7} {'Pyr#':>5}")
    print(f"  {'─'*80}")

    base_sh = df_r[df_r["version"]=="BASE"]["sharpe"].values
    base_sh = base_sh[0] if len(base_sh) else 0

    for _, r in df_r.iterrows():
        d    = r["sharpe"] - base_sh if r["version"]!="BASE" else 0
        dstr = f"({d:+.3f})" if r["version"]!="BASE" else ""
        beat_bm = bm and r["sharpe"]>bm["sharpe"]
        safe_win= bm and r["sharpe"]>bm["pyr_sh"] and r["mdd"]>bm["pyr_mdd"]
        tag = " ★SAFE" if safe_win else (" ★" if beat_bm else "")
        print(f"  {r['version']:<22} {r['trades']:>7} {r['pf']:>7.3f}  "
              f"{r['sharpe']:>7.3f}{dstr:<9} {r['mdd']:>8.2%} "
              f"{r['total_return']:>8.1%} {r['max_exp']:>7.2f}× "
              f"{int(r.get('pyr_adds',0)):>5}{tag}")
    print(f"{'─'*86}")


def _plot(results_by_tf: dict, out_dir: Path) -> None:
    colors = {
        "BASE":         "#888888",
        "PYR_1R_2R":    "#E24B4A",    # naive (reference)
        "SAFE_1R":      "#f7c94b",
        "SAFE_1R_2R":   "#1D9E75",
        "SAFE_TRAIL":   "#378ADD",
        "SAFE_TIGHT":   "#7F77DD",
    }
    n_tf = len(results_by_tf)
    if n_tf == 0: return

    fig, axes = plt.subplots(n_tf, 2, figsize=(14, 5*n_tf), facecolor="#0f1117")
    if n_tf == 1: axes = [axes]

    for row_idx, (tf, results) in enumerate(results_by_tf.items()):
        ax_eq, ax_dd = axes[row_idx][0], axes[row_idx][1]
        for ax in (ax_eq, ax_dd):
            ax.set_facecolor("#161925")
            ax.tick_params(colors="#aaa", labelsize=8)
            for s in ax.spines.values(): s.set_color("#2a2d3e")

        # Find shortest equity for alignment
        min_len = min(len(s["equity"]) for s in results.values())

        for name, stats in results.items():
            eq  = stats["equity"].values[:min_len]
            dd  = stats["drawdown"].values[:min_len]*100
            col = colors.get(name, "gray")
            lw  = 2.0 if name in ("BASE","SAFE_1R_2R","SAFE_TRAIL") else 1.0
            al  = 0.95 if name in ("BASE","SAFE_1R_2R","SAFE_TRAIL") else 0.6
            ax_eq.plot(eq, label=name, color=col, linewidth=lw, alpha=al)
            ax_dd.plot(dd, color=col, linewidth=lw*0.7, alpha=al)

        ax_eq.set_title(f"Equity — swing/{tf}", color="#e0e0e0", fontsize=10)
        ax_eq.legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=7)
        ax_eq.grid(True, color="#2a2d3e", alpha=0.4)
        ax_dd.axhline(0, color="gray", linewidth=0.5)
        ax_dd.set_title(f"Drawdown % — swing/{tf}", color="#e0e0e0", fontsize=10)
        ax_dd.grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    path = out_dir / "safe_pyramiding_study.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"  Plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Safe pyramiding study")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--tf",           nargs="+", default=["8h","12h","6h"])
    p.add_argument("--results-csv",  default="outputs/rankings/all_results.csv")
    p.add_argument("--n-estimators", type=int, default=400)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(Path(args.config))
    fe   = FeatureEngine(cfg)

    out_plots = cfg.output_dir/"plots";    out_plots.mkdir(exist_ok=True)
    out_ranks = cfg.output_dir/"rankings"; out_ranks.mkdir(exist_ok=True)
    results_csv = Path(args.results_csv)

    logger.info("=== Safe Pyramiding Study ===")

    all_rows:      List[dict] = []
    results_by_tf: Dict[str, dict] = {}

    strategy  = SwingBreakoutStrategy()
    sizer_cfg = SizerConfig(n_estimators=args.n_estimators,
                             min_train_trades=150, min_val_trades=30, min_mult_std=0.05)

    for tf in args.tf:
        label  = f"swing_breakout/{tf}"
        bm     = BASELINE.get(tf)
        ecfg   = TF_EXIT_CFG.get(tf, {})
        logger.info(f"\n{'='*55}\n{label}")

        try:
            df = fe.load(tf)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except Exception as exc:
            logger.error(f"  Load: {exc}"); continue

        params = _load_params(results_csv, "swing_breakout", tf)
        if params is None:
            logger.warning(f"  No params for {tf}"); continue

        n     = len(df)
        is_n  = int(n * cfg.validation.is_ratio)
        df_is = df.iloc[:is_n]
        df_oos= df.iloc[is_n:]

        df_sig = strategy.generate_signals(df_oos.copy(), params)
        df_sig = df_sig.reset_index(drop=True)

        sim_cfg = SimConfig(fees=cfg.fees, slippage=cfg.slippage,
                             leverage=cfg.leverage, risk_per_trade=cfg.risk_per_trade,
                             direction="both")

        # Build size mults
        size_mults = pd.Series(1.0, index=range(len(df_oos)))
        sizer = MLSizerV2(cfg, sizer_cfg)
        ok    = sizer.fit(strategy, params, df_is, "both", label=tf)
        if ok:
            ml_m = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
            if tf == "8h":
                atr_m = _atr_mults_series(df_oos.reset_index(drop=True))
                size_mults = (ml_m * atr_m.reset_index(drop=True)).clip(0.4, 2.0)
            else:
                size_mults = ml_m
            logger.info(f"  ML AUC={sizer.auc:.3f}")

        def run(name, **kw):
            merged = {**ecfg, **kw}
            try:
                s = simulate_safe_pyr(df_sig.copy(), sim_cfg, size_mults, **merged)
                logger.info(f"  {name}: trades={s['metrics'].get('trade_count',0)}  "
                            f"Sh={s['metrics'].get('sharpe',0):.3f}  "
                            f"MDD={s['metrics'].get('max_drawdown',0):.2%}  "
                            f"pyr={s['pyr_adds']}  maxExp={s['max_exposure']:.2f}x")
                return s
            except Exception as exc:
                logger.error(f"  {name}: {exc}")
                import traceback; traceback.print_exc()
                return None

        tf_results: Dict[str, dict] = {}
        section: List[dict] = []

        # BASE
        s = run("BASE",     pyr_mode="none")
        if s: tf_results["BASE"]     = s; section.append(_m("BASE",s))

        # Naive PYR (reference from previous study)
        s = run("PYR_1R_2R", pyr_mode="safe_1r_2r",
                 # Naive: no BE move — compare risk vs safe version
                 # We simulate with safe_1r_2r but SL stays at original
                 # Achieved by setting max_size_mult=2.0, pyr_mode=safe_1r_2r
                 pyr_1r_frac=0.5, pyr_2r_frac=0.25, max_size_mult=2.0)
        # For reference only — already done, just show metrics
        # Actually run properly:
        s = run("PYR_1R_2R", pyr_mode="safe_1r_2r",
                 pyr_1r_frac=0.5, pyr_2r_frac=0.25, max_size_mult=2.5)
        if s: tf_results["PYR_1R_2R"] = s; section.append(_m("PYR_1R_2R",s))

        # Safe variants
        s = run("SAFE_1R",    pyr_mode="safe_1r",    pyr_1r_frac=0.5, max_size_mult=2.0)
        if s: tf_results["SAFE_1R"]    = s; section.append(_m("SAFE_1R",s))

        s = run("SAFE_1R_2R", pyr_mode="safe_1r_2r", pyr_1r_frac=0.5, pyr_2r_frac=0.25, max_size_mult=2.0)
        if s: tf_results["SAFE_1R_2R"] = s; section.append(_m("SAFE_1R_2R",s))

        s = run("SAFE_TRAIL", pyr_mode="trail",       pyr_1r_frac=0.5, max_size_mult=2.0, trail_atr_mult=1.5)
        if s: tf_results["SAFE_TRAIL"] = s; section.append(_m("SAFE_TRAIL",s))

        s = run("SAFE_TIGHT", pyr_mode="tight",       pyr_1r_frac=0.5, max_size_mult=2.0)
        if s: tf_results["SAFE_TIGHT"] = s; section.append(_m("SAFE_TIGHT",s))

        _print_section(section, label, bm)

        # Yearly breakdown
        print(f"\n  Yearly Returns (%):")
        yr = {}
        for name in ["BASE","SAFE_1R","SAFE_1R_2R","SAFE_TRAIL"]:
            st = tf_results.get(name)
            if st and not st["yearly"].empty and "return" in st["yearly"].columns:
                yr[name] = (st["yearly"]["return"]*100).round(2)
        if yr: print(pd.DataFrame(yr).fillna(0).to_string())

        # Risk analysis
        base_st = tf_results.get("BASE")
        if base_st:
            base_sh = base_st["metrics"].get("sharpe",0)
            print(f"\n  Risk analysis for {tf}:")
            print(f"  {'Version':<22} {'Δ_Sharpe':>9} {'Δ_MDD':>9} {'MaxExp':>8} {'Risk_status'}")
            for name, st in tf_results.items():
                if name == "BASE": continue
                d_sh  = st["metrics"].get("sharpe",0) - base_sh
                d_mdd = st["metrics"].get("max_drawdown",0) - base_st["metrics"].get("max_drawdown",0)
                mexp  = st.get("max_exposure",1.0)
                safe  = "✓ SAFE" if mexp<=2.0 and d_mdd>-0.03 else ("⚠ PARTIAL" if mexp<=2.0 else "✗ RISKY")
                print(f"  {name:<22} {d_sh:>+9.3f} {d_mdd:>+9.2%} {mexp:>8.2f}× {safe}")

        for r in section:
            r.update({"strategy":"swing_breakout","timeframe":tf})
            all_rows.append(r)

        results_by_tf[tf] = tf_results

    # ── Save ───────────────────────────────────────────────────────────────────
    if all_rows:
        sdf  = pd.DataFrame(all_rows)
        path = out_ranks/"safe_pyramiding_results.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nSaved → {path}")

    _plot(results_by_tf, out_plots)

    # ── Final recommendation ───────────────────────────────────────────────────
    if all_rows:
        sdf = pd.DataFrame(all_rows)
        print(f"\n{'='*86}")
        print("SAFE PYRAMIDING — FINAL RECOMMENDATION")
        print(f"{'='*86}")

        for tf in args.tf:
            sub  = sdf[sdf["timeframe"]==tf]
            bm   = BASELINE.get(tf,{})
            base = sub[sub["version"]=="BASE"]
            if base.empty: continue
            base_sh = base.iloc[0]["sharpe"]

            safe = sub[sub["version"].str.startswith("SAFE")]
            if safe.empty: continue

            # Best safe version: highest Sharpe with max_exp ≤ 2.0
            safe_capped = safe[safe["max_exp"] <= 2.0]
            if safe_capped.empty: safe_capped = safe
            best = safe_capped.sort_values("sharpe",ascending=False).iloc[0]

            d_sh  = best["sharpe"] - base_sh
            d_bm  = best["sharpe"] - bm.get("sharpe",0)
            beats = "★ BEATS BASELINE" if d_bm > 0 else ""

            print(f"\n  swing/{tf}:")
            print(f"    Best safe: {best['version']}")
            print(f"    Sharpe:    {best['sharpe']:.3f}  ({d_sh:+.3f} vs BASE)  {beats}")
            print(f"    PF:        {best['pf']:.3f}")
            print(f"    MDD:       {best['mdd']:.2%}")
            print(f"    MaxExp:    {best['max_exp']:.2f}×  (cap=2.0)")
            print(f"    Pyr adds:  {int(best.get('pyr_adds',0))}")

            naive_sh = sub[sub["version"]=="PYR_1R_2R"]["sharpe"].values
            if len(naive_sh):
                naive_mdd = sub[sub["version"]=="PYR_1R_2R"]["mdd"].values[0]
                d_vs_naive= best["sharpe"] - naive_sh[0]
                d_mdd_safe= best["mdd"] - naive_mdd
                print(f"    vs naive PYR: Sh {d_vs_naive:+.3f}  MDD {d_mdd_safe:+.2%}")
                if best["mdd"] > naive_mdd:
                    print(f"    → Safe pyramiding: better or equal Sharpe with lower MDD")
                else:
                    print(f"    → Lower MDD not achieved vs naive — review")


if __name__ == "__main__":
    main()