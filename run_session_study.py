#!/usr/bin/env python3
"""
run_session_study.py
─────────────────────
Session-based analysis for BTC trading strategies.

Tests three approaches:
    A) Session filter   — trade only within one session (may cut trades)
    B) Session sizing   — size modifier per session, no trades skipped
    C) Session + ML     — session as additional feature in ML position sizer

Sessions (UTC, fixed):
    Asia   : 00:00 – 08:00
    London : 08:00 – 16:00
    NY     : 13:00 – 21:00
    (NY and London overlap 13:00-16:00 — counted as London+NY)

Baseline (DO NOT MODIFY):
    swing_breakout / 8h  + ML_SIZE  : PF=1.948  Sharpe=1.772
    swing_breakout / 12h + ML_VOL   : PF=2.018  Sharpe=1.789

Usage
─────
    python run_session_study.py
    python run_session_study.py --tf 8h 12h --strategy swing_breakout
    python run_session_study.py --no-filter    # skip session filter (A)
    python run_session_study.py --no-ml-sess   # skip session+ML (C)
    python run_session_study.py --asia-mult 0.5 --london-mult 1.3 --ny-mult 1.1
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
from src.backtest.engine import BacktestEngine, SimConfig
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig, FEATURES
from src.ml.dynamic_engine import run_dynamic
from src.strategies.trend import DonchianBreakoutStrategy
from src.strategies.structure import SwingBreakoutStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/session_study.log"))

# ── Session definitions (UTC hours) ──────────────────────────────────────────
SESSION_DEFS = {
    "asia":   (0,  8),   # 00:00 – 08:00 UTC
    "london": (8,  16),  # 08:00 – 16:00 UTC
    "ny":     (13, 21),  # 13:00 – 21:00 UTC
}

BASELINE = {
    "swing_breakout/8h+ML":   {"pf": 1.948, "sharpe": 1.772, "mdd": -0.0272, "trades": 246},
    "swing_breakout/12h+ML":  {"pf": 2.018, "sharpe": 1.789, "mdd": -0.0186, "trades": 209},
}

STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}

ML_ELIGIBLE = {"6h", "8h", "12h", "3h"}


# ── Session helpers ───────────────────────────────────────────────────────────

def assign_session(ts: pd.Series) -> pd.Series:
    """
    Assign primary session label to each bar based on candle open hour (UTC).

    Priority when sessions overlap (13:00-16:00): London (already active).
    Returns: Series of strings ['asia', 'london', 'ny', 'off']
    """
    hour = pd.to_datetime(ts).dt.hour
    sess = pd.Series("off", index=ts.index)

    asia_s,   asia_e   = SESSION_DEFS["asia"]
    london_s, london_e = SESSION_DEFS["london"]
    ny_s,     ny_e     = SESSION_DEFS["ny"]

    # Order matters: overlap (13-16) → London takes priority
    sess[(hour >= asia_s)   & (hour < asia_e)]   = "asia"
    sess[(hour >= london_s) & (hour < london_e)] = "london"
    sess[(hour >= ny_s)     & (hour < ny_e)]     = "ny"

    return sess


def session_mults_series(
    df:           pd.DataFrame,
    asia_mult:    float,
    london_mult:  float,
    ny_mult:      float,
) -> pd.Series:
    """
    Return per-bar size multiplier based on session at signal bar (prior bar, no lookahead).
    Off-session → mult 1.0 (neutral, won't skip anything).
    """
    ts_prev = pd.to_datetime(df["timestamp"]).shift(1).bfill()
    sess  = assign_session(ts_prev)

    mults = pd.Series(1.0, index=df.index, dtype=float)
    mults[sess == "asia"]   = asia_mult
    mults[sess == "london"] = london_mult
    mults[sess == "ny"]     = ny_mult
    return mults


def session_filter_signals(
    df_sig:  pd.DataFrame,
    session: str,
) -> pd.DataFrame:
    """
    Zero out signals outside the target session.
    Uses prior bar timestamp (no lookahead).
    """
    df_out = df_sig.copy()
    ts_prev = pd.to_datetime(df_sig["timestamp"]).shift(1).bfill()
    sess = assign_session(ts_prev)

    mask = (df_out["signal"] != 0) & (sess != session)
    df_out.loc[mask, "signal"] = 0

    if "sl_price" in df_out.columns:
        df_out.loc[mask, "sl_price"] = np.nan
    if "tp_price" in df_out.columns:
        df_out.loc[mask, "tp_price"] = np.nan

    n_kept = int((df_out["signal"] != 0).sum())
    n_orig = int((df_sig["signal"] != 0).sum())
    logger.info(f"  Filter '{session}': kept {n_kept}/{n_orig} signals ({n_kept/max(n_orig,1):.1%})")
    return df_out


# ── Session distribution analysis ────────────────────────────────────────────

def session_breakdown(
    trades_df: pd.DataFrame,
    label:     str,
) -> pd.DataFrame:
    """Analyse trade performance by session using entry_time."""
    if trades_df.empty or "entry_time" not in trades_df.columns:
        return pd.DataFrame()

    trades = trades_df.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["session"]    = assign_session(trades["entry_time"])

    rows = []
    for sess in ["asia", "london", "ny", "off", "ALL"]:
        sub = trades if sess == "ALL" else trades[trades["session"] == sess]
        if len(sub) == 0:
            continue
        wins  = (sub["pnl"] > 0).sum()
        gp    = sub[sub["pnl"] > 0]["pnl"].sum()
        gl    = abs(sub[sub["pnl"] < 0]["pnl"].sum())
        rows.append({
            "label":      label,
            "session":    sess,
            "trades":     len(sub),
            "win_rate":   wins / len(sub),
            "pf":         gp / gl if gl > 0 else np.nan,
            "avg_pnl":    sub["pnl"].mean(),
            "total_pnl":  sub["pnl"].sum(),
            "avg_r_mult": sub["r_multiple"].mean() if "r_multiple" in sub.columns else np.nan,
        })
    return pd.DataFrame(rows)


# ── Metrics helper ────────────────────────────────────────────────────────────

def _m(name: str, res, extra: dict = None) -> dict:
    m = res.metrics
    row = {
        "version":     name,
        "trades":      m.get("trade_count",   0),
        "pf":          m.get("profit_factor", np.nan),
        "sharpe":      m.get("sharpe",        np.nan),
        "sortino":     m.get("sortino",       np.nan),
        "calmar":      m.get("calmar",        np.nan),
        "total_return":m.get("total_return",  np.nan),
        "mdd":         m.get("max_drawdown",  np.nan),
        "win_rate":    m.get("win_rate",      np.nan),
    }
    if extra:
        row.update(extra)
    return row


def _print_table(rows: List[dict], label: str, baseline_key: str = None) -> None:
    df  = pd.DataFrame(rows)
    b   = BASELINE.get(baseline_key, {})
    print(f"\n{'─'*80}")
    print(f"  {label}")
    if b:
        print(f"  Baseline: PF={b['pf']:.3f}  Sharpe={b['sharpe']:.3f}  "
              f"MDD={b['mdd']:.2%}  Trades={b['trades']}")
    print(f"{'─'*80}")
    print(f"{'Version':<22} {'Trades':>7} {'PF':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'MDD':>9} {'Return':>9}")
    print(f"{'─'*80}")
    base = df[df["version"]=="BASE"].iloc[0] if "BASE" in df["version"].values else None
    for _, r in df.iterrows():
        dpf = f"({r['pf']-base['pf']:+.3f})"     if base is not None and r["version"]!="BASE" else ""
        dsh = f"({r['sharpe']-base['sharpe']:+.3f})" if base is not None and r["version"]!="BASE" else ""
        vs_b = ""
        if b and r["sharpe"] > b.get("sharpe", 0) and r["trades"] >= b.get("trades", 0)*0.9:
            vs_b = " ★ BEATS BASELINE"
        print(f"{r['version']:<22} {int(r['trades'] or 0):>7} "
              f"{r['pf']:>7.3f}{dpf:<9} {r['sharpe']:>7.3f}{dsh:<9} "
              f"{r['mdd']:>8.2%} {r['total_return']:>8.1%}{vs_b}")
    print(f"{'─'*80}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_session_breakdown(breakdown: pd.DataFrame, label: str, out_dir: Path) -> None:
    if breakdown.empty:
        return

    sess_order = ["asia", "london", "ny", "off", "ALL"]

    # 🔥 FIX: агрегируем дубликаты
    df = (
        breakdown
        .groupby("session", as_index=False)
        .agg({
            "pf": "mean",
            "win_rate": "mean",
            "trades": "sum"
        })
        .set_index("session")
        .reindex(sess_order)
        .dropna(subset=["trades"])
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0f1117")
    colors = {"asia":"#7F77DD","london":"#1D9E75","ny":"#378ADD","off":"#888","ALL":"#f7c94b"}

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#2a2d3e")

    c = [colors.get(s,"#aaa") for s in df.index]

    axes[0].bar(df.index, df["pf"].fillna(0), color=c, alpha=0.85)
    axes[0].axhline(1, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_title("Profit Factor by Session", color="#e0e0e0")

    axes[1].bar(df.index, df["win_rate"]*100, color=c, alpha=0.85)
    axes[1].axhline(50, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("Win Rate % by Session", color="#e0e0e0")

    axes[2].bar(df.index, df["trades"], color=c, alpha=0.85)
    axes[2].set_title("Trade Count by Session", color="#e0e0e0")

    plt.suptitle(f"Session Breakdown — {label}", color="#e0e0e0", fontsize=12)
    plt.tight_layout()

    clean = label.replace("/","_").replace(" ","_")
    path  = out_dir / f"session_breakdown_{clean}.png"

    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)

    logger.info(f"  Plot → {path}")


def _plot_equity_comparison(
    results: Dict[str, object],
    df_oos:  pd.DataFrame,
    label:   str,
    out_dir: Path,
) -> None:
    colors = {
        "BASE":          "#888888",
        "SESSION_SIZE":  "#1D9E75",
        "ML_SIZE":       "#378ADD",
        "ML_SESS":       "#f7c94b",
        "SESS_ASIA":     "#7F77DD",
        "SESS_LONDON":   "#5DCAA5",
        "SESS_NY":       "#85B7EB",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="#0f1117",
                              gridspec_kw={"height_ratios":[3,1.5]})
    ts = df_oos["timestamp"].reset_index(drop=True)

    for ax in axes:
        ax.set_facecolor("#161925")
        ax.tick_params(colors="#aaa", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#2a2d3e")

    for name, res in results.items():
        if not hasattr(res, 'equity') or res.equity.empty: continue
        eq = res.equity.reset_index(drop=True)
        lw = 2.0 if name in ("SESSION_SIZE","ML_SESS") else 1.0
        axes[0].plot(ts[:len(eq)], eq, label=name,
                     color=colors.get(name,"white"), linewidth=lw, alpha=0.9)

    axes[0].set_title(f"Equity — {label}", color="#e0e0e0")
    axes[0].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
    axes[0].grid(True, color="#2a2d3e", alpha=0.4)

    for name, res in results.items():
        if not hasattr(res, 'drawdown') or res.drawdown.empty: continue
        dd = res.drawdown.reset_index(drop=True)*100
        axes[1].plot(ts[:len(dd)], dd, label=name,
                     color=colors.get(name,"white"), linewidth=1.2, alpha=0.85)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_title("Drawdown %", color="#e0e0e0", fontsize=10)
    axes[1].legend(facecolor="#1e2235", labelcolor="#e0e0e0", fontsize=8)
    axes[1].grid(True, color="#2a2d3e", alpha=0.4)

    plt.tight_layout()
    clean = label.replace("/","_")
    path  = out_dir / f"session_equity_{clean}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)


# ── Load params ───────────────────────────────────────────────────────────────

def _load_params(csv: Path, strategy: str, tf: str) -> Optional[dict]:
    if not csv.exists():
        return None
    df   = pd.read_csv(csv)
    mask = (df["strategy"] == strategy) & (df["timeframe"] == tf)
    sub  = df[mask]
    if sub.empty:
        return None
    best   = sub.sort_values("robustness", ascending=False).iloc[0]
    p_cols = [c for c in best.index if c.startswith("p_") and not pd.isna(best[c])]
    out    = {}
    for c in p_cols:
        v = best[c]; k = c[2:]
        out[k] = int(v) if isinstance(v, float) and v == int(v) else v
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session study for BTC strategies")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--tf",           nargs="+", default=["8h","12h","6h","3h"])
    p.add_argument("--strategy",     nargs="+",
                   default=["swing_breakout","donchian_breakout"])
    p.add_argument("--direction",    default="both")
    p.add_argument("--results-csv",  default="outputs/rankings/all_results.csv")
    p.add_argument("--asia-mult",    type=float, default=0.7)
    p.add_argument("--london-mult",  type=float, default=1.2)
    p.add_argument("--ny-mult",      type=float, default=1.1)
    p.add_argument("--no-filter",    action="store_true",
                   help="Skip section A (session filter)")
    p.add_argument("--no-ml-sess",   action="store_true",
                   help="Skip section C (session+ML)")
    p.add_argument("--n-estimators", type=int, default=400)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(Path(args.config))
    fe   = FeatureEngine(cfg)

    out_plots = cfg.output_dir / "plots";    out_plots.mkdir(exist_ok=True)
    out_ranks = cfg.output_dir / "rankings"; out_ranks.mkdir(exist_ok=True)
    results_csv = Path(args.results_csv)

    sim_cfg = SimConfig(
        fees=cfg.fees, slippage=cfg.slippage,
        leverage=cfg.leverage, risk_per_trade=cfg.risk_per_trade,
        direction=args.direction,
    )
    sizer_cfg = SizerConfig(
        n_estimators=args.n_estimators,
        min_train_trades=150, min_val_trades=30, min_mult_std=0.05,
    )

    logger.info("=== Session Study ===")
    logger.info(f"Sessions (UTC): Asia={SESSION_DEFS['asia']}  "
                f"London={SESSION_DEFS['london']}  NY={SESSION_DEFS['ny']}")
    logger.info(f"Session sizing: Asia×{args.asia_mult}  "
                f"London×{args.london_mult}  NY×{args.ny_mult}")
    logger.info(f"Strategies: {args.strategy}  TFs: {args.tf}")

    all_rows:       List[dict] = []
    breakdown_rows: List[dict] = []
    yearly_data:    dict       = {}

    for strat_name in args.strategy:
        if strat_name not in STRATEGY_MAP:
            logger.warning(f"Unknown: {strat_name}"); continue
        strategy = STRATEGY_MAP[strat_name]()

        for tf in args.tf:
            label     = f"{strat_name}/{tf}"
            bm_key    = f"{strat_name}/{tf}+ML"
            logger.info(f"\n{'='*55}\n{label}")

            # Load features
            try:
                df = fe.load(tf)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                logger.error(f"  Feature load: {exc}"); continue

            # Add session column (useful for ML feature)
            sess_series = assign_session(df["timestamp"])
            df["session_asia"]   = (sess_series == "asia").astype(int)
            df["session_london"] = (sess_series == "london").astype(int)
            df["session_ny"]     = (sess_series == "ny").astype(int)

            params = _load_params(results_csv, strat_name, tf)
            if params is None:
                logger.warning(f"  No params found"); continue

            n     = len(df)
            is_n  = int(n * cfg.validation.is_ratio)
            df_is = df.iloc[:is_n]
            df_oos= df.iloc[is_n:]
            df_oos_sig = strategy.generate_signals(df_oos.copy(), params)

            results: Dict[str, object] = {}

            # ── BASE ──────────────────────────────────────────────────────────
            logger.info("  Running BASE...")
            res_base = BacktestEngine(sim_cfg).run(
                df_oos_sig.copy(), strat_name, tf, params
            )
            results["BASE"] = res_base

            # Session breakdown on BASE trades
            bd = session_breakdown(res_base.trades, label)
            if not bd.empty:
                breakdown_rows.append(bd)
                logger.info("  Session breakdown (BASE trades):")
                for _, r in bd.iterrows():
                    logger.info(f"    {r['session']:>8}: trades={int(r['trades']):>4}  "
                                f"PF={r['pf']:.3f}  WR={r['win_rate']:.1%}")

            # ── A: Session filter ─────────────────────────────────────────────
            if not args.no_filter:
                for sess in ["asia", "london", "ny"]:
                    ver = f"SESS_{sess.upper()}"
                    logger.info(f"  Running {ver}...")
                    df_filt = session_filter_signals(df_oos_sig.copy(), sess)
                    results[ver] = BacktestEngine(sim_cfg).run(
                        df_filt, strat_name, tf, params
                    )

            # ── B: Session sizing ─────────────────────────────────────────────
            logger.info("  Running SESSION_SIZE...")
            sess_mults = session_mults_series(
                df_oos, args.asia_mult, args.london_mult, args.ny_mult
            ).reset_index(drop=True)
            results["SESSION_SIZE"] = run_dynamic(
                df_oos_sig.copy(), strat_name, tf, sim_cfg, sess_mults, params
            )

            # ── ML_SIZE ───────────────────────────────────────────────────────
            ml_mults   = pd.Series(1.0, index=range(len(df_oos)))
            ml_trained = False

            if tf in ML_ELIGIBLE:
                logger.info("  Training ML sizer...")
                sizer = MLSizerV2(cfg, sizer_cfg)
                ok    = sizer.fit(strategy, params, df_is, args.direction, label=label)
                if ok:
                    ml_mults   = sizer.predict(df_oos, strategy, params).reset_index(drop=True)
                    ml_trained = True
                    results["ML_SIZE"] = run_dynamic(
                        df_oos_sig.copy(), strat_name, tf, sim_cfg, ml_mults, params
                    )

            # ── B+: ML × Session sizing ───────────────────────────────────────
            logger.info("  Running ML_SESS (ML × session mult)...")
            if ml_trained:
                combined = (ml_mults * sess_mults).clip(0.4, 2.0)
            else:
                combined = sess_mults.clip(0.4, 2.0)
            results["ML_SESS"] = run_dynamic(
                df_oos_sig.copy(), strat_name, tf, sim_cfg, combined, params
            )

            # ── C: Session+ML — session as ML feature ─────────────────────────
            if not args.no_ml_sess and tf in ML_ELIGIBLE:
                logger.info("  Training ML_SESS_FEAT (session features in ML)...")

                # Temporarily extend FEATURES with session columns
                sess_feat_cols = ["session_asia","session_london","session_ny"]
                extended_feats = FEATURES + sess_feat_cols

                import src.ml.ml_sizer_v2 as sizer_mod
                orig_features = sizer_mod.FEATURES

                # Monkey-patch temporarily
                sizer_mod.FEATURES = extended_feats
                sizer_ext = MLSizerV2(cfg, sizer_cfg)
                ok_ext    = sizer_ext.fit(strategy, params, df_is, args.direction,
                                          label=f"{label}+sess_feat")
                sizer_mod.FEATURES = orig_features  # restore

                if ok_ext:
                    ml_sess_mults = sizer_ext.predict(df_oos, strategy, params).reset_index(drop=True)
                    results["ML_SESS_FEAT"] = run_dynamic(
                        df_oos_sig.copy(), strat_name, tf, sim_cfg, ml_sess_mults, params
                    )
                    logger.info(
                        f"  ML_SESS_FEAT: AUC={sizer_ext.auc:.3f}  "
                        f"std(mult)={sizer_ext.diagnostics.get('mult_std',0):.3f}"
                    )

            # ── Print results ─────────────────────────────────────────────────
            rows = [_m(k, v, {"strategy":strat_name,"timeframe":tf}) for k,v in results.items()]
            _print_table(rows, label, bm_key)

            # Yearly breakdown for key versions
            print(f"\n  Yearly Returns (%):")
            yr_rows = {}
            for name in ["BASE","SESSION_SIZE","ML_SIZE","ML_SESS"]:
                res = results.get(name)
                if res and not res.yearly.empty and "return" in res.yearly.columns:
                    yr_rows[name] = (res.yearly["return"]*100).round(2)
            if yr_rows:
                print(pd.DataFrame(yr_rows).to_string())

            # Save
            for r in rows:
                all_rows.append(r)

            # Plots
            _plot_session_breakdown(
                pd.concat(breakdown_rows) if breakdown_rows else pd.DataFrame(),
                label, out_plots
            )
            _plot_equity_comparison(results, df_oos, label, out_plots)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    if all_rows:
        sdf  = pd.DataFrame(all_rows)
        path = out_ranks / "session_study_results.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

    if breakdown_rows:
        bdf  = pd.concat(breakdown_rows, ignore_index=True)
        path = out_ranks / "session_breakdown.csv"
        bdf.to_csv(path, index=False)
        logger.info(f"Breakdown → {path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    if all_rows:
        sdf = pd.DataFrame(all_rows)
        print(f"\n{'='*80}")
        print("SESSION STUDY — FINAL SUMMARY vs BASELINE")
        print(f"{'='*80}")

        for (strat, tf), grp in sdf.groupby(["strategy","timeframe"]):
            bm_key = f"{strat}/{tf}+ML"
            b      = BASELINE.get(bm_key, {})
            base   = grp[grp["version"]=="BASE"]
            if base.empty: continue
            base = base.iloc[0]

            print(f"\n{strat}/{tf}:  baseline Sh={b.get('sharpe','?')}  PF={b.get('pf','?')}")
            print(f"  {'Version':<22} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9} {'vs_base_Sh':>12}")
            for _, r in grp.sort_values("sharpe",ascending=False).iterrows():
                d_sh = r["sharpe"] - base["sharpe"]
                beat = " ★" if b and r["sharpe"] > b.get("sharpe",0) and r["trades"]>=b.get("trades",0)*0.9 else ""
                print(f"  {r['version']:<22} {int(r['trades'] or 0):>7} "
                      f"{r['pf']:>8.3f} {r['sharpe']:>8.3f} {r['mdd']:>8.2%} "
                      f"{d_sh:>+11.3f}{beat}")

        # Final verdict
        print(f"\n{'='*80}")
        print("VERDICT:")
        if all_rows:
            sdf_key = sdf[sdf["version"].isin(["SESSION_SIZE","ML_SESS","ML_SESS_FEAT"])]
            beats   = 0
            for _, r in sdf_key.iterrows():
                bm_key = f"{r['strategy']}/{r['timeframe']}+ML"
                b      = BASELINE.get(bm_key, {})
                if b and r["sharpe"] > b.get("sharpe",0):
                    beats += 1
                    print(f"  ★ {r['version']} on {r['strategy']}/{r['timeframe']} "
                          f"beats baseline: Sh={r['sharpe']:.3f} > {b['sharpe']}")
            if beats == 0:
                print("  No session-based version beats the baseline.")
                print("  Session sizing adds diversification but not outperformance.")


if __name__ == "__main__":
    main()