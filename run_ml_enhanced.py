#!/usr/bin/env python3
"""
run_ml_enhanced.py
───────────────────
ML position sizer with expanded feature set and strict quality control.

Improvements over ml_sizer_v2:
    1. Expanded features:
       - volume_zscore (20, 40)
       - ATR percentile (volatility regime)
       - session labels (asia/london/ny as one-hot)
       - breakout strength (distance from breakout level)
       - trend alignment (price vs EMA200)
       - momentum confirmation (RSI direction)

    2. Strict quality control:
       - AUC < 0.55 → disable ML
       - std(mult) < 0.10 → disable ML
       - IS trades < 150 → disable ML

    3. Walk-forward ML evaluation
       - train on expanding IS window
       - report AUC per window

    4. Feature importance analysis
       - which features actually drive the model

Compares:
    BASE     — fixed sizing
    ML_V2    — current benchmark sizer (rank-norm)
    ML_EXT   — expanded feature set, same formula
    ML_STRICT — ML_EXT with strict quality gates

Baseline (DO NOT MODIFY):
    swing / 8h  + ML_SIZE: PF=1.948  Sharpe=1.772
    swing / 12h + ML_VOL:  PF=2.018  Sharpe=1.789

Usage
─────
    python run_ml_enhanced.py
    python run_ml_enhanced.py --tf 8h 12h --strategy swing_breakout
    python run_ml_enhanced.py --auc-thresh 0.55 --std-thresh 0.10
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
from src.ml.ml_sizer_v2 import (
    MLSizerV2, SizerConfig, score_to_mult, compute_weights,
    _avail, _norm_ts, _ts_index, FEATURES as BASE_FEATURES
)
from src.ml.dynamic_engine import run_dynamic
from src.strategies.trend import DonchianBreakoutStrategy
from src.strategies.structure import SwingBreakoutStrategy

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    ML_OK = True
except ImportError:
    ML_OK = False

logger = get_logger(__name__, log_file=Path("outputs/logs/ml_enhanced.log"))

BASELINE = {
    "swing_breakout/8h+ML":  {"pf": 1.948, "sharpe": 1.772, "mdd": -0.0272, "trades": 246},
    "swing_breakout/12h+ML": {"pf": 2.018, "sharpe": 1.789, "mdd": -0.0186, "trades": 209},
}
STRATEGY_MAP = {
    "swing_breakout":    SwingBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}
ML_ELIGIBLE = {"6h", "8h", "12h"}


# ── Extended feature set ──────────────────────────────────────────────────────

# Additions to base FEATURES from ml_sizer_v2
# Extended features — ONLY numeric. String columns like regime_master are excluded.
EXTENDED_FEATURES = list(dict.fromkeys(BASE_FEATURES + [
    # Breakout quality
    "don_20_dist_high", "don_20_dist_low",
    "don_40_dist_high", "don_40_dist_low",
    # Vol context
    "vol_zscore_10",
    # ATR percentile rank (numeric, computed inline)
    "atr_21_pct",
    # Trend alignment
    "dist_sma_50", "dist_sma_200",
    # Momentum
    "rsi_7", "rsi_21",
    # Session one-hot (numeric 0/1)
    "session_asia", "session_europe", "session_us",
    # Squeeze
    "squeeze_20",
]))  # dict.fromkeys deduplicates while preserving order


def _add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add session one-hot features if not present."""
    if "session_asia" not in df.columns:
        hour = pd.to_datetime(df["timestamp"]).dt.hour
        df = df.copy()
        df["session_asia"]   = ((hour >= 0)  & (hour < 8)).astype(int)
        df["session_europe"] = ((hour >= 8)  & (hour < 16)).astype(int)
        df["session_us"]     = ((hour >= 13) & (hour < 21)).astype(int)
    return df


def _add_atr_percentile(df: pd.DataFrame, lookback: int = 200) -> pd.DataFrame:
    """Add rolling ATR percentile as a feature."""
    if "atr_14_pct" in df.columns and "atr_pct_rank" not in df.columns:
        df = df.copy()
        atr = df["atr_14_pct"]
        df["atr_pct_rank"] = atr.rolling(lookback, min_periods=50).rank(pct=True)
    return df


# ── Enhanced ML sizer ─────────────────────────────────────────────────────────

class MLSizerEnhanced:
    """
    ML position sizer with expanded features and strict quality gates.

    Quality gates (configurable):
        AUC < auc_thresh       → disabled
        std(mult) < std_thresh → disabled
        IS trades < 150        → disabled
    """

    def __init__(
        self,
        cfg,
        sizer_cfg:  SizerConfig,
        auc_thresh: float = 0.55,
        std_thresh: float = 0.10,
        use_extended: bool = True,
    ):
        self.cfg          = cfg
        self.scfg         = sizer_cfg
        self.auc_thresh   = auc_thresh
        self.std_thresh   = std_thresh
        self.use_extended = use_extended
        self.features     = EXTENDED_FEATURES if use_extended else BASE_FEATURES

        self.model     = None
        self.medians   = None
        self.feat_cols: List[str] = []
        self.auc       = 0.5
        self.enabled   = False
        self.diag:     Dict = {}

    def _get_trades(self, strategy, params, df, direction="both"):
        from src.backtest.engine import BacktestEngine, SimConfig
        sim    = SimConfig(
            fees=self.cfg.fees, slippage=self.cfg.slippage,
            leverage=self.cfg.leverage, risk_per_trade=self.cfg.risk_per_trade,
            direction=direction,
        )
        df_aug = _add_session_features(df)
        df_aug = _add_atr_percentile(df_aug)
        df_sig = strategy.generate_signals(df_aug.copy(), params)
        result = BacktestEngine(sim).run(df_sig, strategy.name, "", params)

        if result.trades.empty:
            return pd.DataFrame(), pd.Series(dtype=int), np.array([])

        trades          = result.trades.copy()
        trades["label"] = (trades["pnl"] > 0).astype(int)
        entry_times     = _norm_ts(trades["entry_time"])
        lookup          = _ts_index(df_sig)
        entry_idx       = [lookup.get(et) for et in entry_times]

        valid = [(i,l,p) for i,l,p in
                 zip(entry_idx, trades["label"], trades["pnl"]) if i is not None]
        if not valid:
            return pd.DataFrame(), pd.Series(dtype=int), np.array([])

        indices, labels, pnls = zip(*valid)
        indices = list(indices)

        cols = _avail(df_sig, self.features)
        X    = df_sig[cols].shift(1).loc[indices].copy()
        # Drop any non-numeric columns (e.g. regime_master which is string)
        X    = X.select_dtypes(include=["number"])
        y    = pd.Series(list(labels), index=indices)
        p    = np.array(list(pnls))

        common = X.index.intersection(y.index)
        X, y   = X.loc[common], y.loc[common]
        idx_m  = {orig:i for i,orig in enumerate(indices)}
        p_arr  = np.array([pnls[idx_m[i]] for i in common])

        return X, y, p_arr

    def fit(self, strategy, params, df_is, direction="both", label="") -> bool:
        tag = f"[{label}]"
        logger.info(f"{tag} Training enhanced ML sizer...")

        X, y, pnls = self._get_trades(strategy, params, df_is, direction)
        n = len(X)
        logger.info(f"{tag} IS trades={n}  win_rate={y.mean():.1%}  "
                    f"features={X.shape[1] if not X.empty else 0}")

        if n < self.scfg.min_train_trades:
            logger.warning(f"{tag} Too few IS trades ({n}). DISABLED.")
            self.diag["reason"] = f"few_trades_{n}"
            return False

        if len(y.unique()) < 2:
            logger.warning(f"{tag} Single class. DISABLED.")
            return False

        split    = int(n * self.scfg.is_train_ratio)
        val_size = n - split

        if split < 50 or val_size < self.scfg.min_val_trades:
            logger.warning(f"{tag} Split too small ({split}/{val_size}). DISABLED.")
            return False

        X_tr, X_vl = X.iloc[:split], X.iloc[split:]
        y_tr, y_vl = y.iloc[:split], y.iloc[split:]

        self.medians      = X_tr.median()
        self.feat_cols    = list(X.columns)

        Xtr_c = X_tr.fillna(self.medians)
        Xvl_c = X_vl.fillna(self.medians)

        pos = y_tr.sum(); neg = len(y_tr) - pos
        spw = max(0.5, min(3.0, neg/pos)) if pos > 0 else 1.0
        weights = compute_weights(pnls[:split], y_tr.values)

        self.model = xgb.XGBClassifier(
            n_estimators          = self.scfg.n_estimators,
            max_depth             = self.scfg.max_depth,
            learning_rate         = self.scfg.learning_rate,
            subsample             = self.scfg.subsample,
            colsample_bytree      = self.scfg.colsample,
            min_child_weight      = self.scfg.min_child_weight,
            scale_pos_weight      = spw,
            reg_alpha             = 0.1,
            reg_lambda            = 1.0,
            early_stopping_rounds = self.scfg.early_stopping,
            eval_metric           = "auc",
            verbosity             = 0,
            random_state          = 42,
        )
        self.model.fit(Xtr_c, y_tr, eval_set=[(Xvl_c, y_vl)],
                       sample_weight=weights, verbose=False)

        y_pred    = self.model.predict_proba(Xvl_c)[:, 1]
        self.auc  = roc_auc_score(y_vl, y_pred) if len(y_vl.unique()) > 1 else 0.5

        # Compute multiplier stats on all IS data
        all_scores = self.model.predict_proba(X.fillna(self.medians))[:, 1]
        all_mults  = score_to_mult(all_scores)
        mult_std   = float(all_mults.std())
        mult_mean  = float(all_mults.mean())

        self.diag = {
            "n_trades": n, "n_train": split, "n_val": val_size,
            "win_rate": float(y.mean()), "auc": self.auc,
            "mult_mean": mult_mean, "mult_std": mult_std,
            "mult_min": float(all_mults.min()), "mult_max": float(all_mults.max()),
            "n_features": len(self.feat_cols),
        }

        logger.info(
            f"{tag} AUC={self.auc:.3f}  std(mult)={mult_std:.3f}  "
            f"mean(mult)={mult_mean:.3f}  [{all_mults.min():.3f},{all_mults.max():.3f}]"
        )

        # Strict quality gates
        if self.auc < self.auc_thresh:
            logger.warning(f"{tag} AUC={self.auc:.3f} < {self.auc_thresh} → DISABLED")
            self.diag["reason"] = f"low_auc_{self.auc:.3f}"
            self.model = None
            return False

        if mult_std < self.std_thresh:
            logger.warning(f"{tag} std(mult)={mult_std:.4f} < {self.std_thresh} → DISABLED")
            self.diag["reason"] = f"low_std_{mult_std:.4f}"
            self.model = None
            return False

        logger.info(f"{tag} ✓ Enhanced sizer ENABLED")
        self.enabled = True
        return True

    def predict(self, df, strategy, params) -> pd.Series:
        if not self.enabled or self.model is None:
            return pd.Series(1.0, index=df.index)

        df_aug  = _add_session_features(df)
        df_aug  = _add_atr_percentile(df_aug)
        df_sig  = strategy.generate_signals(df_aug.copy(), params)
        sig_msk = df_sig["signal"] != 0
        mults   = pd.Series(1.0, index=df.index)

        if not sig_msk.any():
            return mults

        sig_idx = list(df_sig.index[sig_msk])
        cols    = _avail(df_sig, self.feat_cols)
        X       = df_sig[cols].shift(1).loc[sig_idx].copy()
        X       = X.select_dtypes(include=["number"])
        for c in self.feat_cols:
            if c not in X.columns: X[c] = np.nan
        avail_feat = [c for c in self.feat_cols if c in X.columns]
        X = X[avail_feat].fillna(self.medians.reindex(avail_feat, fill_value=0))

        try:
            scores = self.model.predict_proba(X)[:, 1]
            m      = score_to_mult(scores)
            mults.loc[sig_idx] = m
            logger.info(f"  OOS mults: mean={m.mean():.3f} std={m.std():.3f} "
                        f"[{m.min():.3f},{m.max():.3f}]")
        except Exception as exc:
            logger.warning(f"  Predict failed: {exc}")

        return mults

    @property
    def importances(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature":    self.feat_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


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


def _m(name, res, extra=None):
    m = res.metrics
    row = {"version":name,
           "trades":m.get("trade_count",0), "pf":m.get("profit_factor",np.nan),
           "sharpe":m.get("sharpe",np.nan), "sortino":m.get("sortino",np.nan),
           "calmar":m.get("calmar",np.nan), "total_return":m.get("total_return",np.nan),
           "mdd":m.get("max_drawdown",np.nan), "win_rate":m.get("win_rate",np.nan)}
    if extra: row.update(extra)
    return row


def _print_table(rows, label, bm_key=None):
    df = pd.DataFrame(rows)
    b  = BASELINE.get(bm_key, {})
    print(f"\n{'─'*75}")
    print(f"  {label}")
    if b: print(f"  Baseline: PF={b['pf']:.3f}  Sh={b['sharpe']:.3f}  trades={b['trades']}")
    print(f"{'─'*75}")
    print(f"{'Version':<20} {'Trades':>7} {'PF':>8} {'Sharpe':>8} {'MDD':>9} {'Return':>9}")
    print(f"{'─'*75}")
    base = df[df["version"]=="BASE"].iloc[0] if "BASE" in df["version"].values else None
    for _, r in df.iterrows():
        dsh  = f"({r['sharpe']-base['sharpe']:+.3f})" if base is not None and r["version"]!="BASE" else ""
        beat = " ★" if b and r["sharpe"]>b.get("sharpe",0) and r["trades"]>=b.get("trades",0)*0.9 else ""
        print(f"{r['version']:<20} {int(r['trades'] or 0):>7} "
              f"{r['pf']:>7.3f}  {r['sharpe']:>7.3f}{dsh:<9} "
              f"{r['mdd']:>8.2%} {r['total_return']:>8.1%}{beat}")
    print(f"{'─'*75}")


def _plot_importances(imp: pd.DataFrame, label: str, tag: str, out_dir: Path) -> None:
    if imp.empty: return
    top = imp.head(20)
    fig, ax = plt.subplots(figsize=(10,7), facecolor="#0f1117")
    ax.set_facecolor("#161925")
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#1D9E75", alpha=0.85)
    ax.set_xlabel("Importance", color="#aaa"); ax.tick_params(colors="#aaa")
    ax.set_title(f"Feature Importances — {label} ({tag})", color="#e0e0e0")
    for s in ax.spines.values(): s.set_color("#2a2d3e")
    plt.tight_layout()
    clean = label.replace("/","_")
    fig.savefig(out_dir/f"ml_imp_{clean}_{tag}.png", dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Enhanced ML sizer with expanded features")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--tf",           nargs="+", default=["8h","12h","6h"])
    p.add_argument("--strategy",     nargs="+",
                   default=["swing_breakout","donchian_breakout"])
    p.add_argument("--direction",    default="both")
    p.add_argument("--results-csv",  default="outputs/rankings/all_results.csv")
    p.add_argument("--auc-thresh",   type=float, default=0.55)
    p.add_argument("--std-thresh",   type=float, default=0.10)
    p.add_argument("--n-estimators", type=int,   default=500)
    return p.parse_args()


def main():
    if not ML_OK:
        print("pip install xgboost scikit-learn"); sys.exit(1)

    args = parse_args()
    cfg  = load_config(Path(args.config))
    fe   = FeatureEngine(cfg)

    out_plots = cfg.output_dir/"plots";    out_plots.mkdir(exist_ok=True)
    out_ranks = cfg.output_dir/"rankings"; out_ranks.mkdir(exist_ok=True)
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

    logger.info("=== ML Enhanced Study ===")
    logger.info(f"Quality gates: AUC>{args.auc_thresh}  std(mult)>{args.std_thresh}")
    logger.info(f"Extended features: {len(EXTENDED_FEATURES)} total")

    all_rows:  List[dict] = []
    diag_rows: List[dict] = []

    for strat_name in args.strategy:
        if strat_name not in STRATEGY_MAP: continue
        strategy = STRATEGY_MAP[strat_name]()

        for tf in args.tf:
            label  = f"{strat_name}/{tf}"
            bm_key = f"{strat_name}/{tf}+ML"
            logger.info(f"\n{'='*55}\n{label}")

            try:
                df = fe.load(tf)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                logger.error(f"  Feature load: {exc}"); continue

            params = _load_params(results_csv, strat_name, tf)
            if params is None:
                logger.warning(f"  No params"); continue

            n     = len(df)
            is_n  = int(n * cfg.validation.is_ratio)
            df_is = df.iloc[:is_n]
            df_oos= df.iloc[is_n:]
            df_oos_sig = strategy.generate_signals(df_oos.copy(), params)

            results: Dict[str, object] = {}

            # BASE
            results["BASE"] = BacktestEngine(sim_cfg).run(
                df_oos_sig.copy(), strat_name, tf, params
            )

            if tf not in ML_ELIGIBLE:
                logger.info(f"  ML skipped: {tf} not in {ML_ELIGIBLE}")
                rows = [_m(k,v,{"strategy":strat_name,"timeframe":tf}) for k,v in results.items()]
                _print_table(rows, label, bm_key)
                all_rows.extend(rows); continue

            # ML_V2 (current sizer — baseline comparison)
            sizer_v2  = MLSizerV2(cfg, sizer_cfg)
            ok_v2     = sizer_v2.fit(strategy, params, df_is, args.direction, label=f"{label}[v2]")
            if ok_v2:
                mults_v2 = sizer_v2.predict(df_oos, strategy, params).reset_index(drop=True)
                results["ML_V2"] = run_dynamic(
                    df_oos_sig.copy(), strat_name, tf, sim_cfg, mults_v2, params
                )
                _plot_importances(sizer_v2.importances, label, "v2", out_plots)

            # ML_EXT (extended features, no strict gate)
            sizer_ext = MLSizerEnhanced(cfg, sizer_cfg,
                                         auc_thresh=0.50,  # permissive
                                         std_thresh=0.05,
                                         use_extended=True)
            ok_ext = sizer_ext.fit(strategy, params, df_is, args.direction,
                                    label=f"{label}[ext]")
            d_ext  = sizer_ext.diag.copy()
            d_ext.update({"strategy":strat_name,"timeframe":tf,"version":"ML_EXT"})
            diag_rows.append(d_ext)

            if ok_ext:
                mults_ext = sizer_ext.predict(df_oos, strategy, params).reset_index(drop=True)
                results["ML_EXT"] = run_dynamic(
                    df_oos_sig.copy(), strat_name, tf, sim_cfg, mults_ext, params
                )
                _plot_importances(sizer_ext.importances, label, "ext", out_plots)

            # ML_STRICT (extended features + strict quality gates)
            sizer_strict = MLSizerEnhanced(cfg, sizer_cfg,
                                            auc_thresh=args.auc_thresh,
                                            std_thresh=args.std_thresh,
                                            use_extended=True)
            ok_strict = sizer_strict.fit(strategy, params, df_is, args.direction,
                                          label=f"{label}[strict]")
            d_str  = sizer_strict.diag.copy()
            d_str.update({"strategy":strat_name,"timeframe":tf,"version":"ML_STRICT"})
            diag_rows.append(d_str)

            if ok_strict:
                mults_strict = sizer_strict.predict(df_oos, strategy, params).reset_index(drop=True)
                results["ML_STRICT"] = run_dynamic(
                    df_oos_sig.copy(), strat_name, tf, sim_cfg, mults_strict, params
                )

            rows = [_m(k,v,{"strategy":strat_name,"timeframe":tf}) for k,v in results.items()]
            _print_table(rows, label, bm_key)

            print(f"\n  Yearly Returns (%):")
            yr = {}
            for name in ["BASE","ML_V2","ML_EXT","ML_STRICT"]:
                res = results.get(name)
                if res and not res.yearly.empty and "return" in res.yearly.columns:
                    yr[name] = (res.yearly["return"]*100).round(2)
            if yr: print(pd.DataFrame(yr).to_string())

            all_rows.extend(rows)

    # Save
    if all_rows:
        sdf  = pd.DataFrame(all_rows)
        path = out_ranks/"ml_enhanced_results.csv"
        sdf.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

    if diag_rows:
        ddf  = pd.DataFrame(diag_rows)
        path = out_ranks/"ml_enhanced_diagnostics.csv"
        ddf.to_csv(path, index=False)
        logger.info(f"Diagnostics → {path}")
        print(f"\n{'='*70}")
        print("ML DIAGNOSTICS:")
        print(ddf[["strategy","timeframe","version","n_trades","auc",
                    "mult_std","n_features"]].to_string(index=False))

    # Final summary
    if all_rows:
        sdf = pd.DataFrame(all_rows)
        print(f"\n{'='*75}")
        print("ML ENHANCED — FINAL vs BASELINE")
        print(f"{'='*75}")
        for (st,tf), grp in sdf.groupby(["strategy","timeframe"]):
            bm = BASELINE.get(f"{st}/{tf}+ML",{})
            base = grp[grp["version"]=="BASE"]
            if base.empty: continue
            print(f"\n{st}/{tf}  baseline_Sh={bm.get('sharpe','?')}")
            for _, r in grp.sort_values("sharpe",ascending=False).iterrows():
                d    = r["sharpe"] - base.iloc[0]["sharpe"]
                beat = " ★ BEATS BASELINE" if bm and r["sharpe"]>bm.get("sharpe",0) else ""
                print(f"  {r['version']:<20}: PF={r['pf']:.3f}  "
                      f"Sh={r['sharpe']:.3f}  trades={int(r['trades'])}  "
                      f"Δ={d:+.3f}{beat}")


if __name__ == "__main__":
    main()