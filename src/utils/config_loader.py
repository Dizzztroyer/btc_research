"""
config_loader.py
────────────────
Loads config/config.yaml and exposes a typed Config dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ── Root of the project (two levels above this file) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class ValidationConfig:
    is_ratio:       float = 0.60
    oos_ratio:      float = 0.20
    test_ratio:     float = 0.20
    wf_windows:     int   = 5
    min_trades:     int   = 30
    min_profit_factor: float = 1.0
    stress_fee_mult:   float = 2.0
    stress_slip_mult:  float = 2.0


@dataclass
class OptimizationConfig:
    method:       str       = "grid"
    random_n:     int       = 200
    n_jobs:       int       = 1
    ema_fast:     List[int] = field(default_factory=lambda: [5, 8, 13, 21])
    ema_slow:     List[int] = field(default_factory=lambda: [21, 34, 55, 89])
    sma_lengths:  List[int] = field(default_factory=lambda: [20, 50, 100, 200])
    rsi_length:   List[int] = field(default_factory=lambda: [7, 9, 14, 21])
    rsi_ob:       List[int] = field(default_factory=lambda: [65, 70, 75, 80])
    rsi_os:       List[int] = field(default_factory=lambda: [20, 25, 30, 35])
    atr_length:   List[int] = field(default_factory=lambda: [7, 10, 14, 21])
    atr_sl_mult:  List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])
    atr_tp_mult:  List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0, 4.0])
    bb_length:    List[int] = field(default_factory=lambda: [14, 20, 30])
    bb_std:       List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    donchian_len: List[int] = field(default_factory=lambda: [10, 20, 40, 55])
    adx_length:   List[int] = field(default_factory=lambda: [10, 14, 20])
    adx_threshold:List[int] = field(default_factory=lambda: [20, 25, 30])
    vol_lookback: List[int] = field(default_factory=lambda: [10, 20, 40])
    squeeze_len:  List[int] = field(default_factory=lambda: [10, 20])
    swing_lookback:List[int]= field(default_factory=lambda: [5, 10, 20])
    sl_pct:       List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.03])
    tp_pct:       List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.07])
    trail_mult:   List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])


@dataclass
class ReportingConfig:
    output_dir:         str  = "outputs"
    top_n:              int  = 20
    plot_equity_curves: bool = True
    plot_heatmaps:      bool = True
    html_report:        bool = True


@dataclass
class Config:
    # Data
    exchange:            str  = "binance"
    symbol:              str  = "BTC/USDT"
    symbol_file:         str  = "BTCUSDT"
    start_date:          str  = "2018-01-01"
    end_date:            Optional[str] = None
    enabled_timeframes:  Optional[List[str]] = None

    # Directories
    data_dir:            str = "data"
    raw_subdir:          str = "raw"
    features_subdir:     str = "features"
    metadata_subdir:     str = "metadata"

    # Simulation
    fees:            float = 0.00075
    slippage:        float = 0.0003
    leverage:        float = 1.0
    risk_per_trade:  float = 0.01

    # Sub-configs
    optimization:  OptimizationConfig  = field(default_factory=OptimizationConfig)
    validation:    ValidationConfig    = field(default_factory=ValidationConfig)
    reporting:     ReportingConfig     = field(default_factory=ReportingConfig)

    # Enabled strategy families
    enabled_strategies: List[str] = field(
        default_factory=lambda: [
            "trend", "mean_reversion", "breakout", "structure", "regime", "ensemble"
        ]
    )

    # ── Derived paths (populated in __post_init__) ────────────────────────────
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    raw_dir:      Path = field(init=False)
    features_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    output_dir:   Path = field(init=False)

    def __post_init__(self) -> None:
        base = self.project_root / self.data_dir
        self.raw_dir      = base / self.raw_subdir      / self.symbol_file
        self.features_dir = base / self.features_subdir / self.symbol_file
        self.metadata_dir = base / self.metadata_subdir
        self.output_dir   = self.project_root / self.reporting.output_dir
        # Create folders
        for d in [self.raw_dir, self.features_dir, self.metadata_dir,
                  self.output_dir / "reports",
                  self.output_dir / "rankings",
                  self.output_dir / "plots",
                  self.output_dir / "logs"]:
            d.mkdir(parents=True, exist_ok=True)


def _nested_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _nested_update(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load and validate configuration from YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    # ── Top-level scalar fields ────────────────────────────────────────────────
    scalar_keys = {
        "exchange", "symbol", "symbol_file", "start_date", "end_date",
        "enabled_timeframes", "data_dir", "raw_subdir", "features_subdir",
        "metadata_subdir", "fees", "slippage", "leverage", "risk_per_trade",
        "enabled_strategies",
    }
    kwargs: dict = {}
    for key in scalar_keys:
        if key in raw:
            kwargs[key] = raw[key]

    # ── Sub-config: optimization ───────────────────────────────────────────────
    opt_raw = raw.get("optimization", {})
    opt_obj = OptimizationConfig()
    for k, v in opt_raw.items():
        if hasattr(opt_obj, k):
            setattr(opt_obj, k, v)
    kwargs["optimization"] = opt_obj

    # ── Sub-config: validation ─────────────────────────────────────────────────
    val_raw = raw.get("validation", {})
    val_obj = ValidationConfig()
    for k, v in val_raw.items():
        if hasattr(val_obj, k):
            setattr(val_obj, k, v)
    kwargs["validation"] = val_obj

    # ── Sub-config: reporting ──────────────────────────────────────────────────
    rep_raw = raw.get("reporting", {})
    rep_obj = ReportingConfig()
    for k, v in rep_raw.items():
        if hasattr(rep_obj, k):
            setattr(rep_obj, k, v)
    # Allow output_dir override from reporting section
    if "output_dir" in rep_raw:
        kwargs.setdefault("reporting", rep_obj)
    kwargs["reporting"] = rep_obj

    return Config(**kwargs)
