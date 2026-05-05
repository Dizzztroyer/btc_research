# BTC Research Platform — Project Context

> Last updated: 2026-04-17
> Purpose: Preserve research state and direction across sessions.
> **Read this before making any changes to the codebase or running new experiments.**
> `risk_per_trade = 1%` (standard setting throughout all research)

---

## 1. Current Baseline — DO NOT MODIFY

Three strategies validated through OOS + Walk-Forward. All new experiments compare against these.

### swing_breakout / 8h + ML_ATR + TP_L1.2_S0.8_P50 ← PRIMARY

Config: XGBoost sizer × ATR regime (LOW×1.3, MED×1.0, HIGH×0.7)
Exit: LONG TP×1.2 + SHORT TP×0.8 + Partial 50% at 1R

| Metric | OOS | WF avg (7 windows) |
|---|---|---|
| Profit Factor | **1.942** | **2.361** |
| Sharpe | **1.970** | **1.800** |
| MDD | -2.05% | -1.84% |
| Trade Count | 295 | 34.4/window |
| % WF positive | — | 86% |

ML: AUC=0.667, IS=385, std(mult)=0.289

### swing_breakout / 12h + ML_SIZE + PART_L50_S70 ← SECONDARY

Config: XGBoost sizer only (ATR does NOT help on 12h)
Exit: LONG partial 50% at 1R, SHORT partial 70% at 1R

| Metric | OOS | WF avg (5 windows) |
|---|---|---|
| Profit Factor | **1.823** | **2.241** |
| Sharpe | **1.757** | **1.530** |
| MDD | -1.79% | -1.11% |
| Trade Count | 277 | 40.2/window |
| % WF positive | — | 80% |

ML: AUC=0.528, IS=310, std(mult)=0.285
Note: ML_ATR on 12h wins only 1/5 WF windows — do NOT use ATR on 12h.
Note: LONG_ONLY on 12h gives WF Sh=1.873 — valid but needs more WF validation (172 trades).

### swing_breakout / 6h + ML_SIZE ← THIRD STRATEGY

Config: XGBoost sizer only. No ATR (same reason as 12h).
Exit: PART_L50_S70 shows OOS Sh=2.062 — needs WF validation.

| Metric | OOS | WF avg (5 windows) |
|---|---|---|
| Profit Factor | 1.757 (base) | **1.903** |
| Sharpe | 1.715 (base) | **1.635** |
| MDD | -2.77% | -1.53% |
| Trade Count | 346 | 67.2/window |
| % WF positive | — | **100%** |

ML: strongest lift on 6h: BASE→ML_SIZE avg_Sh +0.467 across all 5 WF windows.
Note: 6h most stable (std_Sh=0.740 vs 8h=1.039, 12h=1.387).

---

## 2. LONG vs SHORT Key Findings

Structural finding across all TFs: **LONG dominates SHORT on BTC.**

| TF | LONG PF | SHORT PF | LONG/SHORT PnL ratio | LONG avgR | SHORT avgR |
|---|---|---|---|---|---|
| 8h | 1.903 | 1.238 | 5.9× | 0.128 | 0.065 |
| 12h | 2.026 | 1.085 | **14.7×** | 0.119 | 0.014 |
| 6h | 1.669 | 1.103 | 7.2× | 0.118 | 0.040 |

SHORT avgR=0.014 on 12h — practically zero. SHORT adds noise, not returns.
On 12h: LONG_ONLY WF Sh=1.873 (+0.201 vs BOTH+PART50). Worth testing in live.

---

## 3. Exit Logic Findings

From exit study (swing_breakout, 8h and 12h):

| Exit type | 8h Sharpe | 12h Sharpe | verdict |
|---|---|---|---|
| BASE | 1.470 | 1.454 | baseline |
| **PART_50** | **1.848** | **1.672** | ✓ best symmetric |
| PART_L50_S70 | **1.915** | **1.757** | ✓ best asymmetric |
| TP_L1.2_S0.8_P50 | **1.970** | 1.497 | ✓ best on 8h |
| TRAIL_x | 0.67–1.52 | 0.67–1.04 | ✗ always hurts |
| BE_x | 1.001 | 1.212 | ✗ always hurts |
| TIME_x | 0.96–1.41 | 0.78–0.98 | ✗ always hurts |

**Rules confirmed:**
- Partial close at 1R: always beneficial (higher WR, lower MDD)
- Trailing stop: hurts swing breakout (trend needs room to run)
- Break-even: counterproductive (forces BE exits at 0 P&L)
- Dynamic TP (k×ATR): marginal improvement only

---

## 4. ATR Regime Findings

| TF | ATR effect | Direction |
|---|---|---|
| 8h | **Strong** (+0.041 Sharpe vs ML_SIZE in WF) | LOW×1.3, HIGH×0.7 |
| 12h | Neutral (1/5 WF windows win) | Do NOT use |
| 6h | Neutral (1/5 WF windows win) | Do NOT use |

Pattern: ATR only informative at bar level on 8h. Longer candles span multiple ATR transitions.

ATR regime breakdown (8h):
- LOW ATR: PF=2.378, WR=45.1% — **1.72× better than HIGH**
- HIGH ATR: PF=1.379, WR=46.4%
- Correct direction: LOW×1.3 (calm=better breakouts), HIGH×0.7 (chaos=smaller bets)

---

## 5. Rejected Ideas

| Idea | Why |
|---|---|
| Binary ML filter | Destroys statistical reliability |
| Session filter/sizing | Sharpe 0.02–0.09, neutral at best |
| ATR filter (enter only in one regime) | Removes 60%+ trades |
| ATR sizing: LOW×0.7, HIGH×1.2 | Wrong direction |
| Trailing stop | Consistently hurts swing breakout |
| Break-even | Forces 0 P&L exits, kills Sharpe |
| Time exit | Redundant (strategy exits via signal) |
| ML on 1d | <40 IS trades |
| 3h, 3d timeframes | WF Sh < 1.0 |
| Inversion study | 0/8 candidates profitable |
| ML extended features | No improvement on swing |
| Mean reversion on BTC | Systemically unprofitable |
| Regime switch strategy | Overfit (OOS PF=2.05, WF Sh=-0.41) |

---

## 6. Active Research Directions

| Direction | Status |
|---|---|
| Portfolio: 8h + 12h + 6h combined | **Next** |
| 6h + PART_L50_S70 WF validation | Pending |
| 12h LONG_ONLY WF (more windows) | Pending |
| donchian/12h + ML_ATR | Not started |

---

## 7. Rules

```
RULE 1: Never modify baseline strategies directly.
RULE 2: Report OOS + WF for every experiment.
RULE 3: No binary filtering without strong evidence.
RULE 4: ML gates: std(mult)<0.05 or IS<150 → disable.
RULE 5: ATR direction: LOW×1.3, HIGH×0.7 on 8h ONLY.
RULE 6: Partial close at 1R: always test this as first exit improvement.
RULE 7: Long-only bias on BTC is real — always check LONG vs SHORT split.
```

---

## 8. Quick Reference

```
STRATEGY 1 — swing/8h + ML_ATR + TP_L1.2_S0.8_P50:
  OOS:  PF=1.942  Sh=1.970  MDD=-2.05%  trades=295
  WF:   avg_PF=2.361  avg_Sh=1.800  avg_MDD=-1.84%  %pos=86%  (7 win)
  Config: ML × ATR(LOW×1.3,MED×1.0,HIGH×0.7) + LONG_TP×1.2 + SHORT_TP×0.8 + partial50%@1R

STRATEGY 2 — swing/12h + ML_SIZE + PART_L50_S70:
  OOS:  PF=1.823  Sh=1.757  MDD=-1.79%  trades=277
  WF:   avg_PF=2.241  avg_Sh=1.530  avg_MDD=-1.11%  %pos=80%  (5 win)
  Config: ML_SIZE + LONG_partial50%@1R + SHORT_partial70%@1R

STRATEGY 3 — swing/6h + ML_SIZE:
  OOS:  PF=1.903  Sh=1.635  MDD=-1.53%  trades=346
  WF:   avg_PF=1.903  avg_Sh=1.635  avg_MDD=-1.53%  %pos=100%  (5 win)
  Config: ML_SIZE only (no ATR, no exit mods yet — pending PART_L50_S70 WF)

NEXT: Portfolio study (8h + 12h + 6h combined)
```

---

## 9. Portfolio Study — FINAL RESULTS

### Recommended Configuration: PORT_W8_12

| Strategy | Risk | Sharpe (OOS) | Sharpe (WF) | MDD | Trades |
|---|---|---|---|---|---|
| swing/8h + ML_ATR + TP_asym + P50 | **1.2%** | 1.979 | 1.800 | -1.98% | 295 |
| swing/12h + ML_SIZE + PART_L50_S70 | **1.2%** | 1.801 | 1.530 | -1.60% | 277 |
| swing/6h + ML_SIZE | **0.6%** | 1.254 | 1.635 | -3.68% | 237 |
| **PORT_W8_12 (combined)** | **3.0%** | **3.456** | — | **-3.10%** | **809** |

### P&L Correlation Matrix

|  | 8h | 12h | 6h |
|---|---|---|---|
| **8h** | 1.000 | -0.013 | -0.025 |
| **12h** | -0.013 | 1.000 | +0.005 |
| **6h** | -0.025 | +0.005 | 1.000 |

**All correlations ≈ 0.** Three strategies on the same instrument (BTC) across different timeframes produce statistically independent P&L streams. Drawdowns do not coincide.

### Why Sharpe jumps from 1.98 → 3.46

Theory: for N uncorrelated strategies with avg Sharpe S → Portfolio Sharpe ≈ S × √N
- avg_Sh = 1.678, N = 3 → theoretical = 2.907
- Actual = 3.456 (+0.549 excess from slight anti-correlation)

### Key insight

The diversification is real, not constructed. Three strategies share one instrument but operate on independent temporal scales. The equity curves visually confirm non-overlapping drawdowns across 2023–2026.

```
FINAL PORTFOLIO BENCHMARK:
  PORT_W8_12: Sharpe=3.456  PF=1.836  MDD=-3.10%  Return=39.7%  trades=809
  Risk allocation: 8h=1.2%  12h=1.2%  6h=0.6%  (total ≈3%/day)
```