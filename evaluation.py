"""
evaluation.py  —  Multi-model evaluation
==========================================
Auto-discovers all trained models (ppo_sol_perp_w{W}_td{D}.zip),
evaluates each across 5 market periods, then prints a master comparison
grid so you can see which (window, trade_duration) combination works best.

Per model per period:
  - Trade chart (price + position + equity + drawdown)
  - Behaviour chart (PnL decomposition, Sharpe, trade R-multiples, durations)
  - Full console metrics

Master grid (all models × all periods):
  - Average return
  - Average Sharpe
  - Average max drawdown

Charts are saved as:
  eval_chart_w{W}_td{D}_{period}.png
  eval_behaviour_w{W}_td{D}_{period}.png
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from sol_env import SolPerpEnv


# ── Configuration ─────────────────────────────────────────────────────────────

CSV_PATH   = "SOLUSDT_perpetual_futures_processed_2020-2026.csv"
MODEL_DIR  = Path("ppo_sol_perp")
SEED       = 42
FEE_RATE   = 0.0005      # Binance taker 0.05 %
MAX_LEV    = 3.0         # must match training
MIN_HOLD   = 12          # must match training

# Train / val / test split boundaries
TRAIN_END  = "2023-12-31"   # training data ends here (inclusive)
VAL_END    = "2024-12-31"   # validation period ends here
TEST_START = "2025-01-01"   # true out-of-sample test starts here

PERIODS = [
    # ── In-sample (training data) ─────────────────────────────────────
    ("[IS] Early Bull  (Sep 2020–Jun 2021)",  "2020-09-01", "2021-06-01"),
    ("[IS] Peak & Crash(Jun 2021–Jun 2022)",  "2021-06-01", "2022-06-01"),
    ("[IS] Bear / FTX  (Jun 2022–Jan 2023)",  "2022-06-01", "2023-01-01"),
    ("[IS] Recovery    (Jan 2023–Jan 2024)",  "2023-01-01", "2024-01-01"),
    # ── Validation ────────────────────────────────────────────────────
    ("[VAL] Bull Run   (Jan 2024–Jan 2025)",  "2024-01-01", "2025-01-01"),
    # ── Out-of-sample test ────────────────────────────────────────────
    ("[OOS] 2025–2026  (Jan 2025–Apr 2026)",  "2025-01-01", "2026-04-08"),
]

df_full = pd.read_csv(CSV_PATH, parse_dates=["open_time"])


# ── Baselines ─────────────────────────────────────────────────────────────────

def compute_bnh_baseline(df, eval_start, eval_end):
    """Buy-and-hold: buy at eval_start, hold until eval_end. Returns metrics dict."""
    prices  = df["close"].values.astype(float)
    seg     = prices[eval_start : eval_end + 1]
    if len(seg) < 2:
        return None
    equity  = 1_000.0 * seg / seg[0]
    total_ret = (equity[-1] / equity[0]) - 1.0
    n = len(equity)
    HPY = 8_760
    log_r = np.diff(np.log(equity + 1e-8))
    sharpe = (log_r.mean() / (log_r.std() + 1e-8)) * np.sqrt(HPY)
    peak   = np.maximum.accumulate(equity)
    max_dd = ((peak - equity) / (peak + 1e-8)).max()
    return dict(total_ret=total_ret, sharpe=sharpe, max_dd=max_dd,
                label="Buy-and-Hold")


def compute_sma_crossover_baseline(df, eval_start, eval_end):
    """
    SMA-20 vs SMA-50 crossover (long/flat only, no shorting).
    Signal: long when SMA-20 > SMA-50, flat otherwise.
    Uses pre-computed sma_20 column; SMA-50 computed on the fly.
    """
    prices = df["close"].values.astype(float)
    sma20  = df["sma_20"].values.astype(float)
    sma50  = pd.Series(prices).rolling(50).mean().values

    equity = 1_000.0
    eq_arr = []
    HPY    = 8_760

    for i in range(eval_start, eval_end):
        p_ret    = (prices[i + 1] - prices[i]) / (prices[i] + 1e-8)
        in_trade = (not np.isnan(sma50[i])) and (sma20[i] > sma50[i])
        if in_trade:
            fee    = FEE_RATE if (i == eval_start or not (
                (not np.isnan(sma50[i-1])) and (sma20[i-1] > sma50[i-1])
            )) else 0.0
            equity = equity * (1 + p_ret) - fee * equity
        eq_arr.append(equity)

    if len(eq_arr) < 2:
        return None
    eq_arr = np.array(eq_arr)
    total_ret = (eq_arr[-1] / eq_arr[0]) - 1.0
    log_r = np.diff(np.log(eq_arr + 1e-8))
    sharpe = (log_r.mean() / (log_r.std() + 1e-8)) * np.sqrt(HPY)
    peak   = np.maximum.accumulate(eq_arr)
    max_dd = ((peak - eq_arr) / (peak + 1e-8)).max()
    return dict(total_ret=total_ret, sharpe=sharpe, max_dd=max_dd,
                label="SMA Crossover (20/50)")


# ── Model discovery ────────────────────────────────────────────────────────────

def discover_models():
    """Return sorted list of (window, duration) for every trained model found."""
    pattern = re.compile(r"ppo_sol_perp_w(\d+)_td(\d+)\.zip$")
    found   = []
    for p in sorted(MODEL_DIR.glob("ppo_sol_perp_w*_td*.zip")):
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    return sorted(found)


def model_files(window, duration):
    tag = f"w{window}_td{duration}"
    return (
        str(MODEL_DIR / f"ppo_sol_perp_{tag}"),
        str(MODEL_DIR / f"vec_normalize_{tag}.pkl"),
    )


# ── Period index lookup ────────────────────────────────────────────────────────

def get_period_indices(df, start_date, end_date, window):
    mask = (df["open_time"] >= start_date) & (df["open_time"] < end_date)
    rows = df.index[mask]
    if len(rows) < window + 10:
        raise ValueError(f"Too few rows ({len(rows)}) for {start_date}–{end_date}")
    return int(max(rows[0], window)), int(rows[-1])


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(model, vecnorm_path, window, trade_dur, eval_start, eval_end):
    """Run one evaluation episode. Returns step-level and trade-level histories."""

    def make_env():
        return SolPerpEnv(
            csv_path=CSV_PATH, window_size=window,
            fee_rate=FEE_RATE, max_leverage=MAX_LEV,
            initial_equity=1_000.0,
            max_trade_duration=trade_dur,
            min_hold_steps=MIN_HOLD,
            eval_start=eval_start, eval_end=eval_end,
        )

    set_random_seed(SEED)
    env = DummyVecEnv([make_env])

    if Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
        env.training    = False
        env.norm_reward = False
    else:
        print(f"    [WARN] {vecnorm_path} not found — no obs normalisation")

    obs = env.reset()

    # Step-level histories
    prices, positions, equity   = [], [], []
    pnl_h, fee_h, fund_h, dd_h = [], [], [], []
    regime_h                    = []

    # Trade-level history
    trades        = []
    prev_state    = "FLAT"
    open_trade    = {}
    open_eq       = 1_000.0

    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_arr, infos = env.step(action)
        done = bool(done_arr[0])
        info = infos[0]

        prices.append(info["price"])
        positions.append(info["position"])
        equity.append(info["equity"])
        pnl_h.append(info["pnl"])
        fee_h.append(info["trade_fee"])
        fund_h.append(info["funding_payment"])
        dd_h.append(info["drawdown"])
        regime_h.append(float(info.get("regime_aligned", False)))

        curr_state = info.get("trade_state", "FLAT")

        # Detect trade open
        if prev_state == "FLAT" and curr_state == "IN_TRADE":
            open_trade = {
                "entry_step":  step,
                "direction":   np.sign(info["position"]),
                "risk_pct":    info.get("trade_risk_pct", 0.0),
                "pos_size":    abs(info["position"]),
                "entry_equity": info["equity"],
            }
            open_eq = info["equity"]

        # Detect trade close
        if info.get("trade_closed", False) and open_trade:
            open_trade["exit_step"]   = step
            open_trade["duration"]    = step - open_trade["entry_step"]
            open_trade["r_multiple"]  = info.get("r_multiple", 0.0)
            open_trade["exit_equity"] = info["equity"]
            open_trade["exit_type"]   = (
                "tp"      if info.get("tp_hit")      else
                "sl"      if info.get("sl_hit")      else
                "timeout"
            )
            trades.append(open_trade)
            open_trade = {}

        prev_state = curr_state
        step      += 1

    timestamps = df_full["open_time"].iloc[eval_start : eval_start + len(equity)].values

    return dict(
        timestamps = timestamps,
        prices     = np.array(prices),
        positions  = np.array(positions),
        equity     = np.array(equity),
        pnl        = np.array(pnl_h),
        fees       = np.array(fee_h),
        funding    = np.array(fund_h),
        drawdown   = np.array(dd_h),
        regime     = np.array(regime_h),
        trades     = trades,
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(h):
    eq   = h["equity"]
    pos  = h["positions"]
    pnl  = h["pnl"]
    tr   = h["trades"]
    n    = len(eq)
    HPY  = 8_760

    total_ret = (eq[-1] / eq[0]) - 1.0
    ann_ret   = (1 + total_ret) ** (HPY / max(n, 1)) - 1

    peak  = np.maximum.accumulate(eq)
    dd_a  = (peak - eq) / (peak + 1e-8)
    max_dd = dd_a.max()

    log_r  = np.diff(np.log(eq + 1e-8))
    sharpe = (log_r.mean() / (log_r.std() + 1e-8)) * np.sqrt(HPY)
    calmar = ann_ret / (max_dd + 1e-8)
    vol    = log_r.std() * np.sqrt(HPY) * 100

    frac_long  = (pos >  0.05).mean()
    frac_short = (pos < -0.05).mean()
    frac_flat  = (np.abs(pos) <= 0.05).mean()

    # ── Trade-level stats ──────────────────────────────────────────────────────
    n_trades   = len(tr)
    if n_trades > 0:
        r_vals      = np.array([t["r_multiple"]  for t in tr])
        dur_vals    = np.array([t["duration"]     for t in tr])
        risk_vals   = np.array([t["risk_pct"]     for t in tr])
        n_tp        = sum(1 for t in tr if t["exit_type"] == "tp")
        n_sl        = sum(1 for t in tr if t["exit_type"] == "sl")
        n_timeout   = sum(1 for t in tr if t["exit_type"] == "timeout")
        trade_wins  = sum(1 for t in tr if t["r_multiple"] > 0)
        trade_wr    = trade_wins / n_trades
        avg_r       = r_vals.mean()
        avg_dur     = dur_vals.mean()
        avg_risk    = risk_vals.mean()
        best_r      = r_vals.max()
        worst_r     = r_vals.min()
    else:
        r_vals = dur_vals = risk_vals = np.array([])
        n_tp = n_sl = n_timeout = trade_wins = 0
        trade_wr = avg_r = avg_dur = avg_risk = best_r = worst_r = 0.0

    trades_per_day = n_trades / (n / 24 + 1e-8)

    # Step-level fees / gross
    gross_pnl        = pnl.sum()
    total_fees       = h["fees"].sum()
    total_funding    = h["funding"].sum()
    fee_drag         = total_fees / (abs(gross_pnl) + 1e-8) * 100

    return dict(
        # Equity
        total_ret=total_ret, ann_ret=ann_ret, max_dd=max_dd,
        sharpe=sharpe, calmar=calmar, vol=vol,
        init_eq=eq[0], final_eq=eq[-1], n_steps=n,
        # Time in market
        frac_long=frac_long, frac_short=frac_short, frac_flat=frac_flat,
        # Trades
        n_trades=n_trades, trade_wr=trade_wr, avg_r=avg_r,
        avg_dur=avg_dur, avg_risk=avg_risk,
        best_r=best_r, worst_r=worst_r,
        n_tp=n_tp, n_sl=n_sl, n_timeout=n_timeout,
        trades_per_day=trades_per_day,
        # Costs
        total_fees=total_fees, total_funding=total_funding,
        gross_pnl=gross_pnl, fee_drag=fee_drag,
        pct_regime=h["regime"].mean(),
        # Raw arrays for charts
        r_vals=r_vals, dur_vals=dur_vals, risk_vals=risk_vals,
    )


# ── Console printer ────────────────────────────────────────────────────────────

def print_metrics(m, label, window, duration):
    W   = 54
    sep = "  " + "─" * W
    print(sep)
    print(f"  {'PERFORMANCE — ' + label:^{W}}")
    print(sep)
    print(f"  {'Initial equity':<34} ${m['init_eq']:>8.2f}")
    print(f"  {'Final equity':<34} ${m['final_eq']:>8.2f}")
    print(f"  {'Total return':<34} {m['total_ret']*100:>+8.2f}%")
    print(f"  {'Annualised return':<34} {m['ann_ret']*100:>+8.2f}%")
    print(f"  {'Annualised volatility':<34} {m['vol']:>8.2f}%")
    print(f"  {'Max drawdown':<34} {m['max_dd']*100:>8.2f}%")
    print(f"  {'Sharpe (annualised)':<34} {m['sharpe']:>8.3f}")
    print(f"  {'Calmar ratio':<34} {m['calmar']:>8.3f}")
    print(sep)
    print(f"  {'Episode length (hours)':<34} {m['n_steps']:>8d}")
    print(f"  {'Time long':<34} {m['frac_long']*100:>8.1f}%")
    print(f"  {'Time short':<34} {m['frac_short']*100:>8.1f}%")
    print(f"  {'Time flat (waiting)':<34} {m['frac_flat']*100:>8.1f}%")
    print(f"  {'Regime-aligned steps':<34} {m['pct_regime']*100:>8.1f}%")
    print(sep)
    print(f"  {'Completed trades':<34} {m['n_trades']:>8d}")
    print(f"  {'Trades per day':<34} {m['trades_per_day']:>8.2f}")
    print(f"  {'Trade win rate':<34} {m['trade_wr']*100:>8.1f}%")
    print(f"  {'Avg R-multiple per trade':<34} {m['avg_r']:>+8.3f}")
    print(f"  {'Best trade (R)':<34} {m['best_r']:>+8.3f}")
    print(f"  {'Worst trade (R)':<34} {m['worst_r']:>+8.3f}")
    print(f"  {'Avg trade duration (h)':<34} {m['avg_dur']:>8.1f}")
    print(f"  {'Avg risk per trade (%)':<34} {m['avg_risk']:>8.2f}%")
    print(f"  {'TP exits':<34} {m['n_tp']:>8d}")
    print(f"  {'SL exits':<34} {m['n_sl']:>8d}")
    print(f"  {'Timeout exits':<34} {m['n_timeout']:>8d}")
    print(sep)
    print(f"  {'Total fees paid':<34} ${m['total_fees']:>8.2f}")
    print(f"  {'Total funding paid':<34} ${m['total_funding']:>8.2f}")
    print(f"  {'Gross PnL (before fees)':<34} ${m['gross_pnl']:>8.2f}")
    print(f"  {'Fee drag (% of gross PnL)':<34} {m['fee_drag']:>8.1f}%")
    print(sep)

    dsep = "  " + "═" * W
    print(dsep)
    print(f"  {'DIAGNOSTICS':^{W}}")
    print(dsep)

    # Committed position: overtrading is structurally prevented.
    # Diagnose trade quality instead.
    if m['n_trades'] == 0:
        print(f"  ⚠  NO TRADES — agent stayed flat the entire period")
    else:
        status = "✓" if m['trade_wr'] >= 0.50 else "⚠"
        print(f"  {status}  Trade win rate {m['trade_wr']*100:.0f}%  "
              f"({'good' if m['trade_wr'] >= 0.50 else 'needs > 50%'})")

        status = "✓" if m['avg_r'] >= 0.5 else "⚠"
        print(f"  {status}  Avg R = {m['avg_r']:+.2f}  "
              f"({'positive edge' if m['avg_r'] >= 0.5 else 'edge not yet positive'})")

        status = "✓" if m['fee_drag'] < 30 else "⚠"
        print(f"  {status}  Fee drag {m['fee_drag']:.1f}%  "
              f"({'acceptable' if m['fee_drag'] < 30 else 'should be < 30%'})")

        status = "✓" if m['frac_flat'] >= 0.20 else "⚠"
        print(f"  {status}  Flat {m['frac_flat']*100:.0f}% of time  "
              f"({'agent waits for opportunities' if m['frac_flat'] >= 0.20 else 'agent rarely waits — may be forcing trades'})")

        if m['max_dd'] > 0.40:
            print(f"  ⚠  Max drawdown {m['max_dd']*100:.1f}% is high — risk sizing needs work")
        else:
            print(f"  ✓  Max drawdown {m['max_dd']*100:.1f}% is controlled")

    print(dsep)


# ── Trade chart ────────────────────────────────────────────────────────────────

def plot_episode(h, m, label, window, duration, save_path):
    prices    = h["prices"]
    positions = h["positions"]
    equity    = h["equity"]
    dd_pct    = h["drawdown"] * 100
    ts        = h["timestamps"]
    x         = np.arange(len(prices))
    peak_eq   = np.maximum.accumulate(equity)

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#f8f9fa")

    title = (
        f"w={window}h / td={duration}h  |  {label}\n"
        f"Return: {m['total_ret']*100:+.1f}%   "
        f"Max DD: {m['max_dd']*100:.1f}%   "
        f"Sharpe: {m['sharpe']:.2f}   "
        f"Trades: {m['n_trades']}   "
        f"Win rate: {m['trade_wr']*100:.0f}%   "
        f"Avg R: {m['avg_r']:+.2f}"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 2, 1],
                           hspace=0.08, left=0.07, right=0.97,
                           top=0.93, bottom=0.07)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(color="#e0e0e0", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    ax1, ax2, ax3, ax4 = axes

    p_min, p_max = prices.min(), prices.max()
    ax1.fill_between(x, p_min, p_max, where=positions >  0.05,
                     color="#00c853", alpha=0.14, label="Long")
    ax1.fill_between(x, p_min, p_max, where=positions < -0.05,
                     color="#d50000", alpha=0.14, label="Short")
    ax1.plot(x, prices, color="#1a1a2e", linewidth=0.9, zorder=3)
    ax1.set_ylabel("SOL Price (USDT)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    ax2.fill_between(x, 0, positions, where=positions >= 0, color="#00c853", alpha=0.8)
    ax2.fill_between(x, 0, positions, where=positions <  0, color="#d50000", alpha=0.8)
    ax2.axhline(0, color="#555555", linewidth=0.7, zorder=4)
    ax2.set_ylabel("Position", fontsize=9)
    ax2.set_ylim(-MAX_LEV * 1.15, MAX_LEV * 1.15)

    ax3.fill_between(x, equity, peak_eq, color="#d50000", alpha=0.18, label="Drawdown")
    ax3.plot(x, peak_eq, color="#9e9e9e", linewidth=0.5, linestyle="--", zorder=2)
    ax3.plot(x, equity,  color="#1565c0", linewidth=1.1, zorder=3, label="Equity")
    ax3.axhline(equity[0], color="#9e9e9e", linewidth=0.8, linestyle=":", zorder=1)
    ax3.set_ylabel("Equity ($)", fontsize=9)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    ax4.fill_between(x, 0, -dd_pct, color="#d50000", alpha=0.55)
    ax4.set_ylabel("Drawdown (%)", fontsize=9)
    ax4.set_xlabel("Time", fontsize=9)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    n_ticks = min(10, len(x))
    tick_pos = np.linspace(0, len(x) - 1, n_ticks, dtype=int)
    ax4.set_xticks(tick_pos)
    ax4.set_xticklabels([str(ts[i])[:10] for i in tick_pos],
                        rotation=25, fontsize=7, ha="right")
    for ax in (ax1, ax2, ax3):
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart → {save_path}")


# ── Behaviour chart (trade-focused) ────────────────────────────────────────────

def plot_behaviour(h, m, label, window, duration, save_path):
    eq      = h["equity"]
    pnl_arr = h["pnl"]
    fee_arr = h["fees"]
    n       = len(eq)
    x       = np.arange(n)

    cum_gross = np.cumsum(pnl_arr)
    cum_fees  = np.cumsum(fee_arr)
    net_delta = eq - eq[0]

    log_ret = pd.Series(np.diff(np.log(eq + 1e-8)))
    roll_s  = (log_ret.rolling(168).mean() /
               (log_ret.rolling(168).std() + 1e-8) * np.sqrt(8_760))
    rs_full = np.full(n, np.nan)
    rs_full[1:] = roll_s.values

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Behaviour Analysis — w={window}h / td={duration}h  |  {label}\n"
        f"Trades: {m['n_trades']}   Win rate: {m['trade_wr']*100:.0f}%   "
        f"Avg R: {m['avg_r']:+.2f}   Avg duration: {m['avg_dur']:.0f}h   "
        f"Avg risk: {m['avg_risk']:.1f}%",
        fontsize=11, fontweight="bold", y=0.99,
    )
    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e0e0e0", linewidth=0.5, alpha=0.7)

    # ── Panel TL: PnL Decomposition ───────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, cum_gross, color="#1565c0", linewidth=1.2, label="Gross PnL")
    ax.plot(x, net_delta, color="#2e7d32", linewidth=1.2, label="Net equity Δ")
    ax.fill_between(x, 0, -cum_fees, color="#d50000", alpha=0.35, label="–Fees")
    ax.axhline(0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_title("PnL Decomposition", fontweight="bold", fontsize=10)
    ax.set_ylabel("$ change from start", fontsize=8)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # ── Panel TR: Rolling 7-day Sharpe ────────────────────────────────────────
    ax = axes[0, 1]
    valid = ~np.isnan(rs_full)
    xv, sv = x[valid], rs_full[valid]
    if len(xv) > 0:
        ax.plot(xv, sv, color="#1565c0", linewidth=1.0)
        ax.fill_between(xv, 0, sv, where=sv >= 0, color="#2e7d32", alpha=0.25)
        ax.fill_between(xv, 0, sv, where=sv <  0, color="#d50000", alpha=0.25)
    ax.axhline(0,   color="#555555", linewidth=0.8, linestyle="--")
    ax.axhline(1.5, color="#1565c0", linewidth=0.8, linestyle="--", label="Target 1.5")
    ax.set_title("Rolling 168h Sharpe", fontweight="bold", fontsize=10)
    ax.set_ylabel("Sharpe (annualised)", fontsize=8)
    ax.legend(fontsize=8)

    # ── Panel BL: R-multiple distribution ────────────────────────────────────
    ax = axes[1, 0]
    if len(m["r_vals"]) > 0:
        rv = m["r_vals"]
        bins = np.linspace(min(-3.5, rv.min() - 0.5), max(5.5, rv.max() + 0.5), 30)
        ax.hist(rv[rv >= 0], bins=bins, color="#2e7d32", alpha=0.75,
                label=f"Wins ({(rv>=0).sum()})")
        ax.hist(rv[rv <  0], bins=bins, color="#d50000", alpha=0.75,
                label=f"Losses ({(rv<0).sum()})")
        ax.axvline(0,         color="#555", linewidth=1.0, linestyle="--")
        ax.axvline(rv.mean(), color="#ff6f00", linewidth=1.5,
                   linestyle="-", label=f"Avg R = {rv.mean():+.2f}")
    ax.set_title("Trade R-Multiple Distribution", fontweight="bold", fontsize=10)
    ax.set_xlabel("R multiple (1R = risked amount)", fontsize=8)
    ax.set_ylabel("# trades", fontsize=8)
    ax.legend(fontsize=8)

    # ── Panel BR: Trade duration distribution ─────────────────────────────────
    ax = axes[1, 1]
    if len(m["dur_vals"]) > 0:
        dv = m["dur_vals"]
        ax.hist(dv, bins=20, color="#1565c0", alpha=0.75, edgecolor="white",
                linewidth=0.4)
        ax.axvline(dv.mean(), color="#ff6f00", linewidth=1.5,
                   linestyle="-", label=f"Avg = {dv.mean():.0f}h")
    ax.set_title("Trade Duration Distribution", fontweight="bold", fontsize=10)
    ax.set_xlabel("Trade duration (hours)", fontsize=8)
    ax.set_ylabel("# trades", fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Behaviour → {save_path}")


# ── Master comparison grid ─────────────────────────────────────────────────────

def print_master_grid(all_results, models_found):
    """
    all_results: {(window, duration): {period_label: metrics_dict}}
    """
    windows   = sorted(set(w for w, _ in models_found))
    durations = sorted(set(d for _, d in models_found))

    def avg_metric(w, d, key):
        res = all_results.get((w, d), {})
        vals = [m[key] for m in res.values() if m is not None]
        return np.mean(vals) if vals else float("nan")

    def grid_block(title, key, fmt, highlight="max", scale=1.0):
        W = 72
        print(f"\n{'='*W}")
        print(f"  {title}")
        print(f"  (average across {len(PERIODS)} periods)")
        print(f"{'='*W}")

        hdr = f"  {'Window':>8}  " + "  ".join(f"td={d:>3}h" for d in durations)
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        # Collect all values to find best (apply scale for display)
        all_vals = {}
        for w in windows:
            for d in durations:
                if (w, d) in models_found:
                    raw = avg_metric(w, d, key)
                    all_vals[(w, d)] = raw * scale if not np.isnan(raw) else float("nan")

        if highlight == "max":
            best_val = max((v for v in all_vals.values() if not np.isnan(v)),
                           default=float("nan"))
        else:
            best_val = min((v for v in all_vals.values() if not np.isnan(v)),
                           default=float("nan"))

        for w in windows:
            row = f"  w={w:>4}h    "
            for d in durations:
                if (w, d) not in models_found:
                    row += f"   {'—':>8}  "
                    continue
                v = all_vals.get((w, d), float("nan"))
                if np.isnan(v):
                    row += f"   {'n/a':>8}  "
                else:
                    cell = fmt.format(v)
                    star = " ★" if abs(v - best_val) < 1e-9 else "  "
                    row += f"   {cell}{star}  "
            print(row)

        print(f"\n  ★ = best value")

    print("\n\n" + "=" * 72)
    print("  MASTER COMPARISON GRID — ALL MODELS × ALL PERIODS")
    print("=" * 72)

    grid_block("AVG TOTAL RETURN (%)",              "total_ret",
               "{:>+7.1f}%", highlight="max", scale=100.0)
    grid_block("AVG ANNUALISED SHARPE",              "sharpe",
               "{:>+7.3f} ", highlight="max", scale=1.0)
    grid_block("AVG MAX DRAWDOWN (%) — lower better","max_dd",
               "{:>7.1f}% ", highlight="min", scale=100.0)
    grid_block("AVG TRADE WIN RATE (%)",             "trade_wr",
               "{:>+7.1f}%", highlight="max", scale=100.0)
    grid_block("AVG R-MULTIPLE PER TRADE",           "avg_r",
               "{:>+7.3f} ", highlight="max", scale=1.0)

    # Best model summary
    print("\n\n" + "=" * 72)
    print("  TOP 5 MODELS  (by average Sharpe)")
    print("=" * 72)
    scored = []
    for (w, d) in models_found:
        s = avg_metric(w, d, "sharpe")
        r = avg_metric(w, d, "total_ret")
        dd = avg_metric(w, d, "max_dd")
        scored.append((s, w, d, r, dd))
    scored.sort(reverse=True)
    print(f"  {'Model':<18} {'Avg Sharpe':>12} {'Avg Return':>12} {'Avg MaxDD':>12}")
    print("  " + "─" * 56)
    for rank, (s, w, d, r, dd) in enumerate(scored[:5], 1):
        marker = " ★" if rank == 1 else "  "
        print(f"  w={w}h / td={d}h{'':<6} {s:>+11.3f}  {r*100:>+10.1f}%  {dd*100:>10.1f}%{marker}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    models_found = discover_models()

    if not models_found:
        print("No trained models found in ppo_sol_perp/. Run train_solo_ppo.py first.")
        raise SystemExit(1)

    # Chart output directory
    CHART_DIR = Path("evaluation pictures")
    CHART_DIR.mkdir(exist_ok=True)

    print(f"\nFound {len(models_found)} models: "
          + ", ".join(f"w{w}_td{d}" for w, d in models_found))
    print(f"Evaluating across {len(PERIODS)} periods each.\n")

    all_results = {}   # (window, duration) → {period_label: metrics}

    for model_idx, (window, duration) in enumerate(models_found, 1):
        model_name, vecnorm_name = model_files(window, duration)
        tag = f"w{window}_td{duration}"

        print(f"\n{'#'*70}")
        print(f"  MODEL {model_idx}/{len(models_found)}:  {model_name}  "
              f"(window={window}h  trade_dur={duration}h)")
        print(f"{'#'*70}")

        try:
            model = PPO.load(model_name)
        except Exception as e:
            print(f"  [ERROR] Cannot load {model_name}: {e}")
            continue

        period_results = {}

        for period_label, start_date, end_date in PERIODS:
            print(f"\n  ── {period_label} ──")
            try:
                eval_start, eval_end = get_period_indices(
                    df_full, start_date, end_date, window
                )
            except ValueError as e:
                print(f"    [SKIP] {e}")
                continue

            n_days = (eval_end - eval_start) / 24
            print(f"    Rows {eval_start}→{eval_end}  (~{n_days:.0f} days)")

            hist    = run_episode(model, vecnorm_name, window, duration,
                                  eval_start, eval_end)
            metrics = compute_metrics(hist)

            print_metrics(metrics, period_label, window, duration)

            # ── Baseline comparison ───────────────────────────────────────────
            bnh = compute_bnh_baseline(df_full, eval_start, eval_end)
            sma = compute_sma_crossover_baseline(df_full, eval_start, eval_end)
            print(f"  {'':─<54}")
            print(f"  {'BASELINE COMPARISON':^54}")
            print(f"  {'':─<54}")
            print(f"  {'Strategy':<28} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8}")
            print(f"  {'PPO Agent':<28} {metrics['total_ret']*100:>+7.1f}% "
                  f"{metrics['sharpe']:>8.3f} {metrics['max_dd']*100:>7.1f}%")
            for b in [bnh, sma]:
                if b is not None:
                    print(f"  {b['label']:<28} {b['total_ret']*100:>+7.1f}% "
                          f"{b['sharpe']:>8.3f} {b['max_dd']*100:>7.1f}%")
            print(f"  {'':─<54}")

            safe_period = start_date[:7].replace("-", "")
            plot_episode(hist, metrics, period_label, window, duration,
                         str(CHART_DIR / f"eval_chart_{tag}_{safe_period}.png"))
            plot_behaviour(hist, metrics, period_label, window, duration,
                           str(CHART_DIR / f"eval_behaviour_{tag}_{safe_period}.png"))

            period_results[period_label] = metrics
            period_results[period_label]["_bnh"] = bnh
            period_results[period_label]["_sma"] = sma

        all_results[(window, duration)] = period_results

        # Per-model cross-period summary
        if period_results:
            print(f"\n  ── {tag} cross-period summary ──")
            print(f"  {'Period':<36} {'PPO Ret':>9} {'PPO Sh':>8} "
                  f"{'BnH Ret':>9} {'BnH Sh':>8} {'SMA Ret':>9} {'SMA Sh':>8}")
            print("  " + "─" * 100)
            for lbl, m in period_results.items():
                bnh = m.get("_bnh") or {}
                sma = m.get("_sma") or {}
                print(f"  {lbl[:34]:<36}"
                      f" {m['total_ret']*100:>+8.1f}%"
                      f" {m['sharpe']:>8.3f}"
                      f" {bnh.get('total_ret', float('nan'))*100:>+8.1f}%"
                      f" {bnh.get('sharpe', float('nan')):>8.3f}"
                      f" {sma.get('total_ret', float('nan'))*100:>+8.1f}%"
                      f" {sma.get('sharpe', float('nan')):>8.3f}")

    # Master comparison grid across all 16 models
    print_master_grid(all_results, models_found)
