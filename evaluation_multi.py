"""
evaluation_multi.py  —  Multi-asset portfolio model evaluation
==============================================================
Auto-discovers all trained multi-asset models (ppo_multi_perp_w{W}_td{D}.zip),
evaluates each across 6 market periods, then prints a master comparison grid.

Portfolio-level metrics per model per period:
  - Total return, annualised Sharpe, max drawdown, Calmar, fee drag
  - Per-asset time long / short / flat
  - Per-asset inferred trade count, win rate, avg duration
  - Baseline comparison: equal-weight BnH, BTC-only BnH

Charts saved as:
  eval_multi_chart_w{W}_td{D}_{YYYYMM}.png     — portfolio equity + positions
  eval_multi_behaviour_w{W}_td{D}_{YYYYMM}.png — PnL decomp + Sharpe + trade analysis
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
from multi_asset_env import MultiAssetPerpEnv


# ── Configuration ─────────────────────────────────────────────────────────────

CSV_PATHS = [
    "BTCUSDT_perpetual_futures_processed_2020-2026.csv",
    "ETHUSDT_perpetual_futures_processed_2020-2026.csv",
    "SOLUSDT_perpetual_futures_processed_2020-2026.csv",
    "BNBUSDT_perpetual_futures_processed_2020-2026.csv",
]
ASSET_NAMES = ["BTC", "ETH", "SOL", "BNB"]
N_ASSETS    = len(CSV_PATHS)

MODEL_DIR   = Path("ppo_multi_perp")
SEED        = 42
FEE_RATE    = 0.0005
MAX_LEV     = 3.0
MIN_HOLD    = 12
INIT_EQUITY = 1_000.0

PERIODS = [
    ("[IS] Early Bull  (Sep 2020–Jun 2021)",  "2020-09-01", "2021-06-01"),
    ("[IS] Peak & Crash(Jun 2021–Jun 2022)",  "2021-06-01", "2022-06-01"),
    ("[IS] Bear / FTX  (Jun 2022–Jan 2023)",  "2022-06-01", "2023-01-01"),
    ("[IS] Recovery    (Jan 2023–Jan 2024)",  "2023-01-01", "2024-01-01"),
    ("[VAL] Bull Run   (Jan 2024–Jan 2025)",  "2024-01-01", "2025-01-01"),
    ("[OOS] 2025–2026  (Jan 2025–Apr 2026)",  "2025-01-01", "2026-04-08"),
]


# ── Aligned data loading ──────────────────────────────────────────────────────
# Mirror the inner-join alignment that MultiAssetPerpEnv does, so period index
# lookups reference the same timestamps the environment uses.

print("Loading and aligning CSV files…")
_raw_dfs = [pd.read_csv(p, parse_dates=["open_time"]) for p in CSV_PATHS]

_common = set(_raw_dfs[0]["open_time"])
for _df in _raw_dfs[1:]:
    _common &= set(_df["open_time"])
_common_times = sorted(_common)
_common_set   = set(_common_times)

# Aligned dataframes per asset — same rows the env sees
ALIGNED_DFS = []
for _df in _raw_dfs:
    _df = (_df[_df["open_time"].isin(_common_set)]
             .sort_values("open_time")
             .reset_index(drop=True))
    ALIGNED_DFS.append(_df)

ALIGNED_TIMES = pd.Series(_common_times)   # for index lookups
print(f"  Aligned rows: {len(ALIGNED_TIMES):,}  "
      f"({str(ALIGNED_TIMES.iloc[0])[:10]} → {str(ALIGNED_TIMES.iloc[-1])[:10]})\n")


# ── Period index lookup ────────────────────────────────────────────────────────

def get_period_indices(start_date, end_date, window):
    """Return (start_idx, end_idx) into the aligned data for this date range."""
    mask = (ALIGNED_TIMES >= start_date) & (ALIGNED_TIMES < end_date)
    rows = ALIGNED_TIMES.index[mask]
    if len(rows) < window + 10:
        raise ValueError(f"Too few rows ({len(rows)}) for {start_date}–{end_date}")
    return int(max(rows[0], window)), int(rows[-1])


# ── Baselines ─────────────────────────────────────────────────────────────────

def compute_equal_weight_bnh(eval_start, eval_end):
    """Equal-weight buy-and-hold across all 4 assets — no rebalancing."""
    HPY    = 8_760
    equity = INIT_EQUITY
    eq_arr = [equity]

    for step in range(eval_start, eval_end):
        total_ret = 0.0
        for df in ALIGNED_DFS:
            p0 = df["close"].iloc[step]
            p1 = df["close"].iloc[step + 1]
            total_ret += (p1 - p0) / (p0 + 1e-8)
        equity *= (1 + total_ret / N_ASSETS)
        eq_arr.append(equity)

    eq_arr = np.array(eq_arr)
    total_ret = (eq_arr[-1] / eq_arr[0]) - 1.0
    log_r     = np.diff(np.log(eq_arr + 1e-8))
    sharpe    = (log_r.mean() / (log_r.std() + 1e-8)) * np.sqrt(HPY)
    peak      = np.maximum.accumulate(eq_arr)
    max_dd    = ((peak - eq_arr) / (peak + 1e-8)).max()
    return dict(total_ret=total_ret, sharpe=sharpe, max_dd=max_dd,
                label="Equal-Weight BnH", equity=eq_arr)


def compute_btc_bnh(eval_start, eval_end):
    """BTC-only buy-and-hold benchmark."""
    HPY    = 8_760
    prices = ALIGNED_DFS[0]["close"].values.astype(float)  # BTC is index 0
    seg    = prices[eval_start : eval_end + 1]
    equity = INIT_EQUITY * seg / seg[0]
    total_ret = (equity[-1] / equity[0]) - 1.0
    log_r     = np.diff(np.log(equity + 1e-8))
    sharpe    = (log_r.mean() / (log_r.std() + 1e-8)) * np.sqrt(HPY)
    peak      = np.maximum.accumulate(equity)
    max_dd    = ((peak - equity) / (peak + 1e-8)).max()
    return dict(total_ret=total_ret, sharpe=sharpe, max_dd=max_dd,
                label="BTC Buy-and-Hold", equity=equity)


# ── Model discovery ────────────────────────────────────────────────────────────

def discover_models():
    pattern = re.compile(r"ppo_multi_perp_w(\d+)_td(\d+)\.zip$")
    found   = []
    for p in sorted(MODEL_DIR.glob("ppo_multi_perp_w*_td*.zip")):
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    return sorted(found)


def model_files(window, duration):
    tag = f"w{window}_td{duration}"
    return (
        str(MODEL_DIR / f"ppo_multi_perp_{tag}"),
        str(MODEL_DIR / f"vec_normalize_{tag}.pkl"),
    )


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(model, vecnorm_path, window, trade_dur, eval_start, eval_end):
    """
    Run one evaluation episode for the multi-asset environment.
    Returns step-level and inferred trade-level history.
    """

    def make_env():
        return MultiAssetPerpEnv(
            csv_paths       = CSV_PATHS,
            asset_names     = ASSET_NAMES,
            window_size     = window,
            fee_rate        = FEE_RATE,
            max_leverage    = MAX_LEV,
            initial_equity  = INIT_EQUITY,
            max_trade_duration = trade_dur,
            min_hold_steps  = MIN_HOLD,
            eval_start      = eval_start,
            eval_end        = eval_end,
        )

    set_random_seed(SEED)
    env = DummyVecEnv([make_env])

    if Path(vecnorm_path).exists():
        env = VecNormalize.load(vecnorm_path, env)
        env.training    = False
        env.norm_reward = False
    else:
        print(f"    [WARN] {vecnorm_path} not found — running without obs normalisation")

    obs = env.reset()

    # ── Step-level tracking ────────────────────────────────────────────────────
    equity_h   = []
    dd_h       = []
    fee_h      = []
    fund_h     = []
    pnl_h      = []
    # Per-asset positions over time: shape (steps, N_ASSETS)
    positions_h = []
    timestamps  = []

    # ── Per-asset price tracking (for trade R-multiple inference) ─────────────
    asset_prices = [df["close"].values.astype(float) for df in ALIGNED_DFS]

    # ── Inferred trade tracking per asset ─────────────────────────────────────
    # prev_positions[i] = last known position for asset i
    prev_positions = [0.0] * N_ASSETS
    asset_trades   = [[] for _ in range(N_ASSETS)]   # list of trade dicts per asset
    open_trades    = [{}  for _ in range(N_ASSETS)]  # in-progress trade per asset

    done = False
    step_idx = eval_start   # global index into aligned data

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_arr, infos = env.step(action)
        done = bool(done_arr[0])
        info = infos[0]

        eq  = float(info["equity"])
        dd  = float(info["drawdown"])
        pos = list(info["positions"])      # list of N_ASSETS floats

        equity_h.append(eq)
        dd_h.append(dd)
        fee_h.append(float(info["total_fees"]))
        fund_h.append(float(info["total_funding"]))
        pnl_h.append(float(info["total_pnl"]))
        positions_h.append(pos[:])

        # Timestamp for this step
        t_idx = min(step_idx, len(ALIGNED_TIMES) - 1)
        timestamps.append(ALIGNED_TIMES.iloc[t_idx])
        step_idx += 1

        # ── Infer trade opens / closes per asset ──────────────────────────────
        for i in range(N_ASSETS):
            prev = prev_positions[i]
            curr = pos[i]

            was_flat = abs(prev) < 0.01
            is_flat  = abs(curr) < 0.01

            # Trade opened this step
            if was_flat and not is_flat:
                price_idx = min(t_idx, len(asset_prices[i]) - 1)
                open_trades[i] = {
                    "entry_step":  len(equity_h) - 1,
                    "entry_price": asset_prices[i][price_idx],
                    "direction":   1 if curr > 0 else -1,
                    "pos_size":    abs(curr),
                }

            # Trade closed this step
            elif not was_flat and is_flat and open_trades[i]:
                ot = open_trades[i]
                price_idx = min(t_idx, len(asset_prices[i]) - 1)
                exit_price = asset_prices[i][price_idx]
                raw_ret = (
                    (exit_price - ot["entry_price"]) / (ot["entry_price"] + 1e-8)
                ) * ot["direction"]
                duration = (len(equity_h) - 1) - ot["entry_step"]

                asset_trades[i].append({
                    "entry_step":  ot["entry_step"],
                    "exit_step":   len(equity_h) - 1,
                    "duration":    duration,
                    "direction":   ot["direction"],
                    "raw_return":  raw_ret,
                    "won":         raw_ret > 0,
                })
                open_trades[i] = {}

            prev_positions[i] = curr

    return dict(
        timestamps   = np.array(timestamps),
        equity       = np.array(equity_h),
        drawdown     = np.array(dd_h),
        fees         = np.array(fee_h),
        funding      = np.array(fund_h),
        pnl          = np.array(pnl_h),
        positions    = np.array(positions_h),   # (steps, N_ASSETS)
        asset_trades = asset_trades,            # list[N_ASSETS] of trade dicts
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(h):
    eq  = h["equity"]
    pos = h["positions"]   # (steps, N_ASSETS)
    HPY = 8_760
    n   = len(eq)

    total_ret = (eq[-1] / eq[0]) - 1.0
    ann_ret   = (1 + total_ret) ** (HPY / max(n, 1)) - 1

    peak   = np.maximum.accumulate(eq)
    dd_a   = (peak - eq) / (peak + 1e-8)
    max_dd = dd_a.max()

    log_r  = np.diff(np.log(eq + 1e-8))
    sharpe = (log_r.mean() / (log_r.std() + 1e-8)) * np.sqrt(HPY)
    calmar = ann_ret / (max_dd + 1e-8)
    vol    = log_r.std() * np.sqrt(HPY) * 100

    # Gross vs fees
    gross_pnl  = h["pnl"].sum()
    total_fees = h["fees"].sum()
    total_fund = h["funding"].sum()
    fee_drag   = total_fees / (abs(gross_pnl) + 1e-8) * 100

    # Per-asset time in market
    per_asset = []
    for i in range(N_ASSETS):
        col          = pos[:, i]
        frac_long    = (col >  0.05).mean()
        frac_short   = (col < -0.05).mean()
        frac_flat    = (np.abs(col) <= 0.05).mean()
        trades       = h["asset_trades"][i]
        n_trades     = len(trades)
        win_rate     = np.mean([t["won"] for t in trades]) if trades else 0.0
        avg_dur      = np.mean([t["duration"] for t in trades]) if trades else 0.0
        per_asset.append(dict(
            name=ASSET_NAMES[i],
            frac_long=frac_long, frac_short=frac_short, frac_flat=frac_flat,
            n_trades=n_trades, win_rate=win_rate, avg_dur=avg_dur,
        ))

    # Overall portfolio time-in-market
    any_pos = (np.abs(pos).sum(axis=1) > 0.05)
    frac_any = any_pos.mean()

    return dict(
        total_ret=total_ret, ann_ret=ann_ret, max_dd=max_dd,
        sharpe=sharpe, calmar=calmar, vol=vol,
        init_eq=eq[0], final_eq=eq[-1], n_steps=n,
        gross_pnl=gross_pnl, total_fees=total_fees, total_fund=total_fund,
        fee_drag=fee_drag,
        frac_any_position=frac_any,
        per_asset=per_asset,
    )


# ── Console printer ────────────────────────────────────────────────────────────

def print_metrics(m, label, window, duration):
    W   = 56
    sep = "  " + "─" * W
    print(sep)
    print(f"  {'PORTFOLIO PERFORMANCE — ' + label[:28]:^{W}}")
    print(sep)
    print(f"  {'Initial equity':<36} ${m['init_eq']:>8.2f}")
    print(f"  {'Final equity':<36} ${m['final_eq']:>8.2f}")
    print(f"  {'Total return':<36} {m['total_ret']*100:>+8.2f}%")
    print(f"  {'Annualised return':<36} {m['ann_ret']*100:>+8.2f}%")
    print(f"  {'Annualised volatility':<36} {m['vol']:>8.2f}%")
    print(f"  {'Max drawdown':<36} {m['max_dd']*100:>8.2f}%")
    print(f"  {'Sharpe (annualised)':<36} {m['sharpe']:>8.3f}")
    print(f"  {'Calmar ratio':<36} {m['calmar']:>8.3f}")
    print(f"  {'Steps evaluated':<36} {m['n_steps']:>8d}")
    print(sep)
    print(f"  {'Gross PnL':<36} ${m['gross_pnl']:>8.2f}")
    print(f"  {'Total fees paid':<36} ${m['total_fees']:>8.2f}")
    print(f"  {'Total funding paid':<36} ${m['total_fund']:>8.2f}")
    print(f"  {'Fee drag (% of gross PnL)':<36} {m['fee_drag']:>8.1f}%")
    print(sep)
    print(f"  {'PER-ASSET BREAKDOWN':^{W}}")
    print(sep)
    print(f"  {'Asset':<8} {'Long%':>7} {'Short%':>8} {'Flat%':>7} "
          f"{'Trades':>7} {'WinR%':>7} {'AvgDur':>8}")
    print("  " + "─" * 56)
    for a in m["per_asset"]:
        print(f"  {a['name']:<8} {a['frac_long']*100:>6.1f}%"
              f" {a['frac_short']*100:>7.1f}%"
              f" {a['frac_flat']*100:>6.1f}%"
              f" {a['n_trades']:>7d}"
              f" {a['win_rate']*100:>6.1f}%"
              f" {a['avg_dur']:>7.1f}h")
    print(sep)

    dsep = "  " + "═" * W
    print(dsep)
    print(f"  {'DIAGNOSTICS':^{W}}")
    print(dsep)
    all_trades = sum(a["n_trades"] for a in m["per_asset"])
    if all_trades == 0:
        print(f"  ⚠  NO TRADES executed — agent stayed flat all period")
    else:
        status = "✓" if m["sharpe"] > 0 else "⚠"
        print(f"  {status}  Portfolio Sharpe {m['sharpe']:+.3f}  "
              f"({'positive' if m['sharpe'] > 0 else 'negative'})")
        status = "✓" if m["max_dd"] <= 0.35 else "⚠"
        print(f"  {status}  Max drawdown {m['max_dd']*100:.1f}%  "
              f"({'controlled' if m['max_dd'] <= 0.35 else 'above 35% — review sizing'})")
        status = "✓" if m["fee_drag"] < 40 else "⚠"
        print(f"  {status}  Fee drag {m['fee_drag']:.1f}%  "
              f"({'acceptable' if m['fee_drag'] < 40 else 'high — fewer/larger trades needed'})")
        status = "✓" if m["frac_any_position"] <= 0.85 else "⚠"
        print(f"  {status}  Portfolio occupied {m['frac_any_position']*100:.0f}% of steps  "
              f"({'balanced' if m['frac_any_position'] <= 0.85 else 'always in trades — consider selectivity'})")
    print(dsep)


# ── Portfolio chart ────────────────────────────────────────────────────────────

def plot_episode(h, m, bnh_ew, bnh_btc, label, window, duration, save_path):
    equity    = h["equity"]
    positions = h["positions"]   # (steps, N_ASSETS)
    dd_pct    = h["drawdown"] * 100
    ts        = h["timestamps"]
    x         = np.arange(len(equity))
    peak_eq   = np.maximum.accumulate(equity)

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#f8f9fa")

    title = (
        f"Multi-Asset PPO  |  w={window}h / td={duration}h  |  {label}\n"
        f"Portfolio Return: {m['total_ret']*100:+.1f}%   "
        f"Max DD: {m['max_dd']*100:.1f}%   "
        f"Sharpe: {m['sharpe']:.2f}   "
        f"BnH(EW): {bnh_ew['total_ret']*100:+.1f}%   "
        f"BnH(BTC): {bnh_btc['total_ret']*100:+.1f}%"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.99)

    # Layout: equity (big) + 4 asset positions + drawdown
    n_rows = 2 + N_ASSETS   # equity row, N_ASSETS position rows, drawdown row
    height_ratios = [3] + [1] * N_ASSETS + [1]
    gs = gridspec.GridSpec(
        n_rows, 1, height_ratios=height_ratios,
        hspace=0.06, left=0.08, right=0.97, top=0.94, bottom=0.06,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(color="#e0e0e0", linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    ax_eq = axes[0]
    asset_axes = axes[1 : 1 + N_ASSETS]
    ax_dd = axes[-1]

    # ── Equity panel with baselines ────────────────────────────────────────────
    # Rescale BnH baselines to same start point
    x_bnh_ew  = np.linspace(0, len(x) - 1, len(bnh_ew["equity"]))
    x_bnh_btc = np.linspace(0, len(x) - 1, len(bnh_btc["equity"]))
    scale_ew  = equity[0] / (bnh_ew["equity"][0]  + 1e-8)
    scale_btc = equity[0] / (bnh_btc["equity"][0] + 1e-8)

    ax_eq.fill_between(x, equity, peak_eq, color="#d50000", alpha=0.15, label="Drawdown area")
    ax_eq.plot(x, peak_eq,                      color="#9e9e9e", linewidth=0.6,
               linestyle="--", zorder=2)
    ax_eq.plot(x, equity,                        color="#1565c0", linewidth=1.3,
               zorder=3, label="PPO Portfolio")
    ax_eq.plot(x_bnh_ew, bnh_ew["equity"] * scale_ew,
               color="#ff8f00", linewidth=1.0, linestyle="--",
               zorder=2, label="Equal-Weight BnH")
    ax_eq.plot(x_bnh_btc, bnh_btc["equity"] * scale_btc,
               color="#6a1b9a", linewidth=1.0, linestyle=":",
               zorder=2, label="BTC-Only BnH")
    ax_eq.axhline(equity[0], color="#9e9e9e", linewidth=0.7, linestyle=":", zorder=1)
    ax_eq.set_ylabel("Portfolio Equity ($)", fontsize=9)
    ax_eq.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # ── Per-asset position panels ──────────────────────────────────────────────
    colours = ["#1a1a2e", "#1565c0", "#6a1b9a", "#ff8f00"]
    for i, (ax, name, col) in enumerate(zip(asset_axes, ASSET_NAMES, colours)):
        p = positions[:, i]
        ax.fill_between(x, 0, p, where=p >= 0, color="#00c853", alpha=0.75)
        ax.fill_between(x, 0, p, where=p <  0, color="#d50000", alpha=0.75)
        ax.axhline(0, color="#555555", linewidth=0.6, zorder=4)
        ax.set_ylabel(f"{name}", fontsize=8)
        ax.set_ylim(-MAX_LEV * 1.15, MAX_LEV * 1.15)
        ax.tick_params(axis="y", labelsize=7)
        n_t = m["per_asset"][i]["n_trades"]
        ax.text(0.99, 0.85, f"{n_t} trades", transform=ax.transAxes,
                ha="right", fontsize=7, color="#555555")

    # ── Drawdown panel ─────────────────────────────────────────────────────────
    ax_dd.fill_between(x, 0, -dd_pct, color="#d50000", alpha=0.55)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=8)
    ax_dd.set_xlabel("Time", fontsize=9)
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    n_ticks  = min(10, len(x))
    tick_pos = np.linspace(0, len(x) - 1, n_ticks, dtype=int)
    ax_dd.set_xticks(tick_pos)
    ax_dd.set_xticklabels([str(ts[i])[:10] for i in tick_pos],
                          rotation=25, fontsize=7, ha="right")
    for ax in [ax_eq] + asset_axes:
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart → {save_path}")


# ── Behaviour chart ────────────────────────────────────────────────────────────

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
    total_trades = sum(a["n_trades"] for a in m["per_asset"])
    fig.suptitle(
        f"Portfolio Behaviour — w={window}h / td={duration}h  |  {label}\n"
        f"Total trades: {total_trades}   "
        + "  |  ".join(
            f"{a['name']}: {a['n_trades']} trades  WR={a['win_rate']*100:.0f}%"
            for a in m["per_asset"]
        ),
        fontsize=9, fontweight="bold", y=0.99,
    )
    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e0e0e0", linewidth=0.5, alpha=0.7)

    # ── TL: PnL Decomposition ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, cum_gross, color="#1565c0", linewidth=1.2, label="Gross PnL")
    ax.plot(x, net_delta, color="#2e7d32", linewidth=1.2, label="Net equity Δ")
    ax.fill_between(x, 0, -cum_fees, color="#d50000", alpha=0.35, label="–Fees")
    ax.axhline(0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_title("Portfolio PnL Decomposition", fontweight="bold", fontsize=10)
    ax.set_ylabel("$ change from start", fontsize=8)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # ── TR: Rolling Sharpe ────────────────────────────────────────────────────
    ax = axes[0, 1]
    valid = ~np.isnan(rs_full)
    xv, sv = x[valid], rs_full[valid]
    if len(xv) > 0:
        ax.plot(xv, sv, color="#1565c0", linewidth=1.0)
        ax.fill_between(xv, 0, sv, where=sv >= 0, color="#2e7d32", alpha=0.25)
        ax.fill_between(xv, 0, sv, where=sv <  0, color="#d50000", alpha=0.25)
    ax.axhline(0,   color="#555555", linewidth=0.8, linestyle="--")
    ax.axhline(1.5, color="#1565c0", linewidth=0.8, linestyle="--", label="Target 1.5")
    ax.set_title("Rolling 168h Portfolio Sharpe", fontweight="bold", fontsize=10)
    ax.set_ylabel("Sharpe (annualised)", fontsize=8)
    ax.legend(fontsize=8)

    # ── BL: Per-asset trade counts and win rates ───────────────────────────────
    ax = axes[1, 0]
    names_list = [a["name"]   for a in m["per_asset"]]
    counts     = [a["n_trades"] for a in m["per_asset"]]
    win_rates  = [a["win_rate"] * 100 for a in m["per_asset"]]
    bar_colours = ["#1a1a2e", "#1565c0", "#6a1b9a", "#ff8f00"]
    bars = ax.bar(names_list, counts, color=bar_colours, alpha=0.8, zorder=3)
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"WR {wr:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Trades per Asset + Win Rate", fontweight="bold", fontsize=10)
    ax.set_ylabel("# completed trades", fontsize=8)

    # ── BR: Per-asset time allocation ─────────────────────────────────────────
    ax = axes[1, 1]
    x_pos   = np.arange(N_ASSETS)
    w       = 0.28
    longs   = [a["frac_long"]  * 100 for a in m["per_asset"]]
    shorts  = [a["frac_short"] * 100 for a in m["per_asset"]]
    flats   = [a["frac_flat"]  * 100 for a in m["per_asset"]]
    ax.bar(x_pos - w, longs,  width=w, label="Long",  color="#00c853", alpha=0.8)
    ax.bar(x_pos,     flats,  width=w, label="Flat",  color="#90a4ae", alpha=0.8)
    ax.bar(x_pos + w, shorts, width=w, label="Short", color="#d50000", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_list)
    ax.set_title("Time Allocation per Asset", fontweight="bold", fontsize=10)
    ax.set_ylabel("% of steps", fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Behaviour → {save_path}")


# ── Master comparison grid ─────────────────────────────────────────────────────

def print_master_grid(all_results, models_found):
    windows   = sorted(set(w for w, _ in models_found))
    durations = sorted(set(d for _, d in models_found))

    def avg_metric(w, d, key):
        res  = all_results.get((w, d), {})
        vals = [m[key] for m in res.values() if m is not None and key in m]
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
                    row += f"   {'—':>9}  "
                    continue
                v = all_vals.get((w, d), float("nan"))
                if np.isnan(v):
                    row += f"   {'n/a':>9}  "
                else:
                    cell = fmt.format(v)
                    star = " ★" if abs(v - best_val) < 1e-9 else "  "
                    row += f"   {cell}{star}  "
            print(row)
        print(f"\n  ★ = best value")

    print("\n\n" + "=" * 72)
    print("  MASTER COMPARISON GRID — MULTI-ASSET MODELS × ALL PERIODS")
    print("=" * 72)

    grid_block("AVG TOTAL RETURN (%)",               "total_ret",
               "{:>+7.1f}%", highlight="max", scale=100.0)
    grid_block("AVG ANNUALISED SHARPE",               "sharpe",
               "{:>+7.3f} ", highlight="max", scale=1.0)
    grid_block("AVG MAX DRAWDOWN (%) — lower better", "max_dd",
               "{:>7.1f}% ", highlight="min", scale=100.0)
    grid_block("AVG FEE DRAG (%) — lower better",    "fee_drag",
               "{:>7.1f}% ", highlight="min", scale=1.0)

    # ── Top 5 by average Sharpe ────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  TOP 5 MULTI-ASSET MODELS  (by average Sharpe)")
    print("=" * 72)
    scored = []
    for (w, d) in models_found:
        s  = avg_metric(w, d, "sharpe")
        r  = avg_metric(w, d, "total_ret")
        dd = avg_metric(w, d, "max_dd")
        scored.append((s, w, d, r, dd))
    scored.sort(reverse=True)
    print(f"  {'Model':<20} {'Avg Sharpe':>12} {'Avg Return':>12} {'Avg MaxDD':>12}")
    print("  " + "─" * 58)
    for rank, (s, w, d, r, dd) in enumerate(scored[:5], 1):
        marker = " ★" if rank == 1 else "  "
        print(f"  w={w}h / td={d}h{'':<8} {s:>+11.3f}  {r*100:>+10.1f}%  {dd*100:>10.1f}%{marker}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    models_found = discover_models()

    if not models_found:
        print("No trained models found in ppo_multi_perp/. Run train_multi_ppo.py first.")
        raise SystemExit(1)

    CHART_DIR = Path("evaluation pictures multi")
    CHART_DIR.mkdir(exist_ok=True)

    print(f"Found {len(models_found)} multi-asset models: "
          + ", ".join(f"w{w}_td{d}" for w, d in models_found))
    print(f"Evaluating across {len(PERIODS)} periods each.\n")

    all_results = {}   # (window, duration) → {period_label: metrics}

    for model_idx, (window, duration) in enumerate(models_found, 1):
        model_name, vecnorm_name = model_files(window, duration)
        tag = f"w{window}_td{duration}"

        print(f"\n{'#'*72}")
        print(f"  MODEL {model_idx}/{len(models_found)}:  {model_name}  "
              f"(window={window}h  trade_dur={duration}h)")
        print(f"{'#'*72}")

        try:
            model = PPO.load(model_name)
        except Exception as e:
            print(f"  [ERROR] Cannot load {model_name}: {e}")
            continue

        period_results = {}

        for period_label, start_date, end_date in PERIODS:
            print(f"\n  ── {period_label} ──")
            try:
                eval_start, eval_end = get_period_indices(start_date, end_date, window)
            except ValueError as e:
                print(f"    [SKIP] {e}")
                continue

            n_days = (eval_end - eval_start) / 24
            print(f"    Rows {eval_start}→{eval_end}  (~{n_days:.0f} days)")

            hist    = run_episode(model, vecnorm_name, window, duration,
                                  eval_start, eval_end)
            metrics = compute_metrics(hist)
            print_metrics(metrics, period_label, window, duration)

            # ── Baselines ─────────────────────────────────────────────────────
            bnh_ew  = compute_equal_weight_bnh(eval_start, eval_end)
            bnh_btc = compute_btc_bnh(eval_start, eval_end)

            print(f"  {'':─<56}")
            print(f"  {'BASELINE COMPARISON':^56}")
            print(f"  {'':─<56}")
            print(f"  {'Strategy':<30} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8}")
            print(f"  {'PPO Portfolio':<30} {metrics['total_ret']*100:>+7.1f}% "
                  f"{metrics['sharpe']:>8.3f} {metrics['max_dd']*100:>7.1f}%")
            for b in [bnh_ew, bnh_btc]:
                print(f"  {b['label']:<30} {b['total_ret']*100:>+7.1f}% "
                      f"{b['sharpe']:>8.3f} {b['max_dd']*100:>7.1f}%")
            print(f"  {'':─<56}")

            safe_period = start_date[:7].replace("-", "")
            plot_episode(hist, metrics, bnh_ew, bnh_btc,
                         period_label, window, duration,
                         str(CHART_DIR / f"eval_multi_chart_{tag}_{safe_period}.png"))
            plot_behaviour(hist, metrics, period_label, window, duration,
                           str(CHART_DIR / f"eval_multi_behaviour_{tag}_{safe_period}.png"))

            period_results[period_label] = metrics
            period_results[period_label]["_bnh_ew"]  = bnh_ew
            period_results[period_label]["_bnh_btc"] = bnh_btc

        all_results[(window, duration)] = period_results

        # Per-model cross-period summary
        if period_results:
            print(f"\n  ── {tag} cross-period summary ──")
            print(f"  {'Period':<38} {'PPO Ret':>9} {'PPO Sh':>8} "
                  f"{'BnH(EW) Ret':>12} {'BnH(EW) Sh':>11} "
                  f"{'BnH(BTC) Ret':>13} {'BnH(BTC) Sh':>12}")
            print("  " + "─" * 105)
            for lbl, m in period_results.items():
                bew  = m.get("_bnh_ew")  or {}
                bbtc = m.get("_bnh_btc") or {}
                print(f"  {lbl[:36]:<38}"
                      f" {m['total_ret']*100:>+8.1f}%"
                      f" {m['sharpe']:>8.3f}"
                      f" {bew.get('total_ret', float('nan'))*100:>+11.1f}%"
                      f" {bew.get('sharpe', float('nan')):>11.3f}"
                      f" {bbtc.get('total_ret', float('nan'))*100:>+12.1f}%"
                      f" {bbtc.get('sharpe', float('nan')):>12.3f}")

    # Final master grid
    print_master_grid(all_results, models_found)
