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
CSV_PATH    = "SOLUSDT_perpetual_futures_processed_2020-2025.csv"
SEED        = 42
WINDOW      = 48
DEFAULT_LEV = 2.0      # realistic perpetual futures leverage
DEFAULT_FEE = 0.0005   # Binance taker fee (0.05%)

MODEL_NAME   = f"ppo_sol_perp_w{WINDOW}"
VECNORM_NAME = f"vec_normalize_w{WINDOW}.pkl"

PERIODS = [
    ("Early Bull  (Sep 2020 – Jun 2021)",   "2020-09-01", "2021-06-01"),
    ("Peak & Crash (Jun 2021 – Jun 2022)",  "2021-06-01", "2022-06-01"),
    ("Bear / FTX  (Jun 2022 – Jan 2023)",   "2022-06-01", "2023-01-01"),
    ("Recovery    (Jan 2023 – Jan 2024)",   "2023-01-01", "2024-01-01"),
    ("Bull Run    (Jan 2024 – Jan 2025)",   "2024-01-01", "2025-01-01"),
]

model   = PPO.load(MODEL_NAME)
df_full = pd.read_csv(CSV_PATH, parse_dates=["open_time"])

# Load VecNormalize statistics if available (produced by train_solo_ppo.py)
_vecnorm_path = Path(VECNORM_NAME)
_USE_VECNORM  = _vecnorm_path.exists()
if _USE_VECNORM:
    print(f"VecNormalize stats loaded from {VECNORM_NAME}")
else:
    print(f"[INFO] {VECNORM_NAME} not found — running without observation normalisation")


# ── Period index lookup ───────────────────────────────────────────────────────

def get_period_indices(df, start_date, end_date):
    mask = (df["open_time"] >= start_date) & (df["open_time"] < end_date)
    rows = df.index[mask]
    if len(rows) < WINDOW + 10:
        raise ValueError(f"Too few rows ({len(rows)}) in {start_date}–{end_date}")
    return int(max(rows[0], WINDOW)), int(rows[-1])


# ── Episode runner (full history) ─────────────────────────────────────────────

def run_episode_full(max_leverage, fee_rate, eval_start, eval_end):
    def make_env():
        return SolPerpEnv(
            csv_path=CSV_PATH, window_size=WINDOW,
            fee_rate=fee_rate, max_leverage=max_leverage,
            initial_equity=1_000.0,
            eval_start=eval_start, eval_end=eval_end,
        )
    set_random_seed(SEED)
    env = DummyVecEnv([make_env])

    if _USE_VECNORM:
        env = VecNormalize.load(VECNORM_NAME, env)
        env.training    = False   # freeze running stats
        env.norm_reward = False

    obs = env.reset()

    prices, positions, equity = [], [], []
    pnl_hist, fee_hist, fund_hist, dd_hist = [], [], [], []
    sl_hits, tp_hits, regime_hist = [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        done = bool(done[0])
        info = infos[0]
        prices.append(info["price"])
        positions.append(info["position"])
        equity.append(info["equity"])
        pnl_hist.append(info["pnl"])
        fee_hist.append(info["trade_fee"])
        fund_hist.append(info["funding_payment"])
        dd_hist.append(info["drawdown"])
        sl_hits.append(float(info.get("sl_hit", False)))
        tp_hits.append(float(info.get("tp_hit", False)))
        regime_hist.append(float(info.get("regime_aligned", False)))

    timestamps = df_full["open_time"].iloc[eval_start:eval_start + len(equity)].values
    return dict(
        timestamps=timestamps,
        prices=np.array(prices),
        positions=np.array(positions),
        equity=np.array(equity),
        pnl=np.array(pnl_hist),
        fees=np.array(fee_hist),
        funding=np.array(fund_hist),
        drawdown=np.array(dd_hist),
        sl_hits=np.array(sl_hits),
        tp_hits=np.array(tp_hits),
        regime=np.array(regime_hist),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(h):
    eq   = h["equity"]
    pos  = h["positions"]
    pnl  = h["pnl"]
    n    = len(eq)
    HPY  = 8_760  # hours per year

    total_ret   = (eq[-1] / eq[0]) - 1.0
    ann_ret     = (1 + total_ret) ** (HPY / n) - 1

    # drawdown
    peak   = np.maximum.accumulate(eq)
    dd_arr = (peak - eq) / (peak + 1e-8)
    max_dd = dd_arr.max()

    # Sharpe (hourly log returns, annualised)
    log_ret  = np.diff(np.log(eq + 1e-8))
    sharpe   = (log_ret.mean() / (log_ret.std() + 1e-8)) * np.sqrt(HPY)

    # Calmar
    calmar = ann_ret / (max_dd + 1e-8)

    # Time in market
    frac_long  = (pos >  0.05).mean()
    frac_short = (pos < -0.05).mean()
    frac_flat  = (np.abs(pos) <= 0.05).mean()

    # Direction changes (trade flips)
    sign_changes = np.diff(np.sign(pos))
    n_flips      = int((sign_changes != 0).sum())

    # Step-level win rate while in a position
    in_pos   = np.abs(pos) > 0.05
    wins     = (pnl > 0) & in_pos
    losses   = (pnl < 0) & in_pos
    win_rate = wins.sum() / (in_pos.sum() + 1e-8)

    gross_profit = pnl[wins].sum()
    gross_loss   = abs(pnl[losses].sum())
    profit_factor = gross_profit / (gross_loss + 1e-8)

    avg_win  = pnl[wins].mean()  if wins.any()  else 0.0
    avg_loss = pnl[losses].mean() if losses.any() else 0.0
    expect   = win_rate * avg_win + (1 - win_rate) * avg_loss  # expectancy per step

    best_step  = log_ret.max()  * 100
    worst_step = log_ret.min()  * 100
    vol_ann    = log_ret.std()  * np.sqrt(HPY) * 100

    # SL / TP statistics
    sl_hits = h.get("sl_hits", np.zeros(n))
    tp_hits = h.get("tp_hits", np.zeros(n))
    regime  = h.get("regime",  np.zeros(n))
    n_sl    = int(sl_hits.sum())
    n_tp    = int(tp_hits.sum())
    pct_regime = regime.mean()

    # ── Extended behaviour metrics ─────────────────────────────────────────────
    turnover_rate    = np.abs(np.diff(pos)).mean()
    avg_fee_per_step = h["fees"].sum() / n
    gross_pnl        = h["pnl"].sum()
    fee_pct_of_gross = h["fees"].sum() / (abs(gross_pnl) + 1e-8) * 100

    # Consecutive win / loss streaks
    in_pos_mask = np.abs(pos) > 0.05
    step_results = np.where(in_pos_mask, np.sign(pnl), 0)  # +1 win, -1 loss, 0 flat
    max_consec_win = max_consec_loss = 0
    cur_win = cur_loss = 0
    for r in step_results:
        if r > 0:
            cur_win  += 1
            cur_loss  = 0
        elif r < 0:
            cur_loss += 1
            cur_win   = 0
        else:
            cur_win = cur_loss = 0
        max_consec_win  = max(max_consec_win,  cur_win)
        max_consec_loss = max(max_consec_loss, cur_loss)

    flip_rate_per_day = n_flips / (n / 24 + 1e-8)

    return dict(
        total_ret=total_ret, ann_ret=ann_ret, max_dd=max_dd,
        sharpe=sharpe, calmar=calmar,
        frac_long=frac_long, frac_short=frac_short, frac_flat=frac_flat,
        n_flips=n_flips, win_rate=win_rate,
        profit_factor=profit_factor, avg_win=avg_win, avg_loss=avg_loss,
        expectancy=expect,
        total_fees=h["fees"].sum(), total_funding=h["funding"].sum(),
        best_step=best_step, worst_step=worst_step, vol_ann=vol_ann,
        n_steps=n, init_eq=eq[0], final_eq=eq[-1],
        n_sl=n_sl, n_tp=n_tp, pct_regime=pct_regime,
        turnover_rate=turnover_rate,
        avg_fee_per_step=avg_fee_per_step,
        gross_pnl=gross_pnl,
        fee_pct_of_gross=fee_pct_of_gross,
        max_consec_win=max_consec_win,
        max_consec_loss=max_consec_loss,
        flip_rate_per_day=flip_rate_per_day,
    )


# ── Console printer ───────────────────────────────────────────────────────────

def print_metrics(m, label):
    W = 54
    sep = "  " + "─" * W
    print(sep)
    print(f"  {'PERFORMANCE METRICS — ' + label:^{W}}")
    print(sep)
    print(f"  {'Initial equity':<32} ${m['init_eq']:>9.2f}")
    print(f"  {'Final equity':<32} ${m['final_eq']:>9.2f}")
    print(f"  {'Total return':<32} {m['total_ret']*100:>+9.2f}%")
    print(f"  {'Annualised return':<32} {m['ann_ret']*100:>+9.2f}%")
    print(f"  {'Annualised volatility':<32} {m['vol_ann']:>9.2f}%")
    print(f"  {'Max drawdown':<32} {m['max_dd']*100:>9.2f}%")
    print(f"  {'Sharpe ratio (annualised)':<32} {m['sharpe']:>9.3f}")
    print(f"  {'Calmar ratio':<32} {m['calmar']:>9.3f}")
    print(sep)
    print(f"  {'Episode length (hours)':<32} {m['n_steps']:>9d}")
    print(f"  {'Time long':<32} {m['frac_long']*100:>9.1f}%")
    print(f"  {'Time short':<32} {m['frac_short']*100:>9.1f}%")
    print(f"  {'Time flat':<32} {m['frac_flat']*100:>9.1f}%")
    print(sep)
    print(f"  {'Direction changes (flips)':<32} {m['n_flips']:>9d}")
    print(f"  {'Step win rate':<32} {m['win_rate']*100:>9.1f}%")
    print(f"  {'Profit factor':<32} {m['profit_factor']:>9.3f}")
    print(f"  {'Avg winning step PnL':<32} ${m['avg_win']:>8.4f}")
    print(f"  {'Avg losing step PnL':<32} ${m['avg_loss']:>8.4f}")
    print(f"  {'Expectancy per step':<32} ${m['expectancy']:>8.4f}")
    print(f"  {'Best single step (log ret)':<32} {m['best_step']:>+9.3f}%")
    print(f"  {'Worst single step (log ret)':<32} {m['worst_step']:>+9.3f}%")
    print(sep)
    print(f"  {'Stop-loss triggers':<32} {m['n_sl']:>9d}")
    print(f"  {'Take-profit triggers':<32} {m['n_tp']:>9d}")
    print(f"  {'Time regime-aligned':<32} {m['pct_regime']*100:>9.1f}%")
    print(sep)
    print(f"  {'Total fees paid':<32} ${m['total_fees']:>9.2f}")
    print(f"  {'Total funding paid':<32} ${m['total_funding']:>9.2f}")
    print(sep)
    print(f"  {'Turnover rate (avg |Δpos|/step)':<32} {m['turnover_rate']:>9.3f}")
    print(f"  {'Avg fee per step':<32} ${m['avg_fee_per_step']:>8.3f}")
    print(f"  {'Gross PnL (before fees)':<32} ${m['gross_pnl']:>9.2f}")
    print(f"  {'Fee drag (% of |gross PnL|)':<32} {m['fee_pct_of_gross']:>9.1f}%")
    print(f"  {'Flip rate':<32} {m['flip_rate_per_day']:>8.1f} flips/day")
    print(f"  {'Max consec. wins':<32} {m['max_consec_win']:>9d} steps")
    print(f"  {'Max consec. losses':<32} {m['max_consec_loss']:>9d} steps")
    print(sep)

    # ── DIAGNOSTICS ───────────────────────────────────────────────────────────
    dsep = "  " + "═" * W
    print(dsep)
    print(f"  {'DIAGNOSTICS':^{W}}")
    print(dsep)

    SUSTAINABLE_FLIPS_PER_DAY = 5.5
    if m['flip_rate_per_day'] > SUSTAINABLE_FLIPS_PER_DAY:
        print(f"  ⚠  OVERTRADING: {m['flip_rate_per_day']:.1f} flips/day — "
              f"sustainable limit ~{SUSTAINABLE_FLIPS_PER_DAY} /day at this cost")
    else:
        print(f"  ✓  Flip rate {m['flip_rate_per_day']:.1f}/day is within sustainable range")

    if m['fee_pct_of_gross'] > 30.0:
        print(f"  ⚠  FEE DRAG: fees = {m['fee_pct_of_gross']:.1f}% of gross PnL "
              f"(should be < 30%)")
    else:
        print(f"  ✓  Fee drag {m['fee_pct_of_gross']:.1f}% of gross PnL is acceptable")

    if m['win_rate'] < 0.50:
        print(f"  ⚠  RANDOM POLICY: win rate {m['win_rate']*100:.0f}% ≈ coin flip "
              f"(need > 50%)")
    else:
        print(f"  ✓  Win rate {m['win_rate']*100:.0f}% > 50%")

    if m['pct_regime'] < 0.60:
        print(f"  ⚠  REGIME BLIND: {m['pct_regime']*100:.0f}% aligned "
              f"(need > 60%)")
    else:
        print(f"  ✓  Regime alignment {m['pct_regime']*100:.0f}% > 60%")

    print(dsep)


# ── Trade chart ───────────────────────────────────────────────────────────────

def plot_episode(h, m, label, save_path):
    prices    = h["prices"]
    positions = h["positions"]
    equity    = h["equity"]
    dd_pct    = h["drawdown"] * 100
    ts        = h["timestamps"]
    x         = np.arange(len(prices))

    # Running equity peak for shading
    peak_eq = np.maximum.accumulate(equity)

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#f8f9fa")

    title = (
        f"{label}\n"
        f"Return: {m['total_ret']*100:+.1f}%   "
        f"Ann.Return: {m['ann_ret']*100:+.1f}%   "
        f"Max DD: {m['max_dd']*100:.1f}%   "
        f"Sharpe: {m['sharpe']:.2f}   "
        f"Calmar: {m['calmar']:.2f}   "
        f"Win Rate: {m['win_rate']*100:.1f}%"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 2, 1],
                           hspace=0.08, left=0.07, right=0.97,
                           top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_facecolor("#ffffff")
        ax.grid(color="#e0e0e0", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 1: Price + position background ──────────────────────────────────
    p_min, p_max = prices.min(), prices.max()
    ax1.fill_between(x, p_min, p_max,
                     where=positions >  0.05, color="#00c853", alpha=0.12, label="Long")
    ax1.fill_between(x, p_min, p_max,
                     where=positions < -0.05, color="#d50000", alpha=0.12, label="Short")
    ax1.plot(x, prices, color="#1a1a2e", linewidth=0.9, zorder=3)
    ax1.set_ylabel("SOL Price (USDT)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # ── Panel 2: Position size ─────────────────────────────────────────────────
    ax2.fill_between(x, 0, positions,
                     where=positions >= 0, color="#00c853", alpha=0.8, label="Long")
    ax2.fill_between(x, 0, positions,
                     where=positions <  0, color="#d50000", alpha=0.8, label="Short")
    ax2.axhline(0, color="#555555", linewidth=0.7, zorder=4)
    ax2.set_ylabel("Position", fontsize=9)
    ax2.set_ylim(-DEFAULT_LEV * 1.15, DEFAULT_LEV * 1.15)
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.8)

    # ── Panel 3: Equity curve with drawdown shading ────────────────────────────
    ax3.fill_between(x, equity, peak_eq, color="#d50000", alpha=0.18, label="Drawdown")
    ax3.plot(x, peak_eq, color="#9e9e9e", linewidth=0.5, linestyle="--", zorder=2)
    ax3.plot(x, equity,  color="#1565c0", linewidth=1.1, zorder=3, label="Equity")
    ax3.axhline(equity[0], color="#9e9e9e", linewidth=0.8, linestyle=":", zorder=1)
    ax3.set_ylabel("Equity ($)", fontsize=9)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # ── Panel 4: Drawdown % ────────────────────────────────────────────────────
    ax4.fill_between(x, 0, -dd_pct, color="#d50000", alpha=0.55)
    ax4.set_ylabel("Drawdown (%)", fontsize=9)
    ax4.set_xlabel("Time", fontsize=9)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # ── X-axis date labels (shared, shown only on bottom panel) ───────────────
    n_ticks = min(10, len(x))
    tick_pos = np.linspace(0, len(x) - 1, n_ticks, dtype=int)
    ax4.set_xticks(tick_pos)
    ax4.set_xticklabels(
        [str(ts[i])[:10] for i in tick_pos],
        rotation=25, fontsize=7, ha="right"
    )
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {save_path}")


# ── Behaviour chart ───────────────────────────────────────────────────────────

def plot_behaviour(h, m, label, save_path):
    """
    2×2 figure analysing trading behaviour in detail.

    Panels:
      Top-left   — PnL decomposition (gross PnL, net equity Δ, cumulative fees)
      Top-right  — Rolling 7-day (168-step) Sharpe ratio
      Bottom-left — Position size distribution (long vs short)
      Bottom-right — Overtrading rate (flips per 48-step rolling window)
    """
    eq       = h["equity"]
    pos      = h["positions"]
    pnl_arr  = h["pnl"]
    fee_arr  = h["fees"]
    n        = len(eq)
    x        = np.arange(n)

    # Derived series
    cum_gross = np.cumsum(pnl_arr)
    cum_fees  = np.cumsum(fee_arr)
    net_delta = eq - eq[0]

    log_ret = pd.Series(np.diff(np.log(eq + 1e-8)))
    roll_sharpe = (
        log_ret.rolling(168).mean() /
        (log_ret.rolling(168).std() + 1e-8) *
        np.sqrt(8_760)
    )
    # Align to length n (diff shortens by 1)
    roll_sharpe_full = np.full(n, np.nan)
    roll_sharpe_full[1:] = roll_sharpe.values

    # Flip indicators: 1 where sign of position changes
    sign_pos   = np.sign(pos)
    flip_mask  = np.concatenate([[0], (np.diff(sign_pos) != 0).astype(float)])
    window_48  = 48
    flip_roll  = pd.Series(flip_mask).rolling(window_48, min_periods=1).sum().values

    # Sustainable flip line: 5.5 flips/day × 48h/24 = 11 flips per 48-step window
    sustainable_48 = 48 * 5.5 / 24  # = 11.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Trading Behaviour Analysis — {label}\n"
        f"Flips/day: {m['flip_rate_per_day']:.1f}   "
        f"Fee drag: {m['fee_pct_of_gross']:.1f}%   "
        f"Win rate: {m['win_rate']*100:.1f}%   "
        f"Regime: {m['pct_regime']*100:.1f}%",
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
    ax.set_title("PnL Decomposition: Where Did the Money Go?", fontweight="bold", fontsize=10)
    ax.set_xlabel("Step index", fontsize=8)
    ax.set_ylabel("$ change from start", fontsize=8)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.1f}"))

    # ── Panel TR: Rolling 7-day Sharpe ────────────────────────────────────────
    ax = axes[0, 1]
    valid = ~np.isnan(roll_sharpe_full)
    xv    = x[valid]
    sv    = roll_sharpe_full[valid]
    ax.plot(xv, sv, color="#1565c0", linewidth=1.0, label="Rolling 168h Sharpe")
    ax.fill_between(xv, 0, sv, where=sv >= 0, color="#2e7d32", alpha=0.25)
    ax.fill_between(xv, 0, sv, where=sv <  0, color="#d50000", alpha=0.25)
    ax.axhline(0,   color="#555555", linewidth=0.8, linestyle="--")
    ax.axhline(2.0, color="#1565c0", linewidth=0.8, linestyle="--", label="Target = 2.0")
    ax.set_title("Rolling 7-day Sharpe Ratio", fontweight="bold", fontsize=10)
    ax.set_xlabel("Step index", fontsize=8)
    ax.set_ylabel("Sharpe (annualised)", fontsize=8)
    ax.legend(fontsize=8)

    # ── Panel BL: Position size distribution ─────────────────────────────────
    ax = axes[1, 0]
    long_pos  = pos[pos >  0.05]
    short_pos = np.abs(pos[pos < -0.05])
    flat_pos  = pos[np.abs(pos) <= 0.05]
    bins = np.linspace(0, pos.max() if pos.max() > 0 else 1, 31)
    if len(long_pos) > 0:
        ax.hist(long_pos,  bins=bins, color="#2e7d32", alpha=0.70,
                label=f"Long  ({len(long_pos)/n*100:.0f}%)", edgecolor="white", linewidth=0.3)
    if len(short_pos) > 0:
        ax.hist(short_pos, bins=bins, color="#d50000", alpha=0.70,
                label=f"Short ({len(short_pos)/n*100:.0f}%)", edgecolor="white", linewidth=0.3)
    if len(flat_pos) > 0:
        ax.axvline(0, color="gray", linewidth=1.0, linestyle="--")
        ax.text(0.01, 0.96, f"Flat: {len(flat_pos)/n*100:.0f}%",
                transform=ax.transAxes, fontsize=8, va="top", color="gray")
    ax.set_title("Position Size Distribution", fontweight="bold", fontsize=10)
    ax.set_xlabel("|Position| (leverage units)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.legend(fontsize=8)

    # ── Panel BR: Overtrading rate ────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(x, flip_roll, color="#e65100", linewidth=0.9, alpha=0.85,
            label=f"Flips per {window_48}h window")
    ax.axhline(sustainable_48, color="#d50000", linewidth=1.2, linestyle="--",
               label=f"Sustainable limit ({sustainable_48:.0f} flips/{window_48}h)")
    ax.fill_between(x, 0, flip_roll,
                    where=flip_roll > sustainable_48,
                    color="#d50000", alpha=0.20, label="Overtrading zone")
    ax.set_title(f"Overtrading Rate (flips per {window_48}h rolling window)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Step index", fontsize=8)
    ax.set_ylabel("# direction changes", fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Behaviour chart saved → {save_path}")


# ── Cross-period summary table ────────────────────────────────────────────────

def print_summary(all_metrics):
    labels = list(all_metrics.keys())
    W = 80
    print(f"\n\n{'='*W}")
    print(f"  {'CROSS-PERIOD SUMMARY':^{W}}")
    print(f"  Leverage: {DEFAULT_LEV}x   Fee: {DEFAULT_FEE*100:.3f}%   Window: {WINDOW}h")
    print(f"{'='*W}")

    cols = ["Return", "Ann.Ret", "MaxDD", "Sharpe", "Calmar", "WinRate", "Flips", "Fees $"]
    hdr  = f"  {'Period':<36}" + "".join(f"{c:>9}" for c in cols)
    print(hdr)
    print("  " + "─" * (W - 2))

    for lbl, m in all_metrics.items():
        short = lbl[:34]
        row = (
            f"  {short:<36}"
            f"{m['total_ret']*100:>+8.1f}%"
            f"{m['ann_ret']*100:>+8.1f}%"
            f"{m['max_dd']*100:>8.1f}%"
            f"{m['sharpe']:>9.2f}"
            f"{m['calmar']:>9.2f}"
            f"{m['win_rate']*100:>8.1f}%"
            f"{m['n_flips']:>8d}"
            f"{m['total_fees']:>8.1f}"
        )
        print(row)

    print(f"{'='*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

all_metrics = {}

for period_label, start_date, end_date in PERIODS:
    print(f"\n{'#'*70}")
    print(f"  PERIOD: {period_label}")
    print(f"{'#'*70}")

    try:
        eval_start, eval_end = get_period_indices(df_full, start_date, end_date)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        continue

    n_days = (eval_end - eval_start) / 24
    print(f"  Rows {eval_start} → {eval_end}  (~{n_days:.0f} days)")

    hist    = run_episode_full(DEFAULT_LEV, DEFAULT_FEE, eval_start, eval_end)
    metrics = compute_metrics(hist)

    print_metrics(metrics, period_label)

    safe_name = start_date[:7].replace("-", "")
    plot_episode(hist, metrics, period_label, f"eval_chart_{safe_name}.png")
    plot_behaviour(hist, metrics, period_label, f"eval_behaviour_{safe_name}.png")

    all_metrics[period_label] = metrics

print_summary(all_metrics)
