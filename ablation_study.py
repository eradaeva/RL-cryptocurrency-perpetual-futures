"""
ablation_study.py  —  Reward function ablation study
=====================================================
Trains 6 variants of the v3.4 reward, each with one component removed.
Each variant runs 500k steps (same as smoke test) then evaluates OOS
(Jan 2025 – Apr 2026) and reports a comparison table.

This directly answers Dr. Yang's suggestion:
  "If you can analyse the findings for reward function design, including
   an ablation study (removing certain component will degrade performance),
   it will be a big plus for your dissertation."

Variants tested:
  0. Full v3.4          — all components active (baseline)
  1. No funding penalty  — FUNDING_SCALE = 0  (agent blind to funding cost)
  2. No trend-hold       — TREND_HOLD_BONUS/PENALTY = 0  (no per-step direction)
  3. No counter-trend    — COUNTER_TREND_PENALTY = 0  (free counter-trend entry)
  4. No flat patience    — STEP_SCALE = 0  (no reward for staying out)
  5. No entry cost       — ENTRY_COST = 0  (no selectivity gate)

Results are written to ablation_results.csv for easy table generation.

Usage:
    python3 ablation_study.py
    # Takes ~90 minutes (6 × ~15 min each, sequential)
"""

import os
import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

import multi_asset_env as mae
from multi_asset_env import MultiAssetPerpEnv


# ── Config (matches train_multi_ppo_parallel.py) ─────────────────────────────

N_ENVS  = 4
N_STEPS = 512

WINDOW_SIZE    = 168
TRADE_DURATION = 168
MIN_HOLD_STEPS = 48
NET_ARCH       = [256, 256]
ABLATION_STEPS = 500_000
TRAIN_END_DATE = "2023-12-31"
INIT_EQUITY    = 1_000.0
SEED           = 42

CSV_PATHS = [
    "BTCUSDT_perpetual_futures_processed_2020-2026.csv",
    "ETHUSDT_perpetual_futures_processed_2020-2026.csv",
    "SOLUSDT_perpetual_futures_processed_2020-2026.csv",
    "BNBUSDT_perpetual_futures_processed_2020-2026.csv",
]
ASSET_NAMES = ["BTC", "ETH", "SOL", "BNB"]

OUT_DIR = Path("ablation_models")
OUT_DIR.mkdir(exist_ok=True)

OOS_START_DATE = "2025-01-01"
OOS_END_DATE   = "2026-04-08"


# ── Ablation variants ─────────────────────────────────────────────────────────
# Each entry overrides specific constants in multi_asset_env before training.
# "label" is used in the results table.
# "overrides" is a dict of {constant_name: new_value}.

ABLATIONS = [
    {
        "label":     "Full v3.4 (baseline)",
        "tag":       "full_v34",
        "overrides": {},   # nothing removed
        "hypothesis": "Best overall — all components working together",
    },
    {
        "label":     "No funding penalty",
        "tag":       "no_funding",
        "overrides": {"FUNDING_SCALE": 0.0},
        "hypothesis": "Agent blind to funding cost → holds longs in high-funding bull markets",
    },
    {
        "label":     "No trend-hold signal",
        "tag":       "no_trend_hold",
        "overrides": {"TREND_HOLD_BONUS": 0.0, "TREND_HOLD_PENALTY": 0.0},
        "hypothesis": "No per-step directional feedback → counter-trend holding returns",
    },
    {
        "label":     "No counter-trend entry penalty",
        "tag":       "no_ct_entry",
        "overrides": {"COUNTER_TREND_PENALTY": 0.0},
        "hypothesis": "Agent enters freely against trend → direction errors at open",
    },
    {
        "label":     "No flat patience",
        "tag":       "no_flat",
        "overrides": {"STEP_SCALE": 0.0},
        "hypothesis": "No reward for staying out → always-in-trades behaviour",
    },
    {
        "label":     "No entry cost",
        "tag":       "no_entry_cost",
        "overrides": {"ENTRY_COST": 0.0},
        "hypothesis": "No selectivity gate → excessive entries, fee drag spikes",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(s):
    s = int(s)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m"
    return f"{s//60}m {s%60}s"


def _apply_overrides(overrides: dict):
    """Monkey-patch multi_asset_env constants for this ablation variant."""
    for name, val in overrides.items():
        setattr(mae, name, val)


def _restore_defaults():
    """Restore all constants to their v3.4 values after each ablation."""
    mae.STEP_SCALE            = 0.004
    mae.ENTRY_COST            = 0.030
    mae.FEE_SCALE             = 8.0
    mae.FUNDING_SCALE         = 4.0
    mae.SHARPE_SCALE          = 0.05
    mae.TREND_THRESHOLD       = 0.01
    mae.COUNTER_TREND_PENALTY = 0.040
    mae.TREND_HOLD_BONUS      = 0.0008
    mae.TREND_HOLD_PENALTY    = 0.0008
    mae.TRADE_SCALE           = 0.05
    mae.COUNTER_TREND_WIN_MULT = 1.3
    mae.TREND_WIN_MULT         = 1.5


class QuietCallback(BaseCallback):
    def __init__(self, total, label):
        super().__init__()
        self.total = total
        self.label = label
        self._last = -1

    def _on_step(self):
        pct = int(self.num_timesteps / self.total * 100)
        if pct % 20 == 0 and pct != self._last:
            print(f"    [{self.label}] {pct}%")
            self._last = pct
        return True


# ── OOS evaluation ─────────────────────────────────────────────────────────────

def _get_oos_indices():
    """Compute OOS start/end indices (Jan 2025 – Apr 2026) in aligned data."""
    raw = [pd.read_csv(p, parse_dates=["open_time"]) for p in CSV_PATHS]
    common = set(raw[0]["open_time"])
    for df in raw[1:]:
        common &= set(df["open_time"])
    ts = pd.Series(sorted(common))
    mask = (ts >= OOS_START_DATE) & (ts < OOS_END_DATE)
    rows = ts.index[mask]
    return int(max(rows[0], WINDOW_SIZE)), int(rows[-1])


OOS_START_IDX, OOS_END_IDX = _get_oos_indices()
print(f"OOS window: idx {OOS_START_IDX} → {OOS_END_IDX}  "
      f"({OOS_START_DATE} – {OOS_END_DATE})\n")


def evaluate_oos(model, vecnorm_path):
    """Run one OOS episode and return key metrics."""

    def make_env():
        return MultiAssetPerpEnv(
            csv_paths          = CSV_PATHS,
            asset_names        = ASSET_NAMES,
            window_size        = WINDOW_SIZE,
            max_trade_duration = TRADE_DURATION,
            min_hold_steps     = MIN_HOLD_STEPS,
            initial_equity     = INIT_EQUITY,
            eval_start         = OOS_START_IDX,
            eval_end           = OOS_END_IDX,
        )

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training    = False
    env.norm_reward = False

    obs = env.reset()

    equity_h, fees_h, fund_h, pnl_h = [], [], [], []
    positions_h = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_arr, infos = env.step(action)
        done = bool(done_arr[0])
        info = infos[0]

        equity_h.append(info["equity"])
        fees_h.append(info["total_fees"])
        fund_h.append(info["total_funding"])
        pnl_h.append(info["total_pnl"])
        positions_h.append(list(info["positions"]))

    eq   = np.array(equity_h)
    pos  = np.array(positions_h)   # shape (T, N_ASSETS)

    total_ret  = (eq[-1] / eq[0]) - 1.0
    log_r      = np.diff(np.log(eq + 1e-8))
    sharpe     = (log_r.mean() / (log_r.std() + 1e-8)) * np.sqrt(8760)
    peak       = np.maximum.accumulate(eq)
    max_dd     = float(((peak - eq) / (peak + 1e-8)).max())

    gross_pnl  = sum(pnl_h)
    total_fees = sum(fees_h)
    total_fund = sum(fund_h)
    fee_drag   = total_fees / (abs(gross_pnl) + 1e-8) * 100

    # Portfolio leverage
    lev          = np.abs(pos).sum(axis=1)
    avg_lev      = float(lev.mean())
    # Flat time = all four assets flat simultaneously
    frac_flat    = float((lev < 0.05 * len(ASSET_NAMES)).mean())

    # Per-asset long%
    per_long = []
    for i in range(len(ASSET_NAMES)):
        long_frac = float((pos[:, i] > 0.05).mean())
        per_long.append(long_frac)

    env.close()

    return {
        "total_ret":   total_ret,
        "sharpe":      float(sharpe),
        "max_dd":      max_dd,
        "fee_drag":    fee_drag,
        "total_fees":  total_fees,
        "total_fund":  total_fund,
        "gross_pnl":   gross_pnl,
        "avg_lev":     avg_lev,
        "frac_flat":   frac_flat,
        "per_long":    per_long,
        "final_eq":    float(eq[-1]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  ABLATION STUDY — v3.4 Reward Components")
    print(f"  {len(ABLATIONS)} variants × {ABLATION_STEPS:,} steps each")
    print(f"  OOS evaluation: {OOS_START_DATE} – {OOS_END_DATE}")
    print("=" * 70)

    for i, ab in enumerate(ABLATIONS):
        print(f"\n{'#'*70}")
        print(f"  [{i+1}/{len(ABLATIONS)}]  {ab['label']}")
        print(f"  Hypothesis: {ab['hypothesis']}")
        if ab["overrides"]:
            print(f"  Overrides:  {ab['overrides']}")
        print(f"{'#'*70}\n")

        # Apply overrides (monkey-patch constants before env creation)
        _restore_defaults()
        _apply_overrides(ab["overrides"])

        tag       = ab["tag"]
        norm_path = str(OUT_DIR / f"ablation_vecnorm_{tag}.pkl")
        model_path = str(OUT_DIR / f"ablation_{tag}")

        def _make_fn():
            return MultiAssetPerpEnv(
                csv_paths          = CSV_PATHS,
                asset_names        = ASSET_NAMES,
                window_size        = WINDOW_SIZE,
                max_trade_duration = TRADE_DURATION,
                min_hold_steps     = MIN_HOLD_STEPS,
                initial_equity     = INIT_EQUITY,
                train_end_date     = TRAIN_END_DATE,
            )

        t0  = time.time()
        # DummyVecEnv (not SubprocVecEnv) is required here: monkey-patching of
        # multi_asset_env constants only affects the main process.  SubprocVecEnv
        # spawns child processes that each reload the module with original defaults,
        # so all ablation variants would train identically.  DummyVecEnv runs all
        # environments in the same process, so the overrides take effect correctly.
        env  = DummyVecEnv([_make_fn] * N_ENVS)
        norm = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model = PPO(
            "MlpPolicy", norm,
            policy_kwargs = {"net_arch": NET_ARCH},
            device        = "cpu",
            verbose       = 0,
            learning_rate = 3e-4,
            n_steps       = N_STEPS,
            batch_size    = 256,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.01,
            seed          = SEED,
        )

        model.learn(
            total_timesteps = ABLATION_STEPS,
            callback        = QuietCallback(ABLATION_STEPS, tag),
        )

        norm.save(norm_path)
        model.save(model_path)
        train_time = time.time() - t0
        print(f"\n  Training done in {_fmt_time(train_time)}")

        # Evaluate OOS
        print(f"  Evaluating OOS…")
        metrics = evaluate_oos(model, norm_path)

        ab["metrics"]    = metrics
        ab["train_time"] = train_time

        print(f"  Total return : {metrics['total_ret']*100:>+7.1f}%")
        print(f"  Sharpe       : {metrics['sharpe']:>+6.3f}")
        print(f"  Max drawdown : {metrics['max_dd']*100:>6.1f}%")
        print(f"  Fee drag     : {metrics['fee_drag']:>6.1f}%")
        print(f"  Frac flat    : {metrics['frac_flat']*100:>5.1f}%")
        print(f"  Total funding: ${metrics['total_fund']:>+8.2f}")

        del model, norm, env
        gc.collect()
        _restore_defaults()

    # ── Results table ─────────────────────────────────────────────────────────

    print("\n\n" + "=" * 70)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 70)

    header = f"  {'Variant':<35}  {'Return':>7}  {'Sharpe':>7}  {'MaxDD':>6}  {'FeeDrag':>8}  {'Flat%':>6}  {'Funding':>9}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    rows = []
    for ab in ABLATIONS:
        m   = ab["metrics"]
        row = {
            "variant":    ab["label"],
            "tag":        ab["tag"],
            "total_ret":  round(m["total_ret"] * 100, 1),
            "sharpe":     round(m["sharpe"], 3),
            "max_dd":     round(m["max_dd"]  * 100, 1),
            "fee_drag":   round(m["fee_drag"], 1),
            "frac_flat":  round(m["frac_flat"] * 100, 1),
            "total_fund": round(m["total_fund"], 2),
            "final_eq":   round(m["final_eq"], 2),
            "hypothesis": ab["hypothesis"],
        }
        rows.append(row)
        print(
            f"  {ab['label']:<35}  "
            f"{m['total_ret']*100:>+7.1f}%  "
            f"{m['sharpe']:>+7.3f}  "
            f"{m['max_dd']*100:>5.1f}%  "
            f"{m['fee_drag']:>7.1f}%  "
            f"{m['frac_flat']*100:>5.1f}%  "
            f"${m['total_fund']:>+8.2f}"
        )

    print("=" * 70)

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv("ablation_results.csv", index=False)
    print(f"\n  Results saved → ablation_results.csv")
    print(f"  Models saved  → {OUT_DIR}/")
    print("\n  Use ablation_results.csv to build Table 4.X in your dissertation.")
    print("  Each row = one ablation; Sharpe drop vs baseline = component value.")
