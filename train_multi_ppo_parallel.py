"""
train_multi_ppo_gpu.py  —  Multi-asset PPO (CPU-parallel, SubprocVecEnv)
=========================================================================
Faster version of train_multi_ppo.py using parallel environment collection
instead of GPU (which is slower for small MlpPolicy networks — see SB3 #1245).

Speedup mechanism — SubprocVecEnv:
  N_ENVS independent environment copies run in separate subprocesses and step
  simultaneously.  Each subprocess uses one CPU core.  The agent collects
  N_ENVS transitions per step instead of 1, giving proportional rollout speedup.

  Original : 1 env  × n_steps=2048 → 2048 transitions per rollout
  This file: 4 envs × n_steps=512  → 2048 transitions per rollout (same size)
                                      but ~3–4× faster collection

  Gradient update phase is unchanged (same total batch, same batch_size).

Expected wall-clock improvement vs original script:
  M1 Max (10 perf cores, N_ENVS=4) : ~2.5–3.5×
  Any modern quad-core CPU          : ~2–3×

Memory note:
  Each subprocess loads all 4 CSVs independently — expect ~4× RAM vs original.
  On M1 Max 64 GB this is not a concern.  Reduce N_ENVS if RAM is tight.
"""

import os
import gc
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from multi_asset_env import MultiAssetPerpEnv


# ── Parallelism ───────────────────────────────────────────────────────────────

N_ENVS = 8   # parallel environment workers
             # safe on M1 Max (10 perf cores); lower to 2 if RAM is a concern

# n_steps per env: keep total rollout size = N_ENVS × N_STEPS = 2048
N_STEPS = 2048 // N_ENVS   # = 512


# ── Assets ────────────────────────────────────────────────────────────────────

CSV_PATHS = [
    "BTCUSDT_perpetual_futures_processed_2020-2026.csv",
    "ETHUSDT_perpetual_futures_processed_2020-2026.csv",
    "SOLUSDT_perpetual_futures_processed_2020-2026.csv",
    "BNBUSDT_perpetual_futures_processed_2020-2026.csv",
]
ASSET_NAMES = ["BTC", "ETH", "SOL", "BNB"]

# ── Grid ──────────────────────────────────────────────────────────────────────

WINDOW_SIZES    = [168, 96, 48, 96, 24]
TRADE_DURATIONS = [168, 48, 48, 96, 168]   # paired: (168,168), (96,48), (48,48), (96,96), (24,168)

TOTAL_STEPS    = 3_000_000
SKIP_EXISTING  = False

TRAIN_END_DATE = "2023-12-31"
FEE_RATE       = 0.0005
MAX_LEVERAGE   = 3.0
MIN_HOLD_STEPS = 12

MODEL_DIR = "ppo_multi_perp"
LOG_DIR   = "training_log_multi"

PRINT_EVERY = 100_000


# ── Progress callback ─────────────────────────────────────────────────────────

class GridProgressCallback(BaseCallback):
    """Prints a live progress bar and key metrics every PRINT_EVERY steps."""

    def __init__(self, total_steps, model_idx, total_models,
                 model_name, log_path, print_every=100_000):
        super().__init__(verbose=0)
        self.total_steps  = total_steps
        self.model_idx    = model_idx
        self.total_models = total_models
        self.model_name   = model_name
        self.log_path     = log_path
        self.print_every  = print_every
        self._last_print  = 0
        self.rows         = []
        self._t0          = None

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        print(f"\n  ▶  Starting  {self.model_name}"
              f"  [{self.model_idx}/{self.total_models}]"
              f"  workers={N_ENVS}  n_steps/env={N_STEPS}")

    def _on_rollout_start(self) -> None:
        nv = self.model.logger.name_to_value
        self.rows.append({
            "timestep":      self.num_timesteps,
            "policy_loss":   nv.get("train/policy_gradient_loss"),
            "value_loss":    nv.get("train/value_loss"),
            "entropy":       nv.get("train/entropy_loss"),
            "explained_var": nv.get("train/explained_variance"),
            "approx_kl":     nv.get("train/approx_kl"),
            "clip_fraction": nv.get("train/clip_fraction"),
            "ep_rew_mean":   nv.get("rollout/ep_rew_mean"),
            "ep_len_mean":   nv.get("rollout/ep_len_mean"),
        })

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print < self.print_every:
            return True
        self._last_print = self.num_timesteps

        elapsed   = time.time() - self._t0
        model_pct = self.num_timesteps / self.total_steps * 100

        steps_done_total = (self.model_idx - 1) * self.total_steps + self.num_timesteps
        overall_pct      = steps_done_total / (self.total_models * self.total_steps) * 100

        rate      = self.num_timesteps / (elapsed + 1e-6)
        eta_this  = (self.total_steps - self.num_timesteps) / (rate + 1e-6)
        remaining = self.total_models - self.model_idx
        eta_total = eta_this + remaining * self.total_steps / (rate + 1e-6)

        nv  = self.model.logger.name_to_value
        ev  = nv.get("train/explained_variance", float("nan"))
        rew = nv.get("rollout/ep_rew_mean",       float("nan"))

        bar_filled = int(30 * model_pct / 100)
        bar        = "█" * bar_filled + "░" * (30 - bar_filled)

        print(
            f"  [{bar}] {model_pct:5.1f}%  "
            f"model {self.model_idx}/{self.total_models}  "
            f"overall {overall_pct:5.1f}%  "
            f"EV={ev:+.3f}  rew={rew:.4f}  "
            f"ETA this={_fmt_time(eta_this)}  "
            f"total={_fmt_time(eta_total)}"
        )
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._t0
        nv = self.model.logger.name_to_value
        ev = nv.get("train/explained_variance", float("nan"))
        print(f"  ✓  Done  {self.model_name}  "
              f"EV={ev:+.3f}  time={_fmt_time(elapsed)}")
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.log_path, index=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m"
    return f"{s//60}m {s%60}s"


def model_paths(window: int, duration: int):
    tag = f"w{window}_td{duration}"
    return (
        os.path.join(MODEL_DIR, f"ppo_multi_perp_{tag}"),
        os.path.join(MODEL_DIR, f"vec_normalize_{tag}.pkl"),
        os.path.join(LOG_DIR,   f"training_log_{tag}.csv"),
    )


def _make_env_fn(window: int, duration: int):
    """Return a zero-argument callable for SubprocVecEnv (avoids closure issues)."""
    def _inner():
        return MultiAssetPerpEnv(
            csv_paths          = CSV_PATHS,
            asset_names        = ASSET_NAMES,
            window_size        = window,
            fee_rate           = FEE_RATE,
            max_leverage       = MAX_LEVERAGE,
            initial_equity     = 1_000.0,
            max_trade_duration = duration,
            min_hold_steps     = MIN_HOLD_STEPS,
            train_end_date     = TRAIN_END_DATE,
        )
    return _inner


def train_one(window: int, duration: int,
              model_idx: int, total_models: int) -> dict:
    """Train a single (window, duration) combination. Returns summary dict."""

    model_path, vecnorm_path, log_path = model_paths(window, duration)

    if SKIP_EXISTING and os.path.exists(f"{model_path}.zip"):
        print(f"  ⏭  Skipping {model_path}  (already exists)")
        return {"window": window, "duration": duration,
                "status": "skipped", "final_ev": None}

    env = SubprocVecEnv([_make_env_fn(window, duration)] * N_ENVS)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        device        = "cpu",
        verbose       = 0,
        learning_rate = 3e-4,
        n_steps       = N_STEPS,   # 512 per env × 4 envs = 2048 total rollout
        batch_size    = 256,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,
    )

    cb = GridProgressCallback(
        total_steps  = TOTAL_STEPS,
        model_idx    = model_idx,
        total_models = total_models,
        model_name   = model_path,
        log_path     = log_path,
        print_every  = PRINT_EVERY,
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

    model.save(model_path)
    env.save(vecnorm_path)

    final_ev = None
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        if "explained_var" in df_log.columns and len(df_log) > 0:
            final_ev = df_log["explained_var"].iloc[-1]

    del model, env
    gc.collect()

    return {"window": window, "duration": duration,
            "status": "trained", "final_ev": final_ev}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    combinations    = list(zip(WINDOW_SIZES, TRADE_DURATIONS))
    total_models    = len(combinations)
    total_steps_all = total_models * TOTAL_STEPS

    print("=" * 70)
    print(f"  PPO Multi-Asset Grid Training (parallel) — {total_models} models")
    print(f"  Workers (SubprocVecEnv) : {N_ENVS}")
    print(f"  n_steps per env         : {N_STEPS}  ({N_ENVS}×{N_STEPS}={N_ENVS*N_STEPS} total/rollout)")
    print(f"  Assets          : {ASSET_NAMES}")
    print(f"  Window sizes    : {WINDOW_SIZES}")
    print(f"  Trade durations : {TRADE_DURATIONS}")
    print(f"  Steps per model : {TOTAL_STEPS:,}")
    print(f"  Total steps     : {total_steps_all:,}")
    print(f"  Skip existing   : {SKIP_EXISTING}")
    print("=" * 70)

    grid_start = time.time()
    results    = []

    for idx, (window, duration) in enumerate(combinations, start=1):
        result = train_one(window, duration, idx, total_models)
        results.append(result)

        pct     = idx / total_models * 100
        elapsed = time.time() - grid_start
        eta     = elapsed / idx * (total_models - idx) if idx < total_models else 0
        print(f"\n  Grid progress: {idx}/{total_models} models  "
              f"({pct:.0f}%)  ETA remaining: {_fmt_time(eta)}\n")

    # ── Summary grid ──────────────────────────────────────────────────────────
    total_time = time.time() - grid_start
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE — {_fmt_time(total_time)}  ({N_ENVS} workers)")
    print("=" * 70)
    print(f"\n  Final Explained Variance Grid  (higher = better learning)\n")

    header = f"  {'Window':>8}  " + "  ".join(f"  td={d:>3}h" for d in TRADE_DURATIONS)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for window in WINDOW_SIZES:
        row = f"  w={window:>4}h    "
        for duration in TRADE_DURATIONS:
            match = next((r for r in results
                          if r["window"] == window and r["duration"] == duration), None)
            if match and match["final_ev"] is not None:
                ev     = match["final_ev"]
                marker = " ★" if ev >= 0.5 else "  "
                row   += f"  {ev:+.3f}{marker}"
            elif match and match["status"] == "skipped":
                row += f"  (skip) "
            else:
                row += f"    n/a  "
        print(row)

    print("\n  ★ = explained variance ≥ 0.5 (good learning)")
    print("\n  Models saved as  ppo_multi_perp/ppo_multi_perp_w{W}_td{D}.zip")
    print("  Run  evaluation_multi.py  to evaluate all models.\n")
