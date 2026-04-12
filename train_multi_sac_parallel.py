"""
train_multi_sac_gpu.py  —  Multi-asset SAC (CPU-parallel, SubprocVecEnv)
=========================================================================
Faster version of train_multi_sac.py using parallel environment collection
instead of GPU (which is slower for small MlpPolicy networks — see SB3 #1245).

Speedup mechanism — SubprocVecEnv:
  N_ENVS environment copies run in separate subprocesses.  Each step call
  advances all N_ENVS envs simultaneously, collecting N_ENVS transitions
  in the time it previously took to collect 1.

  SAC specifics with N_ENVS > 1:
    - train_freq=1 means: do 1 gradient update after each collect call
      (which now yields N_ENVS transitions instead of 1)
    - gradient_steps=N_ENVS preserves the original update-to-data ratio
      (1 gradient step per transition collected), preventing under-fitting
      that would occur if gradient_steps stayed at 1

Expected wall-clock improvement vs original script:
  M1 Max (10 perf cores, N_ENVS=4) : ~2–3×
  Any modern quad-core CPU          : ~1.8–2.5×

  SAC sees slightly less speedup than PPO because gradient updates
  (256-unit network) are relatively more expensive than the env steps.

Memory note:
  Each subprocess loads all 4 CSVs + maintains its own replay buffer slice.
  The main process holds the 1M-transition replay buffer.
  On M1 Max 64 GB this is not a concern.  Reduce N_ENVS if RAM is tight.
"""

import os
import gc
import time
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from multi_asset_env import MultiAssetPerpEnv


# ── Parallelism ───────────────────────────────────────────────────────────────

N_ENVS = 4   # parallel environment workers


# ── Assets ────────────────────────────────────────────────────────────────────

CSV_PATHS = [
    "BTCUSDT_perpetual_futures_processed_2020-2026.csv",
    "ETHUSDT_perpetual_futures_processed_2020-2026.csv",
    "SOLUSDT_perpetual_futures_processed_2020-2026.csv",
    "BNBUSDT_perpetual_futures_processed_2020-2026.csv",
]
ASSET_NAMES = ["BTC", "ETH", "SOL", "BNB"]

# ── Configs ───────────────────────────────────────────────────────────────────

CONFIGS = [
    (168, 168),   # mirrors single-asset winner
    ( 96,  48),   # best drawdown in PPO grid
    ( 24, 168),   # best Sharpe in PPO grid
]

TOTAL_STEPS    = 3_000_000
TRAIN_END_DATE = "2023-12-31"
FEE_RATE       = 0.0005
MAX_LEVERAGE   = 3.0
MIN_HOLD_STEPS = 24   # raised from 12 — forces commitment, halves max trade frequency

MODEL_DIR = "ppo_multi_perp"
LOG_DIR   = "training_log_multi"
PRINT_EVERY = 100_000


# ── Progress callback ─────────────────────────────────────────────────────────

class SACProgressCallback(BaseCallback):
    def __init__(self, total_steps, model_idx, total_models, model_name,
                 log_path, print_every=100_000):
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

    def _on_training_start(self):
        self._t0 = time.time()
        print(f"\n  ▶  SAC starting  {self.model_name}"
              f"  [{self.model_idx}/{self.total_models}]"
              f"  workers={N_ENVS}  gradient_steps/collect={N_ENVS}")

    def _on_step(self):
        if self.num_timesteps - self._last_print < self.print_every:
            return True
        self._last_print = self.num_timesteps

        elapsed   = time.time() - self._t0
        model_pct = self.num_timesteps / self.total_steps * 100
        steps_done_total = (self.model_idx - 1) * self.total_steps + self.num_timesteps
        overall_pct = steps_done_total / (self.total_models * self.total_steps) * 100

        rate      = self.num_timesteps / (elapsed + 1e-6)
        eta_this  = (self.total_steps - self.num_timesteps) / (rate + 1e-6)
        remaining = self.total_models - self.model_idx
        eta_total = eta_this + remaining * self.total_steps / (rate + 1e-6)

        nv  = self.model.logger.name_to_value
        rew = nv.get("rollout/ep_rew_mean",   float("nan"))
        ent = nv.get("train/ent_coef",        float("nan"))
        ql  = nv.get("train/critic_loss",     float("nan"))

        self.rows.append({
            "timestep":    self.num_timesteps,
            "ep_rew_mean": rew,
            "ent_coef":    ent,
            "critic_loss": ql,
        })

        bar_filled = int(30 * model_pct / 100)
        bar = "█" * bar_filled + "░" * (30 - bar_filled)
        print(
            f"  [{bar}] {model_pct:5.1f}%  "
            f"model {self.model_idx}/{self.total_models}  "
            f"overall {overall_pct:5.1f}%  "
            f"rew={rew:.4f}  ent={ent:.3f}  "
            f"ETA this={_fmt(eta_this)}  total={_fmt(eta_total)}"
        )
        return True

    def _on_training_end(self):
        elapsed = time.time() - self._t0
        print(f"  ✓  SAC done  {self.model_name}  time={_fmt(elapsed)}"
              f"  workers={N_ENVS}")
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.log_path, index=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(seconds):
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m"
    return f"{s//60}m {s%60}s"


def model_paths(window, duration):
    tag = f"w{window}_td{duration}"
    return (
        os.path.join(MODEL_DIR, f"sac_multi_perp_{tag}"),
        os.path.join(LOG_DIR,   f"sac_log_{tag}.csv"),
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


# ── Train one config ──────────────────────────────────────────────────────────

def train_one(window, duration, model_idx, total_models):
    model_path, log_path = model_paths(window, duration)

    env = SubprocVecEnv([_make_env_fn(window, duration)] * N_ENVS)

    model = SAC(
        "MlpPolicy",
        env,
        device           = "cpu",
        verbose          = 0,
        learning_rate    = 3e-4,
        buffer_size      = 1_000_000,
        learning_starts  = 10_000,
        batch_size       = 256,
        tau              = 0.005,
        gamma            = 0.99,
        train_freq       = 1,
        gradient_steps   = N_ENVS,   # 1 grad step per transition collected
        ent_coef         = "auto",
        target_entropy   = "auto",
        policy_kwargs    = dict(net_arch=[256, 256]),
        seed             = 42,
    )

    cb = SACProgressCallback(
        total_steps  = TOTAL_STEPS,
        model_idx    = model_idx,
        total_models = total_models,
        model_name   = model_path,
        log_path     = log_path,
        print_every  = PRINT_EVERY,
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)
    model.save(model_path)

    del model, env
    gc.collect()

    return {"window": window, "duration": duration, "status": "trained"}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    total_models    = len(CONFIGS)
    total_steps_all = total_models * TOTAL_STEPS

    print("=" * 70)
    print(f"  SAC Multi-Asset Training (parallel) — {total_models} configs")
    print(f"  Workers (SubprocVecEnv) : {N_ENVS}")
    print(f"  gradient_steps          : {N_ENVS}  (matches workers — 1 per transition)")
    print(f"  Assets          : {ASSET_NAMES}")
    print(f"  Configs         : {CONFIGS}")
    print(f"  Steps per model : {TOTAL_STEPS:,}")
    print(f"  Total steps     : {total_steps_all:,}")
    print(f"  Algorithm       : SAC (off-policy, replay buffer 1M)")
    print("=" * 70)

    grid_start = time.time()
    results    = []

    for idx, (window, duration) in enumerate(CONFIGS, start=1):
        result = train_one(window, duration, idx, total_models)
        results.append(result)

        pct     = idx / total_models * 100
        elapsed = time.time() - grid_start
        eta     = elapsed / idx * (total_models - idx) if idx < total_models else 0
        print(f"\n  Grid progress: {idx}/{total_models}  ({pct:.0f}%)"
              f"  ETA remaining: {_fmt(eta)}\n")

    total_time = time.time() - grid_start
    print("\n" + "=" * 70)
    print(f"  SAC TRAINING COMPLETE — {_fmt(total_time)}  ({N_ENVS} workers)")
    print(f"  Models saved as  {MODEL_DIR}/sac_multi_perp_w{{W}}_td{{D}}.zip")
    print(f"  Run  evaluation_multi_sac.py  to evaluate.")
    print("=" * 70)
