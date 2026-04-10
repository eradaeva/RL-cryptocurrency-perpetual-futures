"""
train_multi_sac.py  —  Multi-asset SAC trainer (top 3 configs)
==============================================================
Trains the same 3 configs as train_multi_ppo.py but using SAC instead of PPO.

Why SAC for this environment:
  - Off-policy: uses a replay buffer, so every transition is reused many times.
    At 3M steps SAC effectively makes 3-5× more useful gradient updates than PPO.
  - Better suited to continuous action spaces (direction, risk, SL, TP).
  - Entropy regularisation encourages exploration — naturally produces more
    diverse action distributions, which may increase flat time.
  - Sample efficiency matters here: the multi-asset obs space is ~13k dims for
    w=168, and PPO at 3M steps is provably insufficient to converge it.

Note: SAC does NOT use VecNormalize (it normalises observations internally via
running stats in the replay buffer preprocessing). Models saved as:
    ppo_multi_perp/sac_multi_perp_w{W}_td{D}.zip
    training_log_multi/sac_log_w{W}_td{D}.csv

Configs trained (same 3 as PPO retrain):
    w=168h / td=168h
    w=96h  / td=48h
    w=24h  / td=168h

Estimated time: ~1.5–2h per model × 3 = 4.5–6h  (run overnight)
"""

import os
import gc
import time
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from multi_asset_env import MultiAssetPerpEnv

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
MIN_HOLD_STEPS = 12

MODEL_DIR = "ppo_multi_perp"    # same dir — easy comparison with PPO models
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
              f"  [{self.model_idx}/{self.total_models}]")

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
        print(f"  ✓  SAC done  {self.model_name}  time={_fmt(elapsed)}")
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


# ── Train one config ──────────────────────────────────────────────────────────

def train_one(window, duration, model_idx, total_models):
    model_path, log_path = model_paths(window, duration)

    def make_env():
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

    # SAC works directly with a single (non-vectorised) env
    # but SB3's SAC also accepts DummyVecEnv
    env = DummyVecEnv([make_env])

    model = SAC(
        "MlpPolicy",
        env,
        verbose          = 0,
        learning_rate    = 3e-4,
        buffer_size      = 1_000_000,   # replay buffer: stores 1M transitions
        learning_starts  = 10_000,      # random exploration before first update
        batch_size       = 256,         # larger than PPO's 64 — stable for off-policy
        tau              = 0.005,       # soft target update
        gamma            = 0.99,
        train_freq       = 1,           # update every step
        gradient_steps   = 1,
        ent_coef         = "auto",      # automatic entropy tuning
        target_entropy   = "auto",
        policy_kwargs    = dict(net_arch=[256, 256]),   # deeper than PPO default
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
    print(f"  SAC Multi-Asset Training — {total_models} configs")
    print(f"  Assets          : {ASSET_NAMES}")
    print(f"  Configs         : {CONFIGS}")
    print(f"  Steps per model : {TOTAL_STEPS:,}")
    print(f"  Total steps     : {total_steps_all:,}")
    print(f"  Algorithm       : SAC (off-policy, replay buffer 1M)")
    print(f"  Flat zone       : ±0.7  (70% of action range)")
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
    print(f"  SAC TRAINING COMPLETE — {_fmt(total_time)}")
    print(f"  Models saved as  {MODEL_DIR}/sac_multi_perp_w{{W}}_td{{D}}.zip")
    print(f"  Run  evaluation_multi_sac.py  to evaluate.")
    print("=" * 70)
