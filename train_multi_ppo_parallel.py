"""
train_multi_ppo_parallel.py  --  Multi-asset PPO (CPU-parallel, SubprocVecEnv)
===============================================================================
Swing-trader configuration -- v3.2 reward, w336/td336 winner.

Smoke test verdict (Apr 2026):
  All short-window configs (w24, w96, w168 with td48/td168) showed fee drag
  3000%+ or near-wipeout at 500k steps. w336/td336 was the only config to
  produce a positive OOS return (+9.4%, $1,093 from $1,000) at 500k steps.

  Why w336/td336 wins:
    - 2-week context window gives the agent enough bars to interpret
      price_vs_sma500 and the v3.2 trend alignment reward meaningfully.
    - 2-week max hold (td=336) prevents forced premature exits and the
      re-entry churn cycle that caused 3000%+ fee drag in short-td configs.
    - min_hold=72 gives trades room to breathe before SL/TP can fire.
    - [512,256] network compresses the 26,910-dim observation space.

  5M steps chosen over 3M because:
    - 10x larger obs space + 25x more network parameters vs previous configs.
    - Trend alignment reward needs many completed trades to shape the policy.
    - With min_hold=72 and td=336, fewer trades per episode -> need more episodes.

Parallelism -- SubprocVecEnv:
  4 envs x 512 steps = 2048 transitions per rollout.
  ~3-4x faster collection than single-env training.

Memory note:
  Each subprocess loads all 4 CSVs independently.
  On M1 Max 64 GB this is not a concern.
"""

import os
import gc
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from multi_asset_env import MultiAssetPerpEnv


# -- Parallelism ---------------------------------------------------------------

N_ENVS  = 4              # parallel environment workers (safe on M1 Max)
N_STEPS = 2048 // N_ENVS  # 512 per env x 4 = 2048 total rollout


# -- Assets --------------------------------------------------------------------

CSV_PATHS = [
    "BTCUSDT_perpetual_futures_processed_2020-2026.csv",
    "ETHUSDT_perpetual_futures_processed_2020-2026.csv",
    "SOLUSDT_perpetual_futures_processed_2020-2026.csv",
    "BNBUSDT_perpetual_futures_processed_2020-2026.csv",
]
ASSET_NAMES = ["BTC", "ETH", "SOL", "BNB"]


# -- Winning config (smoke test Apr 2026) --------------------------------------

WINDOW_SIZE    = 336        # 2-week context window
TRADE_DURATION = 336        # 2-week max hold
MIN_HOLD_STEPS = 72         # 3-day minimum hold before SL/TP can fire
NET_ARCH       = [512, 256] # network sized for 26,910-dim obs space

TOTAL_STEPS    = 5_000_000  # 5M -- needed for [512,256] net on 26k-dim obs
TRAIN_END_DATE = "2023-12-31"
FEE_RATE       = 0.0005
MAX_LEVERAGE   = 3.0

MODEL_DIR = "ppo_multi_perp"
LOG_DIR   = "training_log_multi"

PRINT_EVERY = 250_000       # progress print every 250k steps (20 prints total)


# -- Progress callback ---------------------------------------------------------

class TrainingCallback(BaseCallback):
    """Prints progress and key metrics every PRINT_EVERY steps."""

    def __init__(self, total_steps, model_name, log_path, print_every=250_000):
        super().__init__(verbose=0)
        self.total_steps = total_steps
        self.model_name  = model_name
        self.log_path    = log_path
        self.print_every = print_every
        self._last_print = 0
        self.rows        = []
        self._t0         = None

    def _on_training_start(self):
        self._t0 = time.time()
        print(f"\n  Starting  {self.model_name}")
        print(f"  Config    : w{WINDOW_SIZE}/td{TRADE_DURATION}/mh{MIN_HOLD_STEPS}"
              f"  net={NET_ARCH}  steps={TOTAL_STEPS:,}")
        print(f"  Workers   : {N_ENVS}  x  {N_STEPS} steps/env  =  {N_ENVS*N_STEPS} total/rollout")

    def _on_rollout_start(self):
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

    def _on_step(self):
        if self.num_timesteps - self._last_print < self.print_every:
            return True
        self._last_print = self.num_timesteps

        elapsed  = time.time() - self._t0
        pct      = self.num_timesteps / self.total_steps * 100
        rate     = self.num_timesteps / (elapsed + 1e-6)
        eta      = (self.total_steps - self.num_timesteps) / (rate + 1e-6)

        nv  = self.model.logger.name_to_value
        ev  = nv.get("train/explained_variance", float("nan"))
        rew = nv.get("rollout/ep_rew_mean",       float("nan"))
        kl  = nv.get("train/approx_kl",           float("nan"))

        bar_filled = int(30 * pct / 100)
        bar        = "=" * bar_filled + "." * (30 - bar_filled)

        print(
            f"  [{bar}] {pct:5.1f}%  "
            f"steps={self.num_timesteps:,}  "
            f"EV={ev:+.3f}  rew={rew:.4f}  kl={kl:.4f}  "
            f"ETA={_fmt_time(eta)}"
        )
        return True

    def _on_training_end(self):
        elapsed = time.time() - self._t0
        nv = self.model.logger.name_to_value
        ev = nv.get("train/explained_variance", float("nan"))
        print(f"\n  Done  {self.model_name}  EV={ev:+.3f}  time={_fmt_time(elapsed)}")
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
            print(f"  Training log -> {self.log_path}")


# -- Helpers -------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m"
    return f"{s//60}m {s%60}s"


def _make_env_fn():
    """Zero-argument callable for SubprocVecEnv (avoids closure capture issues)."""
    def _inner():
        return MultiAssetPerpEnv(
            csv_paths          = CSV_PATHS,
            asset_names        = ASSET_NAMES,
            window_size        = WINDOW_SIZE,
            fee_rate           = FEE_RATE,
            max_leverage       = MAX_LEVERAGE,
            initial_equity     = 1_000.0,
            max_trade_duration = TRADE_DURATION,
            min_hold_steps     = MIN_HOLD_STEPS,
            train_end_date     = TRAIN_END_DATE,
        )
    return _inner


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    tag         = f"w{WINDOW_SIZE}_td{TRADE_DURATION}_mh{MIN_HOLD_STEPS}"
    model_path  = os.path.join(MODEL_DIR, f"ppo_multi_perp_{tag}")
    vecnorm_path = os.path.join(MODEL_DIR, f"vec_normalize_{tag}.pkl")
    log_path    = os.path.join(LOG_DIR,   f"training_log_{tag}.csv")

    print("=" * 70)
    print("  PPO Multi-Asset -- Swing Trader -- 5M steps")
    print("=" * 70)
    print(f"  Config       : w{WINDOW_SIZE} / td{TRADE_DURATION} / min_hold={MIN_HOLD_STEPS}")
    print(f"  Network      : {NET_ARCH}  (obs ~26,910 dims)")
    print(f"  Total steps  : {TOTAL_STEPS:,}")
    print(f"  Workers      : {N_ENVS}  x  {N_STEPS} n_steps/env  =  {N_ENVS*N_STEPS}/rollout")
    print(f"  Assets       : {ASSET_NAMES}")
    print(f"  Train period : up to {TRAIN_END_DATE}")
    print(f"  Reward       : v3.2 (trend alignment via price_vs_sma500)")
    print(f"  Model out    : {model_path}.zip")
    print("=" * 70)

    env   = SubprocVecEnv([_make_env_fn()] * N_ENVS)
    norm  = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        norm,
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
    )

    cb = TrainingCallback(
        total_steps = TOTAL_STEPS,
        model_name  = tag,
        log_path    = log_path,
        print_every = PRINT_EVERY,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

    model.save(model_path)
    norm.save(vecnorm_path)

    total_time = time.time() - t0
    print(f"\n  Model saved  -> {model_path}.zip")
    print(f"  VecNormalize -> {vecnorm_path}")
    print(f"  Total time   : {_fmt_time(total_time)}")
    print(f"\n  Next: run evaluation_multi.py to evaluate OOS performance.")

    del model, norm, env
    gc.collect()
