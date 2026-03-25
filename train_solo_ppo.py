"""
train_solo_ppo.py  —  Automated grid trainer
=============================================
Trains all combinations of window size × trade duration automatically.

Grid:
    WINDOW_SIZES    = [24, 48, 96, 168]   hours of market lookback
    TRADE_DURATIONS = [24, 48, 72, 168]   max hours per committed trade
    Total           = 16 models

Each model is saved as:
    ppo_sol_perp_w{W}_td{D}.zip
    vec_normalize_w{W}_td{D}.pkl
    training_log_w{W}_td{D}.csv

To train a single model instead of the full grid:
    Set  WINDOW_SIZES = [48]  and  TRADE_DURATIONS = [48]

Estimated time (CPU):
    ~25–40 min per model × 16 = 7–11 hours  (run overnight)

SKIP_EXISTING = True  means already-saved models are skipped automatically,
so you can safely re-run after a crash and it will continue from where it left off.
"""

import os
import gc
import time
import itertools
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sol_env import SolPerpEnv


# ── Grid ──────────────────────────────────────────────────────────────────────

WINDOW_SIZES    = [24, 48, 96, 168]
TRADE_DURATIONS = [24, 48, 72, 168]

TOTAL_STEPS  = 2_000_000    # per model — increase to 3M for sharper convergence
SKIP_EXISTING = True        # skip already-trained models (safe to re-run)

CSV_PATH     = "SOLUSDT_perpetual_futures_processed_2020-2025.csv"
FEE_RATE     = 0.0005
MAX_LEVERAGE = 3.0

PRINT_EVERY  = 100_000      # progress report every N training steps


# ── Progress callback ─────────────────────────────────────────────────────────

class GridProgressCallback(BaseCallback):
    """
    Prints a live progress bar and key metrics every PRINT_EVERY steps.
    Tracks both per-model progress and overall grid progress.
    Saves training metrics to CSV on completion.
    """

    def __init__(self, total_steps, model_idx, total_models,
                 model_name, log_path, print_every=100_000):
        super().__init__(verbose=0)
        self.total_steps  = total_steps
        self.model_idx    = model_idx       # 1-based
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
              f"  [{self.model_idx}/{self.total_models}]")

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

        elapsed     = time.time() - self._t0
        model_pct   = self.num_timesteps / self.total_steps * 100

        # Overall grid percentage
        steps_done_total = (self.model_idx - 1) * self.total_steps \
                           + self.num_timesteps
        overall_pct = steps_done_total / (self.total_models * self.total_steps) * 100

        # ETA calculations
        rate = self.num_timesteps / (elapsed + 1e-6)          # steps/sec
        eta_this  = (self.total_steps - self.num_timesteps) / (rate + 1e-6)
        remaining_models = self.total_models - self.model_idx
        eta_total = eta_this + remaining_models * self.total_steps / (rate + 1e-6)

        nv = self.model.logger.name_to_value
        ev = nv.get("train/explained_variance", float("nan"))
        rew = nv.get("rollout/ep_rew_mean",      float("nan"))

        # ASCII progress bar (30 chars)
        bar_filled = int(30 * model_pct / 100)
        bar = "█" * bar_filled + "░" * (30 - bar_filled)

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
    """Format seconds as  Xh Ym  or  Ym Zs."""
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m"
    return f"{s//60}m {s%60}s"


def model_names(window: int, duration: int):
    tag = f"w{window}_td{duration}"
    return (
        f"ppo_sol_perp_{tag}",
        f"vec_normalize_{tag}.pkl",
        f"training_log_{tag}.csv",
    )


def train_one(window: int, duration: int,
              model_idx: int, total_models: int) -> dict:
    """Train a single (window, duration) combination. Returns summary dict."""

    model_name, vecnorm_name, log_name = model_names(window, duration)

    if SKIP_EXISTING and os.path.exists(f"{model_name}.zip"):
        print(f"  ⏭  Skipping {model_name}  (already exists)")
        return {"window": window, "duration": duration,
                "status": "skipped", "final_ev": None}

    def make_env():
        return SolPerpEnv(
            csv_path           = CSV_PATH,
            window_size        = window,
            fee_rate           = FEE_RATE,
            max_leverage       = MAX_LEVERAGE,
            initial_equity     = 1_000.0,
            max_trade_duration = duration,
        )

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose       = 0,          # suppressed — callback handles output
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 64,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,
    )

    cb = GridProgressCallback(
        total_steps   = TOTAL_STEPS,
        model_idx     = model_idx,
        total_models  = total_models,
        model_name    = model_name,
        log_path      = log_name,
        print_every   = PRINT_EVERY,
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

    model.save(model_name)
    env.save(vecnorm_name)

    # Read final explained variance from log
    final_ev = None
    if os.path.exists(log_name):
        df_log = pd.read_csv(log_name)
        if "explained_var" in df_log.columns and len(df_log) > 0:
            final_ev = df_log["explained_var"].iloc[-1]

    # Free memory before next model
    del model, env
    gc.collect()

    return {"window": window, "duration": duration,
            "status": "trained", "final_ev": final_ev}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    combinations  = list(itertools.product(WINDOW_SIZES, TRADE_DURATIONS))
    total_models  = len(combinations)
    total_steps_all = total_models * TOTAL_STEPS

    print("=" * 70)
    print(f"  PPO Grid Training — {total_models} models")
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

        # Print overall grid progress after each model
        pct = idx / total_models * 100
        elapsed = time.time() - grid_start
        eta = elapsed / idx * (total_models - idx) if idx < total_models else 0
        print(f"\n  Grid progress: {idx}/{total_models} models  "
              f"({pct:.0f}%)  ETA remaining: {_fmt_time(eta)}\n")

    # ── Summary grid ──────────────────────────────────────────────────────────
    total_time = time.time() - grid_start
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE — {_fmt_time(total_time)}")
    print("=" * 70)
    print(f"\n  Final Explained Variance Grid  (higher = better learning)\n")

    # Header row
    header = f"  {'Window':>8}  " + "  ".join(f"  td={d:>3}h" for d in TRADE_DURATIONS)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for window in WINDOW_SIZES:
        row = f"  w={window:>4}h    "
        for duration in TRADE_DURATIONS:
            match = next((r for r in results
                          if r["window"] == window and r["duration"] == duration), None)
            if match and match["final_ev"] is not None:
                ev = match["final_ev"]
                marker = " ★" if ev >= 0.5 else "  "
                row += f"  {ev:+.3f}{marker}"
            elif match and match["status"] == "skipped":
                row += f"  (skip) "
            else:
                row += f"    n/a  "
        print(row)

    print("\n  ★ = explained variance ≥ 0.5 (good learning)")
    print("\n  Models saved as  ppo_sol_perp_w{W}_td{D}.zip")
    print("  Run  evaluation.py  and point it at any model to compare.\n")
