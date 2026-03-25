"""
train_solo_ppo.py
=================
Train a PPO agent on SOL perpetual futures.

Window size experiment:
    Change WINDOW_SIZE below and re-run to train a model for each window.
    Suggested values to compare: 24, 48, 96, 168
    Each trained model is saved as  ppo_sol_perp_w{WINDOW_SIZE}.zip
    The evaluation sweep in evaluation.py will pick them up automatically.

Recommended timesteps: 1_000_000 – 2_000_000 for meaningful convergence.
"""

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sol_env import SolPerpEnv

# ── Configurable parameters ───────────────────────────────────────────────────

WINDOW_SIZE  = 48          # lookback window in hours — change to 24/96/168 to test
TOTAL_STEPS  = 2_000_000   # 2M needed for meaningful convergence with new reward

CSV_PATH     = "SOLUSDT_perpetual_futures_processed_2020-2025.csv"
FEE_RATE     = 0.0005      # Binance taker fee (0.05%)
MAX_LEVERAGE = 2.0

MODEL_NAME   = f"ppo_sol_perp_w{WINDOW_SIZE}"
VECNORM_NAME = f"vec_normalize_w{WINDOW_SIZE}.pkl"
LOG_NAME     = f"training_log_w{WINDOW_SIZE}.csv"

# ── MetricsLogger callback ────────────────────────────────────────────────────

class MetricsLogger(BaseCallback):
    """
    Logs key PPO training metrics to a CSV file after each rollout.

    Captured keys (from model.logger.name_to_value):
        train/policy_gradient_loss, train/value_loss, train/entropy_loss,
        train/explained_variance, train/approx_kl, train/clip_fraction,
        rollout/ep_rew_mean, rollout/ep_len_mean

    CSV is written to  training_log_w{WINDOW_SIZE}.csv  at training end.
    Progress is printed to stdout every 50,000 timesteps.
    """

    def __init__(self, log_path: str, print_every: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_path    = log_path
        self.print_every = print_every
        self.rows        = []
        self._last_print = 0

    def _on_rollout_start(self) -> None:
        nv = self.model.logger.name_to_value
        row = {
            "timestep":      self.num_timesteps,
            "policy_loss":   nv.get("train/policy_gradient_loss"),
            "value_loss":    nv.get("train/value_loss"),
            "entropy":       nv.get("train/entropy_loss"),
            "explained_var": nv.get("train/explained_variance"),
            "approx_kl":     nv.get("train/approx_kl"),
            "clip_fraction": nv.get("train/clip_fraction"),
            "ep_rew_mean":   nv.get("rollout/ep_rew_mean"),
            "ep_len_mean":   nv.get("rollout/ep_len_mean"),
        }
        self.rows.append(row)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.print_every:
            self._last_print = self.num_timesteps
            nv  = self.model.logger.name_to_value
            ev  = nv.get("train/explained_variance", float("nan"))
            rew = nv.get("rollout/ep_rew_mean",       float("nan"))
            print(
                f"  [MetricsLogger] step={self.num_timesteps:>8,}"
                f"  exp_var={ev:+.3f}"
                f"  ep_rew={rew:.4f}"
            )
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        df.to_csv(self.log_path, index=False)
        print(f"  [MetricsLogger] Saved {len(df)} rows → {self.log_path}")


# ── Environment ───────────────────────────────────────────────────────────────

def make_env():
    return SolPerpEnv(
        csv_path=CSV_PATH,
        window_size=WINDOW_SIZE,
        fee_rate=FEE_RATE,
        max_leverage=MAX_LEVERAGE,
        initial_equity=1_000.0,
    )

env = DummyVecEnv([make_env])

# VecNormalize: normalises observations to zero mean / unit variance.
# norm_reward=False keeps the tiered reward signal intact.
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# ── PPO model ─────────────────────────────────────────────────────────────────

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,     # encourages exploration; reduce to 0.001 once policy matures
)

# ── Train ─────────────────────────────────────────────────────────────────────

print(f"Training window={WINDOW_SIZE}h  steps={TOTAL_STEPS:,}  model={MODEL_NAME}")
_metrics_logger = MetricsLogger(log_path=LOG_NAME, print_every=50_000)
model.learn(total_timesteps=TOTAL_STEPS, callback=_metrics_logger)

model.save(MODEL_NAME)
env.save(VECNORM_NAME)
print(f"Saved  {MODEL_NAME}.zip  and  {VECNORM_NAME}")

# ── Quick deterministic test run ─────────────────────────────────────────────

obs = env.reset()
done = False
steps = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    done = bool(done[0])
    steps += 1
    if steps % 1000 == 0:
        print(info[0])

print(f"Test episode finished in {steps} steps")
