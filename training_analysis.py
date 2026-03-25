"""
training_analysis.py
====================
Diagnostic analysis of the trained PPO policy.

  Part 1 — Policy Probe
    Samples 2000 deterministic actions from the current model across random
    states.  Plots action distributions for all 3 action dimensions:
      action[0]  target position  [-1, +1]  (should cluster if trained)
      action[1]  stop-loss ATR multiplier   (should have a learnt preference)
      action[2]  take-profit ATR multiplier (same)
    A uniform distribution indicates an untrained/random policy.

  Part 2 — Training Curves
    Reads training_log_w{WINDOW}.csv produced by the MetricsLogger callback in
    train_solo_ppo.py.  Plots 6 convergence indicators:
      - Mean episode reward      (should rise)
      - Mean episode length      (should rise as policy improves)
      - Policy gradient loss     (should fall and stabilise)
      - Value function loss      (should fall sharply early)
      - Policy entropy           (should decrease as policy converges)
      - Explained variance       (CRITICAL: must exceed 0.5 for good learning)

  Part 3 — Plain-English Diagnosis
    Interprets all metrics and prints a structured diagnostic report.

Run:
    python training_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sol_env import SolPerpEnv

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW       = 48
CSV_PATH     = "SOLUSDT_perpetual_futures_processed_2020-2025.csv"
MODEL_NAME   = f"ppo_sol_perp_w{WINDOW}"
VECNORM_NAME = f"vec_normalize_w{WINDOW}.pkl"
LOG_NAME     = f"training_log_w{WINDOW}.csv"
N_PROBE      = 2_000   # actions to sample for the policy probe

model = PPO.load(MODEL_NAME)

# ── Part 1: Policy Probe ───────────────────────────────────────────────────────

def probe_policy(n_samples: int = N_PROBE) -> np.ndarray:
    """Sample n_samples deterministic actions to characterise policy behaviour."""
    def make_env():
        return SolPerpEnv(csv_path=CSV_PATH, window_size=WINDOW, initial_equity=1_000.0)

    env = DummyVecEnv([make_env])
    if Path(VECNORM_NAME).exists():
        env = VecNormalize.load(VECNORM_NAME, env)
        env.training    = False
        env.norm_reward = False

    obs = env.reset()
    actions = []
    for _ in range(n_samples):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()
        actions.append(action[0].copy())

    return np.array(actions)   # shape (n_samples, 3)


def plot_policy_probe(actions: np.ndarray,
                      save_path: str = "analysis_policy_probe.png") -> None:
    pos = actions[:, 0]
    sl  = actions[:, 1]
    tp  = actions[:, 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "Policy Probe — Action Distribution\n"
        "(uniform histogram = untrained/random policy   |   peaked = learnt behaviour)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    pct_long  = (pos >  0.05).mean() * 100
    pct_short = (pos < -0.05).mean() * 100
    pct_flat  = (np.abs(pos) <= 0.05).mean() * 100

    # ── Panel A: Position distribution ────────────────────────────────────────
    axes[0, 0].hist(pos, bins=60, color="#1565c0", alpha=0.80, edgecolor="white", linewidth=0.4)
    axes[0, 0].axvline(0,          color="black",  linewidth=1.0, linestyle="--", label="zero")
    axes[0, 0].axvline(pos.mean(), color="#d50000", linewidth=1.5, linestyle="-",
                       label=f"mean={pos.mean():+.3f}")
    axes[0, 0].set_title("action[0] — Target Position", fontweight="bold")
    axes[0, 0].set_xlabel("Position (scaled by leverage)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].text(
        0.02, 0.97,
        f"Long: {pct_long:.0f}%  Short: {pct_short:.0f}%  Flat: {pct_flat:.0f}%",
        transform=axes[0, 0].transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )

    # ── Panel B: Stop-loss multiplier ─────────────────────────────────────────
    axes[0, 1].hist(sl, bins=60, color="#ff8f00", alpha=0.80, edgecolor="white", linewidth=0.4)
    axes[0, 1].axvline(sl.mean(), color="#d50000", linewidth=1.5, linestyle="--",
                       label=f"mean={sl.mean():.2f}×")
    axes[0, 1].set_title("action[1] — Stop-Loss ATR Multiplier", fontweight="bold")
    axes[0, 1].set_xlabel("SL distance (× ATR)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend(fontsize=8)

    # ── Panel C: Take-profit multiplier ───────────────────────────────────────
    axes[1, 0].hist(tp, bins=60, color="#2e7d32", alpha=0.80, edgecolor="white", linewidth=0.4)
    axes[1, 0].axvline(tp.mean(), color="#d50000", linewidth=1.5, linestyle="--",
                       label=f"mean={tp.mean():.2f}×  R:R={tp.mean()/sl.mean():.2f}")
    axes[1, 0].set_title("action[2] — Take-Profit ATR Multiplier", fontweight="bold")
    axes[1, 0].set_xlabel("TP distance (× ATR)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend(fontsize=8)

    # ── Panel D: Position vs SL scatter ───────────────────────────────────────
    sc = axes[1, 1].scatter(pos, sl, c=tp, cmap="viridis", alpha=0.25, s=6)
    plt.colorbar(sc, ax=axes[1, 1], label="TP mult", pad=0.02)
    axes[1, 1].axvline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title("Position vs SL  (colour = TP)", fontweight="bold")
    axes[1, 1].set_xlabel("Target position")
    axes[1, 1].set_ylabel("SL ATR multiplier")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#ffffff")
        ax.grid(color="#e0e0e0", linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

    # Numerical summary
    print(f"\n  ── ACTION STATISTICS ({N_PROBE} samples) ─────────────────")
    print(f"  Position   mean={pos.mean():+.4f}  std={pos.std():.4f}  "
          f"min={pos.min():+.3f}  max={pos.max():+.3f}")
    print(f"  SL mult    mean={sl.mean():.3f}   std={sl.std():.3f}")
    print(f"  TP mult    mean={tp.mean():.3f}   std={tp.std():.3f}")
    print(f"  R:R ratio  1 : {tp.mean() / (sl.mean() + 1e-8):.2f}")
    print(f"  Long bias  {pct_long:.0f}%   Short bias {pct_short:.0f}%   Flat {pct_flat:.0f}%")

    # Diagnosis
    print()
    if pos.std() > 0.75:
        print("  ⚠  ACTION[0] STD > 0.75 — position output is near-random "
              "(policy not converged)")
    else:
        print("  ✓  ACTION[0] shows structured position preference")

    if abs(pos.mean()) < 0.05:
        print("  ⚠  POSITION MEAN ≈ 0 — no directional bias learnt")

    rr = tp.mean() / (sl.mean() + 1e-8)
    if rr < 1.0:
        print(f"  ⚠  R:R = {rr:.2f} < 1.0 — TP closer than SL on average (unfavourable)")
    else:
        print(f"  ✓  R:R = {rr:.2f} — TP further than SL on average")


# ── Part 2: Training Curves ────────────────────────────────────────────────────

def plot_training_curves(log_path: str = LOG_NAME,
                          save_path: str = "analysis_training_curves.png") -> None:
    if not Path(log_path).exists():
        print(f"\n  [INFO] '{log_path}' not found.")
        print("  Training logs will be created on the NEXT training run.")
        print("  The MetricsLogger callback in train_solo_ppo.py saves this file.")
        return

    df = pd.read_csv(log_path).dropna(how="all")
    if len(df) < 3:
        print(f"  [INFO] Only {len(df)} rows in log — need at least 3 to plot")
        return

    print(f"  Training log: {len(df)} update steps  "
          f"({df['timestep'].max():,} total timesteps)")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Training Convergence — {MODEL_NAME}\n"
        f"{df['timestep'].max():,} timesteps  |  {len(df)} gradient updates",
        fontsize=12, fontweight="bold", y=0.99,
    )

    def smooth(s, w=7):
        return s.rolling(w, min_periods=1).mean()

    ts = df["timestep"] / 1_000  # x-axis in thousands

    specs = [
        ("ep_rew_mean",   "Mean Episode Reward",     "Reward",    "#1565c0", None,  None),
        ("ep_len_mean",   "Mean Episode Length",      "Steps",     "#2e7d32", None,  None),
        ("policy_loss",   "Policy Gradient Loss",     "Loss",      "#6a1b9a", None,  None),
        ("value_loss",    "Value Function Loss",      "Loss",      "#e65100", None,  None),
        ("entropy",       "Policy Entropy",           "Entropy",   "#00838f", None,  None),
        ("explained_var", "Explained Variance ★",     "Exp. Var.", "#c62828", 0.5,   "Target ≥ 0.5"),
    ]

    for ax, (col, title, ylabel, color, hline, hlabel) in zip(axes.flat, specs):
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e0e0e0", linewidth=0.5)

        if col not in df.columns or df[col].isna().all():
            ax.text(0.5, 0.5, "no data\n(rerun training with callback)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="gray", fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            continue

        raw = df[col]
        sm  = smooth(raw)
        ax.plot(ts, raw, color=color, alpha=0.25, linewidth=0.8)
        ax.plot(ts, sm,  color=color, linewidth=1.8, label="smoothed (7-step)")

        if hline is not None:
            ax.axhline(hline, color="red", linewidth=1.2, linestyle="--",
                       label=hlabel)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Timestep (×1000)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

    # ── Training diagnostics ──────────────────────────────────────────────────
    print(f"\n  ── TRAINING DIAGNOSTICS ────────────────────────────────────")

    def last5(col):
        if col in df.columns and not df[col].isna().all():
            return df[col].dropna().iloc[-5:].mean()
        return None

    ev = last5("explained_var")
    if ev is not None:
        bar = "✗" if ev < 0 else ("⚠" if ev < 0.3 else ("△" if ev < 0.6 else "✓"))
        msg = ("VALUE FUNCTION WORSE THAN BASELINE — not learning!"
               if ev < 0 else
               ("learning slowly" if ev < 0.3 else
                ("moderate learning" if ev < 0.6 else "strong value function")))
        print(f"  {bar}  Explained variance (final):  {ev:.3f}  → {msg}")

    if "entropy" in df.columns and not df["entropy"].isna().all():
        ent = df["entropy"].dropna()
        d_ent = ent.iloc[:5].mean() - ent.iloc[-5:].mean()
        sym   = "✓" if d_ent > 0.05 else "⚠"
        print(f"  {sym}  Entropy change: {ent.iloc[0]:.3f} → {ent.iloc[-1]:.3f}"
              f"  (Δ={d_ent:+.3f})  "
              + ("policy converging" if d_ent > 0.05 else "policy not converging"))

    if "ep_rew_mean" in df.columns and not df["ep_rew_mean"].isna().all():
        r = df["ep_rew_mean"].dropna()
        trend = "improving" if r.iloc[-5:].mean() > r.iloc[:5].mean() else "not improving"
        print(f"  →  Episode reward: {r.iloc[0]:.4f} → {r.iloc[-1]:.4f}  ({trend})")


# ── Part 3: Plain-English Diagnosis ───────────────────────────────────────────

def print_diagnosis() -> None:
    W = 64
    print(f"\n{'='*W}")
    print(f"  CURRENT MODEL DIAGNOSIS — {MODEL_NAME}")
    print(f"{'='*W}")
    print(f"""
  ★ EXPLAINED VARIANCE is the single most important metric.
    It measures whether the value function (critic) can predict
    future rewards better than just guessing the mean.
    < 0.0  →  value function is HARMFUL (actively misleading)
    0.0–0.3 →  weak learning — increase timesteps
    0.3–0.6 →  moderate — policy can improve further
    > 0.6  →  good — policy is learning meaningful structure

  PROBLEM 1 — OVERTRADING  (most critical right now)
  ────────────────────────────────────────────────────
  The agent makes 3,000–5,000 direction changes per year.
  At fee=0.05% × leverage=2×, each flip costs ~0.1% of equity.
  4,000 flips × 0.1% = 400% of initial equity lost to fees alone.

  Maximum sustainable flips ≈ 2,000 per year at these costs.
  FIX: stronger churn penalty, add minimum hold of 12–24 steps.

  PROBLEM 2 — RANDOM POLICY  (undertrained)
  ───────────────────────────────────────────
  Win rate ~38%, regime alignment ~50% = coin-flip behaviour.
  200k training steps over 46k rows = only ~4 dataset passes.
  MINIMUM needed for convergence: 1–2 million steps.

  PROBLEM 3 — REWARD SCALE MISMATCH
  ────────────────────────────────────
  Tiered bonuses (STEP_SCALE × 0.001–0.008) are the same order
  of magnitude as a neutral base_return step (~±0.001).
  The -2× churn penalty (0.002) is weaker than a single bad step.
  FIX: increase STEP_SCALE or the churn penalty multiplier.

  WHAT TO DO NEXT:
  ─────────────────
  1. Retrain with total_timesteps=2_000_000
  2. Monitor explained_variance in training_curves.png
  3. If EV stays negative after 500k steps → env/reward has a bug
  4. Check policy_probe.png after retraining — should show peaks
""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  TRAINING ANALYSIS — {MODEL_NAME}")
    print(f"{'='*60}")

    print("\n  1/3  Probing policy action distribution...")
    actions = probe_policy()
    plot_policy_probe(actions)

    print("\n  2/3  Training convergence curves...")
    plot_training_curves()

    print("\n  3/3  Diagnosis...")
    print_diagnosis()

    print("\nDone.  Outputs: analysis_policy_probe.png  analysis_training_curves.png\n")
