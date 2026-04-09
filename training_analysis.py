"""
training_analysis.py  —  Multi-model grid analysis
====================================================
Reads training logs from training_log/ and models from ppo_sol_perp/
for all 16 (window × trade_duration) combinations.

  Part 1 — EV Grid
    4×4 plot of explained variance curves for every model.
    EV is the single most important convergence indicator.
    Target: final EV > 0.5

  Part 2 — Training Curves (best model)
    Full 6-panel convergence chart for the model with highest final EV.

  Part 3 — Policy Probe (best model)
    Sample 2 000 deterministic actions to inspect what the policy
    has learnt about direction, SL size, TP size, and R:R ratio.

  Part 4 — Plain-English Diagnosis
    Interprets all metrics and prints a structured report.

Run:
    python3 training_analysis.py
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sol_env import SolPerpEnv

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH   = "SOLUSDT_perpetual_futures_processed_2020-2025.csv"
MODEL_DIR  = Path("ppo_sol_perp")
LOG_DIR    = Path("training_log")
N_PROBE    = 2_000

WINDOW_SIZES    = [24, 48, 96, 168]
TRADE_DURATIONS = [24, 48, 72, 168]


# ── Path helpers ───────────────────────────────────────────────────────────────

def model_path(w, d):
    return MODEL_DIR / f"ppo_sol_perp_w{w}_td{d}.zip"

def vecnorm_path(w, d):
    return MODEL_DIR / f"vec_normalize_w{w}_td{d}.pkl"

def log_path(w, d):
    return LOG_DIR / f"training_log_w{w}_td{d}.csv"


def discover_models():
    """Return sorted list of (window, duration) for every .zip found."""
    pattern = re.compile(r"ppo_sol_perp_w(\d+)_td(\d+)\.zip$")
    found = []
    for p in sorted(MODEL_DIR.glob("ppo_sol_perp_w*_td*.zip")):
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    return sorted(found)


def load_log(w, d):
    p = log_path(w, d)
    if not p.exists():
        return None
    df = pd.read_csv(p).dropna(how="all")
    return df if len(df) >= 2 else None


# ── Part 1: EV Grid ─────────────────────────────────────────────────────────────

def plot_ev_grid(models_found, save_path="analysis_ev_grid.png"):
    """4×4 grid of explained-variance curves, one panel per (window, duration)."""
    fig, axes = plt.subplots(
        len(WINDOW_SIZES), len(TRADE_DURATIONS),
        figsize=(18, 12), sharex=False, sharey=True,
    )
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "Training Convergence — Explained Variance (EV) Grid\n"
        "Target: final EV > 0.5  |  Red dashed = 0.5 threshold",
        fontsize=13, fontweight="bold", y=0.99,
    )

    best_ev   = -9999
    best_key  = None
    ev_summary = {}

    for ri, w in enumerate(WINDOW_SIZES):
        for ci, d in enumerate(TRADE_DURATIONS):
            ax = axes[ri][ci]
            ax.set_facecolor("#ffffff")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="#e0e0e0", linewidth=0.4)
            ax.axhline(0.5, color="#d50000", linewidth=0.9,
                       linestyle="--", alpha=0.7)
            ax.axhline(0.0, color="#555555", linewidth=0.7, linestyle=":")

            tag = f"w{w} / td{d}h"
            ax.set_title(tag, fontsize=8, fontweight="bold", pad=3)

            if (w, d) not in models_found:
                ax.text(0.5, 0.5, "not trained",
                        transform=ax.transAxes, ha="center",
                        va="center", color="gray", fontsize=8)
                continue

            df = load_log(w, d)
            if df is None or "explained_var" not in df.columns:
                ax.text(0.5, 0.5, "no log",
                        transform=ax.transAxes, ha="center",
                        va="center", color="gray", fontsize=8)
                continue

            ev  = df["explained_var"].dropna()
            ts  = df["timestep"].iloc[ev.index] / 1_000
            sm  = ev.rolling(7, min_periods=1).mean()

            final_ev = float(ev.iloc[-5:].mean())
            ev_summary[(w, d)] = final_ev
            if final_ev > best_ev:
                best_ev  = final_ev
                best_key = (w, d)

            color = ("#2e7d32" if final_ev > 0.5 else
                     "#ff8f00" if final_ev > 0.2 else "#d50000")
            ax.plot(ts, ev, color=color, alpha=0.20, linewidth=0.7)
            ax.plot(ts, sm, color=color, linewidth=1.5)
            ax.text(0.97, 0.05,
                    f"EV={final_ev:+.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", alpha=0.8))

            if ri == len(WINDOW_SIZES) - 1:
                ax.set_xlabel("Steps (×1k)", fontsize=7)
            if ci == 0:
                ax.set_ylabel("Expl. Var.", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_ylim(-0.5, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

    # Print EV table
    print(f"\n  Final EV Grid  (avg last-5 checkpoints)\n")
    hdr = f"  {'Window':>8}  " + "  ".join(f"  td={d:>3}h" for d in TRADE_DURATIONS)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for w in WINDOW_SIZES:
        row = f"  w={w:>4}h    "
        for d in TRADE_DURATIONS:
            ev = ev_summary.get((w, d), None)
            if ev is None:
                row += f"     {'—':>6}   "
            else:
                star = " ★" if (w, d) == best_key else "  "
                tag  = ("✓" if ev > 0.5 else ("△" if ev > 0.2 else "✗"))
                row += f"  {tag} {ev:+.3f}{star}  "
        print(row)
    print(f"\n  ★ = best model  ✓ = good (>0.5)  △ = moderate  ✗ = poor\n")

    return best_key, ev_summary


# ── Part 2: Full training curves for one model ─────────────────────────────────

def plot_training_curves(w, d, save_path=None):
    if save_path is None:
        save_path = f"analysis_training_curves_w{w}_td{d}.png"
    df = load_log(w, d)
    if df is None:
        print(f"  [WARN] No log for w{w}_td{d}")
        return

    print(f"  Training log: {len(df)} update steps  "
          f"({df['timestep'].max():,} total timesteps)")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Training Convergence — w={w}h / td={d}h\n"
        f"{df['timestep'].max():,} timesteps  |  {len(df)} gradient updates",
        fontsize=12, fontweight="bold", y=0.99,
    )

    def smooth(s, w=7):
        return s.rolling(w, min_periods=1).mean()

    ts = df["timestep"] / 1_000

    specs = [
        ("ep_rew_mean",   "Mean Episode Reward",   "Reward",    "#1565c0", None,  None),
        ("ep_len_mean",   "Mean Episode Length",   "Steps",     "#2e7d32", None,  None),
        ("policy_loss",   "Policy Gradient Loss",  "Loss",      "#6a1b9a", None,  None),
        ("value_loss",    "Value Function Loss",   "Loss",      "#e65100", None,  None),
        ("entropy",       "Policy Entropy",        "Entropy",   "#00838f", None,  None),
        ("explained_var", "Explained Variance ★",  "Exp. Var.", "#c62828", 0.5,   "Target ≥ 0.5"),
    ]

    for ax, (col, title, ylabel, color, hline, hlabel) in zip(axes.flat, specs):
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e0e0e0", linewidth=0.5)

        if col not in df.columns or df[col].isna().all():
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            continue

        raw = df[col]
        sm  = smooth(raw)
        ax.plot(ts, raw, color=color, alpha=0.25, linewidth=0.8)
        ax.plot(ts, sm,  color=color, linewidth=1.8, label="smoothed (7-step)")
        if hline is not None:
            ax.axhline(hline, color="red", linewidth=1.2,
                       linestyle="--", label=hlabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Timestep (×1000)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

    # Diagnostics
    print(f"\n  ── TRAINING DIAGNOSTICS  w={w}h / td={d}h ─────────────────")

    def last5(col):
        if col in df.columns and not df[col].isna().all():
            return df[col].dropna().iloc[-5:].mean()
        return None

    ev = last5("explained_var")
    if ev is not None:
        bar = "✗" if ev < 0 else ("⚠" if ev < 0.3 else ("△" if ev < 0.6 else "✓"))
        msg = ("value function HARMFUL — not learning!"
               if ev < 0 else
               ("learning slowly" if ev < 0.3 else
                ("moderate" if ev < 0.6 else "strong value function")))
        print(f"  {bar}  Explained variance (final): {ev:+.3f}  → {msg}")

    if "entropy" in df.columns and not df["entropy"].isna().all():
        ent = df["entropy"].dropna()
        d_ent = ent.iloc[:5].mean() - ent.iloc[-5:].mean()
        sym   = "✓" if d_ent > 0.05 else "⚠"
        print(f"  {sym}  Entropy: {ent.iloc[0]:.3f} → {ent.iloc[-1]:.3f}"
              f"  (Δ={d_ent:+.3f})  "
              + ("policy converging" if d_ent > 0.05 else "policy not converging"))

    if "ep_rew_mean" in df.columns and not df["ep_rew_mean"].isna().all():
        r = df["ep_rew_mean"].dropna()
        trend = "improving" if r.iloc[-5:].mean() > r.iloc[:5].mean() else "not improving"
        print(f"  →  Episode reward: {r.iloc[0]:.4f} → {r.iloc[-1]:.4f}  ({trend})")


# ── Part 3: Policy probe ───────────────────────────────────────────────────────

def probe_policy(w, d, n_samples=N_PROBE):
    mp = model_path(w, d)
    vp = vecnorm_path(w, d)
    if not mp.exists():
        print(f"  [WARN] Model {mp} not found")
        return None

    mdl = PPO.load(str(mp))

    def make_env():
        return SolPerpEnv(csv_path=CSV_PATH, window_size=w,
                          initial_equity=1_000.0)

    env = DummyVecEnv([make_env])
    if vp.exists():
        env = VecNormalize.load(str(vp), env)
        env.training    = False
        env.norm_reward = False

    obs = env.reset()
    actions = []
    for _ in range(n_samples):
        action, _ = mdl.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()
        actions.append(action[0].copy())

    return np.array(actions)   # (n_samples, 4)


def plot_policy_probe(actions, w, d,
                      save_path=None):
    if save_path is None:
        save_path = f"analysis_policy_probe_w{w}_td{d}.png"

    pos  = actions[:, 0]   # direction signal
    risk = actions[:, 1]   # risk %
    sl   = actions[:, 2]   # SL ATR mult
    tp   = actions[:, 3]   # TP ATR mult

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Policy Probe — Action Distribution  (w={w}h / td={d}h)\n"
        f"uniform = random policy   |   peaked = learnt behaviour",
        fontsize=12, fontweight="bold", y=0.98,
    )

    pct_long  = (pos >  0.33).mean() * 100
    pct_short = (pos < -0.33).mean() * 100
    pct_flat  = (np.abs(pos) <=0.33).mean() * 100

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#ffffff")
        ax.grid(color="#e0e0e0", linewidth=0.5)

    # Panel A: Direction signal
    axes[0, 0].hist(pos, bins=60, color="#1565c0", alpha=0.80,
                    edgecolor="white", linewidth=0.4)
    axes[0, 0].axvline( 0.33, color="gray",    linewidth=1.0,
                        linestyle="--", label="LONG threshold")
    axes[0, 0].axvline(-0.33, color="gray",    linewidth=1.0,
                        linestyle="--", label="SHORT threshold")
    axes[0, 0].axvline(pos.mean(), color="#d50000", linewidth=1.5,
                       label=f"mean={pos.mean():+.3f}")
    axes[0, 0].set_title("action[0] — Direction Signal  [-1, +1]",
                          fontweight="bold")
    axes[0, 0].set_xlabel("< -0.33=SHORT  |  -0.33..+0.33=FLAT  |  >+0.33=LONG")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].text(
        0.02, 0.97,
        f"LONG {pct_long:.0f}%  |  FLAT {pct_flat:.0f}%  |  SHORT {pct_short:.0f}%",
        transform=axes[0, 0].transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )

    # Panel B: Risk %
    axes[0, 1].hist(risk, bins=40, color="#ff8f00", alpha=0.80,
                    edgecolor="white", linewidth=0.4)
    axes[0, 1].axvline(risk.mean(), color="#d50000", linewidth=1.5,
                       linestyle="--", label=f"mean={risk.mean():.2f}%")
    axes[0, 1].set_title("action[1] — Risk per Trade (%)",
                          fontweight="bold")
    axes[0, 1].set_xlabel("% of equity risked per trade  [0.5, 5.0]")
    axes[0, 1].legend(fontsize=8)

    # Panel C: SL / TP ATR multipliers
    rr = tp.mean() / (sl.mean() + 1e-8)
    axes[1, 0].hist(sl, bins=40, color="#d50000", alpha=0.65,
                    edgecolor="white", linewidth=0.4, label=f"SL mean={sl.mean():.2f}×")
    axes[1, 0].hist(tp, bins=40, color="#2e7d32", alpha=0.65,
                    edgecolor="white", linewidth=0.4, label=f"TP mean={tp.mean():.2f}×")
    axes[1, 0].set_title(f"SL & TP ATR Multipliers  (R:R = 1:{rr:.2f})",
                          fontweight="bold")
    axes[1, 0].set_xlabel("ATR multiples  SL [1.5, 5.0]  TP [1.5, 10.0]")
    axes[1, 0].legend(fontsize=8)

    # Panel D: Direction vs SL scatter (colour = TP)
    sc = axes[1, 1].scatter(pos, sl, c=tp, cmap="viridis", alpha=0.25, s=6)
    plt.colorbar(sc, ax=axes[1, 1], label="TP mult", pad=0.02)
    axes[1, 1].axvline( 0.33, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 1].axvline(-0.33, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title("Direction vs SL  (colour = TP)", fontweight="bold")
    axes[1, 1].set_xlabel("Direction signal")
    axes[1, 1].set_ylabel("SL ATR multiplier")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")

    # Numerical summary
    print(f"\n  ── ACTION STATISTICS  w={w}h / td={d}h  ({N_PROBE} samples) ──")
    print(f"  Direction  LONG {pct_long:.0f}%  FLAT {pct_flat:.0f}%  SHORT {pct_short:.0f}%")
    print(f"  Risk %%    mean={risk.mean():.2f}%%  std={risk.std():.2f}%%")
    print(f"  SL mult    mean={sl.mean():.2f}×   std={sl.std():.2f}×")
    print(f"  TP mult    mean={tp.mean():.2f}×   std={tp.std():.2f}×")
    print(f"  R:R ratio  1 : {rr:.2f}")

    if pos.std() > 0.6:
        print("  ⚠  Direction std > 0.6 → policy still near-random")
    else:
        print("  ✓  Direction output shows learnt structure")
    if rr < 1.0:
        print(f"  ⚠  R:R {rr:.2f} < 1.0 — TP closer than SL on average")
    else:
        print(f"  ✓  R:R {rr:.2f} — TP further than SL (good)")
    if sl.mean() < 2.0:
        print(f"  ⚠  Avg SL {sl.mean():.2f}× < 2.0 — still tight, possible scalping")
    else:
        print(f"  ✓  Avg SL {sl.mean():.2f}× — wide enough to breathe")


# ── Part 4: Plain-English Diagnosis ───────────────────────────────────────────

def print_diagnosis(best_key, ev_summary):
    w, d = best_key
    best_ev = ev_summary.get(best_key, float("nan"))
    n_good  = sum(1 for ev in ev_summary.values() if ev > 0.5)
    n_total = len(ev_summary)

    W = 66
    print(f"\n{'='*W}")
    print(f"  GRID DIAGNOSIS")
    print(f"{'='*W}")
    print(f"""
  ★ EXPLAINED VARIANCE (EV) — the single most important metric
    EV measures whether the value function (critic) predicts
    future rewards better than just guessing the mean.
    < 0.0  →  value function HARMFUL (actively misleading)
    0.0–0.3  →  weak learning — needs more timesteps
    0.3–0.6  →  moderate — policy can improve further
    > 0.6  →  strong — policy is learning meaningful structure

  Best model   : w={w}h / td={d}h   EV = {best_ev:+.3f}
  Models > 0.5 EV : {n_good} / {n_total}
""")

    if best_ev > 0.5:
        print("  ✓ At least one model converged well (EV > 0.5).")
        print("  → Run evaluation.py to see out-of-sample performance.")
    elif best_ev > 0.2:
        print("  △ Moderate convergence. Consider training 3–5M steps.")
    else:
        print("  ✗ Poor convergence across all models.")
        print("  → Check sol_env.py reward scale or increase total steps.")

    print(f"""
  KEY CHANGES vs previous run
  ────────────────────────────
  SL minimum raised 0.5 → 1.5× ATR.
  This prevents the scalping pattern (2–6h avg trade) that
  produced 100–500% fee drag in the previous evaluation.
  Expected result: fewer trades (1–2/day), wider stops,
  lower fee drag, better directional learning.

  WHAT TO RUN NEXT
  ─────────────────
  1.  python3 evaluation.py   — out-of-sample performance
  2.  Implement SAC with same sol_env.py for comparison
  3.  SHAP explainability on the best model
""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*66}")
    print(f"  TRAINING ANALYSIS — Multi-model Grid")
    print(f"{'='*66}\n")

    models_found = discover_models()
    if not models_found:
        print("  No models found in ppo_sol_perp/. Run train_solo_ppo.py first.")
        raise SystemExit(1)

    print(f"  Found {len(models_found)} models: "
          + ", ".join(f"w{w}_td{d}" for w, d in models_found))
    print()

    # Part 1: EV grid
    print("  1/4  Explained-variance grid...")
    best_key, ev_summary = plot_ev_grid(models_found)

    if best_key is None:
        print("  No training logs found — cannot continue.")
        raise SystemExit(1)

    # Part 2: Full curves for best model
    bw, bd = best_key
    print(f"\n  2/4  Full training curves for best model: w={bw}h / td={bd}h ...")
    plot_training_curves(bw, bd)

    # Part 3: Policy probe for best model
    print(f"\n  3/4  Policy probe for w={bw}h / td={bd}h ...")
    actions = probe_policy(bw, bd)
    if actions is not None:
        plot_policy_probe(actions, bw, bd)

    # Part 4: Diagnosis
    print("\n  4/4  Diagnosis...")
    print_diagnosis(best_key, ev_summary)
