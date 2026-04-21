"""
shap_analysis.py  —  SHAP explainability for PPO and SAC multi-asset models
============================================================================
Explains which observation features most influence each agent's trading
decisions using SHAP (SHapley Additive exPlanations).
Runs on both PPO and SAC so feature importance can be compared directly.

Approach: Feature-grouped KernelExplainer
  The raw observation is 26,903-dimensional (336-step window x 20 features
  x 4 assets + 23 portfolio scalars).  KernelExplainer on 26,903 features
  would take weeks.  Instead we compress each observation to 103 interpretable
  grouped features:

    - For each of 20 feature types x 4 assets: mean across the 336-step window
      -> 80 values capturing the average market state per asset
    - Plus 23 portfolio scalars unchanged (position, equity, etc.)
    -> Total: 103 grouped features

  The SHAP wrapper reconstructs a full-dim observation from the 103 grouped
  features (by broadcasting the means across the window) and returns the
  agent's direction signal for each asset.

  This is a valid approximation: SHAP perturbs the mean-level features and
  measures how much each feature's average value influences the action.

Outputs (saved to shap_output/):
  {algo}_shap_importance_bar.png  -- top 20 features by mean |SHAP|
  {algo}_shap_per_asset.png       -- per-asset beeswarm (top 15 per asset)
  {algo}_shap_values_{asset}.csv  -- raw SHAP values for all explained samples
  shap_comparison.png             -- side-by-side PPO vs SAC top features
  shap_comparison.csv             -- combined importance table for dissertation

Runtime: ~15-30 min per model (KernelExplainer, 103 features, 200 samples)

Usage:
    python3 shap_analysis.py
"""

import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap
import torch
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from multi_asset_env import MultiAssetPerpEnv, FEATURE_COLS


# ── Models to analyse ─────────────────────────────────────────────────────────
# SAC does not use VecNormalize — set vecnorm to None for SAC.

MODELS = [
    {
        "label":      "PPO",
        "algo":       "ppo",
        "loader":     PPO,
        "model_path": "ppo_multi_perp/ppo_multi_perp_w336_td336_mh72",
        "vecnorm":    "ppo_multi_perp/vec_normalize_w336_td336_mh72.pkl",
    },
    {
        "label":      "SAC",
        "algo":       "sac",
        "loader":     SAC,
        "model_path": "ppo_multi_perp/sac_multi_perp_w336_td336",
        "vecnorm":    None,
    },
]


# ── Shared config ─────────────────────────────────────────────────────────────

CSV_PATHS = [
    "BTCUSDT_perpetual_futures_processed_2020-2026.csv",
    "ETHUSDT_perpetual_futures_processed_2020-2026.csv",
    "SOLUSDT_perpetual_futures_processed_2020-2026.csv",
    "BNBUSDT_perpetual_futures_processed_2020-2026.csv",
]
ASSET_NAMES = ["BTC", "ETH", "SOL", "BNB"]
N_ASSETS    = len(ASSET_NAMES)
N_FEATURES  = len(FEATURE_COLS)   # 20
WINDOW_SIZE = 336
TRADE_DUR   = 336
MIN_HOLD    = 72

OOS_START   = "2025-01-01"
OOS_END     = "2026-04-08"

N_BACKGROUND = 50    # k-means centroids for SHAP background
N_EXPLAIN    = 200   # observations to explain
NSAMPLES     = 512   # KernelExplainer samples per point (speed vs accuracy)

OUT_DIR = Path("shap_output")
OUT_DIR.mkdir(exist_ok=True)


# ── Observation layout ────────────────────────────────────────────────────────
# obs = [BTC_window(6720), ETH_window(6720), SOL_window(6720), BNB_window(6720),
#        global_scalars(3), per_asset_scalars(5 x 4)]
# Total: 26,880 + 23 = 26,903

MARKET_DIM = WINDOW_SIZE * N_FEATURES   # 6720 per asset
SCALAR_DIM = 3 + N_ASSETS * 5           # 23

GROUPED_NAMES = (
    [f"{feat}_mean_{asset}" for asset in ASSET_NAMES for feat in FEATURE_COLS]
    + ["norm_equity", "consec_wins_norm", "consec_losses_norm"]
    + [f"{name}_{asset}"
       for asset in ASSET_NAMES
       for name in ["position", "trade_state", "steps_in_trade",
                    "unrealized_pnl", "current_risk"]]
)
N_GROUPED = len(GROUPED_NAMES)   # 80 + 23 = 103
print(f"Grouped feature space: {N_GROUPED}  (80 market means + 23 portfolio scalars)\n")


# ── OOS index lookup ──────────────────────────────────────────────────────────

def _oos_indices():
    dfs    = [pd.read_csv(p, parse_dates=["open_time"]) for p in CSV_PATHS]
    common = set(dfs[0]["open_time"])
    for df in dfs[1:]:
        common &= set(df["open_time"])
    ts   = pd.Series(sorted(common))
    mask = (ts >= OOS_START) & (ts < OOS_END)
    rows = ts.index[mask]
    return int(max(rows[0], WINDOW_SIZE)), int(rows[-1])

OOS_START_IDX, OOS_END_IDX = _oos_indices()
print(f"OOS window: idx {OOS_START_IDX} -> {OOS_END_IDX}  ({OOS_START} - {OOS_END})\n")


def make_oos_env():
    return MultiAssetPerpEnv(
        csv_paths          = CSV_PATHS,
        asset_names        = ASSET_NAMES,
        window_size        = WINDOW_SIZE,
        max_trade_duration = TRADE_DUR,
        min_hold_steps     = MIN_HOLD,
        initial_equity     = 1_000.0,
        eval_start         = OOS_START_IDX,
        eval_end           = OOS_END_IDX,
    )


# ── Helper functions ──────────────────────────────────────────────────────────

def compress(obs_batch: np.ndarray) -> np.ndarray:
    """(N, 26903) -> (N, 103): mean per (asset, feature) across window + scalars."""
    N   = obs_batch.shape[0]
    out = np.zeros((N, N_GROUPED), dtype=np.float32)
    for a in range(N_ASSETS):
        block        = obs_batch[:, a * MARKET_DIM : (a + 1) * MARKET_DIM]
        window_block = block.reshape(N, WINDOW_SIZE, N_FEATURES)
        out[:, a * N_FEATURES : (a + 1) * N_FEATURES] = window_block.mean(axis=1)
    out[:, N_ASSETS * N_FEATURES:] = obs_batch[:, -SCALAR_DIM:]
    return out


def bar_colour(name: str) -> str:
    if any(s in name for s in ["position", "trade_state", "unrealized",
                                "risk", "equity", "consec", "steps_in"]):
        return "#d50000"   # portfolio state
    if "funding" in name or "basis" in name:
        return "#ff8f00"   # funding-related
    if "sma" in name or "momentum" in name or "donchian" in name:
        return "#1565c0"   # trend / SMA
    if "rsi" in name or "bb_" in name or "macd" in name:
        return "#6a1b9a"   # momentum / oscillator
    return "#2e7d32"       # volatility / volume


LEGEND_PATCHES = [
    Patch(facecolor="#d50000", label="Portfolio state"),
    Patch(facecolor="#1565c0", label="Trend / SMA"),
    Patch(facecolor="#6a1b9a", label="Momentum / oscillator"),
    Patch(facecolor="#ff8f00", label="Funding / basis"),
    Patch(facecolor="#2e7d32", label="Volatility / volume"),
]


# ── Per-model analysis loop ───────────────────────────────────────────────────

all_importance = {}   # label -> importance DataFrame (for comparison plot)

for model_cfg in MODELS:
    label      = model_cfg["label"]
    algo       = model_cfg["algo"]
    loader     = model_cfg["loader"]
    model_path = model_cfg["model_path"]
    vecnorm    = model_cfg["vecnorm"]

    print("=" * 65)
    print(f"  SHAP ANALYSIS — {label}")
    print("=" * 65)

    # Load model
    print(f"  Loading {label} from {model_path}…")
    model  = loader.load(model_path)
    policy = model.policy
    policy.set_training_mode(False)

    # Build OOS environment
    env = DummyVecEnv([make_oos_env])
    if vecnorm and Path(vecnorm).exists():
        env = VecNormalize.load(vecnorm, env)
        env.training    = False
        env.norm_reward = False
        print(f"  VecNormalize: {vecnorm}")
    else:
        print(f"  No VecNormalize")

    # Collect observations
    print(f"  Collecting OOS observations…")
    obs_list = []
    raw_obs  = env.reset()
    done = False
    while not done:
        obs_list.append(raw_obs[0].copy())
        action, _ = model.predict(raw_obs, deterministic=True)
        raw_obs, _, done_arr, _ = env.step(action)
        done = bool(done_arr[0])
    env.close()

    obs_array = np.array(obs_list, dtype=np.float32)
    X_grouped = compress(obs_array)
    print(f"  Collected {len(obs_array):,} obs -> compressed to {X_grouped.shape}")

    # SHAP prediction wrapper — closure over policy
    def make_predictor(pol):
        def predict_directions(X: np.ndarray) -> np.ndarray:
            """(N, 103) -> (N, 4) direction signals per asset."""
            N        = X.shape[0]
            full_obs = np.zeros((N, MARKET_DIM * N_ASSETS + SCALAR_DIM), dtype=np.float32)
            for a in range(N_ASSETS):
                feat_means = X[:, a * N_FEATURES : (a + 1) * N_FEATURES]
                tiled = np.broadcast_to(
                    feat_means[:, np.newaxis, :], (N, WINDOW_SIZE, N_FEATURES)
                ).copy().reshape(N, MARKET_DIM)
                full_obs[:, a * MARKET_DIM : (a + 1) * MARKET_DIM] = tiled
            full_obs[:, -SCALAR_DIM:] = X[:, N_ASSETS * N_FEATURES:]
            with torch.no_grad():
                actions, _, _ = pol(torch.FloatTensor(full_obs))
                actions_np = actions.cpu().numpy()
            return actions_np[:, [i * 4 for i in range(N_ASSETS)]]
        return predict_directions

    predict_directions = make_predictor(policy)

    # Sanity check
    test = predict_directions(X_grouped[:3])
    print(f"  Sanity check (3 steps): {test.round(3)}")

    # Run SHAP
    print(f"\n  Building background ({N_BACKGROUND} k-means centroids)…")
    background  = shap.kmeans(X_grouped, N_BACKGROUND)
    explainer   = shap.KernelExplainer(predict_directions, background)
    explain_idx = np.linspace(0, len(X_grouped) - 1, N_EXPLAIN, dtype=int)
    X_explain   = X_grouped[explain_idx]

    print(f"  Computing SHAP values ({N_EXPLAIN} samples, nsamples={NSAMPLES})…")
    print(f"  Estimated runtime: 15-30 minutes…")
    shap_vals = explainer.shap_values(X_explain, nsamples=NSAMPLES)
    print(f"  Done!")

    # Save raw values
    for i, asset in enumerate(ASSET_NAMES):
        pd.DataFrame(shap_vals[i], columns=GROUPED_NAMES).to_csv(
            OUT_DIR / f"{algo}_shap_values_{asset}.csv", index=False
        )

    # Importance DataFrame
    mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    GROUPED_NAMES,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(OUT_DIR / f"{algo}_shap_importance.csv", index=False)
    all_importance[label] = importance_df

    # Plot 1: Global importance bar chart
    top_n      = 20
    top        = importance_df.head(top_n)
    bar_labels = top["feature"].tolist()[::-1]
    bar_vals   = top["importance"].tolist()[::-1]
    colours    = [bar_colour(l) for l in bar_labels]

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    ax.barh(range(top_n), bar_vals, color=colours, alpha=0.85, height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(bar_labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| — average impact on direction signal", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances — {label} w336/td336\n"
        f"OOS Jan 2025 - Apr 2026  |  aggregated across BTC, ETH, SOL, BNB",
        fontweight="bold", fontsize=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.35, color="#cccccc")
    ax.legend(handles=LEGEND_PATCHES, loc="lower right", fontsize=9, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{algo}_shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart -> {OUT_DIR}/{algo}_shap_importance_bar.png")

    # Plot 2: Per-asset beeswarm
    TOP_PER_ASSET = 15
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"SHAP Values per Asset — {label} Direction Signal  (OOS Jan 2025 - Apr 2026)\n"
        "Each dot = one OOS observation.  Colour = feature value (blue=low, red=high)",
        fontweight="bold", fontsize=13,
    )
    rng = np.random.default_rng(seed=42)
    for i, (ax, asset) in enumerate(zip(axes.flat, ASSET_NAMES)):
        ax.set_facecolor("#ffffff")
        sv      = shap_vals[i]
        top_idx = np.argsort(np.abs(sv).mean(axis=0))[-TOP_PER_ASSET:][::-1]
        top_names = [GROUPED_NAMES[j] for j in top_idx]
        top_sv    = sv[:, top_idx]
        top_xvals = X_explain[:, top_idx]
        for k in range(TOP_PER_ASSET):
            shap_col = top_sv[:, k]
            val_col  = top_xvals[:, k]
            jitter   = rng.uniform(-0.25, 0.25, len(shap_col))
            norm_val = (val_col - val_col.min()) / (val_col.ptp() + 1e-8)
            ax.scatter(shap_col, k + jitter, c=plt.cm.RdBu_r(norm_val),
                       s=10, alpha=0.55, linewidths=0, zorder=3)
        ax.axvline(0, color="#333333", linewidth=0.8, zorder=4)
        ax.set_yticks(range(TOP_PER_ASSET))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel("SHAP value  (+ = long,  - = short)", fontsize=9)
        ax.set_title(f"{asset}  direction signal", fontweight="bold", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3, color="#cccccc")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_DIR / f"{algo}_shap_per_asset.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-asset beeswarm -> {OUT_DIR}/{algo}_shap_per_asset.png")

    # Print top 10
    print(f"\n  Top 10 features ({label}):")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:<45}  {row['importance']:.5f}")
    print()

    del model, policy
    gc.collect()


# ── PPO vs SAC comparison plot ────────────────────────────────────────────────

if len(all_importance) == 2:
    print("=" * 65)
    print("  PPO vs SAC COMPARISON PLOT")
    print("=" * 65)

    labels_list = list(all_importance.keys())
    imp_a = all_importance[labels_list[0]].set_index("feature")["importance"]
    imp_b = all_importance[labels_list[1]].set_index("feature")["importance"]

    # Union of top 20 from each model, ranked by combined importance
    top_a = set(all_importance[labels_list[0]].head(20)["feature"])
    top_b = set(all_importance[labels_list[1]].head(20)["feature"])
    union = sorted(top_a | top_b,
                   key=lambda f: -(imp_a.get(f, 0) + imp_b.get(f, 0)))[:25]

    vals_a = [float(imp_a.get(f, 0)) for f in union]
    vals_b = [float(imp_b.get(f, 0)) for f in union]

    y, h = np.arange(len(union)), 0.38
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    ax.barh(y + h/2, vals_a, height=h, color="#1565c0", alpha=0.85, label=labels_list[0])
    ax.barh(y - h/2, vals_b, height=h, color="#d50000",  alpha=0.85, label=labels_list[1])
    ax.set_yticks(y)
    ax.set_yticklabels(union, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        "PPO vs SAC — Feature Importance Comparison\n"
        "OOS Jan 2025 - Apr 2026  |  Union of top 20 features from each algorithm",
        fontweight="bold", fontsize=12,
    )
    ax.legend(fontsize=11, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.35, color="#cccccc")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot -> {OUT_DIR}/shap_comparison.png")

    comp_df = pd.DataFrame({
        "feature":                          union,
        f"importance_{labels_list[0]}":     vals_a,
        f"importance_{labels_list[1]}":     vals_b,
        "diff_PPO_minus_SAC":               [a - b for a, b in zip(vals_a, vals_b)],
    })
    comp_df.to_csv(OUT_DIR / "shap_comparison.csv", index=False)
    print(f"  Comparison CSV  -> {OUT_DIR}/shap_comparison.csv")


print(f"\n{'='*65}")
print(f"  SHAP ANALYSIS COMPLETE")
print(f"  All outputs in: {OUT_DIR}/")
print(f"  Key dissertation figures:")
print(f"    ppo_shap_importance_bar.png   PPO feature importance")
print(f"    sac_shap_importance_bar.png   SAC feature importance")
print(f"    ppo_shap_per_asset.png        PPO per-asset beeswarm")
print(f"    sac_shap_per_asset.png        SAC per-asset beeswarm")
print(f"    shap_comparison.png           PPO vs SAC side-by-side")
print(f"    shap_comparison.csv           numbers for Table X")
print(f"{'='*65}")
