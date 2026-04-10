"""
smoke_test_gpu.py  —  Parallel smoke test (SubprocVecEnv, CPU)
==============================================================
Faster version of smoke_test.py using parallel environment collection
instead of GPU (which is slower for small MlpPolicy networks — see SB3 #1245).

  Original : 1 env  × n_steps=2048 → 2048 transitions per rollout
  This file: 4 envs × n_steps=512  → 2048 transitions per rollout (same size)
                                      but ~2.5–3.5× faster collection

Pass criteria are identical to smoke_test.py:
  ✓ Flat time > 10%
  ✓ Portfolio leverage peak ≤ 7.0×
  ✓ Fee drag < 200%
  ✓ No wipeout (final equity > $100 on $1,000 start)

If all pass → launch full overnight retraining with train_multi_ppo_gpu.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from multi_asset_env import MultiAssetPerpEnv


# ── Parallelism ───────────────────────────────────────────────────────────────

N_ENVS  = 4              # parallel training workers
N_STEPS = 2048 // N_ENVS  # 512 per env × 4 = 2048 total rollout


# ── Config ─────────────────────────────────────────────────────────────────────

SMOKE_STEPS  = 500_000
TRAIN_END    = "2023-12-31"
INIT_EQUITY  = 1_000.0
SEED         = 42

MULTI_CSV = [
    "BTCUSDT_perpetual_futures_processed_2020-2026.csv",
    "ETHUSDT_perpetual_futures_processed_2020-2026.csv",
    "SOLUSDT_perpetual_futures_processed_2020-2026.csv",
    "BNBUSDT_perpetual_futures_processed_2020-2026.csv",
]
ASSET_NAMES = ["BTC", "ETH", "SOL", "BNB"]

CONFIGS = [
    {"window": 168, "td": 168, "label": "w168/td168 (mirrors single-asset winner)"},
    {"window":  96, "td":  48, "label": "w96/td48   (best multi-asset drawdown)"},
    {"window":  24, "td": 168, "label": "w24/td168  (multi-asset Sharpe winner)"},
]

OUT_DIR = Path("smoke_test_models")
OUT_DIR.mkdir(exist_ok=True)

PPO_KWARGS = dict(
    policy        = "MlpPolicy",
    device        = "cpu",
    learning_rate = 3e-4,
    n_steps       = N_STEPS,   # 512; × N_ENVS = 2048 total rollout
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 0,
    seed          = SEED,
)


# ── Progress callback ──────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    def __init__(self, total_steps, label):
        super().__init__()
        self.total = total_steps
        self.label = label
        self._last_pct = -1

    def _on_step(self):
        pct = int(self.num_timesteps / self.total * 100)
        if pct % 10 == 0 and pct != self._last_pct:
            print(f"    [{self.label}] {pct}%  ({self.num_timesteps:,}/{self.total:,})")
            self._last_pct = pct
        return True


# ── OOS evaluator ──────────────────────────────────────────────────────────────

def quick_eval_multi(model, vecnorm_path, window, td):
    """Run one OOS episode (Jan 2025 – Apr 2026) on a single env (no parallelism)."""
    raw = [pd.read_csv(p, parse_dates=["open_time"]) for p in MULTI_CSV]
    common = set(raw[0]["open_time"])
    for df in raw[1:]:
        common &= set(df["open_time"])
    ts_series = pd.Series(sorted(common))

    start_dt = pd.Timestamp("2025-01-01")
    end_dt   = pd.Timestamp("2026-04-08")
    mask     = (ts_series >= start_dt) & (ts_series < end_dt)
    rows     = ts_series.index[mask]
    eval_start = int(max(rows[0], window))
    eval_end   = int(rows[-1])

    def make_env():
        return MultiAssetPerpEnv(
            csv_paths=MULTI_CSV, asset_names=ASSET_NAMES,
            window_size=window, max_trade_duration=td,
            initial_equity=INIT_EQUITY,
            eval_start=eval_start, eval_end=eval_end,
        )

    # Single env for evaluation — deterministic episode, no parallelism needed
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()
    equity_h, lev_h, fee_h, pnl_h = [], [], [], []
    per_asset_flat = [[] for _ in ASSET_NAMES]

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_arr, infos = env.step(action)
        done = bool(done_arr[0])
        info = infos[0]

        equity_h.append(info["equity"])
        pos_list = info["positions"]
        lev_h.append(sum(abs(p) for p in pos_list))
        fee_h.append(info["total_fees"])
        pnl_h.append(info["total_pnl"])
        for i in range(len(ASSET_NAMES)):
            per_asset_flat[i].append(abs(pos_list[i]) < 0.05)

    eq    = np.array(equity_h)
    lev   = np.array(lev_h)
    fees  = np.array(fee_h)
    gross = np.array(pnl_h)

    total_ret  = (eq[-1] / eq[0]) - 1.0
    avg_lev    = lev.mean()
    max_lev    = lev.max()
    fee_drag   = fees.sum() / (abs(gross.sum()) + 1e-8) * 100
    frac_all_flat = (lev < 0.05 * len(ASSET_NAMES)).mean()

    per_asset_stats = []
    for i, name in enumerate(ASSET_NAMES):
        flat_arr = np.array(per_asset_flat[i])
        per_asset_stats.append({"name": name, "frac_flat": flat_arr.mean()})

    return dict(
        total_ret=total_ret, final_eq=eq[-1],
        avg_lev=avg_lev, max_lev=max_lev,
        fee_drag=fee_drag,
        frac_all_flat=frac_all_flat,
        per_asset=per_asset_stats,
    )


# ── Pass/fail checker ──────────────────────────────────────────────────────────

def check_pass(label, m):
    W = 60
    print(f"\n  {'─'*W}")
    print(f"  SMOKE RESULTS — {label}")
    print(f"  {'─'*W}")
    print(f"  Final equity  : ${m['final_eq']:>8.2f}   (started $1,000)")
    print(f"  Total return  : {m['total_ret']*100:>+7.1f}%")
    print(f"  Avg leverage  : {m['avg_lev']:>6.2f}×  (max {m['max_lev']:.2f}×, hard cap 6.0)")
    print(f"  Fee drag      : {m['fee_drag']:>6.1f}%")
    print(f"  All-flat steps: {m['frac_all_flat']*100:>5.1f}%")
    print(f"  Per-asset flat: " +
          "  ".join(f"{a['name']} {a['frac_flat']*100:.0f}%" for a in m["per_asset"]))

    checks = [
        ("Any asset flat > 10%",
         any(a["frac_flat"] > 0.10 for a in m["per_asset"]),
         f"max {max(a['frac_flat'] for a in m['per_asset'])*100:.1f}%"),
        ("Portfolio leverage peak ≤ 7.0×",
         m["max_lev"] <= 7.0,
         f"{m['max_lev']:.2f}×"),
        ("Fee drag < 200%",
         m["fee_drag"] < 200.0,
         f"{m['fee_drag']:.1f}%"),
        ("No wipeout (equity > $100)",
         m["final_eq"] > 100.0,
         f"${m['final_eq']:.2f}"),
    ]

    all_pass = True
    print()
    for name, passed, val in checks:
        icon = "✓" if passed else "✗"
        print(f"  {icon}  {name:<40} {val}")
        if not passed:
            all_pass = False

    return all_pass


# ── Env factory (avoids closure capture issues with SubprocVecEnv) ─────────────

def _make_env_fn(w, td):
    def _inner():
        return MultiAssetPerpEnv(
            csv_paths=MULTI_CSV, asset_names=ASSET_NAMES,
            window_size=w, max_trade_duration=td,
            initial_equity=INIT_EQUITY, train_end_date=TRAIN_END,
        )
    return _inner


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  MULTI-ASSET SMOKE TEST (parallel) — 500k steps")
    print(f"  Workers: {N_ENVS}  ×  n_steps/env: {N_STEPS}  =  {N_ENVS*N_STEPS} total/rollout")
    print("  NOTE: Single-asset SOL not retrained — existing results stand")
    print("=" * 65)
    print(f"  Configs: {len(CONFIGS)}  ×  multi-asset only  =  {len(CONFIGS)} runs")
    print(f"  Estimated time: ~5-10 minutes\n")

    verdict = {}

    for cfg in CONFIGS:
        w   = cfg["window"]
        td  = cfg["td"]
        lbl = cfg["label"]
        tag = f"w{w}_td{td}"

        print(f"\n{'#'*65}")
        print(f"  CONFIG: {lbl}")
        print(f"{'#'*65}")
        print(f"\n  Training multi-asset ({SMOKE_STEPS:,} steps, {N_ENVS} workers)…")

        env  = SubprocVecEnv([_make_env_fn(w, td)] * N_ENVS)
        norm = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(env=norm, **PPO_KWARGS)
        model.learn(
            total_timesteps=SMOKE_STEPS,
            callback=ProgressCallback(SMOKE_STEPS, f"MULTI {tag}"),
        )

        norm_path = str(OUT_DIR / f"smoke_multi_vecnorm_{tag}.pkl")
        norm.save(norm_path)
        model.save(str(OUT_DIR / f"smoke_multi_{tag}"))
        print(f"  Saved → {OUT_DIR}/smoke_multi_{tag}.zip")

        print(f"\n  Evaluating OOS (Jan 2025 – Apr 2026)…")
        metrics = quick_eval_multi(model, norm_path, w, td)
        passed  = check_pass(lbl, metrics)
        verdict[tag] = passed

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  FINAL VERDICT")
    print("=" * 65)
    all_clear = all(verdict.values())
    for tag, passed in verdict.items():
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {icon}  {tag}")

    print()
    if all_clear:
        print("  ✓ ALL CONFIGS PASSED")
        print("  Next: launch full overnight retraining")
        print("        python3 train_multi_ppo_gpu.py")
        print("        python3 train_multi_sac_gpu.py")
    else:
        print("  ✗ SOME CONFIGS FAILED")
        print("  Review the checks above. Common fixes:")
        print("  - Flat time too low  → increase flat patience or funding penalty")
        print("  - Fee drag too high  → lower TRADE_SCALE or increase flat patience")
        print("  - Leverage too high  → lower MAX_PORTFOLIO_LEVERAGE in multi_asset_env.py")
        print("  - Wipeout            → tighten drawdown threshold or increase penalty coeff")
        print("  Then re-run: python3 smoke_test_gpu.py")
    print("=" * 65)
