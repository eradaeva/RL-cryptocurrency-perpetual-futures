"""
multi_asset_env.py
==================
Multi-asset perpetual-futures environment — shared capital, 4 independent position managers.

Extends the single-asset SolPerpEnv design to trade BTC, ETH, SOL, BNB simultaneously.
Each asset runs its own FLAT / IN_TRADE state machine. Capital is shared — risk_pct for
each new trade is a fraction of total portfolio equity at that moment.

──────────────────────────────────────────────────────────────────────────────
Action space  Box(n_assets × 4)   ordered [BTC, ETH, SOL, BNB] by default

  For each asset i:
    action[i*4 + 0]   direction signal    [-1.0, +1.0]
                      < -0.50  →  SHORT
                      > +0.50  →  LONG
                      else     →  FLAT (stay out)
    action[i*4 + 1]   risk_pct            [0.5,  5.0]   % of total equity to risk
    action[i*4 + 2]   sl_atr_mult         [1.5,  5.0]
    action[i*4 + 3]   tp_atr_mult         [1.5, 10.0]

──────────────────────────────────────────────────────────────────────────────
Observation  (window_size × 20 × n_assets  +  portfolio_scalars)

  Market window (per asset, concatenated, flattened):
    Same 20 features as SolPerpEnv — close_norm, rsi_14, atr_14, … price_vs_sma500

  Portfolio scalars:
    Global (3):
      norm_equity          equity / initial_equity
      consec_wins_norm     global consecutive wins / 10
      consec_losses_norm   global consecutive losses / 10

    Per-asset (5 × n_assets):
      position_i           current leveraged position (0 when flat)
      trade_state_i        0 = FLAT, 1 = IN_TRADE
      steps_in_trade_norm_i
      unrealized_pnl_norm_i  price-based R-multiples (direction-adjusted)
      current_risk_norm_i  trade_risk_pct / 5.0 (0 when flat)

──────────────────────────────────────────────────────────────────────────────
Position sizing  (identical logic to SolPerpEnv, applied per asset)

  risk_amount   = (risk_pct / 100) × equity
  sl_distance   = sl_mult × ATR
  sl_fraction   = sl_distance / entry_price
  position_size = (risk_pct/100) / sl_fraction   capped at max_leverage

  Total portfolio leverage = sum of |position_i| across open trades.
  The drawdown penalty (>30%) naturally discourages simultaneous over-exposure.

──────────────────────────────────────────────────────────────────────────────
Reward  (simplified, per-asset isolated + portfolio-level tiers)

  Per-asset sub-reward (computed independently for each asset — fixes credit
  assignment: each asset gets a clean signal about its own behaviour only)

    +STEP_SCALE          flat patience — waiting is explicitly rewarded per asset
    +TRADE_SCALE × R     trade-close PnL (price-based R-multiple, capped [−3,+5])
                         positive R → bonus, negative R → penalty

  Portfolio-level reward (shared, fired once per step)

    Rolling fee threshold (168-step window):
      No per-step funding penalty — fees are a cost of doing business.
      Penalty fires only when cumulative fees (trading + funding) in the last
      168 steps exceed 2% of current equity — i.e. the agent is overtrading.
      Penalty ∝ excess above threshold, scaled to equity.

    Tiered drawdown penalties (continuous, escalating per 5% band):
      0–15%  DD  →  no penalty
      15–20% DD  →  small    (× 4)
      20–25% DD  →  medium   (× 8,  additive)
      25–30% DD  →  large    (×16,  additive)
      30%+   DD  →  critical (×32,  additive)

    Resettable gain tier bonuses (portfolio equity vs episode-start equity):
      Triggers on upward crossing of each 5% tier (5/10/15/20%).
      Tier unclaims when equity falls back below threshold → fires again
      on recovery. Moving forward to higher tiers gives proportionally
      larger bonuses — only way to earn more is to push higher.
      +5%  tier  →  +5  × STEP_SCALE
      +10% tier  →  +10 × STEP_SCALE
      +15% tier  →  +15 × STEP_SCALE
      +20% tier  →  +20 × STEP_SCALE

  Portfolio leverage cap  (hard mechanical constraint, not reward)
    Entry refused if sum(|positions|) + new_size > MAX_PORTFOLIO_LEVERAGE (6.0)

  Removed vs previous version:
    - Per-step funding penalty (replaced by rolling fee threshold)
    - Entry fee multiplier (12×)
    - Rolling Sharpe bonus (noisy, hard to learn from)
    - Per-trade milestone bonuses (1R, 2R)
    - Streak bonuses / penalties

──────────────────────────────────────────────────────────────────────────────
Data alignment

  All CSVs are inner-joined on open_time. Only hours present in ALL assets are used.
  Sampling and episode boundaries reference the aligned index.
  Bull/bear balanced sampling uses the first asset (BTC by convention) as regime proxy.
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# ── Constants ─────────────────────────────────────────────────────────────────
FLAT     = 0
IN_TRADE = 1

DIR_SHORT = -1
DIR_FLAT  =  0
DIR_LONG  =  1

SHORT_THRESHOLD = -0.7   # raised from -0.5: agent needs more conviction to go short
LONG_THRESHOLD  =  0.7   # raised from +0.5: agent needs more conviction to go long
# Flat zone is now 70% of the action range (-0.7 to +0.7) vs 50% before.
# This structurally forces more selectivity without relying purely on reward signals.

FEATURE_COLS = [
    "close", "rsi_14", "atr_14", "volatility_20",
    "log_returns", "momentum_pct",
    "last_funding_rate", "funding_cost_annualized",
    "price_vs_sma200", "price_vs_sma20",
    "bb_pct", "bb_width",
    "macd_hist", "volume_zscore", "funding_rate_zscore",
    "basis_pct",
    "donchian_pct", "donchian_width",
    "atr_ratio",
    "price_vs_sma500",
]
N_FEATURES = len(FEATURE_COLS)   # 20
ATR_IDX    = FEATURE_COLS.index("atr_14")
SMA200_IDX = FEATURE_COLS.index("price_vs_sma200")
SMA500_IDX = FEATURE_COLS.index("price_vs_sma500")

STEP_SCALE  = 0.001
TRADE_SCALE = 0.04              # raised from 0.015 — trade close is the primary signal
MAX_PORTFOLIO_LEVERAGE = 6.0    # hard cap on sum(|positions|) across all assets


class MultiAssetPerpEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_paths,                          # list of CSV paths, one per asset
        asset_names     = None,             # e.g. ["BTC", "ETH", "SOL", "BNB"]
        window_size     = 48,
        fee_rate        = 0.0005,           # Binance taker 0.05%
        max_leverage    = 3.0,
        initial_equity  = 1_000.0,
        max_trade_duration = 48,            # hours before force-exit
        min_hold_steps  = 12,              # candles before SL/TP can fire
        train_end_date  = None,            # "YYYY-MM-DD" — episodes end here during training
        eval_start      = None,            # fixed start row for evaluation
        eval_end        = None,            # fixed end   row for evaluation
    ):
        super().__init__()

        self.n_assets           = len(csv_paths)
        self.asset_names        = asset_names or [f"asset_{i}" for i in range(self.n_assets)]
        self.window_size        = window_size
        self.fee_rate           = fee_rate
        self.max_leverage       = max_leverage
        self.initial_equity     = initial_equity
        self.max_trade_duration = max_trade_duration
        self.min_hold_steps     = min_hold_steps
        self.eval_start         = eval_start

        self._load_and_align(csv_paths)

        T = len(self.common_times)

        if train_end_date is not None:
            end_dt             = pd.Timestamp(train_end_date)
            self.train_end_idx = int(sum(t <= end_dt for t in self.common_times)) - 1
        else:
            self.train_end_idx = None

        self.eval_end = eval_end if eval_end is not None else T - 1

        # ── Action space: n_assets × 4 ────────────────────────────────────────
        low  = np.tile(np.array([-1.0, 0.5, 1.5,  1.5], dtype=np.float32), self.n_assets)
        high = np.tile(np.array([ 1.0, 5.0, 5.0, 10.0], dtype=np.float32), self.n_assets)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Observation space ─────────────────────────────────────────────────
        n_portfolio_scalars = 3 + 5 * self.n_assets
        obs_dim = self.window_size * N_FEATURES * self.n_assets + n_portfolio_scalars
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = None

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_and_align(self, csv_paths):
        raw_dfs = [pd.read_csv(p, parse_dates=["open_time"]) for p in csv_paths]

        # Inner join on timestamp — only rows present in all assets
        common = set(raw_dfs[0]["open_time"])
        for df in raw_dfs[1:]:
            common &= set(df["open_time"])
        self.common_times = sorted(common)
        common_set = set(self.common_times)

        self.features = []
        self.prices   = []
        self.highs    = []
        self.lows     = []
        self.funding  = []

        for df in raw_dfs:
            df = (df[df["open_time"].isin(common_set)]
                    .sort_values("open_time")
                    .reset_index(drop=True))
            self.features.append(df[FEATURE_COLS].values.astype(np.float32))
            self.prices.append(df["close"].values.astype(np.float32))
            self.highs.append(df["high"].values.astype(np.float32))
            self.lows.append(df["low"].values.astype(np.float32))
            self.funding.append(df["last_funding_rate"].values.astype(np.float32))

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        T         = len(self.common_times)
        min_start = max(self.window_size, 500)   # need 500 bars for SMA-500 warmup

        if self.eval_start is not None:
            self.start_idx    = self.eval_start
            self._episode_end = self.eval_end
        else:
            upper_bound       = self.train_end_idx if self.train_end_idx is not None else T - 2
            self._episode_end = upper_bound

            # Balanced bull/bear sampling — use first asset (BTC) as regime proxy
            sma200_vals  = self.features[0][min_start:upper_bound, SMA200_IDX]
            bull_offsets = np.where(sma200_vals >  0)[0]
            bear_offsets = np.where(sma200_vals <= 0)[0]

            if len(bull_offsets) > 0 and len(bear_offsets) > 0:
                if self.np_random.random() < 0.5:
                    offset = int(self.np_random.choice(bull_offsets))
                else:
                    offset = int(self.np_random.choice(bear_offsets))
                self.start_idx = min_start + offset
            else:
                self.start_idx = int(self.np_random.integers(min_start, upper_bound))

        self.idx         = self.start_idx
        self.equity      = self.initial_equity
        self.peak_equity = self.equity

        # ── Per-asset trade state ─────────────────────────────────────────────
        self.trade_states      = [FLAT]          * self.n_assets
        self.positions         = [0.0]           * self.n_assets
        self.entry_prices      = [0.0]           * self.n_assets
        self.sl_prices         = [0.0]           * self.n_assets
        self.tp_prices         = [0.0]           * self.n_assets
        self.trade_entry_steps = [0]             * self.n_assets
        self.trade_risk_pcts   = [0.0]           * self.n_assets
        self.trade_directions  = [DIR_FLAT]      * self.n_assets
        self.equity_at_entries = [self.equity]   * self.n_assets
        self.just_entered      = [False]         * self.n_assets

        # ── Portfolio gain tier tracking (resettable triggers) ───────────────
        # Tiers: [+5%, +10%, +15%, +20%] from episode-start equity.
        # Claimed → True when bonus fires. Unclaimed → False when equity
        # drops back below threshold, enabling the trigger to fire again.
        self.gain_tiers_claimed = [False] * 4

        # ── Rolling fee window (168 steps = 1 week of hourly bars) ───────────
        # Tracks cumulative fees + funding to detect overtrading.
        self.fee_window = []

        # ── Global streak counter (kept for observation only) ─────────────────
        self.consec_wins   = 0
        self.consec_losses = 0

        obs  = self._get_observation()
        info = {"equity": self.equity, "n_assets": self.n_assets}
        return obs, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        # Market windows — one per asset, each normalised to its own last close
        windows = []
        for i in range(self.n_assets):
            window = self.features[i][self.idx - self.window_size : self.idx].copy()
            ref    = window[-1, 0]
            if ref > 0:
                window[:, 0] = window[:, 0] / ref - 1.0
            windows.append(window.flatten())

        market_obs = np.concatenate(windows)

        # Global portfolio scalars (3)
        global_scalars = np.array([
            self.equity / self.initial_equity,
            min(self.consec_wins,   10) / 10.0,
            min(self.consec_losses, 10) / 10.0,
        ], dtype=np.float32)

        # Per-asset scalars (5 × n_assets)
        per_asset = []
        for i in range(self.n_assets):
            in_trade = self.trade_states[i] == IN_TRADE

            steps_in_trade = (self.idx - self.trade_entry_steps[i]) if in_trade else 0
            steps_norm     = min(steps_in_trade / max(self.max_trade_duration, 1), 1.0)

            # Price-based unrealized R-multiple (independent of other assets' equity moves)
            if in_trade and self.trade_risk_pcts[i] > 0:
                curr_price = self.prices[i][self.idx]
                if self.trade_directions[i] == DIR_LONG:
                    price_move = (curr_price - self.entry_prices[i]) / (self.entry_prices[i] + 1e-8)
                else:
                    price_move = (self.entry_prices[i] - curr_price) / (self.entry_prices[i] + 1e-8)
                unrealized_norm = float(np.clip(
                    price_move * abs(self.positions[i]) / (self.trade_risk_pcts[i] / 100.0),
                    -5.0, 5.0,
                ))
            else:
                unrealized_norm = 0.0

            per_asset.extend([
                self.positions[i],
                float(in_trade),
                steps_norm,
                unrealized_norm,
                (self.trade_risk_pcts[i] / 5.0) if in_trade else 0.0,
            ])

        portfolio_obs = np.concatenate([global_scalars, np.array(per_asset, dtype=np.float32)])
        return np.concatenate([market_obs, portfolio_obs]).astype(np.float32)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        old_equity    = self.equity
        next_idx      = min(self.idx + 1, len(self.common_times) - 1)

        total_pnl     = 0.0
        total_fees    = 0.0
        total_funding = 0.0
        bonus         = 0.0
        penalty       = 0.0

        for i in range(self.n_assets):
            current_price = self.prices[i][self.idx]
            candle_high   = self.highs[i][self.idx]
            candle_low    = self.lows[i][self.idx]
            atr_val       = max(float(self.features[i][self.idx, ATR_IDX]), 1e-6)

            trade_fee    = 0.0
            trade_closed = False
            sl_hit = tp_hit = timeout_hit = False

            # ── FLAT: evaluate signal, potentially enter ───────────────────────
            if self.trade_states[i] == FLAT:
                a          = i * 4
                dir_signal = float(action[a])
                risk_pct   = float(np.clip(action[a + 1], 0.5,  5.0))
                sl_mult    = float(np.clip(action[a + 2], 1.5,  5.0))
                tp_mult    = float(np.clip(action[a + 3], 1.5, 10.0))

                if dir_signal < SHORT_THRESHOLD:
                    direction = DIR_SHORT
                elif dir_signal > LONG_THRESHOLD:
                    direction = DIR_LONG
                else:
                    direction = DIR_FLAT

                if direction != DIR_FLAT:
                    sl_distance = sl_mult * atr_val
                    tp_distance = max(tp_mult * atr_val, 2.0 * sl_distance)

                    sl_fraction   = sl_distance / (current_price + 1e-6)
                    position_size = min(
                        (risk_pct / 100.0) / (sl_fraction + 1e-8),
                        self.max_leverage,
                    )

                    # ── Portfolio leverage cap ─────────────────────────────────
                    # Refuse entry if total portfolio exposure would exceed limit.
                    # Prevents catastrophic 12× simultaneous leverage across 4 assets.
                    total_current = sum(abs(p) for p in self.positions)
                    if total_current + position_size > MAX_PORTFOLIO_LEVERAGE:
                        direction = DIR_FLAT   # skip this entry

                if direction != DIR_FLAT:
                    self.positions[i]         = direction * position_size
                    self.entry_prices[i]      = current_price
                    self.equity_at_entries[i] = self.equity
                    self.trade_risk_pcts[i]   = risk_pct
                    self.trade_directions[i]  = direction
                    self.trade_entry_steps[i] = self.idx
                    self.trade_states[i]      = IN_TRADE
                    self.just_entered[i]      = True

                    if direction == DIR_LONG:
                        self.sl_prices[i] = current_price - sl_distance
                        self.tp_prices[i] = current_price + tp_distance
                    else:
                        self.sl_prices[i] = current_price + sl_distance
                        self.tp_prices[i] = current_price - tp_distance

                    trade_fee = abs(self.positions[i]) * self.equity * self.fee_rate
                    # Reset just_entered flag — no longer used for fee penalty
                    # (entry cost is captured by the rolling fee threshold instead)

            # ── IN_TRADE: check exit conditions (actions ignored) ──────────────
            elif self.trade_states[i] == IN_TRADE:
                steps_held = self.idx - self.trade_entry_steps[i]

                if self.positions[i] > 0:   # long
                    sl_hit = candle_low  <= self.sl_prices[i]
                    tp_hit = candle_high >= self.tp_prices[i]
                else:                        # short
                    sl_hit = candle_high >= self.sl_prices[i]
                    tp_hit = candle_low  <= self.tp_prices[i]

                if steps_held < self.min_hold_steps:
                    sl_hit = False
                    tp_hit = False

                if sl_hit and tp_hit:        # both triggered → SL takes priority
                    tp_hit = False

                timeout_hit = steps_held >= self.max_trade_duration

                if sl_hit or tp_hit or timeout_hit:
                    trade_fee             = abs(self.positions[i]) * self.equity * self.fee_rate
                    self.positions[i]     = 0.0
                    self.trade_states[i]  = FLAT
                    trade_closed          = True

            # ── Price mechanics for this asset ────────────────────────────────
            next_price   = self.prices[i][next_idx]
            price_return = (next_price - current_price) / (current_price + 1e-8)

            funding_rate    = self.funding[i][self.idx]
            funding_payment = self.positions[i] * self.equity * funding_rate

            total_pnl     += self.positions[i] * self.equity * price_return
            total_fees    += trade_fee
            total_funding += funding_payment

            # ── Per-asset reward components (isolated — no cross-asset contamination)

            # ── Flat patience ──────────────────────────────────────────────────
            # Each asset independently rewarded for staying out of the market.
            # Waiting is a valid strategy; this makes it competitive with trading.
            if self.trade_states[i] == FLAT and not self.just_entered[i]:
                bonus += STEP_SCALE

            # Clear just_entered flag (entry fee penalty removed — rolling fee
            # threshold handles overtrading signal at the portfolio level instead)
            if self.just_entered[i]:
                self.just_entered[i] = False

            # ── Trade-close: per-asset price-based R-multiple ──────────────────
            # Isolated from other assets — this asset's price move only.
            # No streak bonuses/penalties (too noisy, removed for clarity).
            if trade_closed and self.trade_risk_pcts[i] > 0:
                if sl_hit:
                    exit_price = self.sl_prices[i]
                elif tp_hit:
                    exit_price = self.tp_prices[i]
                else:
                    exit_price = next_price

                if self.trade_directions[i] == DIR_LONG:
                    raw_return = (exit_price - self.entry_prices[i]) / (self.entry_prices[i] + 1e-8)
                else:
                    raw_return = (self.entry_prices[i] - exit_price) / (self.entry_prices[i] + 1e-8)

                r_multiple = raw_return * abs(self.positions[i]) / (self.trade_risk_pcts[i] / 100.0 + 1e-8)
                r_clipped  = float(np.clip(r_multiple, -3.0, 5.0))

                # Update streak counter (used for observation, not reward)
                if raw_return > 0:
                    self.consec_wins   += 1
                    self.consec_losses  = 0
                else:
                    self.consec_wins    = 0
                    self.consec_losses += 1

                if r_clipped >= 0:
                    bonus   += TRADE_SCALE * r_clipped
                else:
                    penalty += TRADE_SCALE * abs(r_clipped)

        # ── Update shared equity ──────────────────────────────────────────────
        self.equity += total_pnl - total_fees - total_funding
        self.equity  = max(self.equity, 1e-6)
        self.idx     = next_idx

        # ── Drawdown tracking ─────────────────────────────────────────────────
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)

        # ── Portfolio-level reward components ─────────────────────────────────
        base_return = np.log(self.equity / (old_equity + 1e-8))

        # ── Rolling fee threshold (168-step window) ────────────────────────────
        # Fees are a cost of doing business — no per-step penalty.
        # Penalty fires only when cumulative fees in the last 168 steps (1 week)
        # exceed 2% of current equity — a sign of destructive overtrading.
        step_fees = total_fees + abs(total_funding)
        self.fee_window.append(step_fees)
        if len(self.fee_window) > 168:
            self.fee_window.pop(0)
        rolling_fees  = sum(self.fee_window)
        fee_threshold = 0.02 * (self.equity + 1e-8)
        if rolling_fees > fee_threshold:
            excess_fees = rolling_fees - fee_threshold
            penalty += (excess_fees / (self.equity + 1e-8)) * 10.0

        # ── Tiered drawdown penalties (escalating per 5% band) ────────────────
        # No penalty below 15%. Each band above adds an increasing layer,
        # creating a smooth but steepening cost curve as drawdown worsens.
        # Asymmetric with gain bonuses — losses hurt more than gains help.
        if current_drawdown > 0.15:
            penalty += (current_drawdown - 0.15) * 4.0
        if current_drawdown > 0.20:
            penalty += (current_drawdown - 0.20) * 8.0
        if current_drawdown > 0.25:
            penalty += (current_drawdown - 0.25) * 16.0
        if current_drawdown > 0.30:
            penalty += (current_drawdown - 0.30) * 32.0

        # ── Resettable gain tier bonuses ───────────────────────────────────────
        # Fires on upward crossing of each 5% equity gain from episode start.
        # Tier unclaims when equity falls back below threshold — the agent can
        # earn the bonus again by recovering, but must push to higher tiers to
        # earn progressively larger rewards. Asymmetric: smaller than DD penalties.
        gain_thresholds = [1.05, 1.10, 1.15, 1.20]
        gain_bonuses    = [5,    10,   15,   20  ]   # × STEP_SCALE
        equity_ratio    = self.equity / (self.initial_equity + 1e-8)

        for t, (threshold, bonus_mult) in enumerate(zip(gain_thresholds, gain_bonuses)):
            if equity_ratio >= threshold:
                if not self.gain_tiers_claimed[t]:
                    bonus += bonus_mult * STEP_SCALE
                    self.gain_tiers_claimed[t] = True
            else:
                # Unclaim — tier becomes available to trigger again on recovery
                self.gain_tiers_claimed[t] = False

        reward = base_return + bonus - penalty
        done   = self.idx >= self._episode_end

        obs  = self._get_observation()
        info = {
            "equity":         self.equity,
            "positions":      list(self.positions),
            "trade_states":   ["IN_TRADE" if s == IN_TRADE else "FLAT" for s in self.trade_states],
            "drawdown":       current_drawdown,
            "total_pnl":      total_pnl,
            "total_fees":     total_fees,
            "total_funding":  total_funding,
        }
        return obs, reward, done, False, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        pos_str = "  ".join(
            f"{name}:{self.positions[i]:>+6.3f}"
            for i, name in enumerate(self.asset_names)
        )
        dd = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)
        print(
            f"t={self.idx}  equity={self.equity:>8.2f}  dd={dd:.1%}  [{pos_str}]"
        )
