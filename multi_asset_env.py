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
Reward  (same structure as SolPerpEnv, aggregated across assets)

  Step-level
    base_return          log(equity_t / equity_{t-1})
    +2×STEP per asset    position regime-aligned with SMA-200 AND SMA-500
    +0.5×STEP per asset  flat patience (reduces overtrading per asset)
    +4×STEP              running 50-step portfolio Sharpe > 1.5
    -8× excess_dd        portfolio drawdown above 30%
    -6× fee_rate         per-trade entry fee penalty (per asset)

  Trade-close (per asset, fired on SL / TP / timeout)
    ± TRADE_SCALE × R    price-based R-multiple (direction-adjusted, capped ±[−3, +5])
    streak_penalty        -2×STEP if 3+ consecutive losses (global)
    streak_bonus          +2×STEP if 3+ consecutive wins   (global)

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

SHORT_THRESHOLD = -0.5
LONG_THRESHOLD  =  0.5

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
TRADE_SCALE = 0.015


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

        # ── Global streak and return history ──────────────────────────────────
        self.consec_wins    = 0
        self.consec_losses  = 0
        self.return_history = []

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

            # ── Per-asset reward components ────────────────────────────────────

            # Trade-close: price-based R-multiple (unaffected by other assets' equity moves)
            if trade_closed and self.trade_risk_pcts[i] > 0:
                if sl_hit:
                    exit_price = self.sl_prices[i]
                elif tp_hit:
                    exit_price = self.tp_prices[i]
                else:                        # timeout — exits at next open (approx. next_price)
                    exit_price = next_price

                if self.trade_directions[i] == DIR_LONG:
                    raw_return = (exit_price - self.entry_prices[i]) / (self.entry_prices[i] + 1e-8)
                else:
                    raw_return = (self.entry_prices[i] - exit_price) / (self.entry_prices[i] + 1e-8)

                r_multiple = raw_return * abs(self.positions[i]) / (self.trade_risk_pcts[i] / 100.0 + 1e-8)

                # Update global streak
                if raw_return > 0:
                    self.consec_wins   += 1
                    self.consec_losses  = 0
                else:
                    self.consec_wins    = 0
                    self.consec_losses += 1

                r_clipped = float(np.clip(r_multiple, -3.0, 5.0))
                if r_clipped >= 0:
                    bonus   += TRADE_SCALE * r_clipped
                else:
                    penalty += TRADE_SCALE * abs(r_clipped)

                if self.consec_losses >= 3:
                    penalty += 2 * STEP_SCALE
                if self.consec_wins >= 3:
                    bonus   += 2 * STEP_SCALE

            # Regime alignment
            sma200 = float(self.features[i][self.idx, SMA200_IDX])
            sma500 = float(self.features[i][self.idx, SMA500_IDX])
            strong_bull    = sma200 > 0 and sma500 > 0
            strong_bear    = sma200 <= 0 and sma500 <= 0
            regime_aligned = (
                (self.positions[i] >  0.05 and strong_bull) or
                (self.positions[i] < -0.05 and strong_bear)
            )
            if regime_aligned:
                bonus += 2 * STEP_SCALE

            # Flat patience per asset
            if self.trade_states[i] == FLAT and not self.just_entered[i]:
                bonus += 0.5 * STEP_SCALE

            # Entry fee penalty (makes cost salient during learning)
            if self.just_entered[i]:
                penalty += abs(self.positions[i]) * self.fee_rate * 6.0
                self.just_entered[i] = False

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

        self.return_history.append(base_return)
        if len(self.return_history) > 100:
            self.return_history.pop(0)

        # Running Sharpe consistency bonus
        if len(self.return_history) >= 50:
            arr    = np.array(self.return_history)
            sharpe = arr.mean() / (arr.std() + 1e-8) * np.sqrt(8760)
            if sharpe > 1.5:
                bonus += 4 * STEP_SCALE

        # Drawdown penalty
        excess_dd = max(0.0, current_drawdown - 0.30)
        penalty  += 8.0 * excess_dd

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
