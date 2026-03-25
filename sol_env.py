"""
sol_env.py
==========
Perpetual-futures trading environment — committed position manager.

The agent operates like a disciplined position trader:

  FLAT state    Reads market signals. Decides direction, risk %, SL and TP.
                If uncertain → outputs direction in (-0.33, +0.33) → stays flat.
                If decided   → enters trade, transitions to IN_TRADE.

  IN_TRADE      Position is LOCKED. Actions are ignored.
                Exits on:  SL hit  |  TP hit  |  max_trade_duration elapsed.
                On exit    → returns to FLAT.

──────────────────────────────────────────────────────────────────────────────
Action space  Box(4,)   — always emitted, only read when FLAT

  action[0]   direction signal    [-1.0, +1.0]
              < -0.33  →  SHORT
              > +0.33  →  LONG
              else     →  FLAT  (stay out)

  action[1]   risk_pct            [0.5, 5.0]   % of equity to risk per trade
  action[2]   sl_atr_mult         [1.5, 5.0]   ATR multiples for stop-loss
  action[3]   tp_atr_mult         [1.5, 10.0]  ATR multiples for take-profit

──────────────────────────────────────────────────────────────────────────────
Position sizing  (risk-based, automatic)

  risk_amount   = (risk_pct / 100) × equity
  sl_distance   = sl_atr_mult × ATR              (price units)
  sl_fraction   = sl_distance / entry_price       (as fraction)
  position_size = (risk_pct/100) / sl_fraction    (leverage ×)
  → capped at max_leverage (default 3.0×)

  Consequence: high ATR → wide SL required → position auto-shrinks.
               The agent cannot accidentally over-leverage in volatile markets.

──────────────────────────────────────────────────────────────────────────────
Observation  (window_size × 16  +  8 portfolio scalars)

  Market window (16 features per bar):
    close_norm, rsi_14, atr_14, volatility_20, log_returns, momentum_pct,
    last_funding_rate, funding_cost_annualized,
    price_vs_sma200, price_vs_sma20, bb_pct, bb_width,
    macd_hist, volume_zscore, funding_rate_zscore, basis_pct

  Portfolio scalars (8):
    position            current leveraged position (0 when flat)
    norm_equity         equity / initial_equity
    trade_state         0 = FLAT, 1 = IN_TRADE
    steps_in_trade_norm steps held / max_trade_duration  (0→1)
    unrealized_pnl_norm (equity - entry_equity) / entry_equity / risk_fraction
                        → how many R multiples is the current trade up/down
    consec_wins_norm    consecutive winning trades / 10  (capped)
    consec_losses_norm  consecutive losing trades  / 10  (capped)
    current_risk_norm   trade_risk_pct / 5.0  (0 when flat)

──────────────────────────────────────────────────────────────────────────────
Reward

  Step-level
    base_return          log(equity_t / equity_{t-1})   ← fees already in equity
    +2×STEP_SCALE        position aligned with SMA-200 regime (per step)
    +4×STEP_SCALE        running 50-step Sharpe > 1.5  (consistency)
    -8× excess_dd        drawdown above 30% threshold

  Trade-close  (fired once when SL / TP / timeout)
    ±TRADE_SCALE × R     R-multiple of trade:  win=+, loss=−
                         R = (equity_exit − equity_entry) /
                             equity_entry / (risk_pct/100)
    Capped: R ∈ [−3, +5]
    streak_penalty       -2×STEP if 3+ consecutive losses
    streak_bonus         +2×STEP if 3+ consecutive wins

  No churn penalties — commitment is enforced structurally (IN_TRADE lock).
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# State constants
FLAT     = 0
IN_TRADE = 1

# Direction constants
DIR_SHORT = -1
DIR_FLAT  =  0
DIR_LONG  =  1

# Direction thresholds on action[0]
SHORT_THRESHOLD = -0.33
LONG_THRESHOLD  =  0.33


class SolPerpEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path,
        window_size         = 48,
        fee_rate            = 0.0005,   # Binance taker 0.05 %
        max_leverage        = 3.0,      # hard cap on position size
        initial_equity      = 1_000.0,
        max_trade_duration  = 48,       # hours before force-exit  ← parameterised
        eval_start          = None,     # fixed start row for evaluation
        eval_end            = None,     # fixed end   row for evaluation
    ):
        super().__init__()

        self.df = pd.read_csv(csv_path, parse_dates=["open_time"])
        self.df = self.df.reset_index(drop=True)

        self.window_size        = window_size
        self.fee_rate           = fee_rate
        self.max_leverage       = max_leverage
        self.initial_equity     = initial_equity
        self.max_trade_duration = max_trade_duration
        self.eval_start         = eval_start
        self.eval_end           = eval_end if eval_end is not None else len(self.df) - 1

        self.feature_cols = [
            # Original 8
            "close", "rsi_14", "atr_14", "volatility_20",
            "log_returns", "momentum_pct",
            "last_funding_rate", "funding_cost_annualized",
            # Trend 2
            "price_vs_sma200", "price_vs_sma20",
            # Volatility / mean-reversion 2
            "bb_pct", "bb_width",
            # Momentum / sentiment 3
            "macd_hist", "volume_zscore", "funding_rate_zscore",
            # Basis 1
            "basis_pct",
        ]

        self._atr_idx    = self.feature_cols.index("atr_14")
        self._sma200_idx = self.feature_cols.index("price_vs_sma200")

        self._prepare_arrays()

        # ── Action space ──────────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low  = np.array([-1.0, 0.5, 1.5,  1.5], dtype=np.float32),
            high = np.array([ 1.0, 5.0, 5.0, 10.0], dtype=np.float32),
            shape=(4,), dtype=np.float32,
        )

        # ── Observation space ─────────────────────────────────────────────────
        n_scalars = 8
        obs_dim   = self.window_size * len(self.feature_cols) + n_scalars
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = None

    # ── Data prep ─────────────────────────────────────────────────────────────

    def _prepare_arrays(self):
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.prices   = self.df["close"].values.astype(np.float32)
        self.highs    = self.df["high"].values.astype(np.float32)
        self.lows     = self.df["low"].values.astype(np.float32)
        self.funding  = self.df["last_funding_rate"].values.astype(np.float32)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        min_start = max(self.window_size, 200)
        if self.eval_start is not None:
            self.start_idx = self.eval_start
        else:
            self.start_idx = int(self.np_random.integers(
                low=min_start, high=len(self.df) - 2
            ))

        self.idx         = self.start_idx
        self.equity      = self.initial_equity
        self.peak_equity = self.equity
        self.last_price  = self.prices[self.idx]

        # ── Trade state machine ───────────────────────────────────────────────
        self.trade_state      = FLAT
        self.position         = 0.0
        self.entry_price      = 0.0
        self.sl_price         = 0.0
        self.tp_price         = 0.0
        self.trade_entry_step = 0
        self.trade_risk_pct   = 0.0
        self.trade_direction  = DIR_FLAT
        self.equity_at_entry  = self.equity

        # ── Streak tracking (trade-level, not step-level) ─────────────────────
        self.consec_wins   = 0
        self.consec_losses = 0

        # ── Step return history for running Sharpe ────────────────────────────
        self.return_history = []

        obs  = self._get_observation()
        info = {"equity": self.equity, "trade_state": "FLAT"}
        return obs, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        start  = self.idx - self.window_size
        window = self.features[start : self.idx].copy()

        # Normalise close price relative to last bar in window
        ref = window[-1, 0]
        if ref > 0:
            window[:, 0] = window[:, 0] / ref - 1.0

        flat = window.flatten()

        # ── Portfolio scalars ─────────────────────────────────────────────────
        norm_equity      = self.equity / self.initial_equity
        trade_state_flag = float(self.trade_state == IN_TRADE)

        steps_in_trade      = (self.idx - self.trade_entry_step) \
                               if self.trade_state == IN_TRADE else 0
        steps_in_trade_norm = min(steps_in_trade / max(self.max_trade_duration, 1), 1.0)

        # Unrealized P&L in R-multiples (how many times the risked amount gained/lost)
        if self.trade_state == IN_TRADE and self.trade_risk_pct > 0:
            unrealized_pct      = (self.equity - self.equity_at_entry) \
                                   / (self.equity_at_entry + 1e-8)
            unrealized_pnl_norm = float(np.clip(
                unrealized_pct / (self.trade_risk_pct / 100.0), -5.0, 5.0
            ))
        else:
            unrealized_pnl_norm = 0.0

        consec_wins_norm   = min(self.consec_wins,   10) / 10.0
        consec_losses_norm = min(self.consec_losses, 10) / 10.0
        current_risk_norm  = (self.trade_risk_pct / 5.0) \
                              if self.trade_state == IN_TRADE else 0.0

        portfolio = np.array([
            self.position,
            norm_equity,
            trade_state_flag,
            steps_in_trade_norm,
            unrealized_pnl_norm,
            consec_wins_norm,
            consec_losses_norm,
            current_risk_norm,
        ], dtype=np.float32)

        return np.concatenate([flat, portfolio]).astype(np.float32)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        current_price = self.prices[self.idx]
        candle_high   = self.highs[self.idx]
        candle_low    = self.lows[self.idx]
        atr_val       = max(float(self.features[self.idx, self._atr_idx]), 1e-6)

        trade_fee    = 0.0
        sl_hit       = False
        tp_hit       = False
        timeout_hit  = False
        trade_closed = False
        r_multiple   = 0.0

        # ── FLAT: evaluate signal, potentially enter ───────────────────────────
        if self.trade_state == FLAT:

            direction_signal = float(action[0])
            risk_pct         = float(np.clip(action[1], 0.5,  5.0))
            sl_mult          = float(np.clip(action[2], 1.5,  5.0))
            tp_mult          = float(np.clip(action[3], 1.5, 10.0))

            if direction_signal < SHORT_THRESHOLD:
                direction = DIR_SHORT
            elif direction_signal > LONG_THRESHOLD:
                direction = DIR_LONG
            else:
                direction = DIR_FLAT   # agent chose to stay out

            if direction != DIR_FLAT:
                # Risk-based position sizing
                sl_distance   = sl_mult * atr_val
                sl_fraction   = sl_distance / (current_price + 1e-6)
                risk_fraction = risk_pct / 100.0
                position_size = min(
                    risk_fraction / (sl_fraction + 1e-8),
                    self.max_leverage,
                )

                self.position         = direction * position_size
                self.entry_price      = current_price
                self.equity_at_entry  = self.equity
                self.trade_risk_pct   = risk_pct
                self.trade_direction  = direction
                self.trade_entry_step = self.idx
                self.trade_state      = IN_TRADE

                if direction == DIR_LONG:
                    self.sl_price = current_price - sl_distance
                    self.tp_price = current_price + tp_mult * atr_val
                else:
                    self.sl_price = current_price + sl_distance
                    self.tp_price = current_price - tp_mult * atr_val

                # Entry fee
                trade_fee = abs(self.position) * self.equity * self.fee_rate

        # ── IN_TRADE: check exit conditions (actions ignored) ─────────────────
        elif self.trade_state == IN_TRADE:

            steps_held  = self.idx - self.trade_entry_step

            # Check SL/TP against candle high and low (not just close)
            # This correctly captures intra-candle SL/TP touches
            if self.position > 0:    # long
                sl_hit = candle_low  <= self.sl_price
                tp_hit = candle_high >= self.tp_price
            else:                     # short
                sl_hit = candle_high >= self.sl_price
                tp_hit = candle_low  <= self.tp_price

            # If both triggered in the same candle → SL takes priority (conservative)
            if sl_hit and tp_hit:
                tp_hit = False

            timeout_hit = steps_held >= self.max_trade_duration

            if sl_hit or tp_hit or timeout_hit:
                trade_fee         = abs(self.position) * self.equity * self.fee_rate
                self.position     = 0.0
                self.trade_state  = FLAT
                trade_closed      = True

        # ── Price mechanics ───────────────────────────────────────────────────
        next_idx      = min(self.idx + 1, len(self.df) - 1)
        next_price    = self.prices[next_idx]
        price_return  = (next_price - current_price) / (current_price + 1e-8)

        funding_rate    = self.funding[self.idx]
        funding_payment = self.position * self.equity * funding_rate

        old_equity   = self.equity
        pnl          = self.position * self.equity * price_return
        self.equity += pnl - trade_fee - funding_payment
        self.equity  = max(self.equity, 1e-6)

        # ── Update trade streak on close ──────────────────────────────────────
        if trade_closed and self.trade_risk_pct > 0:
            trade_pnl_pct = (self.equity - self.equity_at_entry) \
                             / (self.equity_at_entry + 1e-8)
            r_multiple    = trade_pnl_pct / (self.trade_risk_pct / 100.0)
            if self.equity > self.equity_at_entry:
                self.consec_wins   += 1
                self.consec_losses  = 0
            else:
                self.consec_wins    = 0
                self.consec_losses += 1

        # ── Advance state ─────────────────────────────────────────────────────
        self.idx        = next_idx
        self.last_price = next_price

        # ── Drawdown tracking ─────────────────────────────────────────────────
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)

        # ── Reward ────────────────────────────────────────────────────────────
        base_return = np.log(self.equity / (old_equity + 1e-8))

        self.return_history.append(base_return)
        if len(self.return_history) > 100:
            self.return_history.pop(0)

        STEP_SCALE  = 0.001
        TRADE_SCALE = 0.015
        bonus   = 0.0
        penalty = 0.0

        # Regime alignment: long above SMA-200, short below (only while in trade)
        price_vs_sma200 = float(self.features[self.idx, self._sma200_idx])
        above_sma200    = price_vs_sma200 > 0
        regime_aligned  = (
            (self.position >  0.05 and     above_sma200) or
            (self.position < -0.05 and not above_sma200)
        )
        if regime_aligned:
            bonus += 2 * STEP_SCALE

        # Running Sharpe consistency bonus (last 50 steps)
        if len(self.return_history) >= 50:
            arr    = np.array(self.return_history)
            sharpe = arr.mean() / (arr.std() + 1e-8) * np.sqrt(8760)
            if sharpe > 1.5:
                bonus += 4 * STEP_SCALE

        # Drawdown penalty
        excess_dd = max(0.0, current_drawdown - 0.30)
        penalty  += 8.0 * excess_dd

        # Trade-close: R-multiple calibration signal
        if trade_closed:
            r_clipped = float(np.clip(r_multiple, -3.0, 5.0))
            if r_clipped >= 0:
                bonus   += TRADE_SCALE * r_clipped
            else:
                penalty += TRADE_SCALE * abs(r_clipped)

            # Streak signals
            if self.consec_losses >= 3:
                penalty += 2 * STEP_SCALE   # calibration is off
            if self.consec_wins   >= 3:
                bonus   += 2 * STEP_SCALE   # agent is reading market well

        reward = base_return + bonus - penalty
        done   = self.idx >= self.eval_end

        obs  = self._get_observation()
        info = {
            "equity":           self.equity,
            "position":         self.position,
            "price":            self.last_price,
            "pnl":              pnl,
            "trade_fee":        trade_fee,
            "funding_payment":  funding_payment,
            "drawdown":         current_drawdown,
            "sl_hit":           sl_hit,
            "tp_hit":           tp_hit,
            "timeout_hit":      timeout_hit,
            "trade_closed":     trade_closed,
            "r_multiple":       r_multiple,
            "regime_aligned":   regime_aligned,
            "trade_state":      "IN_TRADE" if self.trade_state == IN_TRADE else "FLAT",
            "trade_risk_pct":   self.trade_risk_pct if self.trade_state == IN_TRADE else 0.0,
        }
        return obs, reward, done, False, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        state = "IN_TRADE" if self.trade_state == IN_TRADE else "FLAT    "
        print(
            f"t={self.idx}  price={self.last_price:>8.2f}  "
            f"equity={self.equity:>8.2f}  pos={self.position:>+6.3f}  "
            f"state={state}  "
            f"sl={self.sl_price:>8.2f}  tp={self.tp_price:>8.2f}  "
            f"W={self.consec_wins}  L={self.consec_losses}"
        )
