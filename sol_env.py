import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class SolPerpEnv(gym.Env):
    """
    Perpetual-futures trading environment for PPO.

    Action space (3D continuous):
        action[0]  target position  [-1, +1]  (scaled by max_leverage)
        action[1]  stop-loss ATR multiplier   [0.5, 5.0]
        action[2]  take-profit ATR multiplier [0.5, 10.0]

    Observation (window_size × 16 features + 6 portfolio scalars):
        Market window features (16):
            close, rsi_14, atr_14, volatility_20, log_returns, momentum_pct,
            last_funding_rate, funding_cost_annualized,
            price_vs_sma200, price_vs_sma20, bb_pct, bb_width,
            macd_hist, volume_zscore, funding_rate_zscore, basis_pct

        Portfolio scalars (6):
            position, norm_equity, norm_consec_profit,
            sl_active, entry_price_norm, running_sharpe_20

    Reward — tiered scale (STEP_SCALE = 0.001):
        Base  : log equity return  (fees already deducted from equity)

        BONUSES
        +8×   : running 100-step Sharpe > 2.0  (consistency)
        +4×   : ≥10 consecutive profitable steps (streak)
        +2×   : position aligned with SMA-200 regime
        +1×   : take-profit hit naturally

        PENALTIES
        -8×·dd: drawdown excess above 30% threshold
        -4×   : stop-loss triggered
        -1×   : quadratic position-size penalty (λ=0.005)
        -0.5× : funding drag when not regime-aligned

        ANTI-CHURN  (3-layer system — the main overtrading fix)
        Layer A — Fee drag amplifier
            Every position change is penalised at FEE_AMPLIFIER × actual fee.
            With FEE_AMPLIFIER=15: a full flip at $1 000 equity costs
            2.0 × 0.0005 × 15 = 0.015 in reward, vs base_return noise ≈ 0.001.
            The agent must expect > 1.5 % equity growth to justify each trade.
        Layer B — Minimum hold enforcement  (MIN_HOLD_STEPS = 8 h)
            Flipping in < 8 steps incurs a scaled penalty up to 0.010.
            Enforces the agent to commit to a direction for at least 8 hours.
        Layer C — Progressive flip rate  (linear above 3 flips / 24 h)
            Each extra flip beyond 3 in a 24-step window adds 0.003 penalty.
            At the old rate of 13 flips/day: +0.030 per step continuously.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path,
        window_size=48,
        fee_rate=0.0005,        # Binance taker fee 0.05%
        max_leverage=2.0,
        initial_equity=1_000.0,
        eval_start=None,        # fixed start row for evaluation
        eval_end=None,          # fixed end row for evaluation
    ):
        super().__init__()

        self.df = pd.read_csv(csv_path, parse_dates=["open_time"])
        self.df = self.df.reset_index(drop=True)

        self.window_size    = window_size
        self.fee_rate       = fee_rate
        self.max_leverage   = max_leverage
        self.initial_equity = initial_equity
        self.eval_start     = eval_start
        self.eval_end       = eval_end if eval_end is not None else len(self.df) - 1

        self.feature_cols = [
            # Original (8)
            "close", "rsi_14", "atr_14", "volatility_20",
            "log_returns", "momentum_pct",
            "last_funding_rate", "funding_cost_annualized",
            # New trend (2)
            "price_vs_sma200", "price_vs_sma20",
            # New volatility / mean-reversion (2)
            "bb_pct", "bb_width",
            # New momentum / sentiment (3)
            "macd_hist", "volume_zscore", "funding_rate_zscore",
            # Basis (1)
            "basis_pct",
        ]

        # Pre-compute feature index lookups
        self._atr_idx    = self.feature_cols.index("atr_14")
        self._sma200_idx = self.feature_cols.index("price_vs_sma200")

        self._prepare_arrays()

        # ── Action space ──────────────────────────────────────────────────────
        # [target_position, sl_atr_mult, tp_atr_mult]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.5,  0.5], dtype=np.float32),
            high=np.array([1.0,  5.0, 10.0], dtype=np.float32),
            shape=(3,), dtype=np.float32,
        )

        # ── Observation space ─────────────────────────────────────────────────
        # window × features + 6 portfolio scalars
        obs_dim = self.window_size * len(self.feature_cols) + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = None

    # ── Data prep ─────────────────────────────────────────────────────────────

    def _prepare_arrays(self):
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.prices   = self.df["close"].values.astype(np.float32)
        self.funding  = self.df["last_funding_rate"].values.astype(np.float32)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Minimum start index: need window history + valid sma_200 (200 bars)
        min_start = max(self.window_size, 200)

        if self.eval_start is not None:
            self.start_idx = self.eval_start
        else:
            self.start_idx = int(self.np_random.integers(
                low=min_start, high=len(self.df) - 2
            ))

        self.idx        = self.start_idx
        self.equity     = self.initial_equity
        self.position   = 0.0
        self.last_price = self.prices[self.idx]
        self.peak_equity = self.equity

        # Stop-loss / take-profit state
        self.entry_price = 0.0
        self.sl_price    = 0.0
        self.tp_price    = 0.0
        self.sl_active   = False
        self.sl_hit      = False
        self.tp_hit      = False

        # Reward tracking
        self.consec_profit    = 0
        self.consec_loss      = 0
        self.return_history   = []    # last 100 step log-returns
        self.flip_history     = []    # recent step-indices of direction flips
        self.steps_since_flip = 0     # steps held in current direction (Layer B)

        obs  = self._get_observation()
        info = {"equity": self.equity}
        return obs, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self):
        start  = self.idx - self.window_size
        window = self.features[start:self.idx].copy()   # (window, n_feat)

        # Normalise close price column relative to last bar
        ref = window[-1, 0]   # close is index 0
        if ref > 0:
            window[:, 0] = window[:, 0] / ref - 1.0

        flat = window.flatten()

        # Portfolio scalars
        norm_equity  = self.equity / self.initial_equity
        norm_consec  = min(self.consec_profit, 50) / 50.0

        sl_active_f  = float(self.sl_active)
        entry_norm   = (self.entry_price / (self.last_price + 1e-8) - 1.0) \
                       if self.sl_active else 0.0

        if len(self.return_history) >= 20:
            arr = np.array(self.return_history[-20:])
            r_sharpe = float(np.clip(
                arr.mean() / (arr.std() + 1e-8) * np.sqrt(8760), -3.0, 3.0
            ))
        else:
            r_sharpe = 0.0

        portfolio = np.array(
            [self.position, norm_equity, norm_consec, sl_active_f, entry_norm, r_sharpe],
            dtype=np.float32,
        )
        return np.concatenate([flat, portfolio]).astype(np.float32)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        target_pos = float(np.clip(action[0], -1.0, 1.0) * self.max_leverage)
        sl_mult    = float(np.clip(action[1],  0.5,  5.0))
        tp_mult    = float(np.clip(action[2],  0.5, 10.0))

        current_price = self.prices[self.idx]
        atr_val       = float(self.features[self.idx, self._atr_idx])

        # ── Check SL/TP against current price ─────────────────────────────────
        self.sl_hit = False
        self.tp_hit = False

        if self.sl_active and abs(self.position) > 0.05:
            if self.position > 0:                           # long
                if current_price <= self.sl_price:
                    self.sl_hit = True;  target_pos = 0.0
                elif current_price >= self.tp_price:
                    self.tp_hit = True;  target_pos = 0.0
            else:                                           # short
                if current_price >= self.sl_price:
                    self.sl_hit = True;  target_pos = 0.0
                elif current_price <= self.tp_price:
                    self.tp_hit = True;  target_pos = 0.0

        # ── Trade mechanics ───────────────────────────────────────────────────
        pos_change = target_pos - self.position
        trade_fee  = abs(pos_change) * self.equity * self.fee_rate

        next_idx     = min(self.idx + 1, len(self.df) - 1)
        next_price   = self.prices[next_idx]
        price_return = (next_price - current_price) / (current_price + 1e-8)

        funding_rate    = self.funding[self.idx]
        funding_payment = self.position * self.equity * funding_rate

        old_equity   = self.equity
        pnl          = self.position * self.equity * price_return
        self.equity += pnl - trade_fee - funding_payment
        self.equity  = max(self.equity, 1e-6)

        # ── Update SL/TP when entering a new meaningful position ──────────────
        if abs(target_pos) > 0.05 and abs(pos_change) > 0.05:
            sl_dist = sl_mult * max(atr_val, 1e-6)
            tp_dist = tp_mult * max(atr_val, 1e-6)
            self.entry_price = current_price
            if target_pos > 0:
                self.sl_price = current_price - sl_dist
                self.tp_price = current_price + tp_dist
            else:
                self.sl_price = current_price + sl_dist
                self.tp_price = current_price - tp_dist
            self.sl_active = True
        elif abs(target_pos) <= 0.05:
            self.sl_active   = False
            self.entry_price = 0.0

        # ── Advance state ─────────────────────────────────────────────────────
        prev_sign     = np.sign(self.position)
        self.position = target_pos
        self.idx      = next_idx
        self.last_price = next_price

        # Track direction flips
        new_sign  = np.sign(target_pos)
        is_flip   = (prev_sign != new_sign and
                     (abs(prev_sign) > 0 or abs(new_sign) > 0))
        if is_flip:
            self.flip_history.append(self.idx)
            self.steps_since_flip = 0
        else:
            self.steps_since_flip += 1
        # Prune flip records older than 48 steps
        self.flip_history = [f for f in self.flip_history if self.idx - f <= 48]

        # ── Drawdown tracking ─────────────────────────────────────────────────
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-8)

        # ── Base reward ───────────────────────────────────────────────────────
        base_return = np.log(self.equity / (old_equity + 1e-8))

        # Update return history
        self.return_history.append(base_return)
        if len(self.return_history) > 100:
            self.return_history.pop(0)

        # Update consecutive streak
        step_pnl = pnl - trade_fee - funding_payment
        if step_pnl > 0:
            self.consec_profit += 1
            self.consec_loss    = 0
        else:
            self.consec_profit  = 0
            self.consec_loss   += 1

        # ── Tiered reward ─────────────────────────────────────────────────────
        STEP_SCALE = 0.001
        bonus   = 0.0
        penalty = 0.0

        # +8×: Running 100-step Sharpe > 2.0 (consistency reward)
        if len(self.return_history) >= 100:
            ret_arr        = np.array(self.return_history)
            running_sharpe = ret_arr.mean() / (ret_arr.std() + 1e-8) * np.sqrt(8760)
            if running_sharpe > 2.0:
                bonus += 8 * STEP_SCALE

        # +4×: Consecutive profitable streak (≥10 steps)
        if self.consec_profit >= 10:
            bonus += 4 * STEP_SCALE

        # +2×: Regime alignment — long above SMA-200, short below
        price_vs_sma200  = float(self.features[self.idx, self._sma200_idx])
        above_sma200     = price_vs_sma200 > 0
        regime_aligned   = (
            (self.position >  0.05 and     above_sma200) or
            (self.position < -0.05 and not above_sma200)
        )
        if regime_aligned:
            bonus += 2 * STEP_SCALE

        # +1×: Take-profit hit
        if self.tp_hit:
            bonus += 1 * STEP_SCALE

        # -8×·excess_dd: Drawdown above 30% threshold
        excess_dd = max(0.0, current_drawdown - 0.30)
        penalty  += 8.0 * excess_dd

        # -4×: Stop-loss hit
        if self.sl_hit:
            penalty += 4 * STEP_SCALE

        # ── Anti-churn  (3 layers) ────────────────────────────────────────────
        FEE_AMPLIFIER  = 15.0    # Layer A: explicit fee signal multiplier
        MIN_HOLD_STEPS = 8       # Layer B: minimum hold before penalty-free flip
        FLIP_THRESHOLD = 3       # Layer C: free flips per 24-step window
        FLIP_RATE_COST = 0.003   # Layer C: penalty per extra flip above threshold

        # Layer A — fee drag amplifier
        # Penalise EVERY position change at 15× the actual fee cost.
        # Forces the agent to expect > 1.5 % equity growth to justify a trade.
        penalty += abs(pos_change) * self.fee_rate * FEE_AMPLIFIER

        # Layer B — minimum hold enforcement
        # If the agent flips before holding for MIN_HOLD_STEPS, scale a penalty
        # proportional to how early the flip happened (0 → 1 as hold → 0).
        if is_flip and 0 < self.steps_since_flip < MIN_HOLD_STEPS:
            early_frac = (MIN_HOLD_STEPS - self.steps_since_flip) / MIN_HOLD_STEPS
            penalty   += 0.010 * early_frac

        # Layer C — progressive flip rate
        # Each flip beyond FLIP_THRESHOLD in the last 24 steps adds a flat cost.
        # Old rate: 13 flips/day → +0.030/step continuously.
        # Target rate: ≤ 4 flips/day → +0.003/step (manageable).
        recent_flips_24h = sum(1 for f in self.flip_history if self.idx - f <= 24)
        if recent_flips_24h > FLIP_THRESHOLD:
            penalty += FLIP_RATE_COST * (recent_flips_24h - FLIP_THRESHOLD)

        # -1×: Quadratic position-size penalty
        penalty += 0.005 * (self.position ** 2)

        # -0.5×: Funding drag when not regime-aligned
        if abs(self.position) > 0.05 and not regime_aligned:
            penalty += 0.5 * abs(funding_payment / (old_equity + 1e-8))

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
            "sl_hit":           self.sl_hit,
            "tp_hit":           self.tp_hit,
            "regime_aligned":   regime_aligned,
        }
        return obs, reward, done, False, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        print(
            f"t={self.idx}  price={self.last_price:.4f}  "
            f"equity={self.equity:.2f}  pos={self.position:.3f}  "
            f"sl_active={self.sl_active}  "
            f"sl={self.sl_price:.4f}  tp={self.tp_price:.4f}"
        )
