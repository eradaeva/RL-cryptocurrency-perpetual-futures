"""
Perpetual Futures Data Pipeline
================================
Single unified script for fetching, merging, and processing perpetual futures
data from Binance. Produces CSVs that are ready for sol_env.py / training.

Data fetched per asset:
  - Perpetual futures OHLCV  (/fapi/v1/klines)
  - Real funding rates        (/fapi/v1/fundingRate)
  - Spot OHLCV for basis     (/api/v3/klines)

Output columns (36):
  open_time, open, high, low, close,
  volume, quote_volume, taker_buy_volume, taker_buy_quote_volume,
  volume_ratio_taker,
  last_funding_rate, funding_interval_hours, funding_cost_annualized,
  sma_20, rsi_14, atr_14, volatility_20,
  log_returns, momentum_pct, price_range_pct,
  basis, basis_pct,
  sma_200, ema_20, price_vs_sma200, price_vs_sma20,
  bb_width, bb_pct, macd_hist, volume_zscore, funding_rate_zscore,
  sma_500, price_vs_sma500,
  donchian_pct, donchian_width, atr_ratio

Usage:
    python perpetual_futures_data_pipeline.py
    Configure ASSETS, START_DATE, END_DATE at the bottom of the file.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
from typing import List, Optional


# ── API endpoints ─────────────────────────────────────────────────────────────
FUTURES_URL = "https://fapi.binance.com"
SPOT_URL    = "https://api.binance.com"
API_DELAY   = 0.2   # seconds between requests


class PerpetualFuturesDataPipeline:
    """
    Unified pipeline: fetch → merge → indicators → save.

    Call process_asset(symbol, start_date, end_date) for each asset.
    """

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Fetchers ──────────────────────────────────────────────────────────────

    def _fetch_klines(self, url: str, symbol: str,
                      start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generic 1-hour kline fetcher (works for both futures and spot endpoints)."""
        all_data  = []
        current   = int(start_date.timestamp() * 1000)
        end_ts    = int(end_date.timestamp()   * 1000)
        batch_n   = 0

        while current < end_ts:
            try:
                resp = requests.get(
                    url,
                    params={"symbol": symbol, "interval": "1h",
                            "startTime": current, "limit": 1000},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                all_data.extend(data)
                batch_n += 1
                if batch_n % 10 == 0:
                    print(f"    {len(all_data):,} candles fetched...")
                current = int(data[-1][0]) + 1
                time.sleep(API_DELAY)
            except Exception as e:
                print(f"    [ERROR] {e}")
                break

        if not all_data:
            return pd.DataFrame()

        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "count",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
        df = pd.DataFrame(all_data, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            df[c] = df[c].astype(float)
        return df[["open_time", "open", "high", "low", "close", "volume",
                   "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]]

    def fetch_futures_ohlcv(self, symbol: str,
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        print(f"  Fetching {symbol} perpetual futures OHLCV...")
        df = self._fetch_klines(
            f"{FUTURES_URL}/fapi/v1/klines", symbol, start_date, end_date
        )
        if not df.empty:
            print(f"    ✓ {len(df):,} candles "
                  f"({df['open_time'].min().date()} → {df['open_time'].max().date()})")
        return df

    def fetch_spot_ohlcv(self, symbol: str,
                         start_date: datetime, end_date: datetime) -> pd.DataFrame:
        print(f"  Fetching {symbol} spot OHLCV (for basis)...")
        df = self._fetch_klines(
            f"{SPOT_URL}/api/v3/klines", symbol, start_date, end_date
        )
        if not df.empty:
            print(f"    ✓ {len(df):,} spot candles")
        return df[["open_time", "close"]].rename(columns={"close": "spot_close"})

    def fetch_funding_rates(self, symbol: str,
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        print(f"  Fetching {symbol} funding rates...")
        all_rates = []
        current   = int(start_date.timestamp() * 1000)
        end_ts    = int(end_date.timestamp()   * 1000)

        while current < end_ts:
            try:
                resp = requests.get(
                    f"{FUTURES_URL}/fapi/v1/fundingRate",
                    params={"symbol": symbol, "startTime": current,
                            "endTime": end_ts, "limit": 1000},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                all_rates.extend(data)
                current = int(data[-1]["fundingTime"]) + 1
                time.sleep(API_DELAY)
            except Exception as e:
                print(f"    [ERROR] funding rates: {e}")
                break

        if not all_rates:
            print("    [WARN] No funding rate data — using 0.0001 placeholder")
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)
        df = df.rename(columns={"fundingTime": "calc_time",
                                 "fundingRate": "last_funding_rate"})
        df["calc_time"]              = pd.to_datetime(df["calc_time"], unit="ms")
        df["last_funding_rate"]      = df["last_funding_rate"].astype(float)
        df["funding_interval_hours"] = 8
        print(f"    ✓ {len(df):,} funding rate records")
        return df[["calc_time", "last_funding_rate", "funding_interval_hours"]]

    # ── Merge ─────────────────────────────────────────────────────────────────

    def merge_funding(self, ohlcv: pd.DataFrame,
                      funding: pd.DataFrame) -> pd.DataFrame:
        """Attach most-recent funding rate to each hourly candle via asof merge."""
        if funding.empty:
            ohlcv["last_funding_rate"]      = 0.0001
            ohlcv["funding_interval_hours"] = 8
            return ohlcv

        ohlcv   = ohlcv.sort_values("open_time").reset_index(drop=True)
        funding = funding.sort_values("calc_time").reset_index(drop=True)

        merged = pd.merge_asof(
            ohlcv.rename(columns={"open_time": "ts"}),
            funding.rename(columns={"calc_time": "ts"}),
            on="ts", direction="backward",
        ).rename(columns={"ts": "open_time"})

        merged["last_funding_rate"]      = (merged["last_funding_rate"]
                                            .bfill().fillna(0.0001))
        merged["funding_interval_hours"] = (merged["funding_interval_hours"]
                                            .fillna(8))
        return merged

    def merge_basis(self, df: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
        """Compute spot-futures basis and merge into main dataframe."""
        if spot.empty:
            df["basis"]     = 0.0
            df["basis_pct"] = 0.0
            return df

        merged = pd.merge(df, spot, on="open_time", how="left")
        merged["spot_close"] = merged["spot_close"].ffill().fillna(merged["close"])
        merged["basis"]      = merged["close"] - merged["spot_close"]
        merged["basis_pct"]  = (merged["basis"] / merged["spot_close"]) * 100
        merged.drop(columns=["spot_close"], inplace=True)
        return merged

    # ── Indicators ────────────────────────────────────────────────────────────

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators (original + new)."""
        close = df["close"]

        # ── Funding-derived ───────────────────────────────────────────────────
        df["funding_cost_annualized"] = (
            df["last_funding_rate"] * (365.25 * 24 / df["funding_interval_hours"])
        )

        # ── Volume ratio ──────────────────────────────────────────────────────
        df["volume_ratio_taker"] = df["taker_buy_volume"] / (df["volume"] + 1e-8)

        # ── Original indicators ───────────────────────────────────────────────
        df["sma_20"]        = close.rolling(20).mean()
        df["rsi_14"]        = self._rsi(close, 14)
        df["atr_14"]        = self._atr(df, 14)
        df["volatility_20"] = close.pct_change().rolling(20).std()
        df["log_returns"]   = np.log(close / close.shift(1))
        df["momentum_pct"]  = (close - close.shift(20)) / close.shift(20) * 100
        df["price_range_pct"] = ((df["high"] - df["low"]) / close) * 100

        # ── Trend (new) ───────────────────────────────────────────────────────
        df["sma_200"]         = close.rolling(200).mean()
        df["ema_20"]          = close.ewm(span=20, adjust=False).mean()
        df["price_vs_sma200"] = (close / df["sma_200"] - 1).clip(-1.0, 1.0)
        df["price_vs_sma20"]  = (close / df["sma_20"]  - 1).clip(-1.0, 1.0)

        # ── Bollinger Bands (new) ─────────────────────────────────────────────
        bb_std          = close.rolling(20).std()
        bb_upper        = df["sma_20"] + 2 * bb_std
        bb_lower        = df["sma_20"] - 2 * bb_std
        bb_range        = (bb_upper - bb_lower)
        df["bb_width"]  = (bb_range / df["sma_20"]).clip(0.0, 1.0)
        df["bb_pct"]    = ((close - bb_lower) / (bb_range + 1e-10)).clip(0.0, 1.0)

        # ── MACD histogram normalised by close (new) ──────────────────────────
        ema_12          = close.ewm(span=12, adjust=False).mean()
        ema_26          = close.ewm(span=26, adjust=False).mean()
        macd            = ema_12 - ema_26
        signal          = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = ((macd - signal) / (close + 1e-10)).clip(-0.1, 0.1)

        # ── Volume Z-score rolling 20 (new) ───────────────────────────────────
        vol_std              = df["volume"].rolling(20).std()
        df["volume_zscore"]  = (
            (df["volume"] - df["volume"].rolling(20).mean()) / (vol_std + 1e-10)
        ).clip(-4.0, 4.0)

        # ── Funding rate Z-score rolling 48 (new) ────────────────────────────
        fr                        = df["last_funding_rate"]
        fr_std                    = fr.rolling(48).std()
        df["funding_rate_zscore"] = (
            (fr - fr.rolling(48).mean()) / (fr_std + 1e-10)
        ).clip(-4.0, 4.0)

        # ── SMA-500 long-term trend ───────────────────────────────────────────
        df["sma_500"]         = close.rolling(500).mean()
        df["price_vs_sma500"] = (close / df["sma_500"] - 1).clip(-1.0, 1.0)

        # ── Donchian channels (20-period) ─────────────────────────────────────
        donchian_high        = df["high"].rolling(20).max()
        donchian_low         = df["low"].rolling(20).min()
        donchian_range       = donchian_high - donchian_low
        df["donchian_pct"]   = ((close - donchian_low) /
                                (donchian_range + 1e-10)).clip(0.0, 1.0)
        df["donchian_width"] = (donchian_range /
                                (close + 1e-10)).clip(0.0, 0.5)

        # ── ATR regime ratio (current ATR vs 30-day average) ─────────────────
        atr_30d_mean      = df["atr_14"].rolling(720).mean()
        df["atr_ratio"]   = (df["atr_14"] /
                             (atr_30d_mean + 1e-10)).clip(0.3, 3.0)

        return df

    @staticmethod
    def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs    = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift(1)).abs()
        lc  = (df["low"]  - df["close"].shift(1)).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def process_asset(self, symbol: str,
                      start_date: datetime, end_date: datetime) -> pd.DataFrame:
        print(f"\n{'='*70}")
        print(f"  {symbol}  ({start_date.date()} → {end_date.date()})")
        print(f"{'='*70}")

        # 1. Fetch all data sources
        futures = self.fetch_futures_ohlcv(symbol, start_date, end_date)
        if futures.empty:
            print(f"  [SKIP] No futures data for {symbol}")
            return pd.DataFrame()

        funding = self.fetch_funding_rates(symbol, start_date, end_date)
        spot    = self.fetch_spot_ohlcv(symbol, start_date, end_date)

        # 2. Merge funding rates
        df = self.merge_funding(futures, funding)

        # 3. Merge basis
        df = self.merge_basis(df, spot)

        # 4. Compute all indicators
        df = self.compute_indicators(df)

        # 5. Drop the leading NaN rows (sma_500 is the largest window at 500 bars)
        df = df.dropna(subset=["sma_500", "rsi_14", "atr_14"])
        df = df.ffill().fillna(0.0)
        df = df.reset_index(drop=True)

        # 6. Select and order final columns
        final_cols = [
            "open_time",
            "open", "high", "low", "close",
            "volume", "quote_volume", "taker_buy_volume",
            "taker_buy_quote_volume", "volume_ratio_taker",
            "last_funding_rate", "funding_interval_hours",
            "funding_cost_annualized",
            "sma_20", "rsi_14", "atr_14", "volatility_20",
            "log_returns", "momentum_pct", "price_range_pct",
            "basis", "basis_pct",
            "sma_200", "ema_20", "price_vs_sma200", "price_vs_sma20",
            "bb_width", "bb_pct", "macd_hist",
            "volume_zscore", "funding_rate_zscore",
            "sma_500", "price_vs_sma500",
            "donchian_pct", "donchian_width",
            "atr_ratio",
        ]
        df = df[[c for c in final_cols if c in df.columns]]

        print(f"\n  ✓ {symbol}: {len(df):,} rows × {len(df.columns)} columns")
        return df

    def save(self, df: pd.DataFrame, symbol: str,
             start_date: datetime, end_date: datetime) -> str:
        fname = (f"{symbol}_perpetual_futures_processed_"
                 f"{start_date.year}-{end_date.year}.csv")
        path  = os.path.join(self.output_dir, fname)
        df.to_csv(path, index=False)
        size  = os.path.getsize(path) / (1024 * 1024)
        print(f"  Saved → {path}  ({size:.1f} MB)")
        return path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Configure here ────────────────────────────────────────────────────────
    START = datetime(2020, 1, 1)
    END   = datetime(2026, 4, 8)

    # Comment out assets you don't need
    ASSETS = [
        "SOLUSDT",
    ]
    # ─────────────────────────────────────────────────────────────────────────

    pipeline = PerpetualFuturesDataPipeline(output_dir=".")

    for symbol in ASSETS:
        df = pipeline.process_asset(symbol, START, END)
        if not df.empty:
            pipeline.save(df, symbol, START, END)

    print("\nAll assets processed.")
