"""
ML training dataset builder.

Builds point-in-time-correct feature matrices from stored tick and bar
data via TickStore.  Features at time T use only data from [0, T].
Forward returns serve as labels and are clearly separated.

Usage::

    store = TickStore(Path("data/databento"))
    builder = MLDatasetBuilder(store)
    df = builder.build_features("NQ", start, end, bar_tf="5m")
    builder.export_dataset("NQ", start, end, "5m", Path("datasets/nq.parquet"))
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.tick_store import TickStore

logger = logging.getLogger(__name__)


class MLDatasetBuilder:
    """Builds ML training datasets from stored tick data."""

    def __init__(self, tick_store: TickStore) -> None:
        self._store = tick_store

    # ── Main API ──────────────────────────────────────────────────

    def build_features(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_tf: str = "5 minutes",
        ob_lookback: int = 100,
    ) -> pd.DataFrame:
        """Build a feature matrix with point-in-time correctness.

        Aggregates ticks into bars of ``bar_tf`` size, then computes
        bar-level and orderbook features.  Forward returns are appended
        as label columns (prefixed ``fwd_ret_``).

        Args:
            symbol: Instrument symbol
            start: Start of training window
            end: End of training window
            bar_tf: DuckDB interval string for bar size
            ob_lookback: Number of recent ticks for orderbook features

        Returns:
            DataFrame with features + forward-return labels
        """
        # Build bars from ticks
        bars = self._store.build_bars_from_ticks(symbol, start, end, bar_tf)
        if bars.empty:
            return pd.DataFrame()

        # Bar-level features
        bar_feats = self.compute_bar_features(bars)

        # Orderbook features per bar boundary (point-in-time)
        ob_rows: list[dict] = []
        for ts in bars.index:
            ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            lookback_start = ts_dt - timedelta(seconds=60)
            ticks = self._store.query_ticks(symbol, lookback_start, ts_dt)
            if ticks.empty:
                ob_rows.append({})
            else:
                tail = ticks.tail(ob_lookback)
                ob_rows.append(self.compute_orderbook_features(tail))
        ob_df = pd.DataFrame(ob_rows, index=bars.index)

        # Forward returns as labels
        fwd = self._compute_forward_returns(bars, horizons=[1, 5, 20])

        # Combine: bars OHLCV + derived features + orderbook + labels
        result = pd.concat([bars, bar_feats, ob_df, fwd], axis=1)
        # Drop warmup rows where core features are NaN
        if "ret_1" in result.columns:
            result = result.dropna(subset=["ret_1"])
        return result

    def export_dataset(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_tf: str = "5 minutes",
        output_path: Path | str = Path("datasets/features.parquet"),
        train_pct: float = 0.7,
        val_pct: float = 0.15,
    ) -> dict[str, int]:
        """Build features and split into train/val/test Parquet files.

        Args:
            symbol: Instrument symbol
            start: Window start
            end: Window end
            bar_tf: Bar size for aggregation
            output_path: Base path for output files
            train_pct: Training set fraction
            val_pct: Validation set fraction (remainder = test)

        Returns:
            Dict with row counts: {"train": N, "val": N, "test": N}
        """
        df = self.build_features(symbol, start, end, bar_tf)
        if df.empty:
            return {"train": 0, "val": 0, "test": 0}

        n = len(df)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        splits = {
            "train": df.iloc[:train_end],
            "val": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:],
        }

        counts = {}
        for name, split_df in splits.items():
            split_path = output.parent / f"{output.stem}_{name}.parquet"
            split_df.to_parquet(split_path)
            counts[name] = len(split_df)
            logger.info("Wrote %d rows to %s", len(split_df), split_path)

        return counts

    # ── Feature computation ───────────────────────────────────────

    @staticmethod
    def compute_orderbook_features(ticks_df: pd.DataFrame) -> dict:
        """Compute orderbook features from MBP-10 tick data.

        Returns a dict of feature name -> value for a single time point.
        """
        features: dict = {}

        if ticks_df.empty:
            return features

        # Level-0 bid/ask (top of book)
        bid_px_col = "bid_px_00"
        ask_px_col = "ask_px_00"
        bid_sz_col = "bid_sz_00"
        ask_sz_col = "ask_sz_00"

        has_book = all(
            c in ticks_df.columns
            for c in [bid_px_col, ask_px_col, bid_sz_col, ask_sz_col]
        )

        if has_book:
            last = ticks_df.iloc[-1]
            bid_px = float(last[bid_px_col])
            ask_px = float(last[ask_px_col])
            bid_sz = float(last[bid_sz_col])
            ask_sz = float(last[ask_sz_col])

            # Bid-ask spread
            features["spread"] = ask_px - bid_px
            if ask_px > 0:
                features["spread_bps"] = (ask_px - bid_px) / ask_px * 10000

            # Microprice
            total_sz = bid_sz + ask_sz
            if total_sz > 0:
                features["microprice"] = (
                    bid_sz * ask_px + ask_sz * bid_px
                ) / total_sz

            # Depth imbalance across all available levels
            total_bid_sz = 0.0
            total_ask_sz = 0.0
            for i in range(10):
                bc = f"bid_sz_{i:02d}"
                ac = f"ask_sz_{i:02d}"
                if bc in last.index and ac in last.index:
                    b = float(last[bc])
                    a = float(last[ac])
                    if b > 0 or a > 0:
                        total_bid_sz += b
                        total_ask_sz += a

            depth_total = total_bid_sz + total_ask_sz
            if depth_total > 0:
                features["depth_imbalance"] = (
                    (total_bid_sz - total_ask_sz) / depth_total
                )

            # Book pressure: volume-weighted mid across levels
            weighted_sum = 0.0
            weight_total = 0.0
            for i in range(10):
                bpc = f"bid_px_{i:02d}"
                apc = f"ask_px_{i:02d}"
                bsc = f"bid_sz_{i:02d}"
                asc = f"ask_sz_{i:02d}"
                if all(c in last.index for c in [bpc, apc, bsc, asc]):
                    bp = float(last[bpc])
                    ap = float(last[apc])
                    bs = float(last[bsc])
                    asz = float(last[asc])
                    if bp > 0 and ap > 0:
                        mid = (bp + ap) / 2
                        w = bs + asz
                        weighted_sum += mid * w
                        weight_total += w
            if weight_total > 0:
                features["book_pressure"] = weighted_sum / weight_total

        # Trade flow: signed volume from price and size columns
        if "price" in ticks_df.columns and "size" in ticks_df.columns:
            prices = ticks_df["price"]
            sizes = ticks_df["size"]
            price_diff = prices.diff()
            signed_vol = sizes * np.sign(price_diff).fillna(0)
            features["trade_flow"] = float(signed_vol.sum())
            total_vol = float(sizes.sum())
            if total_vol > 0:
                features["trade_flow_ratio"] = (
                    float(signed_vol.sum()) / total_vol
                )

        return features

    @staticmethod
    def compute_bar_features(bars: pd.DataFrame) -> pd.DataFrame:
        """Compute standard OHLCV bar features.

        All features use only past data (lookback windows, no future).
        """
        if bars.empty:
            return pd.DataFrame()

        feats = pd.DataFrame(index=bars.index)

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        volume = bars["volume"]

        # Returns at multiple horizons
        for h in [1, 3, 5, 10]:
            feats[f"ret_{h}"] = close.pct_change(h)

        # Volatility (rolling std of returns)
        ret_1 = close.pct_change(1)
        for w in [10, 20, 50]:
            feats[f"vol_{w}"] = ret_1.rolling(w).std()

        # Volume z-score
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std().replace(0, np.nan)
        feats["volume_zscore"] = (volume - vol_mean) / vol_std

        # Bar range / ATR proxy
        bar_range = high - low
        atr_20 = bar_range.rolling(20).mean().replace(0, np.nan)
        feats["range_atr_ratio"] = bar_range / atr_20

        # Body ratio (close-open vs high-low)
        body = (close - bars["open"]).abs()
        feats["body_ratio"] = body / bar_range.replace(0, np.nan)

        # High-low position (where close sits in the bar)
        feats["hl_position"] = (close - low) / bar_range.replace(0, np.nan)

        return feats

    # ── Labels ────────────────────────────────────────────────────

    @staticmethod
    def _compute_forward_returns(
        bars: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute forward returns as supervised learning labels.

        These are explicitly future-looking and must only be used
        as labels, never as features.
        """
        if horizons is None:
            horizons = [1, 5, 20]

        close = bars["close"]
        fwd = pd.DataFrame(index=bars.index)
        for h in horizons:
            fwd[f"fwd_ret_{h}"] = close.shift(-h) / close - 1
        return fwd
