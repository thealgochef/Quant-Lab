"""
Category 15: Volume Profile Detector.

Identifies Point of Control (POC), Value Area High/Low,
and volume nodes as support/resistance levels.

Signal composition:
- direction: +1 (near support / VAL), -1 (near resistance / VAH), 0 (none)
- strength: combines volume concentration, proximity to node, and price reaction
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 50


def _build_volume_histogram(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    num_bins: int,
) -> dict:
    """Build volume-at-price histogram for a window of bars.

    Returns dict with: bin_edges, bin_volumes, poc_price, poc_volume,
    vah, val, total_volume.
    """
    window = df.iloc[start_idx:end_idx]
    if len(window) == 0:
        return {"poc_price": np.nan, "vah": np.nan, "val": np.nan,
                "total_volume": 0.0, "poc_volume": 0.0}

    price_low = float(window["low"].min())
    price_high = float(window["high"].max())

    if price_high <= price_low:
        mid = price_low
        return {"poc_price": mid, "vah": mid, "val": mid,
                "total_volume": float(window["volume"].sum()),
                "poc_volume": float(window["volume"].sum())}

    bin_edges = np.linspace(price_low, price_high, num_bins + 1)
    bin_volumes = np.zeros(num_bins)

    # Distribute each bar's volume across bins its range covers
    for _, row in window.iterrows():
        bar_lo = row["low"]
        bar_hi = row["high"]
        vol = row["volume"]
        if vol <= 0 or bar_hi <= bar_lo:
            continue

        for b in range(num_bins):
            overlap_lo = max(bar_lo, bin_edges[b])
            overlap_hi = min(bar_hi, bin_edges[b + 1])
            if overlap_hi > overlap_lo:
                fraction = (overlap_hi - overlap_lo) / (bar_hi - bar_lo)
                bin_volumes[b] += vol * fraction

    total_volume = bin_volumes.sum()
    if total_volume <= 0:
        mid = (price_low + price_high) / 2
        return {"poc_price": mid, "vah": mid, "val": mid,
                "total_volume": 0.0, "poc_volume": 0.0}

    # POC: bin with highest volume
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0
    poc_volume = bin_volumes[poc_idx]

    # Value area: expand from POC until 70% of volume captured
    va_lo_idx = poc_idx
    va_hi_idx = poc_idx
    va_volume = bin_volumes[poc_idx]
    target = total_volume * 0.70

    while va_volume < target and (va_lo_idx > 0 or va_hi_idx < num_bins - 1):
        vol_below = bin_volumes[va_lo_idx - 1] if va_lo_idx > 0 else 0
        vol_above = bin_volumes[va_hi_idx + 1] if va_hi_idx < num_bins - 1 else 0

        if vol_below >= vol_above and va_lo_idx > 0:
            va_lo_idx -= 1
            va_volume += bin_volumes[va_lo_idx]
        elif va_hi_idx < num_bins - 1:
            va_hi_idx += 1
            va_volume += bin_volumes[va_hi_idx]
        elif va_lo_idx > 0:
            va_lo_idx -= 1
            va_volume += bin_volumes[va_lo_idx]
        else:
            break

    val_price = bin_edges[va_lo_idx]
    vah_price = bin_edges[va_hi_idx + 1]

    return {
        "poc_price": poc_price,
        "poc_volume": poc_volume,
        "vah": vah_price,
        "val": val_price,
        "total_volume": total_volume,
    }


class VolumeProfileDetector(SignalDetector):
    """Volume Profile: POC, value area, volume nodes as S/R."""

    detector_id = "volume_profile"
    category = "volume_profile"
    tier = SignalTier.COMPOSITE
    timeframes = [
        tf.value for tf in [
            Timeframe.M15, Timeframe.M30, Timeframe.H1,
            Timeframe.H4, Timeframe.D1,
        ]
    ]

    def __init__(
        self,
        lookback_bars: int = 100,
        num_bins: int = 50,
        value_area_pct: float = 0.70,
        proximity_atr: float = 0.5,
    ) -> None:
        self.lookback_bars = lookback_bars
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.proximity_atr = proximity_atr

    def validate_inputs(self, data: DataBundle) -> bool:
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if (
                isinstance(df, pd.DataFrame)
                and len(df) > _MIN_BARS
                and "volume" in df.columns
            ):
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            if "volume" not in df.columns:
                continue
            sv = self._compute_timeframe(df, tf, data.instrument)
            if sv is not None:
                signals.append(sv)
        return signals

    def _compute_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        instrument: str,
    ) -> SignalVector | None:
        close = df["close"]

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        close_vals = close.values
        atr_vals = atr_safe.values

        lookback = min(self.lookback_bars, len(df))

        for i in range(lookback, len(df)):
            a = atr_vals[i]
            if np.isnan(a) or a <= 0:
                continue

            start = max(0, i - lookback)
            profile = _build_volume_histogram(df, start, i, self.num_bins)

            poc = profile["poc_price"]
            vah = profile["vah"]
            val_ = profile["val"]
            total_vol = profile["total_volume"]
            poc_vol = profile["poc_volume"]

            if np.isnan(poc) or total_vol <= 0:
                continue

            c = close_vals[i]
            prox = self.proximity_atr * a

            # Determine direction based on proximity to levels
            near_val = abs(c - val_) <= prox
            near_vah = abs(c - vah) <= prox
            near_poc = abs(c - poc) <= prox

            d = 0
            nearest_level = np.nan
            if near_val and c >= val_:
                d = 1  # Support bounce
                nearest_level = val_
            elif near_vah and c <= vah:
                d = -1  # Resistance rejection
                nearest_level = vah
            elif near_poc:
                # POC: direction based on which side of POC
                d = 1 if c > poc else -1
                nearest_level = poc

            if d == 0:
                continue

            # Strength components
            vol_concentration = min(poc_vol / max(total_vol, 1.0) * 5.0, 1.0)
            dist_to_level = abs(c - nearest_level)
            proximity_score = max(1.0 - dist_to_level / prox, 0.0)

            # Reaction: is price moving away from level?
            if i > 0:
                prev_c = close_vals[i - 1]
                reaction = abs(c - nearest_level) - abs(prev_c - nearest_level)
                reaction_score = min(max(reaction / a, 0.0), 1.0)
            else:
                reaction_score = 0.0

            s = 0.35 * vol_concentration + 0.35 * proximity_score + 0.30 * reaction_score

            direction.iloc[i] = d
            strength.iloc[i] = round(min(s, 1.0), 6)
            formation_idx.iloc[i] = i

        # Forward-fill with exponential decay
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        dir_filled = direction.replace(0, np.nan).ffill().fillna(0).astype(int)
        str_filled = strength.replace(0.0, np.nan).ffill().fillna(0.0)
        has_signal = formation_idx > 0
        direction = dir_filled.where(has_signal, 0)
        strength = str_filled.where(has_signal, 0.0).clip(0.0, 1.0)

        # Decay strength: halve every 20 bars from formation
        bars_since = pd.Series(np.arange(len(df)), index=df.index) - formation_idx
        decay = np.power(0.5, bars_since.clip(lower=0) / 20.0)
        strength = (strength * decay).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_VOLUME_PROFILE_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "lookback_bars": self.lookback_bars,
                "num_bins": self.num_bins,
                "value_area_pct": self.value_area_pct,
                "proximity_atr": self.proximity_atr,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Volume-at-price S/R levels: POC, VAH, VAL",
                "bars_processed": len(df),
            },
        )
