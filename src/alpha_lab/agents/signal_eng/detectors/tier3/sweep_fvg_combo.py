"""
Category 17: Sweep + FVG Combo Detector.

High conviction signal: liquidity sweep followed by FVG formation.
Combines sweep detection with Fair Value Gap confirmation.

Signal composition:
- direction: +1 (bullish sweep + bullish FVG), -1 (bearish combo), 0 (none)
- strength: combines sweep depth, FVG size, and temporal proximity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.detectors.tier2._fvg_helpers import detect_fvgs
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class SweepFVGComboDetector(SignalDetector):
    """Sweep + FVG Combo: sweep followed by FVG, high conviction."""

    detector_id = "sweep_fvg_combo"
    category = "sweep_fvg_combo"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]]

    def __init__(
        self,
        sweep_lookback: int = 20,
        combo_window: int = 5,
        min_gap_atr: float = 0.3,
        min_sweep_atr: float = 0.2,
    ) -> None:
        self.sweep_lookback = sweep_lookback
        self.combo_window = combo_window
        self.min_gap_atr = min_gap_atr
        self.min_sweep_atr = min_sweep_atr

    def validate_inputs(self, data: DataBundle) -> bool:
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > _MIN_BARS:
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
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
        high = df["high"]
        low = df["low"]

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # Rolling highs and lows for sweep detection
        rolling_hi = high.rolling(window=self.sweep_lookback, min_periods=1).max().shift(1)
        rolling_lo = low.rolling(window=self.sweep_lookback, min_periods=1).min().shift(1)

        close_vals = close.values
        high_vals = high.values
        low_vals = low.values
        atr_vals = atr_safe.values
        rhi_vals = rolling_hi.values
        rlo_vals = rolling_lo.values

        # Sweep events: (bar_idx, type, depth)
        sweeps: list[tuple[int, str, float]] = []
        for i in range(self.sweep_lookback, len(df)):
            a = atr_vals[i]
            if np.isnan(a) or a <= 0 or np.isnan(rhi_vals[i]) or np.isnan(rlo_vals[i]):
                continue

            # Bullish sweep: wick below rolling low, close back above
            if low_vals[i] < rlo_vals[i] and close_vals[i] > rlo_vals[i]:
                depth = (rlo_vals[i] - low_vals[i]) / a
                if depth >= self.min_sweep_atr:
                    sweeps.append((i, "bullish", depth))

            # Bearish sweep: wick above rolling high, close back below
            if high_vals[i] > rhi_vals[i] and close_vals[i] < rhi_vals[i]:
                depth = (high_vals[i] - rhi_vals[i]) / a
                if depth >= self.min_sweep_atr:
                    sweeps.append((i, "bearish", depth))

        # Detect FVGs
        fvgs = detect_fvgs(df, min_gap_atr=self.min_gap_atr)
        fvg_lookup: dict[int, list[dict]] = {}
        for fvg in fvgs:
            idx = fvg["idx"]
            fvg_lookup.setdefault(idx, []).append(fvg)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        # Match sweeps with FVGs
        for sweep_bar, sweep_type, sweep_depth in sweeps:
            for fvg_bar in range(sweep_bar, min(sweep_bar + self.combo_window + 1, len(df))):
                if fvg_bar not in fvg_lookup:
                    continue
                for fvg in fvg_lookup[fvg_bar]:
                    # Match direction
                    if sweep_type == "bullish" and fvg["type"] == "bullish":
                        d = 1
                    elif sweep_type == "bearish" and fvg["type"] == "bearish":
                        d = -1
                    else:
                        continue

                    # Signal fires at the FVG bar (point-in-time)
                    signal_bar = fvg_bar + 1
                    if signal_bar >= len(df):
                        continue

                    depth_score = min(sweep_depth / 2.0, 1.0)
                    fvg_score = min(fvg["size_atr"] / 2.0, 1.0)
                    proximity = max(1.0 - (fvg_bar - sweep_bar) / max(self.combo_window, 1), 0.0)

                    s = 0.35 * depth_score + 0.35 * fvg_score + 0.30 * proximity

                    if s > strength.iloc[signal_bar]:
                        direction.iloc[signal_bar] = d
                        strength.iloc[signal_bar] = round(min(s, 1.0), 6)
                        formation_idx.iloc[signal_bar] = signal_bar

        # Forward-fill
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        dir_filled = direction.replace(0, np.nan).ffill().fillna(0).astype(int)
        str_filled = strength.replace(0.0, np.nan).ffill().fillna(0.0)
        has_signal = formation_idx > 0
        direction = dir_filled.where(has_signal, 0)
        strength = str_filled.where(has_signal, 0.0).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_SWEEP_FVG_COMBO_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "sweep_lookback": self.sweep_lookback,
                "combo_window": self.combo_window,
                "min_gap_atr": self.min_gap_atr,
                "min_sweep_atr": self.min_sweep_atr,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Liquidity sweep confirmed by subsequent FVG formation",
                "bars_processed": len(df),
            },
        )
