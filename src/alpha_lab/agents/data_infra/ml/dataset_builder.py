"""
Orchestrator: builds labeled feature matrices from tick data.

Combines extrema detection, labeling, and all three feature extractors
(PL microstructure, MS momentum, signal detectors) into a single
DataFrame suitable for ML training.

Usage::

    store = TickStore(Path("data/databento"))
    builder = ExtremaDatasetBuilder(store, config)
    df = builder.build_dataset("NQ", start, end)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from alpha_lab.agents.data_infra.ml.config import MLPipelineConfig
from alpha_lab.agents.data_infra.ml.extrema_detection import detect_extrema
from alpha_lab.agents.data_infra.ml.features_microstructure import extract_pl_features
from alpha_lab.agents.data_infra.ml.features_momentum import extract_ms_features
from alpha_lab.agents.data_infra.ml.features_signals import (
    extract_signal_features_batch,
)
from alpha_lab.agents.data_infra.ml.labeling import (
    build_label_dataframe,
    label_extrema,
)
from alpha_lab.agents.data_infra.tick_store import TickStore
from alpha_lab.core.contracts import SignalBundle

logger = logging.getLogger(__name__)


class ExtremaDatasetBuilder:
    """Builds ML training datasets from tick data via TickStore.

    Pipeline: query_ticks → detect_extrema → label → extract features → DataFrame
    """

    def __init__(
        self,
        tick_store: TickStore,
        config: MLPipelineConfig | None = None,
        signal_bundle: SignalBundle | None = None,
    ) -> None:
        self._store = tick_store
        self._config = config or MLPipelineConfig()
        self._signal_bundle = signal_bundle

    @property
    def config(self) -> MLPipelineConfig:
        return self._config

    def build_dataset(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Build a complete feature matrix from tick data.

        Queries ticks in one shot. For large date ranges, prefer
        ``build_dataset_daily`` to manage memory.

        Args:
            symbol: Instrument symbol (e.g. "NQ").
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            DataFrame with one row per extremum, feature columns + labels.
        """
        ticks = self._store.query_tick_feature_rows(symbol, start, end)
        if ticks.empty:
            logger.warning("No ticks found for %s [%s, %s]", symbol, start, end)
            return pd.DataFrame()

        return self._process_ticks(ticks, symbol)

    def build_dataset_daily(
        self,
        symbol: str,
        dates: list[str],
    ) -> pd.DataFrame:
        """Build dataset one date at a time to manage memory.

        Args:
            symbol: Instrument symbol.
            dates: List of date strings (e.g. ["2026-02-20", "2026-02-21"]).

        Returns:
            Combined DataFrame across all dates.
        """
        frames: list[pd.DataFrame] = []

        for date_str in dates:
            start = datetime.fromisoformat(f"{date_str}T00:00:00")
            end = start + timedelta(days=1) - timedelta(microseconds=1)

            ticks = self._store.query_tick_feature_rows(symbol, start, end)
            if ticks.empty:
                logger.debug("No ticks for %s on %s", symbol, date_str)
                continue

            df = self._process_ticks(ticks, symbol)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def export_dataset(
        self, df: pd.DataFrame, output_path: str | Path,
    ) -> None:
        """Write the feature matrix to Parquet.

        Args:
            df: Feature DataFrame from build_dataset.
            output_path: Path for the output Parquet file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        logger.info("Exported %d rows to %s", len(df), output)

    def _process_ticks(
        self, ticks: pd.DataFrame, symbol: str,
    ) -> pd.DataFrame:
        """Core pipeline: ticks → extrema → labels → features → DataFrame."""
        cfg = self._config

        # Extract price and timestamp series
        if "price" not in ticks.columns or "ts_event" not in ticks.columns:
            logger.warning("Ticks missing 'price' or 'ts_event' columns")
            return pd.DataFrame()

        tick_prices = ticks["price"].reset_index(drop=True)
        tick_timestamps = ticks["ts_event"].reset_index(drop=True)
        tick_volumes = (
            ticks["size"].reset_index(drop=True)
            if "size" in ticks.columns else None
        )

        # 1. Detect extrema
        extrema = detect_extrema(
            tick_prices, tick_timestamps,
            cfg.extrema, cfg.tick_size,
        )
        if not extrema:
            logger.debug("No extrema detected for %s", symbol)
            return pd.DataFrame()

        # 2. Label extrema
        labeled = label_extrema(extrema, tick_prices, cfg.labeling, cfg.tick_size)
        label_df = build_label_dataframe(labeled)
        if label_df.empty:
            return pd.DataFrame()

        # 3. Extract PL features
        pl_rows: list[dict[str, float]] = []
        ob_lookback = 100
        for ext in extrema:
            start_idx = max(0, ext.index - ob_lookback)
            ticks_slice = ticks.iloc[start_idx: ext.index + 1]
            pl_feats = extract_pl_features(
                ext, ticks_slice, cfg.features, cfg.tick_size,
            )
            pl_rows.append(pl_feats)

        # 4. Extract MS features (pre-convert to numpy for faster slicing)
        import numpy as _np
        prices_np = tick_prices.values.astype(_np.float64)
        volumes_np = (
            tick_volumes.values.astype(_np.float64) if tick_volumes is not None
            else None
        )
        ms_rows: list[dict[str, float]] = []
        for ext in extrema:
            ms_feats = extract_ms_features(
                ext, prices_np, volumes_np, cfg.features,
            )
            ms_rows.append(ms_feats)

        # 5. Extract signal features (hybrid)
        if cfg.features.include_signal_features and self._signal_bundle:
            sig_rows = extract_signal_features_batch(
                extrema, self._signal_bundle, cfg.features,
            )
        else:
            sig_rows = [{} for _ in extrema]

        # 6. Combine all features
        pl_df = pd.DataFrame(pl_rows)
        ms_df = pd.DataFrame(ms_rows)
        sig_df = pd.DataFrame(sig_rows)

        result = pd.concat(
            [label_df.reset_index(drop=True),
             pl_df.reset_index(drop=True),
             ms_df.reset_index(drop=True),
             sig_df.reset_index(drop=True)],
            axis=1,
        )

        # Add metadata columns
        result["symbol"] = symbol

        return result
