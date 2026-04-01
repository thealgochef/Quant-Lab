"""Mode-specific dataset builder for dashboard utility (+15/-30) training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.dashboard_utility_labeling import label_event_15_30
from alpha_lab.agents.data_infra.ml.extrema_detection import Extremum
from alpha_lab.agents.data_infra.ml.features_microstructure import extract_pl_features
from alpha_lab.agents.data_infra.ml.features_momentum import extract_ms_features
from alpha_lab.agents.data_infra.tick_store import TickStore
from alpha_lab.experiment.event_detection import detect_all_events
from alpha_lab.experiment.key_levels import compute_key_levels


@dataclass
class DashboardUtilityConfig:
    tp_ticks: int = 15
    sl_ticks: int = 30
    lookback_minutes: int = 30
    forward_minutes: int = 360
    tick_size: float = 0.25


class DashboardUtilityDatasetBuilder:
    """Build touch-anchored utility dataset using MBP-capable tick rows."""

    def __init__(
        self,
        data_dir: Path,
        config: DashboardUtilityConfig | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config or DashboardUtilityConfig()

    def build_for_dates(self, symbol: str, dates: list[str]) -> pd.DataFrame:
        if not dates:
            return pd.DataFrame()

        levels_df = compute_key_levels(self.data_dir, symbol=symbol)
        if levels_df.empty:
            return pd.DataFrame()
        levels_df = levels_df[levels_df["date"].isin(dates)]
        if levels_df.empty:
            return pd.DataFrame()

        tmp_levels = self.data_dir.parent / "experiment" / "_tmp_dashboard_levels.parquet"
        tmp_levels.parent.mkdir(parents=True, exist_ok=True)
        levels_df.to_parquet(tmp_levels, index=False)

        events = detect_all_events(
            levels_path=tmp_levels,
            data_dir=self.data_dir,
            output_path=None,
        )
        if events.empty:
            return pd.DataFrame()
        events = events[events["date"].isin(dates)].reset_index(drop=True)

        store = TickStore(self.data_dir)
        try:
            for d in dates:
                store.register_symbol_date(symbol, d)
            rows: list[dict[str, object]] = []
            for _, ev in events.iterrows():
                bar_touch_ts = pd.Timestamp(ev["event_ts"])
                refined_touch_ts = self._refine_intra_bar_touch_timestamp(
                    store=store,
                    symbol=symbol,
                    bar_touch_ts=bar_touch_ts,
                    representative_price=float(ev["representative_price"]),
                )
                event_ts = refined_touch_ts if refined_touch_ts is not None else bar_touch_ts

                start = (event_ts - timedelta(minutes=self.config.lookback_minutes)).to_pydatetime()
                end = (event_ts + timedelta(minutes=self.config.forward_minutes)).to_pydatetime()
                ticks = store.query_tick_feature_rows(symbol, start, end)
                if ticks.empty or "ts_event" not in ticks.columns:
                    continue

                ticks["ts_event"] = pd.to_datetime(ticks["ts_event"], utc=True)
                pre = ticks[ticks["ts_event"] <= event_ts].reset_index(drop=True)
                post = ticks[ticks["ts_event"] > event_ts].reset_index(drop=True)
                if len(pre) < 20:
                    continue

                ext_idx = len(pre) - 1
                ext = Extremum(
                    index=ext_idx,
                    timestamp=event_ts,
                    price=float(pre.iloc[ext_idx]["price"]),
                    extremum_type="trough" if str(ev["direction"]).upper() == "LONG" else "peak",
                    prominence=0.0,
                    width=0.0,
                )

                pl_slice = pre.iloc[max(0, ext_idx - 100): ext_idx + 1]
                pl = extract_pl_features(
                    ext,
                    pl_slice,
                    tick_size=self.config.tick_size,
                )
                ms = extract_ms_features(
                    ext,
                    pre["price"].values.astype(np.float64),
                    pre["size"].values.astype(np.float64) if "size" in pre.columns else None,
                )
                lbl = label_event_15_30(
                    direction=str(ev["direction"]),
                    event_ts=event_ts,
                    forward_ticks=post,
                    tick_size=self.config.tick_size,
                    tp_ticks=self.config.tp_ticks,
                    sl_ticks=self.config.sl_ticks,
                )
                if lbl["label_15_30"] is None:
                    continue

                row = {
                    "timestamp": event_ts,
                    "tick_index": ext_idx,
                    "price": float(pre.iloc[ext_idx]["price"]),
                    "extremum_type": ext.extremum_type,
                    "symbol": symbol,
                    "event_direction": str(ev["direction"]),
                    "direction": str(ev["direction"]),
                    "approach_direction": str(ev.get("approach_direction", "")),
                    "event_anchor": (
                        "experiment_first_touch_zone_intersection_1m_bar_refined_mbp_timestamp"
                    ),
                    "event_ts_1m_bar": bar_touch_ts,
                    "event_ts_refined": event_ts,
                    "zone_id": str(ev.get("zone_id", "")),
                    "level_names": str(ev.get("level_names", "")),
                    "representative_level": float(ev["representative_price"]),
                    "session": self._classify_session(event_ts),
                    "label_15_30": int(lbl["label_15_30"]),
                    "outcome_type": lbl["outcome_type"],
                    "time_to_tp": lbl["time_to_tp"],
                    "time_to_sl": lbl["time_to_sl"],
                    "mfe": lbl["mfe"],
                    "mae": lbl["mae"],
                }
                row.update(pl)
                row.update(ms)
                rows.append(row)
        finally:
            store.close()

        return pd.DataFrame(rows)

    @staticmethod
    def _classify_session(ts: pd.Timestamp) -> str:
        """Classify timestamp into NY RTH vs ETH for downstream filtering."""
        ts_et = pd.Timestamp(ts).tz_convert("US/Eastern")
        is_rth = (
            (ts_et.hour > 9 or (ts_et.hour == 9 and ts_et.minute >= 30))
            and ts_et.hour < 16
        )
        return "ny_rth" if is_rth else "eth"

    @staticmethod
    def _refine_intra_bar_touch_timestamp(
        *,
        store: TickStore,
        symbol: str,
        bar_touch_ts: pd.Timestamp,
        representative_price: float,
    ) -> pd.Timestamp | None:
        """Resolve first intra-bar MBP touch for representative level."""
        bar_ts = pd.Timestamp(bar_touch_ts)
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("US/Eastern")
        bar_start_utc = bar_ts.tz_convert("UTC")
        bar_end_utc = bar_start_utc + timedelta(minutes=1)

        bar_ticks = store.query_tick_feature_rows(
            symbol, bar_start_utc.to_pydatetime(), bar_end_utc.to_pydatetime(),
        )
        if bar_ticks.empty or "ts_event" not in bar_ticks.columns:
            return None

        bar_ticks["ts_event"] = pd.to_datetime(bar_ticks["ts_event"], utc=True)
        if "bid_px_00" in bar_ticks.columns and "ask_px_00" in bar_ticks.columns:
            quote_touch = (
                (bar_ticks["bid_px_00"] <= representative_price)
                & (bar_ticks["ask_px_00"] >= representative_price)
            )
            touched = bar_ticks[quote_touch]
            if not touched.empty:
                return pd.Timestamp(touched.iloc[0]["ts_event"])

        if "price" in bar_ticks.columns:
            px = bar_ticks["price"].astype(float)
            crossed = (
                ((px.shift(1) < representative_price) & (px >= representative_price))
                | ((px.shift(1) > representative_price) & (px <= representative_price))
                | (px == representative_price)
            )
            touched = bar_ticks[crossed.fillna(False)]
            if not touched.empty:
                return pd.Timestamp(touched.iloc[0]["ts_event"])

        return None
