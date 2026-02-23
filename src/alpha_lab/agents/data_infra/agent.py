"""
Data Infrastructure Agent (DATA-001) — Market Data Engineer.

Owns the entire data pipeline from raw exchange tick feeds through
clean, session-tagged OHLCV bars at all timeframes.

See docs/agent_prompts/DATA-001.md for full system prompt.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd

from alpha_lab.agents.data_infra.aggregation import (
    aggregate_tick_bars,
    aggregate_time_bars,
)
from alpha_lab.agents.data_infra.providers import create_provider
from alpha_lab.agents.data_infra.quality import run_quality_checks
from alpha_lab.agents.data_infra.sessions import tag_killzones, tag_sessions
from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import (
    DataBundle,
    PreviousDayLevels,
    SessionMetadata,
)
from alpha_lab.core.enums import AgentID, AgentState, MessageType, Timeframe
from alpha_lab.core.message import MessageBus, MessageEnvelope


class DataInfraAgent(BaseAgent):
    """DATA-001: Market Data Engineer.

    Responsibilities:
    - Time bar construction (1m through 1D via Polygon)
    - Session tagging (RTH, GLOBEX, killzones)
    - Previous day/week level computation
    - Data quality validation
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(AgentID.DATA_INFRA, "Data Infrastructure", bus)

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle DATA_REQUEST messages from Orchestrator."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s (request_id=%s)",
            envelope.message_type.value,
            envelope.request_id,
        )

        if envelope.message_type == MessageType.DATA_REQUEST:
            self.send_ack(envelope)
            try:
                bundle = self.build_data_bundle(
                    instrument=envelope.payload["instrument"],
                    date_range=tuple(envelope.payload["date_range"]),
                    timeframes=envelope.payload.get(
                        "timeframes",
                        ["1m", "3m", "5m", "10m", "15m", "30m", "1H", "4H", "1D"],
                    ),
                )
                self.send_message(
                    receiver=AgentID.ORCHESTRATOR,
                    message_type=MessageType.DATA_BUNDLE,
                    payload={"bundle": bundle.model_dump()},
                    request_id=envelope.request_id,
                )
            except Exception:
                self.logger.exception("Failed to build DataBundle")
                self.send_nack(envelope, "DataBundle build failed")
        else:
            self.send_nack(
                envelope,
                f"Unexpected message type: {envelope.message_type.value}",
            )

        self.transition_state(AgentState.IDLE)

    def build_data_bundle(
        self,
        instrument: str,
        date_range: tuple[str, str],
        timeframes: list[str],
    ) -> DataBundle:
        """Build a complete DataBundle for the given parameters.

        1. Fetch 1m bars from Polygon
        2. Tag sessions and killzones
        3. Aggregate to all requested timeframes
        4. Compute previous day/week levels
        5. Run quality validation
        6. Return DataBundle with quality report
        """
        start = datetime.fromisoformat(date_range[0])
        end = datetime.fromisoformat(date_range[1])

        provider = self._create_provider()
        provider.connect()

        try:
            # Fetch 1m bars for the full date range
            bars_1m = provider.get_ohlcv(instrument, Timeframe.M1, start, end)

            # Tag sessions and killzones on 1m bars
            bars_1m = tag_sessions(bars_1m, instrument)
            bars_1m = tag_killzones(bars_1m)

            session_boundaries = {
                "rth_open": "09:30",
                "rth_close": "16:15",
                "globex_open": "18:00",
                "globex_close": "17:00",
            }

            # Aggregate to all requested timeframes
            bars_dict: dict[str, pd.DataFrame] = {}
            for tf_str in timeframes:
                tf = Timeframe(tf_str)
                if tf in (Timeframe.TICK_987, Timeframe.TICK_2000):
                    continue
                if tf == Timeframe.M1:
                    bars_dict[tf_str] = bars_1m
                else:
                    agg = aggregate_time_bars(bars_1m, tf, session_boundaries)
                    agg = tag_sessions(agg, instrument)
                    agg = tag_killzones(agg)
                    bars_dict[tf_str] = agg

            # Tick bar aggregation (if provider supports ticks)
            tick_tfs = [
                tf for tf in timeframes
                if tf in (Timeframe.TICK_987, Timeframe.TICK_2000)
            ]
            if tick_tfs:
                try:
                    ticks = provider.get_ticks(instrument, start, end)
                    if not ticks.empty:
                        for tf_str in tick_tfs:
                            tick_count = 987 if tf_str == Timeframe.TICK_987 else 2000
                            tick_bars = aggregate_tick_bars(ticks, tick_count)
                            if not tick_bars.empty:
                                bars_dict[tf_str] = tick_bars
                except NotImplementedError:
                    self.logger.debug(
                        "Provider %s does not support tick data",
                        provider.provider_name,
                    )

            # Compute previous day/week levels
            pd_levels = _compute_pd_levels(
                bars_dict.get("1D", pd.DataFrame()),
                bars_1m,
                instrument,
            )

            # Build session metadata
            sessions = _build_session_metadata(bars_1m)

            # Run quality checks
            quality = run_quality_checks(bars_dict, instrument)

            return DataBundle(
                request_id=str(uuid.uuid4()),
                instrument=instrument,
                bars=bars_dict,
                sessions=sessions,
                pd_levels=pd_levels,
                quality=quality,
                date_range=date_range,
                metadata={
                    "provider": provider.provider_name,
                    "timeframes": timeframes,
                },
            )
        finally:
            provider.disconnect()

    @staticmethod
    def _create_provider(provider_name: str = "polygon"):
        """Create a DataProvider by name using the factory."""
        return create_provider(provider_name)


# ── Helpers ───────────────────────────────────────────────────────


def _compute_pd_levels(
    daily_df: pd.DataFrame,
    bars_1m: pd.DataFrame,
    instrument: str,
) -> dict[str, PreviousDayLevels]:
    """Compute previous day/week levels for each trading date.

    For each date (except the first), returns:
    - pd_high/low/mid/close from previous RTH session
    - pw_high/low from previous 5 trading days
    - overnight_high/low from GLOBEX bars before today's RTH
    """
    levels: dict[str, PreviousDayLevels] = {}

    if bars_1m.empty or "session_id" not in bars_1m.columns:
        return levels

    # Extract unique RTH dates
    rth_ids = [s for s in bars_1m["session_id"].unique() if s.endswith("_RTH")]
    dates = sorted({sid.split("_")[1] for sid in rth_ids})

    for i, date_str in enumerate(dates):
        if i == 0:
            continue  # No previous day for first date

        prev_date = dates[i - 1]
        prev_rth = bars_1m[bars_1m["session_id"] == f"{instrument}_{prev_date}_RTH"]
        if prev_rth.empty:
            continue

        pd_high = float(prev_rth["high"].max())
        pd_low = float(prev_rth["low"].min())
        pd_close = float(prev_rth["close"].iloc[-1])
        pd_mid = (pd_high + pd_low) / 2

        # Previous week: look back up to 5 trading days
        week_start = max(0, i - 5)
        week_dates = dates[week_start:i]
        week_sids = [f"{instrument}_{d}_RTH" for d in week_dates]
        week_bars = bars_1m[bars_1m["session_id"].isin(week_sids)]
        pw_high = float(week_bars["high"].max()) if not week_bars.empty else pd_high
        pw_low = float(week_bars["low"].min()) if not week_bars.empty else pd_low

        # Overnight: GLOBEX bars for current date
        globex_sid = f"{instrument}_{date_str}_GLOBEX"
        overnight = bars_1m[bars_1m["session_id"] == globex_sid]
        on_high = float(overnight["high"].max()) if not overnight.empty else pd_high
        on_low = float(overnight["low"].min()) if not overnight.empty else pd_low

        levels[date_str] = PreviousDayLevels(
            pd_high=pd_high,
            pd_low=pd_low,
            pd_mid=pd_mid,
            pd_close=pd_close,
            pw_high=pw_high,
            pw_low=pw_low,
            overnight_high=on_high,
            overnight_low=on_low,
        )

    return levels


def _build_session_metadata(bars_1m: pd.DataFrame) -> list[SessionMetadata]:
    """Build SessionMetadata for each unique session in the data."""
    sessions: list[SessionMetadata] = []

    if "session_id" not in bars_1m.columns:
        return sessions

    for sid in sorted(bars_1m["session_id"].unique()):
        session_bars = bars_1m[bars_1m["session_id"] == sid]
        parts = sid.split("_")  # e.g. ["NQ", "2026-02-21", "RTH"]
        if len(parts) < 3:
            continue

        date_str = parts[1]
        stype = parts[2]
        kz = (
            session_bars["killzone"].iloc[0]
            if "killzone" in session_bars.columns
            else "NONE"
        )

        sessions.append(
            SessionMetadata(
                session_id=sid,
                session_type=stype,
                killzone=kz,
                rth_open=f"{date_str}T09:30:00-05:00",
                rth_close=f"{date_str}T16:15:00-05:00",
            )
        )

    return sessions
