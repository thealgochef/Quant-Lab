"""Shared test fixtures for the Alpha Signal Research Lab."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alpha_lab.core.contracts import (
    DataBundle,
    PreviousDayLevels,
    QualityReport,
    SessionMetadata,
    SignalBundle,
    SignalVector,
)
from alpha_lab.core.enums import AgentID, MessageType, Priority
from alpha_lab.core.message import MessageBus, MessageEnvelope


@pytest.fixture
def message_bus() -> MessageBus:
    """Fresh MessageBus instance."""
    return MessageBus()


@pytest.fixture
def sample_quality_report() -> QualityReport:
    """A passing quality report."""
    return QualityReport(
        passed=True,
        total_bars=5000,
        gaps_found=0,
        gaps_detail=[],
        volume_zeros=0,
        ohlc_violations=0,
        cross_tf_mismatches=0,
        timestamp_coverage=1.0,
        report_generated_at=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def sample_session_metadata() -> SessionMetadata:
    """A sample RTH session."""
    return SessionMetadata(
        session_id="NQ_2026-02-21_RTH",
        session_type="RTH",
        killzone="NEW_YORK",
        rth_open="2026-02-21T09:30:00-05:00",
        rth_close="2026-02-21T16:15:00-05:00",
    )


@pytest.fixture
def sample_pd_levels() -> PreviousDayLevels:
    """Sample previous day levels for NQ."""
    return PreviousDayLevels(
        pd_high=22150.0,
        pd_low=21980.0,
        pd_mid=22065.0,
        pd_close=22100.0,
        pw_high=22200.0,
        pw_low=21850.0,
        overnight_high=22130.0,
        overnight_low=22050.0,
    )


@pytest.fixture
def sample_data_bundle(
    sample_quality_report: QualityReport,
    sample_session_metadata: SessionMetadata,
    sample_pd_levels: PreviousDayLevels,
) -> DataBundle:
    """A minimal DataBundle for testing."""
    return DataBundle(
        instrument="NQ",
        bars={"5m": {"open": [100], "high": [101], "low": [99], "close": [100.5]}},
        sessions=[sample_session_metadata],
        pd_levels={"2026-02-21": sample_pd_levels},
        quality=sample_quality_report,
        date_range=("2026-02-21", "2026-02-21"),
    )


@pytest.fixture
def sample_signal_vector() -> SignalVector:
    """A single valid signal vector."""
    return SignalVector(
        signal_id="SIG_EMA_CONFLUENCE_5m_v1",
        category="ema_confluence",
        timeframe="5m",
        version=1,
        direction=[1, 0, -1, 1, 0],
        strength=[0.8, 0.0, 0.6, 0.9, 0.0],
        formation_idx=[0, 1, 2, 3, 4],
        parameters={"ema_fast": 13, "ema_mid": 48, "ema_slow": 200},
        metadata={"intuition": "EMA alignment with expanding spread"},
    )


@pytest.fixture
def sample_signal_bundle(sample_signal_vector: SignalVector) -> SignalBundle:
    """A minimal SignalBundle for testing."""
    return SignalBundle(
        instrument="NQ",
        signals=[sample_signal_vector],
        composite_scores={},
        timeframes_covered=["5m"],
        total_signals=1,
        generation_timestamp=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def sample_envelope() -> MessageEnvelope:
    """A sample message envelope."""
    return MessageEnvelope(
        request_id="test-req-001",
        sender=AgentID.ORCHESTRATOR,
        receiver=AgentID.DATA_INFRA,
        message_type=MessageType.DATA_REQUEST,
        priority=Priority.NORMAL,
        payload={"instrument": "NQ", "date_range": ("2026-02-01", "2026-02-21")},
    )
