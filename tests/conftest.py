"""Shared test fixtures for the Alpha Signal Research Lab."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.agent import DataInfraAgent
from alpha_lab.agents.execution.agent import ExecutionAgent
from alpha_lab.agents.monitoring.agent import MonitoringAgent
from alpha_lab.agents.orchestrator.agent import OrchestratorAgent
from alpha_lab.agents.signal_eng.agent import SignalEngineeringAgent
from alpha_lab.agents.validation.agent import ValidationAgent
from alpha_lab.core.config import InstrumentSpec, PropFirmProfile
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


# ─── Integration Test Fixtures ────────────────────────────────


def _generate_synthetic_bars(
    n_bars: int = 500,
    base_price: float = 22000.0,
    volatility: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic NQ-like 5m OHLCV bars for integration tests."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00005, volatility, n_bars)
    # Add regime shifts
    regime = np.zeros(n_bars)
    regime[100:150] = 0.0003
    regime[200:250] = -0.0005
    regime[350:400] = 0.0002
    returns += regime
    # Volatility bursts
    for burst_start in [80, 180, 300]:
        burst_end = min(burst_start + 30, n_bars)
        returns[burst_start:burst_end] *= 2.5

    close = base_price * np.cumprod(1 + returns)
    bar_range = np.abs(rng.normal(0, volatility * base_price * 0.5, n_bars))
    high = close + bar_range * 0.6
    low = close - bar_range * 0.4
    open_price = close + rng.normal(0, volatility * base_price * 0.2, n_bars)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    volume = rng.poisson(5000, n_bars).astype(float)

    start = datetime(2026, 1, 5, 9, 30)
    dates = []
    current = start
    for _ in range(n_bars):
        dates.append(current)
        current += timedelta(minutes=5)
        if current.hour >= 16 and current.minute >= 15:
            current = current.replace(hour=9, minute=30) + timedelta(days=1)
            while current.weekday() >= 5:
                current += timedelta(days=1)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "session_id": [f"NQ_{d.strftime('%Y-%m-%d')}_RTH" for d in dates],
        },
        index=pd.DatetimeIndex(dates),
    )


@pytest.fixture(scope="session")
def synthetic_bars_5m() -> pd.DataFrame:
    """500 bars of synthetic NQ 5m data (session-scoped for speed)."""
    return _generate_synthetic_bars(n_bars=500)


@pytest.fixture
def synthetic_data_bundle(synthetic_bars_5m: pd.DataFrame) -> DataBundle:
    """DataBundle wrapping synthetic 5m bars."""
    now = datetime.now(UTC)
    return DataBundle(
        instrument="NQ",
        bars={"5m": synthetic_bars_5m},
        sessions=[
            SessionMetadata(
                session_id="NQ_2026-01-05_RTH",
                session_type="RTH",
                killzone="NEW_YORK",
                rth_open="2026-01-05T09:30:00-05:00",
                rth_close="2026-01-05T16:15:00-05:00",
            ),
        ],
        pd_levels={
            "2026-01-05": PreviousDayLevels(
                pd_high=22150.0,
                pd_low=21980.0,
                pd_mid=22065.0,
                pd_close=22100.0,
                pw_high=22200.0,
                pw_low=21850.0,
                overnight_high=22130.0,
                overnight_low=22050.0,
            ),
        },
        quality=QualityReport(
            passed=True,
            total_bars=len(synthetic_bars_5m),
            gaps_found=0,
            gaps_detail=[],
            volume_zeros=0,
            ohlc_violations=0,
            cross_tf_mismatches=0,
            timestamp_coverage=1.0,
            report_generated_at=now.isoformat(),
        ),
        date_range=("2026-01-05", "2026-01-12"),
    )


@pytest.fixture
def nq_instrument() -> InstrumentSpec:
    """NQ instrument specification."""
    return InstrumentSpec(
        full_name="E-mini Nasdaq-100 Futures",
        exchange="CME",
        tick_size=0.25,
        tick_value=5.00,
        point_value=20.00,
        exchange_nfa_per_side=2.14,
        broker_commission_per_side=0.50,
        avg_slippage_ticks=0.5,
        avg_slippage_per_side=2.50,
        total_round_turn=7.78,
        session_open="18:00",
        session_close="17:00",
        rth_open="09:30",
        rth_close="16:15",
    )


@pytest.fixture
def apex_50k_profile() -> PropFirmProfile:
    """Apex Trader Funding 50K profile."""
    return PropFirmProfile(
        name="Apex Trader Funding 50K",
        account_size=50000,
        daily_loss_limit=None,
        trailing_max_drawdown=2500,
        drawdown_type="real_time",
        max_contracts=4,
        consistency_rule_pct=30,
        profit_target=3000,
    )


@pytest.fixture
def full_pipeline_bus(nq_instrument, apex_50k_profile):
    """MessageBus with all 6 agents registered. Returns (bus, agents_dict)."""
    bus = MessageBus()
    agents = {
        "orch": OrchestratorAgent(bus),
        "data": DataInfraAgent(bus),
        "sig": SignalEngineeringAgent(bus),
        "val": ValidationAgent(bus),
        "exec": ExecutionAgent(bus, instrument=nq_instrument, prop_firm=apex_50k_profile),
        "mon": MonitoringAgent(bus),
    }
    return bus, agents
