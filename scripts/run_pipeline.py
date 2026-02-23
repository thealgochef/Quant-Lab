#!/usr/bin/env python3
"""
End-to-end pipeline runner with synthetic data.

Demonstrates the full DATA -> SIG -> VAL -> EXEC flow using
synthetic NQ price data (no API key required).

Usage:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

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
    SignalVerdict,
    ValidationReport,
)
from alpha_lab.core.message import MessageBus

# ─── Logging Setup ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-30s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_runner")


# ─── Synthetic Data Generation ─────────────────────────────────


def generate_synthetic_bars(
    n_bars: int = 2000,
    base_price: float = 22000.0,
    volatility: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic NQ-like 5m OHLCV bars with trending behavior.

    Creates a random walk with mild momentum and mean-reversion to
    produce realistic-looking price action with volume.
    """
    rng = np.random.default_rng(seed)

    # Generate returns: mostly noise with weak mean-reversion
    # This produces realistic signal-to-noise (low IC, moderate hit rates)
    returns = rng.normal(0.00005, volatility, n_bars)
    # Add regime shifts (not smooth trend — prevents high momentum corr)
    regime = np.zeros(n_bars)
    regime[400:600] = 0.0003
    regime[800:900] = -0.0005
    regime[1200:1400] = 0.0002
    regime[1600:1700] = -0.0003
    returns += regime
    # Add noise bursts (volatility clustering)
    for burst_start in [300, 700, 1100, 1500]:
        burst_end = min(burst_start + 50, n_bars)
        returns[burst_start:burst_end] *= 2.5

    # Build close prices
    close = base_price * np.cumprod(1 + returns)

    # Build OHLCV
    bar_range = np.abs(rng.normal(0, volatility * base_price * 0.5, n_bars))
    high = close + bar_range * 0.6
    low = close - bar_range * 0.4
    open_price = close + rng.normal(0, volatility * base_price * 0.2, n_bars)

    # Ensure OHLC integrity
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Volume with intraday pattern
    base_vol = rng.poisson(5000, n_bars).astype(float)
    intraday_pattern = 1 + 0.5 * np.sin(
        np.linspace(0, 2 * np.pi * (n_bars / 78), n_bars)
    )
    volume = base_vol * intraday_pattern

    # Create DatetimeIndex (5m bars during RTH)
    start = datetime(2026, 1, 5, 9, 30)  # Monday
    dates = []
    current = start
    for _i in range(n_bars):
        dates.append(current)
        current += timedelta(minutes=5)
        # Skip to next day if past 16:15
        if current.hour >= 16 and current.minute >= 15:
            current = current.replace(hour=9, minute=30) + timedelta(days=1)
            # Skip weekends
            while current.weekday() >= 5:
                current += timedelta(days=1)

    index = pd.DatetimeIndex(dates)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "session_id": [
                f"NQ_{d.strftime('%Y-%m-%d')}_RTH" for d in dates
            ],
        },
        index=index,
    )

    return df


def build_synthetic_data_bundle(bars_5m: pd.DataFrame) -> DataBundle:
    """Wrap synthetic bars into a proper DataBundle."""
    now = datetime.now(UTC)

    quality = QualityReport(
        passed=True,
        total_bars=len(bars_5m),
        gaps_found=0,
        gaps_detail=[],
        volume_zeros=0,
        ohlc_violations=0,
        cross_tf_mismatches=0,
        timestamp_coverage=1.0,
        report_generated_at=now.isoformat(),
    )

    sessions = [
        SessionMetadata(
            session_id="NQ_2026-01-05_RTH",
            session_type="RTH",
            killzone="NEW_YORK",
            rth_open="2026-01-05T09:30:00-05:00",
            rth_close="2026-01-05T16:15:00-05:00",
        ),
    ]

    pd_levels = {
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
    }

    return DataBundle(
        instrument="NQ",
        bars={"5m": bars_5m},
        sessions=sessions,
        pd_levels=pd_levels,
        quality=quality,
        date_range=("2026-01-05", "2026-02-20"),
    )


# ─── NQ Instrument + Apex Profile ──────────────────────────────

NQ_SPEC = InstrumentSpec(
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

APEX_50K = PropFirmProfile(
    name="Apex Trader Funding 50K",
    account_size=50000,
    daily_loss_limit=None,
    trailing_max_drawdown=2500,
    drawdown_type="real_time",
    max_contracts=4,
    consistency_rule_pct=30,
    profit_target=3000,
)


# ─── Pipeline Runner ───────────────────────────────────────────


def run_pipeline():
    """Run the full end-to-end pipeline with synthetic data."""
    print("=" * 70)
    print("  Alpha Signal Research Lab — End-to-End Pipeline Demo")
    print("=" * 70)
    print()

    # 1. Create message bus and register all agents
    print("[1/6] Setting up agents...")
    bus = MessageBus()

    orch = OrchestratorAgent(bus)
    # DATA-001 registered but we'll bypass it with synthetic data
    DataInfraAgent(bus)
    sig = SignalEngineeringAgent(bus)
    val = ValidationAgent(bus)
    exec_agent = ExecutionAgent(bus, instrument=NQ_SPEC, prop_firm=APEX_50K)
    MonitoringAgent(bus)

    print(f"  Pipeline state: {orch.pipeline.current_state.value}")
    print(f"  Active agents: {[a.value for a in orch.pipeline.active_agents]}")
    print()

    # 2. Generate synthetic data
    print("[2/6] Generating synthetic NQ 5m bars...")
    bars_5m = generate_synthetic_bars(n_bars=2000)
    bundle = build_synthetic_data_bundle(bars_5m)
    print(f"  Bars: {len(bars_5m)} ({bars_5m.index[0]} to {bars_5m.index[-1]})")
    print(f"  Price range: {bars_5m['close'].min():.2f} - {bars_5m['close'].max():.2f}")
    print(f"  Quality: {'PASS' if bundle.quality.passed else 'FAIL'}")
    print()

    # 3. Feed DATA_BUNDLE to orchestrator (simulating DATA-001 output)
    print("[3/6] Running signal generation (SIG-001)...")
    # We directly call SIG-001's generate_signals to get the bundle
    signal_bundle = sig.generate_signals(bundle)
    n_signals = signal_bundle.total_signals
    tfs = signal_bundle.timeframes_covered
    print(f"  Signals generated: {n_signals}")
    print(f"  Timeframes: {tfs}")
    for sv in signal_bundle.signals:
        n_long = int((sv.direction == 1).sum()) if hasattr(sv.direction, 'sum') else 0
        n_short = int((sv.direction == -1).sum()) if hasattr(sv.direction, 'sum') else 0
        n_neutral = int((sv.direction == 0).sum()) if hasattr(sv.direction, 'sum') else 0
        print(f"    {sv.signal_id}: {n_long} long / {n_short} short / {n_neutral} neutral")
    print()

    # 4. Run validation (VAL-001)
    print("[4/6] Running statistical validation (VAL-001)...")
    price_data = {"5m": bars_5m}
    val_report = val.validate_signal_bundle(signal_bundle, price_data)
    print(f"  {val_report.overall_assessment}")
    print(f"  Bonferroni adjusted: {val_report.bonferroni_adjusted}")
    for v in val_report.verdicts:
        print(
            f"    {v.signal_id}: {v.verdict}"
            f" (IC={v.ic:.4f}, t={v.ic_tstat:.2f},"
            f" hit={v.hit_rate:.1%}, Sharpe={v.sharpe:.2f},"
            f" decay={v.decay_class})"
        )
        if v.failed_metrics:
            for fm in v.failed_metrics:
                print(
                    f"      FAIL: {fm['metric']}={fm['value']:.4f}"
                    f" (threshold={fm['threshold']})"
                )
    print()

    # 5. Run execution analysis (EXEC-001)
    print("[5/6] Running execution & risk analysis (EXEC-001)...")
    exec_report = exec_agent.analyze_signals(val_report, price_data)
    n_approved = len(exec_report.approved_signals)
    n_vetoed = len(exec_report.vetoed_signals)
    print(f"  Approved: {n_approved}, Vetoed: {n_vetoed}")

    # If no DEPLOY signals (expected with synthetic data — the firewall catches
    # high factor correlation and look-ahead bias), demo EXEC-001 directly
    # with a synthetic DEPLOY verdict to show the full cost/risk pipeline.
    if n_approved == 0 and n_vetoed == 0 and val_report.verdicts:
        print()
        print("  NOTE: All signals rejected by VAL-001 (firewall working correctly).")
        print("  Running EXEC-001 demo with synthetic DEPLOY verdict...")
        print()

        demo_verdict = val_report.verdicts[0]
        demo_report = ValidationReport(
            request_id=val_report.request_id,
            signal_bundle_id=val_report.signal_bundle_id,
            verdicts=[
                SignalVerdict(
                    **{
                        **demo_verdict.model_dump(),
                        "verdict": "DEPLOY",
                        "max_factor_corr": 0.15,
                    }
                ),
            ],
            deploy_count=1,
            refine_count=0,
            reject_count=0,
            bonferroni_adjusted=False,
            overall_assessment="1 signals evaluated: 1 DEPLOY (demo override)",
            timestamp=val_report.timestamp,
        )
        exec_report = exec_agent.analyze_signals(demo_report, price_data)
        n_approved = len(exec_report.approved_signals)
        n_vetoed = len(exec_report.vetoed_signals)
        print(f"  Approved: {n_approved}, Vetoed: {n_vetoed}")

    for v in exec_report.approved_signals:
        print(
            f"    APPROVED {v.signal_id}:"
            f" net_sharpe={v.costs.net_sharpe:.2f},"
            f" kelly={v.prop_firm.kelly_fraction:.3f},"
            f" contracts={v.risk_parameters.get('max_contracts', 0)},"
            f" MC_ruin={v.prop_firm.mc_ruin_probability:.1%}"
        )
    for v in exec_report.vetoed_signals:
        print(f"    VETOED  {v.signal_id}: {v.veto_reason}")
    print()

    # 6. Summary
    print("[6/6] Pipeline Summary")
    print("=" * 70)
    status = orch.get_pipeline_status()
    print(f"  Phase: {status['current_phase']}")
    print(f"  Decisions logged: {status['decisions_made']}")
    print(f"  Signals generated: {n_signals}")
    print(f"  Validation: {val_report.deploy_count} DEPLOY,"
          f" {val_report.refine_count} REFINE,"
          f" {val_report.reject_count} REJECT")
    print(f"  Execution: {n_approved} APPROVED, {n_vetoed} VETOED")

    portfolio = exec_report.portfolio_risk
    if portfolio.get("total_signals", 0) > 0:
        print(f"  Portfolio: {portfolio['total_signals']} signals,"
              f" {portfolio['total_contracts']} contracts,"
              f" net_sharpe={portfolio['combined_net_sharpe']:.2f}")
    print("=" * 70)

    return exec_report


if __name__ == "__main__":
    run_pipeline()
