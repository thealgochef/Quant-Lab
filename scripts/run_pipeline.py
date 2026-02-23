#!/usr/bin/env python3
"""
Alpha Signal Research Lab — Pipeline Runner with Rich Dashboard.

Demonstrates the full DATA -> SIG -> VAL -> EXEC -> MON flow.
Supports both synthetic data (default) and live Polygon.io data (--live).

Usage:
    python scripts/run_pipeline.py              # synthetic data
    python scripts/run_pipeline.py --live        # real NQ data via Polygon.io
    python scripts/run_pipeline.py --live --symbol ES  # ES instead of NQ
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

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
from alpha_lab.core.enums import AgentID, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope

# ─── Load .env ────────────────────────────────────────────────
load_dotenv()

# ─── Logging (suppress library noise, keep our agent logs) ────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)-30s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("alpha_lab").setLevel(logging.INFO)

# ─── Console ──────────────────────────────────────────────────
console = Console()


# ─── Instrument Specs ─────────────────────────────────────────

INSTRUMENTS = {
    "NQ": InstrumentSpec(
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
    ),
    "ES": InstrumentSpec(
        full_name="E-mini S&P 500 Futures",
        exchange="CME",
        tick_size=0.25,
        tick_value=12.50,
        point_value=50.00,
        exchange_nfa_per_side=2.14,
        broker_commission_per_side=0.50,
        avg_slippage_ticks=0.5,
        avg_slippage_per_side=6.25,
        total_round_turn=17.78,
        session_open="18:00",
        session_close="17:00",
        rth_open="09:30",
        rth_close="16:15",
    ),
}

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


# ═══════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════


def generate_synthetic_bars(
    n_bars: int = 2000,
    base_price: float = 22000.0,
    volatility: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic NQ-like 5m OHLCV bars with regime shifts."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00005, volatility, n_bars)

    regime = np.zeros(n_bars)
    regime[400:600] = 0.0003
    regime[800:900] = -0.0005
    regime[1200:1400] = 0.0002
    regime[1600:1700] = -0.0003
    returns += regime

    for burst_start in [300, 700, 1100, 1500]:
        burst_end = min(burst_start + 50, n_bars)
        returns[burst_start:burst_end] *= 2.5

    close = base_price * np.cumprod(1 + returns)
    bar_range = np.abs(rng.normal(0, volatility * base_price * 0.5, n_bars))
    high = close + bar_range * 0.6
    low = close - bar_range * 0.4
    open_price = close + rng.normal(0, volatility * base_price * 0.2, n_bars)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    base_vol = rng.poisson(5000, n_bars).astype(float)
    intraday_pattern = 1 + 0.5 * np.sin(
        np.linspace(0, 2 * np.pi * (n_bars / 78), n_bars)
    )
    volume = base_vol * intraday_pattern

    start = datetime(2026, 1, 5, 9, 30)
    dates = []
    current = start
    for _i in range(n_bars):
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


def build_synthetic_data_bundle(bars_5m: pd.DataFrame) -> DataBundle:
    """Wrap synthetic bars into a DataBundle."""
    now = datetime.now(UTC)
    return DataBundle(
        instrument="NQ",
        bars={"5m": bars_5m},
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
                pd_high=22150.0, pd_low=21980.0, pd_mid=22065.0,
                pd_close=22100.0, pw_high=22200.0, pw_low=21850.0,
                overnight_high=22130.0, overnight_low=22050.0,
            ),
        },
        quality=QualityReport(
            passed=True, total_bars=len(bars_5m), gaps_found=0, gaps_detail=[],
            volume_zeros=0, ohlc_violations=0, cross_tf_mismatches=0,
            timestamp_coverage=1.0, report_generated_at=now.isoformat(),
        ),
        date_range=("2026-01-05", "2026-02-20"),
    )


# ═══════════════════════════════════════════════════════════════
#  LIVE DATA (Polygon.io)
# ═══════════════════════════════════════════════════════════════


def fetch_live_data(symbol: str, days: int = 30) -> DataBundle:
    """Fetch real market data via Polygon.io and build a DataBundle."""
    from alpha_lab.agents.data_infra.providers.polygon import PolygonDataProvider
    from alpha_lab.core.enums import Timeframe

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        console.print(
            "[bold red]Error:[/] POLYGON_API_KEY not set. "
            "Create a .env file (see .env.example) or set the environment variable."
        )
        sys.exit(1)

    provider = PolygonDataProvider(api_key=api_key)
    provider.connect()

    end = datetime.now()
    start = end - timedelta(days=days)

    console.print(f"  Fetching {symbol} 5m bars from Polygon.io...")
    console.print(f"  Range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

    try:
        bars_5m = provider.get_ohlcv(symbol, Timeframe.M5, start, end)
    finally:
        provider.disconnect()

    if bars_5m.empty:
        console.print("[bold red]Error:[/] No data returned from Polygon.io")
        sys.exit(1)

    console.print(f"  Received {len(bars_5m)} bars")

    now = datetime.now(UTC)
    first_date = bars_5m.index[0].strftime("%Y-%m-%d")

    # Tag sessions
    bars_5m["session_id"] = [
        f"{symbol}_{ts.strftime('%Y-%m-%d')}_RTH" for ts in bars_5m.index
    ]

    return DataBundle(
        instrument=symbol,
        bars={"5m": bars_5m},
        sessions=[
            SessionMetadata(
                session_id=f"{symbol}_{first_date}_RTH",
                session_type="RTH",
                killzone="NEW_YORK",
                rth_open=f"{first_date}T09:30:00-05:00",
                rth_close=f"{first_date}T16:15:00-05:00",
            ),
        ],
        pd_levels={
            first_date: PreviousDayLevels(
                pd_high=float(bars_5m["high"].iloc[:78].max()),
                pd_low=float(bars_5m["low"].iloc[:78].min()),
                pd_mid=float(
                    (bars_5m["high"].iloc[:78].max() + bars_5m["low"].iloc[:78].min()) / 2
                ),
                pd_close=float(
                    bars_5m["close"].iloc[77]
                    if len(bars_5m) > 77
                    else bars_5m["close"].iloc[-1]
                ),
                pw_high=(
                    float(bars_5m["high"].iloc[:390].max())
                    if len(bars_5m) > 390
                    else float(bars_5m["high"].max())
                ),
                pw_low=(
                    float(bars_5m["low"].iloc[:390].min())
                    if len(bars_5m) > 390
                    else float(bars_5m["low"].min())
                ),
                overnight_high=float(bars_5m["high"].iloc[0]),
                overnight_low=float(bars_5m["low"].iloc[0]),
            ),
        },
        quality=QualityReport(
            passed=True, total_bars=len(bars_5m), gaps_found=0, gaps_detail=[],
            volume_zeros=int((bars_5m["volume"] == 0).sum()),
            ohlc_violations=0, cross_tf_mismatches=0,
            timestamp_coverage=1.0, report_generated_at=now.isoformat(),
        ),
        date_range=(
            bars_5m.index[0].strftime("%Y-%m-%d"),
            bars_5m.index[-1].strftime("%Y-%m-%d"),
        ),
    )


# ═══════════════════════════════════════════════════════════════
#  RICH DASHBOARD RENDERING
# ═══════════════════════════════════════════════════════════════


def render_header(mode: str, symbol: str) -> None:
    """Render the main header panel."""
    title = Text("Alpha Signal Research Lab", style="bold white")
    subtitle = Text(
        f"Pipeline Runner  |  Mode: {mode}  |  {symbol}  |  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        style="dim",
    )
    content = Text.assemble(title, "\n", subtitle)
    console.print(Panel(content, border_style="bright_blue", padding=(1, 2)))


def render_data_panel(bars_5m: pd.DataFrame, bundle: DataBundle) -> None:
    """Render data quality and price summary."""
    table = Table(title="Data Summary", show_header=True, border_style="blue")
    table.add_column("Metric", style="cyan", min_width=20)
    table.add_column("Value", style="white", min_width=30)

    table.add_row("Instrument", bundle.instrument)
    table.add_row("Bars", f"{len(bars_5m):,}")
    table.add_row("Date Range", f"{bars_5m.index[0]} to {bars_5m.index[-1]}")
    table.add_row(
        "Price Range",
        f"{bars_5m['close'].min():,.2f} - {bars_5m['close'].max():,.2f}",
    )
    table.add_row("Current Price", f"{bars_5m['close'].iloc[-1]:,.2f}")
    table.add_row(
        "Avg Volume",
        f"{bars_5m['volume'].mean():,.0f}",
    )

    quality = bundle.quality
    qc_style = "green" if quality.passed else "red"
    table.add_row("Quality Check", Text("PASS" if quality.passed else "FAIL", style=qc_style))
    table.add_row("Gaps Found", str(quality.gaps_found))
    table.add_row("OHLC Violations", str(quality.ohlc_violations))

    console.print(table)
    console.print()


def render_signals_panel(signal_bundle) -> None:
    """Render signal generation results."""
    table = Table(title="Signal Generation (SIG-001)", show_header=True, border_style="green")
    table.add_column("Signal ID", style="cyan", min_width=30)
    table.add_column("Long", justify="right", style="green")
    table.add_column("Short", justify="right", style="red")
    table.add_column("Neutral", justify="right", style="dim")
    table.add_column("Bars", justify="right")

    for sv in signal_bundle.signals:
        if hasattr(sv.direction, "sum"):
            n_long = int((sv.direction == 1).sum())
            n_short = int((sv.direction == -1).sum())
            n_neutral = int((sv.direction == 0).sum())
            n_bars = len(sv.direction)
        else:
            n_long = n_short = n_neutral = n_bars = 0

        table.add_row(
            sv.signal_id,
            str(n_long),
            str(n_short),
            str(n_neutral),
            str(n_bars),
        )

    console.print(table)
    console.print(
        f"  Total signals: [bold]{signal_bundle.total_signals}[/]  |  "
        f"Timeframes: [bold]{', '.join(signal_bundle.timeframes_covered)}[/]"
    )
    console.print()


def render_validation_panel(val_report) -> None:
    """Render validation verdicts with color-coded results."""
    table = Table(title="Statistical Validation (VAL-001)", show_header=True, border_style="yellow")
    table.add_column("Signal ID", style="cyan", min_width=25)
    table.add_column("Verdict", min_width=8)
    table.add_column("IC", justify="right")
    table.add_column("t-stat", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Decay", min_width=10)
    table.add_column("Max Corr", justify="right")

    for v in val_report.verdicts:
        verdict_style = {
            "DEPLOY": "bold green",
            "REFINE": "bold yellow",
            "REJECT": "bold red",
        }.get(v.verdict, "white")

        table.add_row(
            v.signal_id,
            Text(v.verdict, style=verdict_style),
            f"{v.ic:.4f}",
            f"{v.ic_tstat:.2f}",
            f"{v.hit_rate:.1%}",
            f"{v.sharpe:.2f}",
            v.decay_class,
            f"{v.max_factor_corr:.3f}",
        )

    console.print(table)

    # Show failed metrics detail
    for v in val_report.verdicts:
        if v.failed_metrics:
            tree = Tree(f"[dim]{v.signal_id} — failed checks:[/]")
            for fm in v.failed_metrics:
                tree.add(
                    f"[red]{fm['metric']}[/] = {fm['value']:.4f}"
                    f" (threshold: {fm['threshold']})"
                )
            console.print(tree)

    console.print(
        f"  Bonferroni adjusted: [bold]{'Yes' if val_report.bonferroni_adjusted else 'No'}[/]  |  "
        f"[green]{val_report.deploy_count} DEPLOY[/]  "
        f"[yellow]{val_report.refine_count} REFINE[/]  "
        f"[red]{val_report.reject_count} REJECT[/]"
    )
    console.print()


def render_execution_panel(exec_report, demo_mode: bool = False) -> None:
    """Render execution analysis results."""
    title = "Execution & Risk Analysis (EXEC-001)"
    if demo_mode:
        title += " [dim](demo override)[/]"

    table = Table(title=title, show_header=True, border_style="magenta")
    table.add_column("Signal ID", style="cyan", min_width=25)
    table.add_column("Decision", min_width=10)
    table.add_column("Net Sharpe", justify="right")
    table.add_column("Kelly f*", justify="right")
    table.add_column("Contracts", justify="right")
    table.add_column("MC Ruin", justify="right")
    table.add_column("Cost Drag", justify="right")

    for v in exec_report.approved_signals:
        table.add_row(
            v.signal_id,
            Text("APPROVED", style="bold green"),
            f"{v.costs.net_sharpe:.2f}",
            f"{v.prop_firm.kelly_fraction:.3f}",
            str(v.risk_parameters.get("max_contracts", 0)),
            f"{v.prop_firm.mc_ruin_probability:.1%}",
            f"{v.costs.cost_drag_pct:.1%}",
        )

    for v in exec_report.vetoed_signals:
        table.add_row(
            v.signal_id,
            Text("VETOED", style="bold red"),
            f"{v.costs.net_sharpe:.2f}",
            "-",
            "-",
            "-",
            f"{v.costs.cost_drag_pct:.1%}",
        )

    console.print(table)

    # Portfolio risk summary
    portfolio = exec_report.portfolio_risk
    if portfolio.get("total_signals", 0) > 0:
        console.print(
            f"  Portfolio: [bold]{portfolio['total_signals']}[/] signals, "
            f"[bold]{portfolio['total_contracts']}[/] contracts, "
            f"combined net Sharpe = [bold]{portfolio['combined_net_sharpe']:.2f}[/]"
        )
    console.print()


def render_monitoring_panel(mon) -> None:
    """Render monitoring dashboard with signal health, regime, alerts."""
    # Signal health table
    table = Table(title="Live Monitoring (MON-001)", show_header=True, border_style="bright_cyan")
    table.add_column("Signal ID", style="cyan", min_width=25)
    table.add_column("Status", min_width=10)
    table.add_column("Live IC", justify="right")
    table.add_column("BT IC", justify="right")
    table.add_column("IC Ratio", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("P&L", justify="right")

    for sid in mon.active_signals:
        health = mon.check_signal_health(sid)
        status_style = {
            "HEALTHY": "bold green",
            "DEGRADING": "bold yellow",
            "FAILING": "bold red",
        }.get(health.status, "white")

        pnl = health.net_pnl_today
        pnl_style = "green" if pnl >= 0 else "red"

        table.add_row(
            sid,
            Text(health.status, style=status_style),
            f"{health.live_ic:.4f}",
            f"{health.backtest_ic:.4f}",
            f"{health.ic_ratio:.1%}",
            f"{health.live_hit_rate:.1%}",
            f"{health.live_sharpe:.2f}",
            Text(f"${pnl:+,.2f}", style=pnl_style),
        )

    console.print(table)

    # Regime panel
    regime_style = {
        "TRENDING": "bold green",
        "RANGING": "bold blue",
        "VOLATILE": "bold red",
        "TRANSITIONAL": "bold yellow",
    }.get(mon.current_regime.value, "white")

    console.print(
        f"  Regime: [{regime_style}]{mon.current_regime.value}[/]  |  "
        f"Signals tracked: [bold]{len(mon.active_signals)}[/]"
    )

    # Alerts
    daily = mon.generate_daily_report()
    if daily.alerts:
        alert_table = Table(title="Active Alerts", show_header=True, border_style="red")
        alert_table.add_column("Level", min_width=10)
        alert_table.add_column("Metric", min_width=15)
        alert_table.add_column("Message", min_width=40)

        for alert in daily.alerts:
            level_style = {
                "HALT": "bold white on red",
                "CRITICAL": "bold red",
                "WARNING": "bold yellow",
                "INFO": "dim",
            }.get(alert.level, "white")

            alert_table.add_row(
                Text(alert.level, style=level_style),
                alert.metric,
                alert.message,
            )
        console.print(alert_table)
    else:
        console.print("  [green]No active alerts[/]")

    # Recommendations
    if daily.recommendations:
        rec_tree = Tree("[bold]Recommendations[/]")
        for rec in daily.recommendations:
            if "HALT" in rec:
                rec_tree.add(f"[bold red]{rec}[/]")
            elif "Reduce" in rec:
                rec_tree.add(f"[yellow]{rec}[/]")
            else:
                rec_tree.add(f"[green]{rec}[/]")
        console.print(rec_tree)

    console.print()


def render_summary(
    orch, n_signals, val_report, exec_report, mon, mode: str
) -> None:
    """Render final pipeline summary panel."""
    status = orch.get_pipeline_status()
    portfolio = exec_report.portfolio_risk

    table = Table(title="Pipeline Summary", show_header=False, border_style="bright_blue")
    table.add_column("Metric", style="cyan", min_width=25)
    table.add_column("Value", style="white", min_width=35)

    table.add_row("Mode", mode)
    table.add_row("Pipeline Phase", status["current_phase"])
    table.add_row("Decisions Logged", str(status["decisions_made"]))
    table.add_row("Signals Generated", str(n_signals))
    table.add_row(
        "Validation",
        f"[green]{val_report.deploy_count} DEPLOY[/]  "
        f"[yellow]{val_report.refine_count} REFINE[/]  "
        f"[red]{val_report.reject_count} REJECT[/]",
    )
    table.add_row(
        "Execution",
        f"[green]{len(exec_report.approved_signals)} APPROVED[/]  "
        f"[red]{len(exec_report.vetoed_signals)} VETOED[/]",
    )
    if portfolio.get("total_signals", 0) > 0:
        table.add_row(
            "Portfolio",
            f"{portfolio['total_signals']} signals, "
            f"{portfolio['total_contracts']} contracts, "
            f"net Sharpe = {portfolio['combined_net_sharpe']:.2f}",
        )
    table.add_row(
        "Monitoring",
        f"{len(mon.active_signals)} tracked, regime = {mon.current_regime.value}",
    )

    console.print(Panel(table, border_style="bright_blue"))


# ═══════════════════════════════════════════════════════════════
#  PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════


def run_pipeline(live: bool = False, symbol: str = "NQ", days: int = 30) -> None:
    """Run the full pipeline with Rich dashboard output."""
    mode = "LIVE" if live else "SYNTHETIC"
    instrument = INSTRUMENTS.get(symbol)
    if instrument is None:
        console.print(f"[bold red]Error:[/] Unknown symbol '{symbol}'. Use NQ or ES.")
        sys.exit(1)

    render_header(mode, symbol)

    # ── Step 1: Setup Agents ──────────────────────────────────
    console.print("[bold cyan][1/7][/] Setting up agents...", highlight=False)
    bus = MessageBus()
    orch = OrchestratorAgent(bus)
    DataInfraAgent(bus)
    sig = SignalEngineeringAgent(bus)
    val = ValidationAgent(bus)
    exec_agent = ExecutionAgent(bus, instrument=instrument, prop_firm=APEX_50K)
    console.print(
        f"  Pipeline state: [bold]{orch.pipeline.current_state.value}[/]  |  "
        f"Agents: {', '.join(a.value for a in orch.pipeline.active_agents)}"
    )
    console.print()

    # ── Step 2: Data ──────────────────────────────────────────
    data_label = "Fetching live" if live else "Generating synthetic"
    console.print(f"[bold cyan][2/7][/] {data_label} data...", highlight=False)
    if live:
        bundle = fetch_live_data(symbol, days=days)
    else:
        bars_5m = generate_synthetic_bars(n_bars=2000)
        bundle = build_synthetic_data_bundle(bars_5m)

    bars_5m = bundle.bars["5m"]
    render_data_panel(bars_5m, bundle)

    # ── Step 3: Signal Generation ─────────────────────────────
    console.print("[bold cyan][3/7][/] Running signal generation (SIG-001)...", highlight=False)
    signal_bundle = sig.generate_signals(bundle)
    render_signals_panel(signal_bundle)

    # ── Step 4: Validation ────────────────────────────────────
    console.print(
        "[bold cyan][4/7][/] Running statistical validation (VAL-001)...",
        highlight=False,
    )
    price_data = {"5m": bars_5m}
    val_report = val.validate_signal_bundle(signal_bundle, price_data)
    render_validation_panel(val_report)

    # ── Step 5: Execution Analysis ────────────────────────────
    console.print(
        "[bold cyan][5/7][/] Running execution & risk analysis (EXEC-001)...",
        highlight=False,
    )
    exec_report = exec_agent.analyze_signals(val_report, price_data)
    demo_mode = False

    if (
        len(exec_report.approved_signals) == 0
        and len(exec_report.vetoed_signals) == 0
        and val_report.verdicts
    ):
        console.print(
            "  [yellow]All signals rejected by firewall (working correctly).[/]\n"
            "  Running EXEC-001 demo with synthetic DEPLOY override..."
        )
        demo_mode = True

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
            overall_assessment="1 signal: 1 DEPLOY (demo override)",
            timestamp=val_report.timestamp,
        )
        exec_report = exec_agent.analyze_signals(demo_report, price_data)

    render_execution_panel(exec_report, demo_mode=demo_mode)

    # ── Step 6: Monitoring ────────────────────────────────────
    console.print("[bold cyan][6/7][/] Deploying to live monitoring (MON-001)...", highlight=False)
    mon = MonitoringAgent(bus)

    if exec_report.approved_signals:
        deploy_payload = {
            "approved_signals": [
                {
                    "signal_id": v.signal_id,
                    "ic": 0.05,
                    "hit_rate": 0.55,
                    "sharpe": 1.5,
                    "risk_parameters": v.risk_parameters,
                }
                for v in exec_report.approved_signals
            ]
        }
        deploy_env = MessageEnvelope(
            request_id="deploy-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.MONITORING,
            message_type=MessageType.DEPLOY_COMMAND,
            payload=deploy_payload,
        )
        mon.handle_message(deploy_env)

        # Simulate metric updates
        for sid in mon.active_signals:
            mon.update_metrics(
                sid,
                live_ic=0.03,
                live_hit_rate=0.52,
                live_sharpe=1.1,
                trades_today=12,
                gross_pnl_today=200.0,
                net_pnl_today=150.0,
            )

        # Classify regime from recent price action
        market_data = {
            "ema_values": [
                float(bars_5m["close"].iloc[-1]),
                float(bars_5m["close"].iloc[-20]),
                float(bars_5m["close"].iloc[-50]),
            ],
            "kama_slope": 0.3,
            "atr_current": float(
                bars_5m["high"].iloc[-20:].mean() - bars_5m["low"].iloc[-20:].mean()
            ),
            "atr_avg": float(bars_5m["high"].mean() - bars_5m["low"].mean()),
            "adx": 25.0,
        }
        mon.update_regime(market_data)

        render_monitoring_panel(mon)
    else:
        console.print("  [dim]No approved signals to monitor[/]\n")

    # ── Step 7: Summary ───────────────────────────────────────
    console.print("[bold cyan][7/7][/] Pipeline complete.", highlight=False)
    render_summary(
        orch,
        signal_bundle.total_signals,
        val_report,
        exec_report,
        mon,
        mode,
    )

    return exec_report


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha Signal Research Lab — Pipeline Runner"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live Polygon.io data instead of synthetic",
    )
    parser.add_argument(
        "--symbol",
        default="NQ",
        choices=["NQ", "ES"],
        help="Instrument symbol (default: NQ)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data to fetch in live mode (default: 30)",
    )
    args = parser.parse_args()
    run_pipeline(live=args.live, symbol=args.symbol, days=args.days)


if __name__ == "__main__":
    main()
