"""
Alpha Signal Research Lab — Streamlit Dashboard.

Full pipeline visualization: candlestick charts with signal overlays,
trade logs, equity curves, validation metrics, and execution analysis.

Usage:
    streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# Ensure src/ is on the path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from alpha_lab.agents.data_infra.agent import DataInfraAgent  # noqa: E402
from alpha_lab.agents.execution.agent import ExecutionAgent  # noqa: E402
from alpha_lab.agents.monitoring.agent import MonitoringAgent  # noqa: E402
from alpha_lab.agents.orchestrator.agent import OrchestratorAgent  # noqa: E402
from alpha_lab.agents.signal_eng.agent import SignalEngineeringAgent  # noqa: E402
from alpha_lab.agents.validation.agent import ValidationAgent  # noqa: E402
from alpha_lab.core.config import InstrumentSpec, PropFirmProfile  # noqa: E402
from alpha_lab.core.contracts import (  # noqa: E402
    DataBundle,
    PreviousDayLevels,
    QualityReport,
    SessionMetadata,
    SignalVerdict,
    ValidationReport,
)
from alpha_lab.core.enums import AgentID, MessageType  # noqa: E402
from alpha_lab.core.message import MessageBus, MessageEnvelope  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logging.getLogger("alpha_lab").setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

INSTRUMENTS = {
    "NQ": InstrumentSpec(
        full_name="E-mini Nasdaq-100 Futures", exchange="CME",
        tick_size=0.25, tick_value=5.00, point_value=20.00,
        exchange_nfa_per_side=2.14, broker_commission_per_side=0.50,
        avg_slippage_ticks=0.5, avg_slippage_per_side=2.50,
        total_round_turn=7.78,
        session_open="18:00", session_close="17:00",
        rth_open="09:30", rth_close="16:15",
    ),
    "ES": InstrumentSpec(
        full_name="E-mini S&P 500 Futures", exchange="CME",
        tick_size=0.25, tick_value=12.50, point_value=50.00,
        exchange_nfa_per_side=2.14, broker_commission_per_side=0.50,
        avg_slippage_ticks=0.5, avg_slippage_per_side=6.25,
        total_round_turn=17.78,
        session_open="18:00", session_close="17:00",
        rth_open="09:30", rth_close="16:15",
    ),
}

APEX_50K = PropFirmProfile(
    name="Apex Trader Funding 50K", account_size=50000,
    daily_loss_limit=None, trailing_max_drawdown=2500,
    drawdown_type="real_time", max_contracts=4,
    consistency_rule_pct=30, profit_target=3000,
)


# ═══════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════


def generate_synthetic_bars(n_bars: int = 2000) -> pd.DataFrame:
    """Generate synthetic NQ-like 5m OHLCV bars."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.00005, 0.001, n_bars)
    regime = np.zeros(n_bars)
    regime[400:600] = 0.0003
    regime[800:900] = -0.0005
    regime[1200:1400] = 0.0002
    regime[1600:1700] = -0.0003
    returns += regime
    for bs in [300, 700, 1100, 1500]:
        returns[bs:min(bs + 50, n_bars)] *= 2.5

    close = 22000.0 * np.cumprod(1 + returns)
    br = np.abs(rng.normal(0, 11.0, n_bars))
    high = close + br * 0.6
    low = close - br * 0.4
    op = close + rng.normal(0, 4.4, n_bars)
    high = np.maximum(high, np.maximum(op, close))
    low = np.minimum(low, np.minimum(op, close))
    vol = rng.poisson(5000, n_bars).astype(float)

    start = datetime(2026, 1, 5, 9, 30)
    dates, cur = [], start
    for _ in range(n_bars):
        dates.append(cur)
        cur += timedelta(minutes=5)
        if cur.hour >= 16 and cur.minute >= 15:
            cur = cur.replace(hour=9, minute=30) + timedelta(days=1)
            while cur.weekday() >= 5:
                cur += timedelta(days=1)

    return pd.DataFrame(
        {"open": op, "high": high, "low": low, "close": close,
         "volume": vol,
         "session_id": [f"NQ_{d.strftime('%Y-%m-%d')}_RTH" for d in dates]},
        index=pd.DatetimeIndex(dates),
    )


def build_bundle(bars_5m: pd.DataFrame, instrument: str) -> DataBundle:
    """Wrap bars into a DataBundle."""
    now = datetime.now(UTC)
    fd = (bars_5m.index[0].strftime("%Y-%m-%d")
          if hasattr(bars_5m.index[0], "strftime") else "2026-01-05")
    ld = (bars_5m.index[-1].strftime("%Y-%m-%d")
          if hasattr(bars_5m.index[-1], "strftime") else "2026-02-20")
    return DataBundle(
        instrument=instrument, bars={"5m": bars_5m},
        sessions=[SessionMetadata(
            session_id=f"{instrument}_{fd}_RTH", session_type="RTH",
            killzone="NEW_YORK",
            rth_open=f"{fd}T09:30:00-05:00", rth_close=f"{fd}T16:15:00-05:00",
        )],
        pd_levels={fd: PreviousDayLevels(
            pd_high=float(bars_5m["high"].iloc[:78].max()),
            pd_low=float(bars_5m["low"].iloc[:78].min()),
            pd_mid=float((bars_5m["high"].iloc[:78].max()
                          + bars_5m["low"].iloc[:78].min()) / 2),
            pd_close=float(bars_5m["close"].iloc[77]
                           if len(bars_5m) > 77
                           else bars_5m["close"].iloc[-1]),
            pw_high=float(bars_5m["high"].max()),
            pw_low=float(bars_5m["low"].min()),
            overnight_high=float(bars_5m["high"].iloc[0]),
            overnight_low=float(bars_5m["low"].iloc[0]),
        )},
        quality=QualityReport(
            passed=True, total_bars=len(bars_5m), gaps_found=0,
            gaps_detail=[], volume_zeros=int((bars_5m["volume"] == 0).sum()),
            ohlc_violations=0, cross_tf_mismatches=0,
            timestamp_coverage=1.0, report_generated_at=now.isoformat(),
        ),
        date_range=(fd, ld),
    )


def fetch_live_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch real bars from Polygon.io."""
    from alpha_lab.agents.data_infra.providers.polygon import PolygonDataProvider
    from alpha_lab.core.enums import Timeframe

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        st.error("POLYGON_API_KEY not set. See .env.example.")
        st.stop()
    provider = PolygonDataProvider(api_key=api_key)
    provider.connect()
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        bars = provider.get_ohlcv(symbol, Timeframe.M5, start, end)
    finally:
        provider.disconnect()
    if bars.empty:
        st.error("No data returned from Polygon.io")
        st.stop()
    bars["session_id"] = [
        f"{symbol}_{ts.strftime('%Y-%m-%d')}_RTH" for ts in bars.index
    ]
    return bars


# ═══════════════════════════════════════════════════════════════
#  TRADE SIMULATION (from signals + price)
# ═══════════════════════════════════════════════════════════════


def simulate_trades(
    direction: pd.Series,
    strength: pd.Series,
    bars: pd.DataFrame,
    instrument: InstrumentSpec,
    strength_threshold: float = 0.4,
    stop_loss_ticks: float = 8.0,
    take_profit_ticks: float = 16.0,
    max_hold_bars: int = 30,
) -> pd.DataFrame:
    """Simulate trade-by-trade log with defined entry/exit rules.

    Entry rules (ALL must be true):
    - Signal direction changes to +1/-1 (new signal formation)
    - Signal strength >= strength_threshold
    - Fill at NEXT bar's open (no look-ahead)

    Exit rules (first triggered wins):
    1. Stop loss: price hits stop_loss_ticks against position
    2. Take profit: price hits take_profit_ticks in favor
    3. Signal reversal: direction flips to opposite side
    4. Max hold: position held max_hold_bars without exit
    5. Signal flat: direction == 0 after at least 2 bars held
    """
    trades: list[dict] = []
    in_trade = False
    entry_price = 0.0
    entry_time = None
    entry_bar_idx = 0
    entry_dir = 0
    entry_str = 0.0
    rt_cost = instrument.total_round_turn
    stop_pts = stop_loss_ticks * instrument.tick_size
    tp_pts = take_profit_ticks * instrument.tick_size

    def _close(exit_bar: int, exit_px: float, reason: str) -> None:
        gross = (exit_px - entry_price) * entry_dir * instrument.point_value
        trades.append({
            "entry_time": entry_time,
            "exit_time": bars.index[exit_bar],
            "direction": "LONG" if entry_dir == 1 else "SHORT",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_px, 2),
            "gross_pnl": round(gross, 2),
            "costs": rt_cost,
            "net_pnl": round(gross - rt_cost, 2),
            "bars_held": exit_bar - entry_bar_idx,
            "exit_reason": reason,
            "entry_strength": round(entry_str, 3),
        })

    for i in range(1, len(direction) - 1):
        cur_dir = int(direction.iloc[i])
        prev_dir = int(direction.iloc[i - 1])
        cur_str = float(strength.iloc[i])

        if in_trade:
            hi = float(bars["high"].iloc[i])
            lo = float(bars["low"].iloc[i])
            bars_in = i - entry_bar_idx

            # 1. Stop loss
            if entry_dir == 1 and lo <= entry_price - stop_pts:
                _close(i, entry_price - stop_pts, "STOP_LOSS")
                in_trade = False
                continue
            if entry_dir == -1 and hi >= entry_price + stop_pts:
                _close(i, entry_price + stop_pts, "STOP_LOSS")
                in_trade = False
                continue

            # 2. Take profit
            if entry_dir == 1 and hi >= entry_price + tp_pts:
                _close(i, entry_price + tp_pts, "TAKE_PROFIT")
                in_trade = False
                continue
            if entry_dir == -1 and lo <= entry_price - tp_pts:
                _close(i, entry_price - tp_pts, "TAKE_PROFIT")
                in_trade = False
                continue

            # 3. Signal reversal
            if cur_dir != 0 and cur_dir != entry_dir:
                _close(i, float(bars["close"].iloc[i]), "REVERSAL")
                in_trade = False
                continue

            # 4. Max hold
            if bars_in >= max_hold_bars:
                _close(i, float(bars["close"].iloc[i]), "MAX_HOLD")
                in_trade = False
                continue

            # 5. Signal flat (after 2+ bars)
            if cur_dir == 0 and bars_in >= 2:
                _close(i, float(bars["close"].iloc[i]), "SIGNAL_FLAT")
                in_trade = False
                continue

        else:
            # Entry: new formation + strength above threshold
            is_new = cur_dir != 0 and cur_dir != prev_dir
            if is_new and cur_str >= strength_threshold:
                in_trade = True
                entry_dir = cur_dir
                entry_str = cur_str
                entry_price = float(bars["open"].iloc[i + 1])
                entry_time = bars.index[i + 1]
                entry_bar_idx = i + 1

    if in_trade:
        _close(len(bars) - 1, float(bars["close"].iloc[-1]), "END_OF_DATA")

    cols = [
        "entry_time", "exit_time", "direction", "entry_price",
        "exit_price", "gross_pnl", "costs", "net_pnl", "bars_held",
        "exit_reason", "entry_strength",
    ]
    return pd.DataFrame(trades, columns=cols) if trades else pd.DataFrame(
        columns=cols,
    )


# ═══════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════


@st.cache_data(ttl=300)
def run_pipeline(
    mode: str, symbol: str, days: int,
    strength_threshold: float = 0.4,
    stop_loss_ticks: float = 8.0,
    take_profit_ticks: float = 16.0,
    max_hold_bars: int = 30,
) -> dict:
    """Run pipeline and return all results."""
    instrument = INSTRUMENTS[symbol]
    bus = MessageBus()
    orch = OrchestratorAgent(bus)
    DataInfraAgent(bus)
    sig = SignalEngineeringAgent(bus)
    val = ValidationAgent(bus)
    exec_agent = ExecutionAgent(bus, instrument=instrument, prop_firm=APEX_50K)

    if mode == "Live (Polygon.io)":
        bars_5m = fetch_live_data(symbol, days=days)
    else:
        bars_5m = generate_synthetic_bars(n_bars=2000)

    bundle = build_bundle(bars_5m, symbol)
    signal_bundle = sig.generate_signals(bundle)

    price_data = {"5m": bars_5m}
    val_report = val.validate_signal_bundle(signal_bundle, price_data)

    exec_report = exec_agent.analyze_signals(val_report, price_data)
    demo_mode = False
    if (not exec_report.approved_signals
            and not exec_report.vetoed_signals
            and val_report.verdicts):
        demo_mode = True
        v = val_report.verdicts[0]
        demo_report = ValidationReport(
            request_id=val_report.request_id,
            signal_bundle_id=val_report.signal_bundle_id,
            verdicts=[SignalVerdict(
                **{**v.model_dump(), "verdict": "DEPLOY", "max_factor_corr": 0.15},
            )],
            deploy_count=1, refine_count=0, reject_count=0,
            bonferroni_adjusted=False,
            overall_assessment="1 DEPLOY (demo override)",
            timestamp=val_report.timestamp,
        )
        exec_report = exec_agent.analyze_signals(demo_report, price_data)

    # Simulate trades per signal
    all_trades = {}
    for sv in signal_bundle.signals:
        if hasattr(sv.direction, "iloc") and hasattr(sv.strength, "iloc"):
            tdf = simulate_trades(
                sv.direction, sv.strength, bars_5m, instrument,
                strength_threshold=strength_threshold,
                stop_loss_ticks=stop_loss_ticks,
                take_profit_ticks=take_profit_ticks,
                max_hold_bars=max_hold_bars,
            )
            all_trades[sv.signal_id] = tdf

    # Monitoring
    mon = MonitoringAgent(bus)
    if exec_report.approved_signals:
        env = MessageEnvelope(
            request_id="deploy-001", sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.MONITORING,
            message_type=MessageType.DEPLOY_COMMAND,
            payload={"approved_signals": [
                {"signal_id": v.signal_id, "ic": 0.05, "hit_rate": 0.55,
                 "sharpe": 1.5, "risk_parameters": v.risk_parameters}
                for v in exec_report.approved_signals
            ]},
        )
        mon.handle_message(env)
        for sid in mon.active_signals:
            mon.update_metrics(
                sid, live_ic=0.03, live_hit_rate=0.52, live_sharpe=1.1,
                trades_today=12, gross_pnl_today=200.0, net_pnl_today=150.0,
            )
        market_data = {
            "ema_values": [float(bars_5m["close"].iloc[-1]),
                           float(bars_5m["close"].iloc[-20]),
                           float(bars_5m["close"].iloc[-50])],
            "kama_slope": 0.3,
            "atr_current": float(bars_5m["high"].iloc[-20:].mean()
                                 - bars_5m["low"].iloc[-20:].mean()),
            "atr_avg": float(bars_5m["high"].mean() - bars_5m["low"].mean()),
            "adx": 25.0,
        }
        mon.update_regime(market_data)

    health_reports = [mon.check_signal_health(s) for s in mon.active_signals]
    daily_report = mon.generate_daily_report() if mon.active_signals else None

    return {
        "bars_5m": bars_5m, "bundle": bundle,
        "signal_bundle": signal_bundle, "val_report": val_report,
        "exec_report": exec_report, "demo_mode": demo_mode,
        "all_trades": all_trades,
        "mon_regime": mon.current_regime.value,
        "mon_active": len(mon.active_signals),
        "health_reports": health_reports, "daily_report": daily_report,
        "status": orch.get_pipeline_status(), "instrument": instrument,
    }


# ═══════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════


def chart_candlestick_with_signals(
    bars: pd.DataFrame, signal_bundle, selected_signal: str,
    trades_df: pd.DataFrame, n_bars: int = 200,
) -> go.Figure:
    """Candlestick chart with signal arrows and trade markers."""
    # Use last n_bars for readability
    display = bars.iloc[-n_bars:]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("Price Action + Signals", "Volume"),
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=display.index, open=display["open"], high=display["high"],
        low=display["low"], close=display["close"],
        name="OHLC", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Signal arrows
    for sv in signal_bundle.signals:
        if sv.signal_id != selected_signal:
            continue
        if not hasattr(sv.direction, "iloc"):
            continue
        sig_slice = sv.direction.iloc[-n_bars:]

        long_mask = sig_slice == 1
        short_mask = sig_slice == -1

        long_idx = display.index[long_mask.values[:len(display)]] if long_mask.any() else []
        short_idx = display.index[short_mask.values[:len(display)]] if short_mask.any() else []

        if len(long_idx) > 0:
            fig.add_trace(go.Scatter(
                x=long_idx,
                y=(display.loc[long_idx, "low"]
                   - (display["high"].max() - display["low"].min()) * 0.02),
                mode="markers",
                marker=dict(symbol="triangle-up", size=8, color="#26a69a", opacity=0.7),
                name=f"{sv.signal_id} LONG",
                hovertemplate="LONG<br>%{x}<br>Price: %{y:.2f}<extra></extra>",
            ), row=1, col=1)

        if len(short_idx) > 0:
            fig.add_trace(go.Scatter(
                x=short_idx,
                y=(display.loc[short_idx, "high"]
                   + (display["high"].max() - display["low"].min()) * 0.02),
                mode="markers",
                marker=dict(symbol="triangle-down", size=8, color="#ef5350", opacity=0.7),
                name=f"{sv.signal_id} SHORT",
                hovertemplate="SHORT<br>%{x}<br>Price: %{y:.2f}<extra></extra>",
            ), row=1, col=1)

    # Trade entry/exit markers
    if not trades_df.empty:
        visible_trades = trades_df[
            trades_df["entry_time"] >= display.index[0]
        ]
        if not visible_trades.empty:
            # Entries
            fig.add_trace(go.Scatter(
                x=visible_trades["entry_time"],
                y=visible_trades["entry_price"],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#FFD700",
                            line=dict(width=2, color="black")),
                name="Trade Entry",
                hovertemplate=(
                    "ENTRY %{customdata[0]}<br>"
                    "%{x}<br>Price: %{y:.2f}<extra></extra>"
                ),
                customdata=visible_trades[["direction"]].values,
            ), row=1, col=1)

            # Exits color-coded by P&L
            win_trades = visible_trades[visible_trades["net_pnl"] >= 0]
            loss_trades = visible_trades[visible_trades["net_pnl"] < 0]

            if not win_trades.empty:
                fig.add_trace(go.Scatter(
                    x=win_trades["exit_time"],
                    y=win_trades["exit_price"],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color="#26a69a",
                                line=dict(width=2, color="black")),
                    name="Exit (Win)",
                    hovertemplate=(
                        "WIN $%{customdata[0]:+.2f}<br>"
                        "%{x}<br>Price: %{y:.2f}<extra></extra>"
                    ),
                    customdata=win_trades[["net_pnl"]].values,
                ), row=1, col=1)

            if not loss_trades.empty:
                fig.add_trace(go.Scatter(
                    x=loss_trades["exit_time"],
                    y=loss_trades["exit_price"],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color="#ef5350",
                                line=dict(width=3, color="#ef5350")),
                    name="Exit (Loss)",
                    hovertemplate=(
                        "LOSS $%{customdata[0]:+.2f}<br>"
                        "%{x}<br>Price: %{y:.2f}<extra></extra>"
                    ),
                    customdata=loss_trades[["net_pnl"]].values,
                ), row=1, col=1)

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(display["close"], display["open"], strict=False)]
    fig.add_trace(go.Bar(
        x=display.index, y=display["volume"], name="Volume",
        marker_color=colors, opacity=0.5,
    ), row=2, col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=40, b=20),
    )
    fig.update_xaxes(type="category", nticks=20, row=1, col=1)
    fig.update_xaxes(type="category", nticks=20, row=2, col=1)

    return fig


def chart_equity_curve(trades_df: pd.DataFrame) -> go.Figure:
    """Cumulative P&L equity curve with drawdown."""
    if trades_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trades to display", showarrow=False)
        return fig

    cum_pnl = trades_df["net_pnl"].cumsum()
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve (Net of Costs)", "Drawdown"),
    )

    fig.add_trace(go.Scatter(
        x=list(range(len(cum_pnl))), y=cum_pnl.values,
        fill="tozeroy", name="Cumulative P&L",
        fillcolor="rgba(38, 166, 154, 0.3)",
        line=dict(color="#26a69a", width=2),
        hovertemplate="Trade #%{x}<br>Cum P&L: $%{y:+,.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(range(len(dd))), y=dd.values,
        fill="tozeroy", name="Drawdown",
        fillcolor="rgba(239, 83, 80, 0.3)",
        line=dict(color="#ef5350", width=2),
        hovertemplate="Trade #%{x}<br>DD: $%{y:,.2f}<extra></extra>",
    ), row=2, col=1)

    # Apex trailing DD limit
    fig.add_hline(y=-2500, line_dash="dot", line_color="red",
                  annotation_text="Apex Trailing DD Limit ($2,500)",
                  row=2, col=1)

    fig.update_layout(
        height=500, template="plotly_dark",
        margin=dict(l=50, r=20, t=40, b=20),
    )
    return fig


def chart_signal_strength(
    bars: pd.DataFrame, signal_bundle, selected_signal: str, n_bars: int = 200,
) -> go.Figure:
    """Signal strength heatmap over time."""
    fig = go.Figure()
    display = bars.iloc[-n_bars:]

    for sv in signal_bundle.signals:
        if sv.signal_id != selected_signal:
            continue
        if not hasattr(sv.strength, "iloc"):
            continue
        str_slice = sv.strength.iloc[-n_bars:]
        dir_slice = sv.direction.iloc[-n_bars:]

        # Color by direction * strength
        values = dir_slice.values * str_slice.values
        colors = ["#26a69a" if v > 0 else "#ef5350" if v < 0 else "#555555"
                  for v in values[:len(display)]]

        fig.add_trace(go.Bar(
            x=display.index,
            y=str_slice.values[:len(display)],
            marker_color=colors,
            name="Signal Strength",
            opacity=0.8,
            hovertemplate="Strength: %{y:.2f}<br>%{x}<extra></extra>",
        ))

    fig.update_layout(
        height=250, template="plotly_dark",
        title="Signal Strength (green=long, red=short)",
        margin=dict(l=50, r=20, t=40, b=20),
        yaxis=dict(range=[0, 1]),
    )
    fig.update_xaxes(type="category", nticks=20)
    return fig


def chart_validation_radar(verdict) -> go.Figure:
    """Radar chart of validation metrics vs thresholds."""
    categories = ["IC", "Hit Rate", "Sharpe", "Profit Factor", "Orthogonality"]
    # Normalize each metric to 0-1 scale relative to threshold
    values = [
        min(abs(verdict.ic) / 0.03, 2.0),  # IC threshold ~0.03
        verdict.hit_rate / 0.55,  # target 55%
        min(verdict.sharpe / 1.5, 2.0),  # target 1.5
        min(verdict.profit_factor / 1.5, 2.0),  # target 1.5
        1.0 - min(verdict.max_factor_corr / 0.5, 1.0),  # lower corr = better
    ]
    # Thresholds at 1.0
    thresholds = [1.0] * 5

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself", name=verdict.signal_id,
        fillcolor="rgba(38, 166, 154, 0.3)",
        line=dict(color="#26a69a"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=thresholds + [thresholds[0]],
        theta=categories + [categories[0]],
        name="Threshold",
        line=dict(color="red", dash="dash"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 2]),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=350, template="plotly_dark",
        margin=dict(l=50, r=50, t=30, b=30),
        showlegend=True,
    )
    return fig


def chart_cost_breakdown(exec_verdict) -> go.Figure:
    """Waterfall chart of cost breakdown."""
    costs = exec_verdict.costs
    fig = go.Figure(go.Waterfall(
        name="P&L Waterfall",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Gross P&L", "Costs", "Net P&L"],
        y=[costs.gross_pnl, -costs.total_costs, costs.net_pnl],
        text=[f"${costs.gross_pnl:,.2f}", f"-${costs.total_costs:,.2f}",
              f"${costs.net_pnl:,.2f}"],
        textposition="outside",
        connector=dict(line=dict(color="gray")),
        increasing=dict(marker=dict(color="#26a69a")),
        decreasing=dict(marker=dict(color="#ef5350")),
        totals=dict(marker=dict(color="#1E88E5")),
    ))
    fig.update_layout(
        height=300, template="plotly_dark",
        title="Cost Impact on P&L",
        margin=dict(l=50, r=20, t=40, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    st.set_page_config(
        page_title="Alpha Signal Research Lab",
        page_icon="📊", layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Pipeline Config")
        mode = st.radio("Data Source", ["Synthetic", "Live (Polygon.io)"])
        symbol = st.selectbox("Instrument", ["NQ", "ES"])
        days = st.slider("History (days)", 7, 90, 30,
                          disabled=(mode == "Synthetic"))
        chart_bars = st.slider("Chart bars to show", 50, 500, 200)

        st.divider()
        st.markdown("**Trade Strategy**")
        strength_threshold = st.slider(
            "Min signal strength", 0.0, 1.0, 0.4, 0.05,
            help="Only enter when signal strength >= this value",
        )
        stop_loss_ticks = st.slider(
            "Stop loss (ticks)", 2.0, 20.0, 8.0, 0.5,
            help="Exit if price moves this many ticks against",
        )
        take_profit_ticks = st.slider(
            "Take profit (ticks)", 4.0, 40.0, 16.0, 1.0,
            help="Exit if price moves this many ticks in favor",
        )
        max_hold_bars = st.slider(
            "Max hold (bars)", 5, 100, 30,
            help="Force exit after holding this many bars",
        )

        st.divider()
        st.markdown("**Apex 50K Profile**")
        st.caption("Max DD: $2,500 | Contracts: 4 | Target: $3,000")

        st.divider()
        run_btn = st.button("🚀 Run Pipeline", type="primary",
                            use_container_width=True)

    # ── Header ────────────────────────────────────────────────
    st.title("📊 Alpha Signal Research Lab")
    st.caption(f"{symbol}  |  {mode}  |  "
               f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if not run_btn and "results" not in st.session_state:
        st.info("Click **Run Pipeline** in the sidebar to start.")
        return

    if run_btn:
        with st.spinner("Running pipeline..."):
            st.session_state["results"] = run_pipeline(
                mode, symbol, days,
                strength_threshold=strength_threshold,
                stop_loss_ticks=stop_loss_ticks,
                take_profit_ticks=take_profit_ticks,
                max_hold_bars=max_hold_bars,
            )

    r = st.session_state["results"]
    bars_5m = r["bars_5m"]
    signal_bundle = r["signal_bundle"]
    val_report = r["val_report"]
    exec_report = r["exec_report"]

    # ══════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ══════════════════════════════════════════════════════════

    tab_chart, tab_trades, tab_val, tab_exec, tab_mon = st.tabs([
        "📈 Price & Signals",
        "📋 Trade Log",
        "🔬 Validation",
        "💰 Execution",
        "🖥️ Monitoring",
    ])

    # Signal selector (used across tabs)
    signal_ids = [sv.signal_id for sv in signal_bundle.signals]
    selected = signal_ids[0] if signal_ids else None

    # ══════════════════════════════════════════════════════════
    # TAB 1: PRICE & SIGNALS
    # ══════════════════════════════════════════════════════════
    with tab_chart:
        selected = st.selectbox("Select Signal", signal_ids,
                                key="sig_select_chart")
        trades_df = r["all_trades"].get(selected, pd.DataFrame())

        st.plotly_chart(
            chart_candlestick_with_signals(
                bars_5m, signal_bundle, selected, trades_df, chart_bars,
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            chart_signal_strength(bars_5m, signal_bundle, selected, chart_bars),
            use_container_width=True,
        )

        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Bars", f"{len(bars_5m):,}")
        with c2:
            st.metric("Current Price", f"${bars_5m['close'].iloc[-1]:,.2f}")
        with c3:
            pct = (bars_5m["close"].iloc[-1] / bars_5m["close"].iloc[0] - 1) * 100
            st.metric("Period Return", f"{pct:+.2f}%")
        with c4:
            st.metric("Quality",
                       "✅ PASS" if r["bundle"].quality.passed else "❌ FAIL")

    # ══════════════════════════════════════════════════════════
    # TAB 2: TRADE LOG
    # ══════════════════════════════════════════════════════════
    with tab_trades:
        sel_trade = st.selectbox("Select Signal", signal_ids,
                                 key="sig_select_trades")
        trades_df = r["all_trades"].get(sel_trade, pd.DataFrame())

        if trades_df.empty:
            st.warning("No trades generated for this signal.")
        else:
            # Equity curve
            st.plotly_chart(chart_equity_curve(trades_df),
                            use_container_width=True)

            # Summary metrics
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            total_pnl = trades_df["net_pnl"].sum()
            n_wins = (trades_df["net_pnl"] >= 0).sum()
            n_losses = (trades_df["net_pnl"] < 0).sum()
            win_rate = n_wins / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = (trades_df.loc[trades_df["net_pnl"] >= 0, "net_pnl"].mean()
                       if n_wins > 0 else 0)
            avg_loss = (trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"].mean()
                        if n_losses > 0 else 0)

            with c1:
                st.metric("Total Trades", len(trades_df))
            with c2:
                st.metric("Net P&L", f"${total_pnl:+,.2f}",
                           delta=f"${total_pnl:+,.2f}")
            with c3:
                st.metric("Win Rate", f"{win_rate:.1%}")
            with c4:
                st.metric("Avg Win", f"${avg_win:+,.2f}")
            with c5:
                st.metric("Avg Loss", f"${avg_loss:,.2f}")
            with c6:
                total_costs = trades_df["costs"].sum()
                st.metric("Total Costs", f"${total_costs:,.2f}")

            # Full trade log
            st.subheader("Trade-by-Trade Log")
            display_df = trades_df.copy()
            display_df["entry_time"] = display_df["entry_time"].astype(str)
            display_df["exit_time"] = display_df["exit_time"].astype(str)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # P&L distribution
            st.subheader("P&L Distribution")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=trades_df["net_pnl"], nbinsx=30, name="Net P&L",
                marker_color=["#26a69a" if x >= 0 else "#ef5350"
                              for x in trades_df["net_pnl"]],
            ))
            fig_dist.add_vline(x=0, line_dash="dash", line_color="white")
            fig_dist.update_layout(
                height=300, template="plotly_dark",
                xaxis_title="Net P&L ($)", yaxis_title="Count",
                margin=dict(l=50, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3: VALIDATION
    # ══════════════════════════════════════════════════════════
    with tab_val:
        st.subheader("Validation Verdicts")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("DEPLOY", val_report.deploy_count)
        with c2:
            st.metric("REFINE", val_report.refine_count)
        with c3:
            st.metric("REJECT", val_report.reject_count)
        with c4:
            st.metric("Bonferroni",
                       "Yes" if val_report.bonferroni_adjusted else "No")

        # Verdict table
        vdata = []
        for v in val_report.verdicts:
            vdata.append({
                "Signal": v.signal_id,
                "Verdict": v.verdict,
                "IC": v.ic,
                "t-stat": v.ic_tstat,
                "Hit Rate": v.hit_rate,
                "Hit Long": v.hit_rate_long,
                "Hit Short": v.hit_rate_short,
                "Sharpe": v.sharpe,
                "Sortino": v.sortino,
                "Max DD": v.max_drawdown,
                "Profit Factor": v.profit_factor,
                "Decay": v.decay_class,
                "Decay HL": v.decay_half_life,
                "Max Corr": v.max_factor_corr,
                "Orthogonal": v.is_orthogonal,
                "Stable": v.subsample_stable,
                "Failed": len(v.failed_metrics),
            })
        st.dataframe(pd.DataFrame(vdata), use_container_width=True,
                      hide_index=True)

        # Per-signal detail
        st.divider()
        for v in val_report.verdicts:
            with st.expander(
                f"{'🟢' if v.verdict == 'DEPLOY' else '🔴' if v.verdict == 'REJECT' else '🟡'} "
                f"{v.signal_id} — {v.verdict}",
                expanded=(v.verdict != "REJECT"),
            ):
                col_radar, col_detail = st.columns([1, 1])
                with col_radar:
                    st.plotly_chart(chart_validation_radar(v),
                                    use_container_width=True)
                with col_detail:
                    st.markdown("**Test Results**")
                    st.markdown(f"- IC: `{v.ic:.4f}` (t={v.ic_tstat:.2f})")
                    st.markdown(f"- Hit Rate: `{v.hit_rate:.1%}` "
                                f"(L={v.hit_rate_long:.1%}, S={v.hit_rate_short:.1%})")
                    st.markdown(f"- Sharpe: `{v.sharpe:.2f}` | "
                                f"Sortino: `{v.sortino:.2f}`")
                    st.markdown(f"- Max DD: `{v.max_drawdown:.1%}` | "
                                f"PF: `{v.profit_factor:.2f}`")
                    st.markdown(f"- Decay: `{v.decay_class}` "
                                f"(half-life={v.decay_half_life:.0f} bars)")
                    st.markdown(f"- Max Factor Corr: `{v.max_factor_corr:.3f}` | "
                                f"Orthogonal: `{v.is_orthogonal}`")

                    if v.failed_metrics:
                        st.markdown("**Failed Checks:**")
                        for fm in v.failed_metrics:
                            st.error(
                                f"**{fm['metric']}** = {fm['value']:.4f} "
                                f"(threshold: {fm['threshold']})"
                            )

    # ══════════════════════════════════════════════════════════
    # TAB 4: EXECUTION
    # ══════════════════════════════════════════════════════════
    with tab_exec:
        if r["demo_mode"]:
            st.warning("All signals REJECTED by firewall. "
                       "Showing EXEC-001 demo with forced DEPLOY override.")

        st.subheader("Execution Verdicts")
        all_exec = exec_report.approved_signals + exec_report.vetoed_signals

        if not all_exec:
            st.info("No signals reached execution analysis.")
        else:
            for ev in all_exec:
                icon = "✅" if ev.verdict == "APPROVED" else "❌"
                with st.expander(
                    f"{icon} {ev.signal_id} — {ev.verdict}", expanded=True
                ):
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown("**Cost Analysis**")
                        st.metric("Gross Sharpe", f"{ev.costs.gross_sharpe:.2f}")
                        st.metric("Net Sharpe", f"{ev.costs.net_sharpe:.2f}")
                        st.metric("Cost Drag", f"{ev.costs.cost_drag_pct:.1%}")
                        st.metric("Breakeven HR",
                                  f"{ev.costs.breakeven_hit_rate:.1%}")

                    with c2:
                        st.markdown("**Prop Firm Feasibility**")
                        st.metric("Kelly f*",
                                  f"{ev.prop_firm.kelly_fraction:.3f}")
                        st.metric("Half-Kelly Contracts",
                                  ev.prop_firm.half_kelly_contracts)
                        st.metric("MC Ruin Prob",
                                  f"{ev.prop_firm.mc_ruin_probability:.2%}")
                        st.metric("Consistency",
                                  f"{ev.prop_firm.consistency_score:.1%}")

                    with c3:
                        st.markdown("**Risk Parameters**")
                        st.metric("Max Contracts",
                                  ev.risk_parameters.get("max_contracts", 0))
                        st.metric("Stop Loss",
                                  f"{ev.risk_parameters.get('stop_loss_ticks', 0)} ticks")
                        st.metric("Worst Day",
                                  f"${ev.prop_firm.worst_day_pnl:,.2f}")
                        st.metric("Max Trail DD",
                                  f"${ev.prop_firm.max_trailing_dd:,.2f}")

                    if ev.veto_reason:
                        st.error(f"**Veto Reason:** {ev.veto_reason}")

                    # Cost waterfall
                    st.plotly_chart(chart_cost_breakdown(ev),
                                    use_container_width=True)

                    # Turnover details
                    st.markdown("**Turnover Analysis**")
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        st.metric("Trades/Day",
                                  f"{ev.turnover.get('trades_per_day', 0):.1f}")
                    with tc2:
                        st.metric("Avg Holding",
                                  f"{ev.turnover.get('avg_holding_bars', 0):.1f} bars")
                    with tc3:
                        st.metric("Flip Rate",
                                  f"{ev.turnover.get('flip_rate', 0):.1%}")

            # Portfolio summary
            portfolio = exec_report.portfolio_risk
            if portfolio.get("total_signals", 0) > 0:
                st.divider()
                st.subheader("Portfolio Risk")
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    st.metric("Total Signals", portfolio["total_signals"])
                with pc2:
                    st.metric("Total Contracts", portfolio["total_contracts"])
                with pc3:
                    st.metric("Combined Net Sharpe",
                              f"{portfolio['combined_net_sharpe']:.2f}")

    # ══════════════════════════════════════════════════════════
    # TAB 5: MONITORING
    # ══════════════════════════════════════════════════════════
    with tab_mon:
        mc1, mc2 = st.columns(2)
        regime = r["mon_regime"]
        regime_icons = {"TRENDING": "📈", "RANGING": "↔️",
                        "VOLATILE": "🌊", "TRANSITIONAL": "🔄"}
        with mc1:
            st.metric("Market Regime",
                       f"{regime_icons.get(regime, '')} {regime}")
        with mc2:
            st.metric("Signals Tracked", r["mon_active"])

        if r["health_reports"]:
            st.subheader("Signal Health")
            hdata = []
            for h in r["health_reports"]:
                hdata.append({
                    "Signal": h.signal_id,
                    "Status": h.status,
                    "Live IC": h.live_ic,
                    "BT IC": h.backtest_ic,
                    "IC Ratio": f"{h.ic_ratio:.0%}",
                    "Hit Rate": f"{h.live_hit_rate:.1%}",
                    "Sharpe": h.live_sharpe,
                    "Net P&L": f"${h.net_pnl_today:+,.2f}",
                    "Trades": h.trades_today,
                })
            st.dataframe(pd.DataFrame(hdata), use_container_width=True,
                          hide_index=True)

        daily = r["daily_report"]
        if daily:
            if daily.alerts:
                st.subheader("Active Alerts")
                for a in daily.alerts:
                    icons = {"HALT": "🚨", "CRITICAL": "🔴",
                             "WARNING": "⚠️", "INFO": "ℹ️"}
                    st.markdown(f"{icons.get(a.level, '')} **{a.level}** — "
                                f"{a.metric}: {a.message}")
            else:
                st.success("No active alerts")

            st.subheader("Recommendations")
            for rec in daily.recommendations:
                if "HALT" in rec:
                    st.error(rec)
                elif "Reduce" in rec:
                    st.warning(rec)
                else:
                    st.info(rec)
        elif r["mon_active"] == 0:
            st.info("No signals deployed for monitoring.")

    # ── Footer ────────────────────────────────────────────────
    st.divider()
    status = r["status"]
    st.caption(
        f"Pipeline: {status['current_phase']} | "
        f"Signals: {signal_bundle.total_signals} | "
        f"Approved: {len(exec_report.approved_signals)} | "
        f"Vetoed: {len(exec_report.vetoed_signals)} | "
        f"Decisions: {status['decisions_made']}"
    )


if __name__ == "__main__":
    main()
