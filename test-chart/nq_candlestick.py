"""
NQ Front-Month Candlestick Chart via Databento.

Supports both time-based bars (5m default) and tick-count bars (987t, 2000t).
Fetches data from Databento, builds bars, and renders an interactive Plotly chart.

Usage:
    python test-chart/nq_candlestick.py              # 5-minute bars (default)
    python test-chart/nq_candlestick.py --tick 2000   # 2000-tick bars
    python test-chart/nq_candlestick.py --tick 987    # 987-tick bars
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path so we can import alpha_lab
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from alpha_lab.agents.data_infra.aggregation import aggregate_tick_bars
from alpha_lab.agents.data_infra.providers.databento import DatabentDataProvider
from alpha_lab.core.enums import Timeframe


def fetch_nq_1m(days_back: int = 2) -> pd.DataFrame:
    """Fetch recent NQ front-month 1-minute bars from Databento."""
    provider = DatabentDataProvider()
    provider.connect()
    try:
        end = datetime.now(UTC) - timedelta(hours=1)
        start = end - timedelta(days=days_back)

        ticker = provider.resolve_front_month_ticker("NQ", as_of=end)
        print(f"Front-month contract: {ticker}")
        print(f"Fetching 1m bars: {start.date()} -> {end.date()}")

        df = provider.get_ohlcv("NQ", Timeframe.M1, start, end)
        print(f"Received {len(df)} 1-minute bars")
        return df
    finally:
        provider.disconnect()


def fetch_nq_trades(days_back: int = 2) -> pd.DataFrame:
    """Fetch recent NQ front-month trade ticks from Databento."""
    provider = DatabentDataProvider()
    provider.connect()
    try:
        end = datetime.now(UTC) - timedelta(hours=1)
        start = end - timedelta(days=days_back)

        ticker = provider.resolve_front_month_ticker("NQ", as_of=end)
        print(f"Front-month contract: {ticker}")
        print(f"Fetching trade ticks: {start.date()} -> {end.date()}")

        df = provider.get_trades("NQ", start, end)
        print(f"Received {len(df)} trade ticks")
        return df
    finally:
        provider.disconnect()


def resample_to_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute bars to 5-minute OHLCV."""
    df = bars_1m.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.DatetimeIndex(df["timestamp"])
        elif "ts_event" in df.columns:
            df.index = pd.DatetimeIndex(df["ts_event"])

    bars_5m = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    print(f"Resampled to {len(bars_5m)} 5-minute bars")
    return bars_5m


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Regular Trading Hours bars (09:30-16:15 ET)."""
    if df.index.tz is None:
        idx = df.index.tz_localize("UTC")
    else:
        idx = df.index

    et = idx.tz_convert("US/Eastern")
    df = df.copy()
    df.index = et

    mask = (et.time >= pd.Timestamp("09:30").time()) & (
        et.time <= pd.Timestamp("16:15").time()
    )
    rth = df[mask]
    print(f"Filtered to {len(rth)} RTH bars")
    return rth


def build_chart(bars: pd.DataFrame, title: str) -> go.Figure:
    """Build a Plotly candlestick + volume chart."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(title, "Volume"),
    )

    # Use integer x-axis with formatted tick labels for clean rendering
    n = len(bars)
    x_int = list(range(n))

    raw_idx = bars.index
    if isinstance(raw_idx, pd.DatetimeIndex):
        fmt_idx = raw_idx.tz_localize(None) if raw_idx.tz is not None else raw_idx
        tick_labels = fmt_idx.strftime("%m/%d %H:%M").tolist()
    else:
        tick_labels = [str(x) for x in raw_idx]

    # -- Candlestick --
    fig.add_trace(
        go.Candlestick(
            x=x_int,
            open=bars["open"].values,
            high=bars["high"].values,
            low=bars["low"].values,
            close=bars["close"].values,
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # -- Volume bars coloured by candle direction --
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for o, c in zip(bars["open"], bars["close"])
    ]

    fig.add_trace(
        go.Bar(
            x=x_int,
            y=bars["volume"].values,
            marker_color=colors,
            opacity=0.6,
            name="Volume",
        ),
        row=2,
        col=1,
    )

    # -- Layout --
    fig.update_layout(
        template="plotly_dark",
        height=800,
        title_text=f"NQ Futures — {title}",
        title_x=0.5,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=80, b=40),
    )

    # Readable tick labels (~25 evenly spaced)
    step = max(1, n // 25)
    t_pos = list(range(0, n, step))
    t_txt = [tick_labels[i] for i in t_pos]
    fig.update_xaxes(tickvals=t_pos, ticktext=t_txt, row=1, col=1)
    fig.update_xaxes(tickvals=t_pos, ticktext=t_txt, row=2, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_xaxes(title_text="Time (ET)", row=2, col=1)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="NQ candlestick chart")
    parser.add_argument(
        "--tick", type=int, default=None,
        help="Tick-count bar size (e.g. 2000 or 987). Omit for 5-min bars.",
    )
    parser.add_argument(
        "--days", type=int, default=2,
        help="Days of history to fetch (default: 2)",
    )
    args = parser.parse_args()

    if args.tick:
        # ── Tick-bar mode ─────────────────────────────────────────
        output_path = Path(__file__).parent / f"nq_{args.tick}t_chart.html"

        trades = fetch_nq_trades(days_back=args.days)
        if trades.empty:
            print("No trade data returned — check API key and date range.")
            sys.exit(1)

        print(f"Building {args.tick}-tick bars...")
        bars = aggregate_tick_bars(trades, tick_count=args.tick)
        print(f"Built {len(bars)} tick bars")

        if bars.empty:
            print("No tick bars produced.")
            sys.exit(1)

        # Convert to ET for RTH filtering
        bars = filter_rth(bars)
        if bars.empty:
            print("No RTH tick bars — using all bars.")
            bars = aggregate_tick_bars(trades, tick_count=args.tick)

        title = f"{args.tick}-Tick Candles (RTH)"

    else:
        # ── Time-bar mode (5m) ────────────────────────────────────
        output_path = Path(__file__).parent / "nq_5m_chart.html"

        bars_1m = fetch_nq_1m(days_back=args.days)
        if bars_1m.empty:
            print("No data returned — check API key and date range.")
            sys.exit(1)

        bars = resample_to_5m(bars_1m)
        bars = filter_rth(bars)
        if bars.empty:
            print("No RTH bars found — using all bars.")
            bars = resample_to_5m(bars_1m)

        title = "5-Min Candles (RTH)"

    # Build and save chart
    fig = build_chart(bars, title)
    fig.write_html(str(output_path))
    print(f"\nChart saved to {output_path}")
    fig.show()


if __name__ == "__main__":
    main()
