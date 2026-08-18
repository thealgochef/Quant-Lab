"""Synthetic fixtures for the search-lane suites (no real source data).

The three-day deterministic walk is adapted from Strategy-Core's replay-parity
harness: day 1 ramps up (bullish HTF gaps), day 2 retraces into them (taps),
day 3 chops — enough to exercise the funnel (taps + setups born) without any
market data. Days sit inside the fixed exploration allowlist so the on-disk
chain tests can run under ``ExplorationDataPolicy`` in a tmp data dir.
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta

import pytest
from strategy_core.candles._ids import make_bar_id
from strategy_core.types import Bar, BarKind, CloseReason

from alpha_lab.agents.data_infra.ifvg.capture_driver import capture_single_date
from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
from alpha_lab.agents.data_infra.ifvg.day_artifacts import DayArtifacts, DaySeeds

SYNTHETIC_DAYS = ("2026-01-13", "2026-01-14", "2026-01-15")
_DAY0 = date(2026, 1, 13)
_DAY_START = datetime(2026, 1, 12, 23, 0, tzinfo=UTC)
_BARS_PER_DAY = 240


def _bar_1m(day_idx: int, index: int, o: int, h: int, low: int, c: int) -> Bar:
    day = _DAY0 + timedelta(days=day_idx)
    open_ts = _DAY_START + timedelta(days=day_idx, seconds=60 * index)
    return Bar(
        timeframe_ticks=60,
        trading_day=day,
        bar_index=index,
        bar_id=make_bar_id(60, day, index, BarKind.TIME),
        open_ts_utc=open_ts,
        close_ts_utc=open_ts + timedelta(seconds=59),
        open_ticks=o,
        high_ticks=h,
        low_ticks=low,
        close_ticks=c,
        volume=3,
        trade_count=2,
        is_complete=index < _BARS_PER_DAY - 1,
        is_partial=index >= _BARS_PER_DAY - 1,
        close_reason=(
            CloseReason.COMPLETE if index < _BARS_PER_DAY - 1 else CloseReason.END_OF_DAY
        ),
        kind=BarKind.TIME,
    )


def _aggregate(bars_1m: list[Bar], seconds: int) -> list[Bar]:
    n = seconds // 60
    out: list[Bar] = []
    day = bars_1m[0].trading_day
    for idx, start in enumerate(range(0, len(bars_1m), n)):
        chunk = bars_1m[start : start + n]
        out.append(
            Bar(
                timeframe_ticks=seconds,
                trading_day=day,
                bar_index=idx,
                bar_id=make_bar_id(seconds, day, idx, BarKind.TIME),
                open_ts_utc=chunk[0].open_ts_utc,
                close_ts_utc=chunk[-1].close_ts_utc,
                open_ticks=chunk[0].open_ticks,
                high_ticks=max(b.high_ticks for b in chunk),
                low_ticks=min(b.low_ticks for b in chunk),
                close_ticks=chunk[-1].close_ticks,
                volume=sum(b.volume for b in chunk),
                trade_count=sum(b.trade_count for b in chunk),
                is_complete=True,
                is_partial=False,
                close_reason=CloseReason.COMPLETE,
                kind=BarKind.TIME,
            )
        )
    return out


def build_synthetic_day_bars(cfg: IfvgCaptureConfig) -> dict[str, list[Bar]]:
    """All timeframes' bars per synthetic day, keyed by ISO date."""

    rng = random.Random(5)
    bases = [
        [20000, 20120, 20240, 20360],
        [20300, 20180, 20060, 19940],
        [20000, 20060, 19990, 20050],
    ]
    result: dict[str, list[Bar]] = {}
    for day_idx, chunk_bases in enumerate(bases):
        bars_1m = []
        for i in range(_BARS_PER_DAY):
            base = chunk_bases[i // 60]
            o = base + rng.randint(-4, 4)
            c = base + rng.randint(-4, 4)
            h = max(o, c) + rng.randint(0, 4)
            low = min(o, c) - rng.randint(0, 4)
            bars_1m.append(_bar_1m(day_idx, i, o, h, low, c))
        bars = list(bars_1m)
        for seconds in cfg.timeframes_seconds():
            if seconds == 60:
                continue
            bars.extend(_aggregate(bars_1m, seconds))
        result[SYNTHETIC_DAYS[day_idx]] = bars
    return result


def day_hl_of(bars: list[Bar]) -> tuple[int, int]:
    ones = [b for b in bars if b.timeframe_ticks == 60]
    return (max(b.high_ticks for b in ones), min(b.low_ticks for b in ones))


def build_artifact_chain(cfg: IfvgCaptureConfig) -> list[DayArtifacts]:
    """DayArtifacts for the three synthetic days with correct entering seeds."""

    bars_by_day = build_synthetic_day_bars(cfg)
    artifacts: list[DayArtifacts] = []
    prev: DayArtifacts | None = None
    for day in SYNTHETIC_DAYS:
        if prev is None:
            seeds = DaySeeds(prev_day=None, prev_full_hl=None, prev_ny_day=None, prev_ny_hl=None)
        else:
            seeds = DaySeeds(
                prev_day=date.fromisoformat(prev.date_str),
                prev_full_hl=prev.day_hl,
                prev_ny_day=None,
                prev_ny_hl=None,
            )
        bars = bars_by_day[day]
        artifacts.append(
            DayArtifacts(
                date_str=day,
                bars=bars,
                level_timeline={},
                seeds=seeds,
                day_hl=day_hl_of(bars),
                ny_hl=None,
                reader_warnings=(),
            )
        )
        prev = artifacts[-1]
    return artifacts


def run_synthetic_chain(
    cfg: IfvgCaptureConfig,
    artifacts: list[DayArtifacts],
    *,
    start_seed=None,
    audit_capture_mode: str = "disabled",
):
    """Chain ``capture_single_date`` over the synthetic artifacts."""

    results = []
    seed = start_seed
    for index, day_artifacts in enumerate(artifacts):
        result = capture_single_date(
            day_artifacts.date_str,
            cfg,
            artifacts=day_artifacts,
            seed=seed,
            dataset_exhausted=index == len(artifacts) - 1,
            audit_capture_mode=audit_capture_mode,
        )
        results.append(result)
        seed = result.end_seed
    return results


@pytest.fixture(scope="session")
def doc_default_cfg() -> IfvgCaptureConfig:
    return IfvgCaptureConfig()


@pytest.fixture(scope="session")
def synthetic_artifacts(doc_default_cfg) -> list[DayArtifacts]:
    return build_artifact_chain(doc_default_cfg)


@pytest.fixture(scope="session")
def synthetic_chain(doc_default_cfg, synthetic_artifacts):
    return run_synthetic_chain(doc_default_cfg, synthetic_artifacts)
