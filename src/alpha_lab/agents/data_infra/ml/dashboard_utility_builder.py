"""
Self-contained dashboard-utility dataset builder.

W1 P4a: the production labeling path is a BATCH DRIVE OF THE SHARED ENGINE —
the canonical Strategy-Core day stream feeds the same StrategyRuntime /
TouchReversalPlugin wiring Trade-Lab serves with; touches are harvested per
bar-close, the 6 SC feature formulas run over the same stream, and
``resolve_honest_outcome`` labels each touch
(``engine_decision.process_single_date_stream``).

The retired legacy stages (bar-close session slicing, builder-local zones /
touches, zones-from-final-levels, DuckDB feature windows) live in
``legacy_decision`` — imported by parity/regression tests only, never here.

All parameters (bar_type, interaction window, approach window, TP/SL)
are configurable via DashboardUtilityConfig and included in the cache hash.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Timezone — SINGLE-SOURCED from the shared engine scheme.
from strategy_core.constants import (
    RESEARCH_SESSION_SCHEME as _SCHEME,
)

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig, MLPipelineConfig
from alpha_lab.agents.data_infra.tick_store import TickStore

logger = logging.getLogger(__name__)

# The 3 canonical dashboard features
DASHBOARD_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
]

_ET = _SCHEME.timezone  # "US/Eastern"

# ── Cache seed provenance (guard against seedless / wrong-seed caches) ────────
# A per-day cache is built by driving the engine with a ``prev_full_hl`` PDH/PDL
# seed. A cache built with the WRONG seed (notably ``None`` on a day that has a
# prior window day — the W3a-P2 standalone-timing-build class of bug) is
# path-correct but content-WRONG: it silently omits PDH/PDL touches. Touched
# levels leave rows but UNtouched PDH/PDL leave none, so the seed cannot be
# recovered from content. We therefore stamp the seed the cache was built with
# into the parquet file metadata and verify it on the trust-existing-cache path.
_SEED_META_KEY = b"ml_utility_prev_full_hl"


def _seed_meta_bytes(seed: tuple[float, float] | None) -> bytes:
    """Serialize a ``prev_full_hl`` seed for parquet metadata (repr → exact float
    round-trip)."""
    if seed is None:
        return b"none"
    return f"{seed[0]!r},{seed[1]!r}".encode()


def _write_day_cache(
    frame: pd.DataFrame, cache_path: Path, seed: tuple[float, float] | None
) -> None:
    """Write a per-day utility cache, stamping the ``prev_full_hl`` seed it was
    built with into parquet file metadata (alongside pandas metadata)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[_SEED_META_KEY] = _seed_meta_bytes(seed)
    table = table.replace_schema_metadata(meta)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, cache_path)


def _cache_seed_matches(
    cache_path: Path, expected_seed: tuple[float, float] | None
) -> bool:
    """True iff the cache was stamped with a seed equal to ``expected_seed``.

    An UNSTAMPED (legacy) or ``none``-stamped cache matches ONLY when no seed is
    expected (``expected_seed is None``). So when a prior window day exists (the
    seed is non-None), a seedless/legacy cache is NOT trusted and is rebuilt —
    exactly the 2026-02-12 stale-cache case.
    """
    import pyarrow.parquet as pq

    try:
        md = pq.read_metadata(cache_path).metadata or {}
    except Exception:
        return False
    stamp = md.get(_SEED_META_KEY)
    if stamp is None or stamp == b"none":
        return expected_seed is None
    try:
        high, low = (float(x) for x in stamp.decode().split(","))
    except Exception:
        return False
    return expected_seed is not None and (high, low) == (
        float(expected_seed[0]),
        float(expected_seed[1]),
    )


def build_utility_dataset(
    dates: list[str],
    data_dir: Path,
    config: MLPipelineConfig,
    progress_fn=None,
    *,
    use_engine: bool = True,
) -> pd.DataFrame:
    """Build a labeled feature dataset for dashboard-utility training.

    Self-contained: builds bars, detects levels, finds touches, labels,
    and computes features directly from raw tick parquet files.

    Args:
        dates: List of date strings to process (e.g. ["2025-06-02", ...]).
        data_dir: Root databento data directory.
        config: Pipeline config (uses dashboard_utility sub-config).
        progress_fn: Optional callable(fraction, text) for progress.
        use_engine: W1 P4a — the production path IS the engine stream drive; the
            legacy decision layer left this module (``legacy_decision``, tests
            only). ``False`` raises.

    Returns:
        DataFrame with one row per labeled touch event.
    """
    if not use_engine:
        raise ValueError(
            "use_engine=False was retired in W1: the legacy decision path moved to "
            "alpha_lab.agents.data_infra.ml.legacy_decision (parity tests only)"
        )
    util_cfg = config.dashboard_utility
    symbol = config.instrument
    cache_tag = config.dataset_config_hash()

    frames: list[pd.DataFrame] = []
    cached_count = 0

    # Track the prior FULL trading day's high/low for PDH/PDL (engine v3) — the
    # cold-start seed for each single-day stream drive.
    prev_full_hl: tuple[float, float] | None = None

    for i, date_str in enumerate(sorted(dates)):
        if progress_fn:
            progress_fn(i / len(dates), f"Processing {date_str} ({i + 1}/{len(dates)})...")

        cache_path = data_dir / symbol / date_str / f"ml_utility_{cache_tag}.parquet"

        # Trust an existing cache ONLY if it was built with the same seed entering
        # this day. A seedless/wrong-seed cache (e.g. a standalone timing build
        # with prev_full_hl=None on a day that HAS a prior window day) is silently
        # missing PDH/PDL touches — rebuild it instead of trusting it.
        if cache_path.exists() and _cache_seed_matches(cache_path, prev_full_hl):
            df = pd.read_parquet(cache_path)
            cached_count += 1
            if not df.empty:
                frames.append(df)
            # Still need this date's full H/L as the next day's PDH/PDL seed.
            prev_full_hl = _get_session_hl_for_date(
                data_dir, symbol, date_str, util_cfg, prev_full_hl
            )
            continue
        if cache_path.exists():
            logger.warning(
                "Rebuilding %s cache for %s: seed stamp does not match prev_full_hl=%s "
                "(stale/seedless cache guard)",
                cache_tag,
                date_str,
                prev_full_hl,
            )

        # Build fresh for this date. Capture the seed ENTERING this day: the cache must
        # be stamped with the seed it was BUILT with (what the :161 trust check compares
        # against on the next run, and the warmer's stamp convention) — stamping the
        # post-update carry (this day's own H/L) made every builder-written cache
        # self-invalidate on the next run (SEED_PARITY_RECON §3(d) rebuild churn).
        entering_seed = prev_full_hl
        df = _process_single_date(
            date_str,
            data_dir,
            symbol,
            util_cfg,
            entering_seed,
        )

        # Update the PDH/PDL carry for the next day
        prev_full_hl = _get_session_hl_for_date(
            data_dir, symbol, date_str, util_cfg, prev_full_hl
        )

        if not df.empty:
            _write_day_cache(df, cache_path, entering_seed)
            frames.append(df)

    if progress_fn:
        progress_fn(1.0, "Done.")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    # Strip any non-live-computable approach features (may exist in old caches)
    from alpha_lab.agents.data_infra.ml.config import (
        LIVE_APPROACH_FEATURES,
        LIVE_INTERACTION_FEATURES,
    )

    live_features = set(LIVE_INTERACTION_FEATURES + LIVE_APPROACH_FEATURES)
    drop_cols = [
        c
        for c in result.columns
        if (c.startswith("app_") or c.startswith("int_")) and c not in live_features
    ]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    logger.info(
        "Utility dataset: %d labeled events from %d dates (%d cached)",
        len(result),
        len(dates),
        cached_count,
    )
    return result


def _get_session_hl_for_date(
    data_dir: Path,
    symbol: str,
    date_str: str,
    util_cfg: DashboardUtilityConfig,
    prev_full_hl: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """This date's FULL trading-day H/L — the NEXT day's PDH/PDL seed.

    Engine v3: max-high / min-low over the entire [18:00, 18:00) ET window (the
    daily-candle extremes). W1 P4b: the asia/london carry is gone — session
    extremes are folded per-trade inside the engine's level state during the
    stream drive, never sliced from bar closes here.
    """
    bars = _build_bars_for_date(data_dir, symbol, date_str, util_cfg)
    if bars.empty:
        return prev_full_hl

    bars_et = _ensure_et_index(bars)
    return (float(bars_et["high"].max()), float(bars_et["low"].min()))


def _process_single_date(
    date_str: str,
    data_dir: Path,
    symbol: str,
    util_cfg: DashboardUtilityConfig,
    prev_full_hl: tuple[float, float] | None,
) -> pd.DataFrame:
    """Process a single date — W1 P4a: a batch drive of the shared SC runtime.

    The canonical day stream feeds the SAME StrategyRuntime/TouchReversalPlugin
    wiring Trade-Lab serves with; ``prev_full_hl`` seeds PDH/PDL exactly like the
    serving cold-start path (``load_prior_day_summary``).
    """

    from alpha_lab.agents.data_infra.ml.engine_decision import process_single_date_stream

    return process_single_date_stream(
        date_str,
        data_dir,
        symbol,
        util_cfg,
        prev_day_hl=prev_full_hl,
    )


# ── Bar Building ──────────────────────────────────────────────────


def _build_bars_for_date(
    data_dir: Path,
    symbol: str,
    date_str: str,
    util_cfg: DashboardUtilityConfig,
) -> pd.DataFrame:
    """Build bars for a single date using the configured bar_type."""
    td = date.fromisoformat(date_str)
    prev_day = td - timedelta(days=1)

    # Session spans 18:00 ET (prev day) to 18:00 ET (current day), DST-aware.
    # Pass tz-aware UTC bounds so DuckDB compares them directly against the TIMESTAMPTZ
    # ts_event column -- NOT a naive datetime that DuckDB reinterprets in its session
    # timezone (the prior 23:00-CT window artifact). build_tick_bars partitions by the
    # same 18:00-ET trading day, so [prev 18:00 ET, cur 18:00 ET) yields this date's bars.
    start_utc = pd.Timestamp(f"{prev_day.isoformat()} 18:00:00", tz="America/New_York").tz_convert(
        "UTC"
    )
    end_utc = pd.Timestamp(f"{td.isoformat()} 18:00:00", tz="America/New_York").tz_convert("UTC")

    store = TickStore(data_dir)
    try:
        # Register both prev day and current day
        for d in [prev_day, td]:
            store.register_symbol_date(symbol, d)

        bar_type = util_cfg.bar_type
        if bar_type == "1m":
            # Check for cached session bars first
            cached = data_dir / symbol / date_str / "ohlcv_1m_session.parquet"
            if cached.exists():
                df = pd.read_parquet(cached)
                if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                return df
            df = store.build_bars_from_ticks(
                symbol,
                start_utc,
                end_utc,
                bar_size="1 minute",
            )
        elif bar_type.endswith("t"):
            tick_count = int(bar_type[:-1])
            # TRADE-BAR CUTOVER (Part 1): the decision/dashboard pipeline now builds
            # TRADE-PRICE tick bars (a tick is a trade print, action='T', OHLC on the
            # 0.25 grid) — the ratified production bar definition
            # (strategy_core.constants.BAR_PRICE_SOURCE="trade_price"). This supersedes
            # the phase-4c book-mid bars. The engine decision path is threaded with
            # tick_size=0.25 to match (engine_decision.process_single_date_engine).
            # DuckDB<->streaming trade-bar parity is proven in strategy-core/validation.
            df = store.build_tick_bars(
                symbol,
                start_utc,
                end_utc,
                tick_count=tick_count,
                price_source="trade",
            )
        else:
            logger.warning("Unknown bar_type: %s, falling back to 987t", bar_type)
            df = store.build_tick_bars(
                symbol,
                start_utc,
                end_utc,
                tick_count=987,
                price_source="trade",
            )
    finally:
        store.close()

    return df


def _ensure_et_index(bars: pd.DataFrame) -> pd.DataFrame:
    """Convert bar index to US/Eastern timezone."""
    if bars.empty:
        return bars
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert(_ET)
    elif str(bars.index.tz) != _ET:
        bars.index = bars.index.tz_convert(_ET)
    return bars
