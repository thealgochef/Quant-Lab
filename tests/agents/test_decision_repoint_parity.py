# ruff: noqa: N803, N812
"""Standing BOOK-MID parity regression: engine decision layer == legacy CQL.

Proves the ``strategy_core`` repoint of the dashboard-utility decision layer
(``engine_decision``) reproduces the legacy duplicate CQL decision code
(``_build_zones`` / ``_detect_touches`` / ``label_touch_event`` /
``_compute_interaction_features`` / ``_compute_approach_features``) EXACTLY when the
engine is run in its REACHABLE BOOK-MID MODE (book-mid bars @0.125, book-mid
interaction feed, LEVEL-price entry). This is the phase-5 engine==legacy proof,
preserved after the trade-bar cutover.

IMPORTANT (post-cutover): the engine's PRODUCTION default is now trade-price bars +
trade-print interaction features + the decision-time honest outcome, which DIFFER
from the legacy book-mid level-entry path BY DESIGN. So this test pins the engine to
its book-mid regression mode explicitly:
``bars_et_to_engine(..., BOOK_MID_TICK)``,
``detect_touches(tick_size=BOOK_MID_TICK)``,
``resolve_outcome(tick_size=BOOK_MID_TICK)`` with the LEVEL entry,
``compute_interaction_features_engine(price_source="book_mid", tick_size=BOOK_MID_TICK)``,
and ``process_single_date_engine(price_source="book_mid", tick_size=BOOK_MID_TICK,
honest_entry=False)``. That book-mid mode must stay byte-identical to legacy; the
NEW trade-path behavior is covered by ``test_engine_trade_path_cutover`` below.

The book-mid diff covers, per sample day:
  * zones (set of (rep_price, side, names))
  * touches (ordered list of (level_type, bar ts, direction, rep_price))
  * labels (class + max_mfe / max_mae, per touch with a resolvable forward window)
  * the 6 model features (int_* book-mid @0.125, app_* trade-grid @0.25); max abs
    diff must be 0
  * the integrated dataset rows: ``process_single_date_engine(book-mid mode)`` ==
    ``_process_single_date(use_engine=False)`` for the model feature columns.

Skips cleanly when the Databento NQ store is absent (CI without the data mount).
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENV_DATABENTO_DIR = "QUANT_LAB_DATABENTO_DIR"
_DEFAULT_DATABENTO_DIR = Path(__file__).resolve().parents[2] / "data" / "databento"


def _resolve_databento_dir() -> Path:
    """Return the Databento root used by the real-data parity tests."""
    override = os.environ.get(ENV_DATABENTO_DIR)
    if override:
        return Path(override).expanduser()
    return _DEFAULT_DATABENTO_DIR


DATA_DIR = _resolve_databento_dir()
SYMBOL = "NQ"
_ET = "US/Eastern"

# Sample of phase-4a book-mid days present in the store; 2025-07-09 follows a
# missing prior calendar day (07-08), exercising that gap. The plumbing is
# day-independent, so a handful of days proves the repoint.
SAMPLE_DAYS = ["2025-07-09", "2025-07-10", "2025-07-11", "2025-07-14", "2025-07-15"]
WARMUP_DAY = "2025-07-07"

MODEL_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
    "app_avg_trade_size",
    "app_large_trade_vol_pct",
    "app_max_spread",
]


def _store_available() -> bool:
    if not DATA_DIR.exists():
        return False
    for d in [WARMUP_DAY, *SAMPLE_DAYS]:
        if not (DATA_DIR / SYMBOL / d / "mbp10.parquet").exists():
            return False
    return True


requires_databento_store = pytest.mark.skipif(
    not _store_available(),
    reason=(
        f"Databento NQ store not available at {DATA_DIR}; "
        f"decision-repoint parity needs real data (set {ENV_DATABENTO_DIR} to override)"
    ),
)


def test_resolve_databento_dir_defaults_to_repo_data(monkeypatch):
    monkeypatch.delenv(ENV_DATABENTO_DIR, raising=False)

    assert _resolve_databento_dir() == _DEFAULT_DATABENTO_DIR


def test_resolve_databento_dir_honors_env_override(monkeypatch, tmp_path):
    override = tmp_path / "custom-databento"
    monkeypatch.setenv(ENV_DATABENTO_DIR, str(override))

    assert _resolve_databento_dir() == override


@pytest.fixture(scope="module")
def cfg():
    from alpha_lab.agents.data_infra.ml.config import MLPipelineConfig

    return MLPipelineConfig(
        training_mode="dashboard_utility",
        instrument="NQ",
        tick_size=0.25,
        dashboard_utility=dict(
            bar_type="147t",
            interaction_window_minutes=5,
            approach_window_minutes=30,
            include_approach_features=True,
            tp_points=15.0,
            sl_points=30.0,
            trap_mfe_min=5.0,
            level_proximity_pts=0.5,
        ),
    ).dashboard_utility


@pytest.fixture(scope="module")
def carry(cfg):
    """Session H/L carry advanced from the warmup day (PDH/PDL for sample day 0)."""
    from alpha_lab.agents.data_infra.ml import dashboard_utility_builder as B

    prev_ny = prev_asia = prev_london = None
    prev_ny, prev_asia, prev_london = B._get_session_hl_for_date(
        DATA_DIR, SYMBOL, WARMUP_DAY, cfg, prev_ny, prev_asia, prev_london
    )
    out = {WARMUP_DAY: (prev_ny, prev_asia, prev_london)}
    # Advance the carry across the sample so each day uses the right prior levels.
    for d in SAMPLE_DAYS:
        out[d] = (prev_ny, prev_asia, prev_london)
        prev_ny, prev_asia, prev_london = B._get_session_hl_for_date(
            DATA_DIR, SYMBOL, d, cfg, prev_ny, prev_asia, prev_london
        )
    return out


def _zone_key(z):
    if isinstance(z, dict):
        return (round(float(z["representative_price"]), 6), z["side"], tuple(z["names"]))
    return (round(float(z.representative_price), 6), z.side.value, tuple(z.names))


def _bookmid_bars(date_str, cfg):
    """Build the LEGACY book-mid 147t bars directly via the TickStore.

    ``_build_bars_for_date`` now hardwires ``price_source="trade"`` (the Part-1
    cutover), so the book-mid regression builds book-mid bars here, mirroring that
    function's window/registration but with ``price_source="book_mid"`` @0.125.
    """
    from datetime import date as _date

    from alpha_lab.agents.data_infra.tick_store import TickStore

    td = _date.fromisoformat(date_str)
    prev_day = td - timedelta(days=1)
    start_utc = pd.Timestamp(f"{prev_day.isoformat()} 18:00:00", tz="America/New_York").tz_convert(
        "UTC"
    )
    end_utc = pd.Timestamp(f"{td.isoformat()} 18:00:00", tz="America/New_York").tz_convert("UTC")
    tick_count = int(cfg.bar_type[:-1])
    store = TickStore(DATA_DIR)
    try:
        for d in (prev_day, td):
            store.register_symbol_date(SYMBOL, d)
        return store.build_tick_bars(
            SYMBOL,
            start_utc,
            end_utc,
            tick_count=tick_count,
            price_source="book_mid",
        )
    finally:
        store.close()


def _legacy_decision_rows(B, bars_et, levels, date_str, cfg):
    """Run the legacy CQL decision loop on given book-mid bars+levels.

    Mirrors ``_process_single_date``'s legacy branch (zones -> touches ->
    label_touch_event -> _compute_interaction_features/_compute_approach_features)
    on the SAME book-mid ``bars_et`` the engine book-mid mode consumes, so the two
    are compared on identical input without going through the cutover bar build.
    """
    from alpha_lab.agents.data_infra.ml.dashboard_utility_labeling import (
        NO_RESOLUTION,
        label_touch_event,
    )

    zones = B._build_zones(levels)
    touches = B._detect_touches(bars_et, zones)
    rth_cutoff = pd.Timestamp(f"{date_str} 16:15:00", tz=_ET)
    rows = []
    for touch in touches:
        forward = bars_et[(bars_et.index > touch["bar_ts"]) & (bars_et.index < rth_cutoff)]
        if forward.empty:
            continue
        label_result = label_touch_event(touch, forward, cfg)
        if label_result["label"] == NO_RESOLUTION:
            continue
        features = B._compute_interaction_features(touch, DATA_DIR, SYMBOL, cfg)
        if features is None:
            continue
        row = {
            "event_ts": touch["bar_ts"],
            "date": date_str,
            "timestamp": touch["bar_ts"],
            "direction": touch["direction"],
            "representative_price": touch["representative_price"],
            "level_type": touch["level_type"],
            "label": label_result["label"],
            "label_encoded": label_result["label_encoded"],
            "max_mfe": label_result["max_mfe"],
            "max_mae": label_result["max_mae"],
        }
        row.update(features)
        if cfg.include_approach_features:
            approach = B._compute_approach_features(touch, DATA_DIR, SYMBOL, cfg)
            if approach:
                row.update(approach)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@requires_databento_store
@pytest.mark.parametrize("date_str", SAMPLE_DAYS)
def test_engine_decision_matches_legacy(date_str, cfg, carry):
    """zones + touches + labels + 6 features: engine == legacy, EXACTLY."""
    from alpha_lab.agents.data_infra.ml import dashboard_utility_builder as B
    from alpha_lab.agents.data_infra.ml import engine_decision as E
    from alpha_lab.agents.data_infra.ml.dashboard_utility_labeling import (
        label_touch_event,
    )

    prev_ny, prev_asia, prev_london = carry[date_str]
    td = date.fromisoformat(date_str)

    # Book-mid bars (the cutover hardwired _build_bars_for_date to trade; build the
    # book-mid regression bars directly so the engine book-mid mode == legacy proof).
    bars = _bookmid_bars(date_str, cfg)
    assert not bars.empty, f"{date_str}: no bars"
    bars_et = B._ensure_et_index(bars.copy())
    levels = B._compute_levels_for_date(bars_et, date_str, prev_ny, prev_asia, prev_london)
    assert levels, f"{date_str}: no levels"

    # ── zones ───────────────────────────────────────────────────────────────
    legacy_zones = B._build_zones(levels)
    eng_zones = E.build_zones(E.levels_to_engine(levels))
    assert sorted(_zone_key(z) for z in legacy_zones) == sorted(_zone_key(z) for z in eng_zones), (
        f"{date_str}: zone set diverged"
    )

    # ── touches (fresh zones each side; detection mutates 'touched') ────────
    legacy_touches = B._detect_touches(bars_et, B._build_zones(levels))
    eng_touches = E.detect_touches(
        E.bars_et_to_engine(bars_et, td, E.BOOK_MID_TICK),
        E.build_zones(E.levels_to_engine(levels)),
        tick_size=E.BOOK_MID_TICK,
        trading_day=td,
    )
    assert len(legacy_touches) == len(eng_touches), (
        f"{date_str}: touch count {len(legacy_touches)} != {len(eng_touches)}"
    )
    for lt, et in zip(legacy_touches, eng_touches, strict=False):
        assert E._to_utc_dt(pd.Timestamp(lt["bar_ts"])) == et.bar_ts_utc
        assert lt["direction"] == et.direction.value
        assert lt["level_type"] == et.level_type
        assert abs(float(lt["representative_price"]) - float(et.representative_price)) < 1e-9

    # ── labels ──────────────────────────────────────────────────────────────
    rth_cutoff = pd.Timestamp(f"{date_str} 16:15:00", tz=_ET)
    for lt, et in zip(legacy_touches, eng_touches, strict=False):
        forward = bars_et[(bars_et.index > lt["bar_ts"]) & (bars_et.index < rth_cutoff)]
        if forward.empty:
            continue
        legacy_lab = label_touch_event(lt, forward, cfg)
        eng_out = E.resolve_outcome(
            entry_points=float(et.representative_price),
            direction=et.direction,
            forward_bars=E.bars_et_to_engine(forward, td, E.BOOK_MID_TICK),
            tick_size=E.BOOK_MID_TICK,
            tp_points=cfg.tp_points,
            sl_points=cfg.sl_points,
            trap_mfe_min=cfg.trap_mfe_min,
        )
        assert legacy_lab["label"] == eng_out.label
        assert legacy_lab["label_encoded"] == eng_out.label_encoded
        assert abs(round(float(legacy_lab["max_mfe"]), 4) - eng_out.max_mfe) < 1e-9
        assert abs(round(float(legacy_lab["max_mae"]), 4) - eng_out.max_mae) < 1e-9

    # ── 6 features ──────────────────────────────────────────────────────────
    for lt, et in zip(legacy_touches, eng_touches, strict=False):
        legacy_int = B._compute_interaction_features(lt, DATA_DIR, SYMBOL, cfg)
        # Pin the engine to its book-mid regression feed (price=(bid+ask)/2 @0.125)
        # to match the legacy book-mid path; the production default is trade-print.
        eng_int = E.compute_interaction_features_engine(
            et,
            DATA_DIR,
            SYMBOL,
            cfg,
            price_source="book_mid",
            tick_size=E.BOOK_MID_TICK,
        )
        # The < 5-row drop must agree, so both keep or both drop the touch.
        assert (legacy_int is None) == (eng_int is None), (
            f"{date_str}: interaction kept/dropped disagreement @ {et.bar_ts_utc}"
        )
        if legacy_int is None:
            continue
        legacy_app = B._compute_approach_features(lt, DATA_DIR, SYMBOL, cfg) or {}
        eng_app = E.compute_approach_features_engine(et, DATA_DIR, SYMBOL, cfg) or {}
        merged_legacy = {**legacy_int, **legacy_app}
        merged_eng = {**eng_int, **eng_app}
        for f in MODEL_FEATURES:
            lv, ev = merged_legacy.get(f), merged_eng.get(f)
            assert lv is not None and ev is not None, f"{date_str}: missing {f}"
            if np.isnan(lv) and np.isnan(ev):
                continue
            assert abs(float(lv) - float(ev)) < 1e-9, (
                f"{date_str}: {f} diverged @ {et.bar_ts_utc}: legacy={lv} eng={ev}"
            )


@requires_databento_store
@pytest.mark.parametrize("date_str", SAMPLE_DAYS)
def test_integrated_dataset_rows_match(date_str, cfg, carry):
    """End-to-end book-mid regression: engine(book-mid mode) == legacy CQL.

    Proves the repoint at the INTEGRATION boundary (the row dicts the builder
    emits), not just the comparators: same row count, same model feature columns.

    Post-cutover, ``_process_single_date(use_engine=True)`` runs the engine's
    PRODUCTION trade-print + honest-entry mode, which differs from legacy by design.
    To keep the engine==legacy book-mid proof, this builds the legacy book-mid bars
    and runs ``process_single_date_engine`` PINNED to its book-mid regression mode
    (price_source="book_mid", tick_size=0.125, honest_entry=False) and compares to
    the legacy book-mid decision path.
    """
    from alpha_lab.agents.data_infra.ml import dashboard_utility_builder as B
    from alpha_lab.agents.data_infra.ml import engine_decision as E

    prev_ny, prev_asia, prev_london = carry[date_str]

    # Legacy book-mid bars + levels (the input both paths consumed pre-cutover).
    bm_bars = _bookmid_bars(date_str, cfg)
    assert not bm_bars.empty, f"{date_str}: no book-mid bars"
    bars_et = B._ensure_et_index(bm_bars.copy())
    levels = B._compute_levels_for_date(bars_et, date_str, prev_ny, prev_asia, prev_london)

    eng_df = E.process_single_date_engine(
        bars_et,
        levels,
        date_str,
        DATA_DIR,
        SYMBOL,
        cfg,
        price_source="book_mid",
        tick_size=E.BOOK_MID_TICK,
        honest_entry=False,
    )
    legacy_df = _legacy_decision_rows(B, bars_et, levels, date_str, cfg)

    assert len(eng_df) == len(legacy_df), f"{date_str}: row count {len(eng_df)} != {len(legacy_df)}"
    if eng_df.empty:
        return

    # Align on the touch timestamp so row order differences (if any) don't matter.
    eng_df = eng_df.sort_values("event_ts").reset_index(drop=True)
    legacy_df = legacy_df.sort_values("event_ts").reset_index(drop=True)

    for col in ["direction", "level_type", "label", "label_encoded"]:
        assert (eng_df[col].values == legacy_df[col].values).all(), (
            f"{date_str}: column {col} diverged"
        )
    for col in ["representative_price", "max_mfe", "max_mae", *MODEL_FEATURES]:
        if col not in eng_df.columns or col not in legacy_df.columns:
            continue
        ev = eng_df[col].to_numpy(dtype=float)
        lv = legacy_df[col].to_numpy(dtype=float)
        both_nan = np.isnan(ev) & np.isnan(lv)
        diff = np.where(both_nan, 0.0, np.abs(ev - lv))
        assert np.nanmax(diff) < 1e-9, f"{date_str}: column {col} max abs diff {np.nanmax(diff)}"


@requires_databento_store
@pytest.mark.parametrize("date_str", SAMPLE_DAYS)
def test_engine_trade_path_cutover(date_str, cfg, carry):
    """NEW trade-path behavior (Parts 1+2): production engine on TRADE bars.

    Asserts the cutover reality (NOT a book-mid match):
      * The engine runs end-to-end in its PRODUCTION default mode on TRADE bars
        (price_source="trade", tick_size=0.25, honest_entry=True) and emits the
        standard dataset schema.
      * The 3 interaction features are present (trade-print sourced).
      * Honest decision-time labeling: every kept touch's decision_time
        (event_ts + DECISION_OFFSET_MINUTES) is BEFORE the flatten (v3: 16:40 ET) and
        before the forward cutoff (17:00 ET) — the executor no-entry rule is honored
        (no kept touch could have a decision at/after the flatten).
      * Classes/thresholds unchanged: labels are within the canonical 3-class set.
    """
    from alpha_lab.agents.data_infra.ml import dashboard_utility_builder as B
    from alpha_lab.agents.data_infra.ml import engine_decision as E

    prev_ny, prev_asia, prev_london = carry[date_str]

    # Production trade bars (the cutover default of _build_bars_for_date).
    bars = B._build_bars_for_date(DATA_DIR, SYMBOL, date_str, cfg)
    assert not bars.empty, f"{date_str}: no trade bars"
    bars_et = B._ensure_et_index(bars.copy())
    levels = B._compute_levels_for_date(bars_et, date_str, prev_ny, prev_asia, prev_london)

    df = E.process_single_date_engine(
        bars_et,
        levels,
        date_str,
        DATA_DIR,
        SYMBOL,
        cfg,
    )  # production defaults: trade / 0.25 / honest_entry=True
    if df.empty:
        pytest.skip(f"{date_str}: no tradeable touches under the honest path")

    for f in ["int_time_beyond_level", "int_time_within_2pts", "int_absorption_ratio"]:
        assert f in df.columns, f"{date_str}: missing trade-print feature {f}"

    valid_labels = {"tradeable_reversal", "trap_reversal", "aggressive_blowthrough"}
    assert set(df["label"].unique()) <= valid_labels, (
        f"{date_str}: unexpected labels {set(df['label'].unique())}"
    )

    # Executor no-entry rule (engine v3): decision_time = event_ts + 5m must be
    # < 16:40 ET (flatten) and < 17:00 ET (forward cutoff) for every KEPT touch
    # (otherwise it should have been dropped). Single-sourced from the engine constants.
    from strategy_core.constants import FLATTEN_TIME, RTH_END

    for ev in pd.to_datetime(df["event_ts"]):
        ev_et = ev.tz_convert(_ET) if ev.tz is not None else ev.tz_localize(_ET)
        decision = ev_et + pd.Timedelta(minutes=E.DECISION_OFFSET_MINUTES)
        assert decision.time() < FLATTEN_TIME, (
            f"{date_str}: kept touch decision {decision} >= flatten {FLATTEN_TIME} ET"
        )
        assert decision.time() < RTH_END, (
            f"{date_str}: kept touch decision {decision} >= forward cutoff {RTH_END} ET"
        )
