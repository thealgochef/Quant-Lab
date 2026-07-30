"""Tests for the IFVG Lab tab's PURE payload builders (``scripts/ifvg_lab_charts``).

Synthetic frames only — no streamlit runtime, no real store reads (the loader
tests write tiny parquets into tmp_path). One optional AppTest smoke test
renders the tab entry-point with a stubbed engine and is skipped when
``streamlit.testing.v1`` is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_lab_charts as charts  # noqa: E402

UTC = "UTC"


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=UTC)


_CAPTURE_DEFAULTS = {
    "kind": None,
    "envelope_setup_id": None,
    "envelope_ts_utc": None,
    "fvg_fvg_id": None,
    "fvg_gap_low_ticks": None,
    "fvg_gap_high_ticks": None,
    "fvg_direction": None,
    "fvg_timeframe_seconds": None,
    "fvg_confirmed_ts_utc": None,
    "selected": None,
    "drop_reason": None,
    "entry_fvg_fvg_id": None,
    "entry_fvg_gap_low_ticks": None,
    "entry_fvg_gap_high_ticks": None,
    "entry_fvg_direction": None,
    "entry_fvg_timeframe_seconds": None,
    "entry_fvg_confirmed_ts_utc": None,
    "entry_family": None,
    "entry_ticks": None,
    "stop_ticks": None,
    "tp_ticks": None,
    "entry_ts_utc": None,
    "resolution": None,
    "mfe_ticks": None,
    "mae_ticks": None,
    "bars_in_trade": None,
    "tap_ts_utc": None,
    "parent_confirmed_ts_utc": None,
    "lock_ts_utc": None,
    "armed_ts_utc": None,
    "inversion_ts_utc": None,
    "penetration_ticks": None,
    "ce_reached": None,
    "conflicted": None,
    "nearest_level_kind": None,
    "sweep_sweep_confirmed": None,
    "sweep_swept_kinds": None,
    "sweep_max_penetration_ticks": None,
    "close_through_margin_ticks": None,
}


def _cap(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_CAPTURE_DEFAULTS, **row} for row in rows])


# ── zone dedup + role/style mapping ───────────────────────────────────────────


def test_dedup_zones_roles_and_last_state() -> None:
    t0, t1, t2 = _ts("2026-01-13 04:00"), _ts("2026-01-13 04:05"), _ts("2026-01-13 04:10")
    capture = _cap(
        [
            # HTF gap tapped twice -> one zone, role htf, selected defaults True.
            {"kind": "htf_tap", "envelope_setup_id": "s1", "envelope_ts_utc": t0,
             "fvg_fvg_id": "H1", "fvg_gap_low_ticks": 100, "fvg_gap_high_ticks": 110,
             "fvg_direction": "bullish", "fvg_timeframe_seconds": 3600,
             "fvg_confirmed_ts_utc": t0},
            {"kind": "htf_tap", "envelope_setup_id": "s1", "envelope_ts_utc": t1,
             "fvg_fvg_id": "H1", "fvg_gap_low_ticks": 100, "fvg_gap_high_ticks": 110,
             "fvg_direction": "bullish", "fvg_timeframe_seconds": 3600,
             "fvg_confirmed_ts_utc": t0},
            # Parent gap: selected then replaced -> LAST state wins.
            {"kind": "parent_candidate", "envelope_setup_id": "s1", "envelope_ts_utc": t0,
             "fvg_fvg_id": "P1", "fvg_gap_low_ticks": 120, "fvg_gap_high_ticks": 126,
             "fvg_direction": "bullish", "fvg_timeframe_seconds": 300,
             "fvg_confirmed_ts_utc": t0, "selected": True},
            {"kind": "parent_candidate", "envelope_setup_id": "s1", "envelope_ts_utc": t1,
             "fvg_fvg_id": "P1", "fvg_gap_low_ticks": 120, "fvg_gap_high_ticks": 126,
             "fvg_direction": "bullish", "fvg_timeframe_seconds": 300,
             "fvg_confirmed_ts_utc": t0, "selected": False,
             "drop_reason": "replaced_by_newer"},
            # Opposing gap armed.
            {"kind": "opposing", "envelope_setup_id": "s1", "envelope_ts_utc": t1,
             "fvg_fvg_id": "O1", "fvg_gap_low_ticks": 130, "fvg_gap_high_ticks": 134,
             "fvg_direction": "bearish", "fvg_timeframe_seconds": 60,
             "fvg_confirmed_ts_utc": t1, "selected": True},
            # Entry candidate carrying the fresh entry FVG.
            {"kind": "entry_candidate", "envelope_setup_id": "s1", "envelope_ts_utc": t2,
             "entry_fvg_fvg_id": "E1", "entry_fvg_gap_low_ticks": 140,
             "entry_fvg_gap_high_ticks": 145, "entry_fvg_direction": "bullish",
             "entry_fvg_timeframe_seconds": 60, "entry_fvg_confirmed_ts_utc": t2,
             "selected": True, "entry_family": "fresh_fvg_continuation",
             "entry_ticks": 146, "stop_ticks": 139},
        ]
    )
    zones = charts.dedup_zones(capture)
    assert len(zones) == 4
    by_id = zones.set_index("fvg_id")
    assert by_id.loc["H1", "role"] == "htf"
    assert bool(by_id.loc["H1", "selected"]) is True  # no explicit flag -> True
    assert by_id.loc["P1", "role"] == "parent"
    assert bool(by_id.loc["P1", "selected"]) is False  # replaced (last state)
    assert by_id.loc["P1", "drop_reason"] == "replaced_by_newer"
    assert by_id.loc["O1", "role"] == "opposing"
    assert by_id.loc["E1", "role"] == "entry_fvg"


def test_zone_style_mapping_is_distinct() -> None:
    selected = charts.zone_style("parent", True)
    replaced = charts.zone_style("parent", False)
    recomputed = charts.zone_style("parent", True, recomputed=True)
    assert selected != replaced
    assert selected["line"]["dash"] == "solid"
    assert replaced["line"]["dash"] == "dash"
    assert recomputed["line"]["dash"] == "dot"  # visually distinct audit overlay
    # role colors differ
    assert charts.zone_style("htf", True) != charts.zone_style("opposing", True)


# ── level series honoring available_from ──────────────────────────────────────


def test_level_series_availability_clipping() -> None:
    t0, t1, t2 = _ts("2026-01-13 09:01"), _ts("2026-01-13 09:02"), _ts("2026-01-13 09:03")
    levels = pd.DataFrame(
        {
            "close_ts_utc": [t0, t1, t2, t0, t1, t2],
            "name": ["ny_high"] * 3 + ["pdh"] * 3,
            "price": [100.0, 100.0, 101.0, 90.0, 90.0, 90.0],
            "side": ["high"] * 6,
            # ny_high only becomes available at t1; pdh has no restriction.
            "available_from": [t1, t1, t1, pd.NaT, pd.NaT, pd.NaT],
        }
    )
    series = charts.level_series(levels)
    ny = series[series["name"] == "ny_high"].sort_values("close_ts_utc")
    assert pd.isna(ny["price"].iloc[0])  # before available_from -> masked
    assert ny["price"].iloc[1] == 100.0
    assert ny["price"].iloc[2] == 101.0
    pdh = series[series["name"] == "pdh"]
    assert pdh["price"].notna().all()  # NaT availability -> always drawn


# ── stage markers ─────────────────────────────────────────────────────────────


def test_stage_markers_order_and_annotations() -> None:
    base = "2026-01-13 "
    capture = _cap(
        [
            {"kind": "htf_tap", "envelope_setup_id": "s1",
             "envelope_ts_utc": _ts(base + "03:41"), "fvg_fvg_id": "H1",
             "penetration_ticks": 9.0, "ce_reached": False, "conflicted": False,
             "nearest_level_kind": "pdh"},
            {"kind": "inversion", "envelope_setup_id": "s1",
             "envelope_ts_utc": _ts(base + "04:29"),
             "sweep_sweep_confirmed": True,
             "sweep_swept_kinds": '["swing_low", "swing_low"]',
             "sweep_max_penetration_ticks": 350.0, "close_through_margin_ticks": 4.0},
            {"kind": "resolution", "envelope_setup_id": "s1",
             "envelope_ts_utc": _ts(base + "04:41"), "resolution": "resolved_sl",
             "tap_ts_utc": _ts(base + "03:41"),
             "parent_confirmed_ts_utc": _ts(base + "04:09"),
             "lock_ts_utc": _ts(base + "04:14"),
             "armed_ts_utc": _ts(base + "04:25"),
             "inversion_ts_utc": _ts(base + "04:29"),
             "entry_ts_utc": _ts(base + "04:30")},
        ]
    )
    markers = charts.stage_markers(capture, "s1")
    stages = [m["stage"] for m in markers]
    assert stages == ["tap", "parent", "lock", "armed", "inversion", "entry", "resolution"]
    ts_order = [m["ts"] for m in markers]
    assert ts_order == sorted(ts_order)
    by_stage = {m["stage"]: m["text"] for m in markers}
    assert "penetration 9t" in by_stage["tap"]
    assert "swing_low" in by_stage["inversion"]
    assert "CONFIRMED" in by_stage["inversion"]
    assert "350" in by_stage["inversion"]
    assert by_stage["resolution"] == "resolved_sl"
    assert charts.stage_markers(capture, "missing") == []


# ── trade overlay math ────────────────────────────────────────────────────────


def _entry_capture(long: bool) -> pd.DataFrame:
    t_entry, t_res = _ts("2026-01-13 04:30"), _ts("2026-01-13 04:41")
    entry, stop = (400, 380) if long else (400, 420)
    return _cap(
        [
            {"kind": "entry_candidate", "envelope_setup_id": "s1",
             "envelope_ts_utc": t_entry, "entry_family": "fresh_fvg_continuation",
             "selected": True, "entry_ticks": entry, "stop_ticks": stop},
            {"kind": "entry_candidate", "envelope_setup_id": "s1",
             "envelope_ts_utc": t_entry, "entry_family": "ifvg_retest",
             "selected": False, "drop_reason": "already_in_trade",
             "entry_ticks": entry, "stop_ticks": stop},
            {"kind": "resolution", "envelope_setup_id": "s1",
             "envelope_ts_utc": t_res, "entry_family": "fresh_fvg_continuation",
             "entry_ts_utc": t_entry, "resolution": "resolved_sl",
             "mfe_ticks": 40.0, "mae_ticks": 8.0, "bars_in_trade": 11},
        ]
    )


def test_trade_overlay_math_long() -> None:
    overlays = charts.trade_overlays(_entry_capture(long=True))
    assert len(overlays) == 2
    sel = next(o for o in overlays if o["selected"])
    assert sel["direction"] == "LONG"
    assert sel["entry_price"] == 100.0
    assert sel["stop_price"] == 95.0
    assert sel["risk_ticks"] == 20.0
    # TP lines at entry + r * risk.
    assert sel["tp_prices"] == {1.0: 105.0, 1.5: 107.5, 2.0: 110.0}
    # Manipulation swing = logged stop + 1 tick (long).
    assert sel["swing_price"] == pytest.approx(381 * 0.25)
    # MFE above / MAE below for a long.
    assert sel["mfe_price"] == pytest.approx(100.0 + 40 * 0.25)
    assert sel["mae_price"] == pytest.approx(100.0 - 8 * 0.25)
    assert sel["resolution"] == "resolved_sl"
    assert sel["bars_in_trade"] == 11
    dropped = next(o for o in overlays if not o["selected"])
    assert dropped["drop_reason"] == "already_in_trade"
    assert dropped["resolution"] is None  # outcome attaches only to the executed family


def test_trade_overlay_math_short() -> None:
    overlays = charts.trade_overlays(_entry_capture(long=False))
    sel = next(o for o in overlays if o["selected"])
    assert sel["direction"] == "SHORT"
    assert sel["stop_price"] == 105.0
    assert sel["tp_prices"] == {1.0: 95.0, 1.5: 92.5, 2.0: 90.0}
    # Swing high = logged stop - 1 tick (short).
    assert sel["swing_price"] == pytest.approx(419 * 0.25)
    # MFE below / MAE above for a short.
    assert sel["mfe_price"] == pytest.approx(100.0 - 40 * 0.25)
    assert sel["mae_price"] == pytest.approx(100.0 + 8 * 0.25)


# ── pre-seal day filtering (in the LOADER, not the widget) ────────────────────


def _write_store(root: Path, days: list[str], atag: str, ctag: str) -> None:
    frame = pd.DataFrame({"x": [1]})
    for day in days:
        day_dir = root / day
        day_dir.mkdir(parents=True)
        frame.to_parquet(day_dir / f"ifvg_tbars_{atag}.parquet")
        frame.to_parquet(day_dir / f"ifvg_levels_{atag}.parquet")
        frame.to_parquet(day_dir / f"ifvg_capture_{ctag}.parquet")


def test_replay_day_listing_excludes_sealed(tmp_path: Path) -> None:
    days = ["2026-06-10", "2026-06-11", "2026-06-12", "2026-07-01"]
    _write_store(tmp_path, days, "AT", "CT")
    # A day missing an artifact never lists.
    (tmp_path / "2026-06-09").mkdir()
    assert charts.list_replay_days(tmp_path, "AT", "CT") == ["2026-06-10", "2026-06-11"]
    assert charts.list_replay_days(tmp_path, "AT", "CT", sealed=True) == [
        "2026-06-12",
        "2026-07-01",
    ]


def test_loader_rejects_sealed_days(tmp_path: Path) -> None:
    _write_store(tmp_path, ["2026-06-11", "2026-06-12"], "AT", "CT")
    payload = charts.load_day_payload(tmp_path, "AT", "CT", "2026-06-11")
    assert payload["day"] == "2026-06-11"
    with pytest.raises(ValueError, match="sealed"):
        charts.load_day_payload(tmp_path, "AT", "CT", "2026-06-12")
    # The explicit ledger-gated path may load it.
    sealed = charts.load_day_payload(tmp_path, "AT", "CT", "2026-06-12", allow_sealed=True)
    assert sealed["day"] == "2026-06-12"


# ── run-scoped trade matching ─────────────────────────────────────────────────


def test_run_scoped_trade_matching() -> None:
    overlays = charts.trade_overlays(_entry_capture(long=True))
    # The saved run admitted only the fresh-family trade (iso string ts, like
    # result.json trade lists).
    run_trades = [
        {
            "setup_id": "s1",
            "entry_family": "fresh_fvg_continuation",
            "entry_ts_utc": "2026-01-13T04:30:00+00:00",
        },
        {"setup_id": "s9", "entry_family": "ifvg_retest", "entry_ts_utc": None},
    ]
    keys = charts.run_trade_keys(run_trades)
    assert len(keys) == 1  # None entry_ts rows are ignored
    scoped = charts.scope_overlays(overlays, keys)
    assert len(scoped) == 1
    assert scoped[0]["entry_family"] == "fresh_fvg_continuation"
    # No scoping -> everything drawn.
    assert charts.scope_overlays(overlays, None) == overlays
    # A different family/ts does not match.
    other = charts.run_trade_keys(
        [{"setup_id": "s1", "entry_family": "ifvg_retest",
          "entry_ts_utc": "2026-01-13T04:31:00+00:00"}]
    )
    assert charts.scope_overlays(overlays, other) == []


# ── session bands ─────────────────────────────────────────────────────────────


def test_session_bands_engine_scheme_placement() -> None:
    start = _ts("2026-01-12 23:00")  # 18:00 ET Jan 12 (trading-day open)
    end = _ts("2026-01-13 22:00")  # 17:00 ET Jan 13
    bands = {
        b["name"]: b
        for b in charts.session_bands("2026-01-13", charts.ENGINE_SESSION_WINDOWS, start, end)
    }
    # asia crosses midnight: 19:00 ET Jan 12 -> 02:45 ET Jan 13 (EST = UTC-5).
    assert bands["asia"]["start"] == _ts("2026-01-13 00:00")
    assert bands["asia"]["end"] == _ts("2026-01-13 07:45")
    assert bands["ny"]["start"] == _ts("2026-01-13 14:00")
    assert bands["ny"]["end"] == _ts("2026-01-13 22:00")


def test_session_bands_clip_to_visible_range() -> None:
    start = _ts("2026-01-12 23:00")
    cutoff = _ts("2026-01-13 01:00")  # scrub truncation before asia's end
    bands = charts.session_bands(
        "2026-01-13", charts.ENGINE_SESSION_WINDOWS, start, cutoff
    )
    names = [b["name"] for b in bands]
    assert names == ["asia"]  # london/ny not started yet -> dropped
    assert bands[0]["end"] == cutoff


# ── config flatten / diff ─────────────────────────────────────────────────────


def test_config_diff_frame_flags_only_differences() -> None:
    a = {"scoring": {"r_family": "r10", "cost_points": 0.75}, "model": None}
    b = {"scoring": {"r_family": "r15", "cost_points": 0.75}, "model": None}
    diff = charts.config_diff_frame(a, b)
    by_field = diff.set_index("field")
    assert bool(by_field.loc["scoring.r_family", "differs"]) is True
    assert bool(by_field.loc["scoring.cost_points", "differs"]) is False
    assert bool(by_field.loc["model", "differs"]) is False


# ── optional streamlit AppTest smoke ──────────────────────────────────────────


def test_tab_smoke_apptest(monkeypatch) -> None:
    apptest = pytest.importorskip("streamlit.testing.v1")
    import ifvg_lab_tab as tab

    ds = pd.DataFrame(
        {
            "trading_day": ["2026-01-13", "2026-06-12"],
            "setup_id": ["ifvg:2026-01-13:0001", "ifvg:2026-06-12:0001"],
            "entry_family": ["fresh_fvg_continuation", "ifvg_retest"],
            "selected": [True, True],
            "entry_ts_utc": [_ts("2026-01-13 04:30"), _ts("2026-06-12 04:30")],
        }
    )
    monkeypatch.setattr(tab, "list_experiments", lambda base_dir=None: [])
    monkeypatch.setattr(tab, "_cached_entry_dataset", lambda path: ds)
    monkeypatch.setattr(tab, "_cached_replay_days", lambda *a, **k: [])

    def _app() -> None:
        # AppTest re-executes this function's SOURCE in a fresh namespace, so
        # import the (already monkeypatched, process-shared) module inside.
        import ifvg_lab_tab

        ifvg_lab_tab.render_ifvg_lab_tab()

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception


# ── sealed-replay gate + global ledger count (regression net) ─────────────────


def test_sealed_replay_available_gate() -> None:
    assert charts.sealed_replay_available(None) is False
    assert charts.sealed_replay_available({"sealed_validations": []}) is False
    assert charts.sealed_replay_available({}) is False
    assert charts.sealed_replay_available({"sealed_validations": [1]}) is True
    assert charts.sealed_replay_available({"sealed_validations": [1, 2]}) is True


def test_sealed_ledger_count_is_global(tmp_path) -> None:
    from alpha_lab.agents.data_infra.ifvg.experiment import sealed_ledger_count

    assert sealed_ledger_count(tmp_path) == 0
    ledger = tmp_path / "sealed_ledger.jsonl"
    ledger.write_text(
        '{"config_hash": "aaa", "seq": 1}\n{"config_hash": "bbb", "seq": 2}\n',
        encoding="utf-8",
    )
    # Two looks across two DIFFERENT configs -> global count 2.
    assert sealed_ledger_count(tmp_path) == 2
