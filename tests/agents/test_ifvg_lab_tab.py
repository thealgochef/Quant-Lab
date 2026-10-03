"""Tests for the IFVG Lab tab's PURE payload builders (``scripts/ifvg_lab_charts``).

Synthetic frames only — no streamlit runtime, no real store reads (the loader
tests write tiny parquets into tmp_path). One optional AppTest smoke test
renders the tab entry-point with a stubbed engine and is skipped when
``streamlit.testing.v1`` is unavailable.
"""

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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


def test_tab_config_diff_is_arrow_safe_for_mixed_contract_values() -> None:
    import ifvg_lab_tab as tab

    diff = tab._config_diff(
        {"dataset": {"artifact_id": "a" * 64}, "train_days": 40},
        {"dataset": {"artifact_id": "b" * 64}, "train_days": 45},
    )

    assert diff["left"].map(type).eq(str).all()
    assert diff["right"].map(type).eq(str).all()
    assert diff["differs"].all()


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
    monkeypatch.setattr(tab, "list_context_run_catalog", lambda catalog_path=None: [])
    monkeypatch.setattr(tab, "_cached_entry_dataset", lambda path: ds)
    monkeypatch.setattr(tab, "_cached_replay_days", lambda *a, **k: [])
    monkeypatch.setattr(tab, "_ready_pair_options", lambda: {})
    monkeypatch.setattr(
        tab,
        "read_preparation_state",
        lambda _root: SimpleNamespace(
            status=SimpleNamespace(value="failed"),
            current_date="2026-03-17",
            error_code="source_unavailable",
        ),
    )

    import ifvg_lab_ui

    def _static_clickable(markup, *, key, st_module=None):
        # AppTest mocks the component registry; the redesign's clickable HTML
        # (navigation links only) is drawn static here.
        import streamlit as st

        (st_module or st).html(str(markup))
        return None

    monkeypatch.setattr(ifvg_lab_ui, "clickable", _static_clickable)

    def _app() -> None:
        # AppTest re-executes this function's SOURCE in a fresh namespace, so
        # import the (already monkeypatched, process-shared) module inside.
        import ifvg_lab_tab

        ifvg_lab_tab.render_ifvg_lab_tab()

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    assert not at.tabs
    rail = ("ifvg_lab_v1_rail_My_studies", "ifvg_lab_v1_rail_Trade_review",
            "ifvg_lab_v1_rail_New_study")
    assert {button.key for button in at.button} >= set(rail)  # the left rail, not a radio
    assert not at.code and not at.json
    # Trade review opens on funded trades; the verified-context reviewer (the
    # earlier Trade review, whose failed-preparation message is checked below)
    # is its "Verified context" source.
    at.session_state["ifvg_lab_v1_review_source_value"] = "Verified context"
    at.button(key="ifvg_lab_v1_rail_Trade_review").click().run()
    assert not at.exception
    assert any("No prepared trade evidence" in item.value for item in at.info)
    assert not at.code and not at.json
    button_labels = {button.label.lower() for button in at.button}
    assert not any(
        unsafe in label
        for label in button_labels
        for unsafe in ("delete", "sealed", "recapture", "promote")
    )


def _apptest_result(state: str) -> dict:
    candidate: dict = {
        "candidate_count": 0,
        "resolved_candidate_count": 0,
        "censored_candidate_count": 0,
        "labels": {},
        "censoring": {},
        "model": {"status": "insufficient_class_coverage"},
    }
    coverage: dict = {
        "candidate_count": 0,
        "feature_count": 0,
        "features": [],
        "m3_status": "model_eligible",
        "anchor_240m_status": "experimental_q40_open",
    }
    status = "insufficient_class_coverage"
    if state == "complete":
        status = "complete"
        candidate.update(
            {
                "candidate_count": 2,
                "resolved_candidate_count": 2,
                "labels": {"0": 1, "1": 1},
                "model": {
                    "metrics": {
                        "status": "complete",
                        "brier_score": 0.2,
                        "brier_skill_score": 0.1,
                        "log_loss": 0.5,
                        "auc": 1.0,
                        "reliability_bins": [
                            {
                                "mean_probability": 0.25,
                                "observed_rate": 0.0,
                                "count": 1,
                            },
                            {
                                "mean_probability": 0.75,
                                "observed_rate": 1.0,
                                "count": 1,
                            },
                        ],
                        "thresholds": [
                            {
                                "threshold": 0.5,
                                "coverage_count": 1,
                                "coverage_fraction": 0.5,
                                "r": {"net_r_sum": 1.0},
                            }
                        ],
                    },
                    "folds": [{"fold_index": 0, "status": "complete"}],
                    "feature_importance": [
                        {
                            "feature": "ctx_gap_count",
                            "permutation_importance_mean": 0.1,
                        }
                    ],
                },
            }
        )
    elif state == "censored":
        candidate.update(
            {
                "candidate_count": 2,
                "censored_candidate_count": 2,
                "censoring": {"development_cutoff": 2},
            }
        )
    elif state == "undefined_auc":
        status = "complete"
        candidate.update(
            {
                "candidate_count": 2,
                "resolved_candidate_count": 2,
                "labels": {"1": 2},
                "model": {
                    "metrics": {
                        "status": "complete",
                        "auc": None,
                        "auc_reason": "single_class_oos",
                    }
                },
            }
        )
    elif state == "descriptive_m3":
        status = "descriptive_only_no_positive_qualification_coverage"
        coverage.update(
            {
                "candidate_count": 2,
                "feature_count": 1,
                "m3_status": status,
                "features": [
                    {
                        "feature": "ctx_sweep_qualifying_link_count",
                        "non_null_count": 2,
                        "missing_count": 0,
                        "coverage_fraction": 1.0,
                        "constant": True,
                        "low_coverage": False,
                    }
                ],
            }
        )
    elif state == "no_valid_fold":
        candidate["model"] = {"status": "no_valid_fold"}
        status = "no_valid_fold"
    return {
        "run_id": "a" * 64,
        "status": status,
        "candidate_research_report": candidate,
        "actual_execution_report": {},
        "feature_coverage_report": coverage,
        "reconciliation_audit_report": {"passed": state != "empty"},
    }


@pytest.mark.parametrize(
    "state",
    (
        "complete",
        "empty",
        "censored",
        "no_valid_fold",
        "undefined_auc",
        "descriptive_m3",
    ),
)
def test_context_report_states_apptest(monkeypatch, state: str) -> None:
    apptest = pytest.importorskip("streamlit.testing.v1")
    import ifvg_lab_tab as tab

    monkeypatch.setattr(tab, "_APPTEST_RESULT", _apptest_result(state), raising=False)

    def _app() -> None:
        import ifvg_lab_tab
        import streamlit as st

        ifvg_lab_tab._render_result(st, ifvg_lab_tab._APPTEST_RESULT)

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    assert [item.label for item in at.tabs] == [
        "Candidate research",
        "Actual execution",
        "Feature coverage",
        "Reconciliation",
    ]


def test_tampered_history_state_apptest(monkeypatch) -> None:
    apptest = pytest.importorskip("streamlit.testing.v1")
    import ifvg_lab_tab as tab

    monkeypatch.setattr(
        tab,
        "list_context_run_catalog",
        lambda **_kwargs: [{"run_id": "a" * 64, "display_name": "Tampered"}],
    )
    monkeypatch.setattr(
        tab,
        "load_context_experiment_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("immutable run artifact was modified")
        ),
    )

    def _app() -> None:
        import ifvg_lab_tab
        import streamlit as st

        ifvg_lab_tab._run_history(st)

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Run verification failed" in item.value for item in at.error)


def test_duplicate_run_is_verified_and_reused_apptest(monkeypatch) -> None:
    apptest = pytest.importorskip("streamlit.testing.v1")
    import ifvg_lab_tab as tab

    from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
        ArtifactReference,
        PairedIfvgArtifactReference,
    )

    references = PairedIfvgArtifactReference(
        v2=ArtifactReference(
            artifact_id="1" * 64,
            manifest_payload_sha256="2" * 64,
            artifact_kind="v2",
            dataset_schema_version=2,
        ),
        v3=ArtifactReference(
            artifact_id="3" * 64,
            manifest_payload_sha256="4" * 64,
            artifact_kind="v3",
            dataset_schema_version=4,
            feature_formula_version="ifvg_context_formula_v2",
        ),
    )
    pair = SimpleNamespace(
        reference=references,
        v3=SimpleNamespace(reference=references.v3),
    )
    view = SimpleNamespace(
        frame=pd.DataFrame(
            {
                "trading_day": ["2026-01-13"],
                "ctx_sweep_qualifying_link_count": [0],
            }
        )
    )
    result_payload = _apptest_result("empty")
    result = SimpleNamespace(
        run_id="a" * 64,
        model_dump=lambda **_kwargs: result_payload,
    )
    stored = SimpleNamespace(
        result=result,
        manifest={"manifest_payload_sha256": "b" * 64},
    )
    cataloged = SimpleNamespace(
        stored_run=stored,
        run_manifest_sha256="b" * 64,
        reused_run=True,
    )
    monkeypatch.setattr(
        tab,
        "_load_selected_pair",
        lambda _st, **_kwargs: (
            pair,
            {
                "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
                "preparation_status": "context_ready",
            },
        ),
    )
    monkeypatch.setattr(tab, "build_candidate_feature_view", lambda _pair: view)
    monkeypatch.setattr(
        tab,
        "run_and_catalog_context_experiment",
        lambda *_args, **_kwargs: cataloged,
    )
    monkeypatch.setattr(
        tab,
        "load_context_experiment_run",
        lambda *_args, **_kwargs: stored,
    )
    monkeypatch.setattr(tab, "_run_history", lambda _st: None)
    monkeypatch.setattr(tab, "_legacy_read_only", lambda _st: None)
    monkeypatch.setattr(tab, "_render_result", lambda _st, _result: None)

    def _app() -> None:
        import ifvg_lab_tab

        ifvg_lab_tab.render_ifvg_experiments_tab()

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    next(button for button in at.button if button.label == "Run deterministic experiment").click()
    at.run()
    assert not at.exception
    assert any("verified and reused" in item.value for item in at.info)
    pointer = at.session_state["ifvg_context_v1_last_run"]
    assert pointer == {
        "run_id": "a" * 64,
        "manifest_payload_sha256": "b" * 64,
    }


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


# ── two-theme palette: colors resolve when a figure is built, never at import ─


_COLOR_LITERAL = re.compile(r"#[0-9A-Fa-f]{6}\b|rgba?\(")


@contextmanager
def _dark_palette():
    """The palette resolver says dark inside the block; the default (light) comes back after."""
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    theme.set_theme_resolver(lambda: "dark")
    try:
        yield theme.DARK_COLORS
    finally:
        theme.set_theme_resolver(None)


def test_funded_inline_styles_carry_no_color_literal() -> None:
    import ifvg_lab_funded as funded

    source = (_SCRIPTS / "ifvg_lab_funded.py").read_text(encoding="utf-8")
    assert not _COLOR_LITERAL.search(source)
    cards = str(funded._window_cards(SimpleNamespace(calendar=[]), None, "Take Profit Trader"))
    gates = str(funded._gate_card(SimpleNamespace(gates=[])))
    assert not _COLOR_LITERAL.search(cards)
    assert not _COLOR_LITERAL.search(gates)
    assert 'style="color:var(--lab-blue)">Selection window' in cards
    assert 'style="color:var(--lab-orange)">Unseen window' in cards
    assert ("border:1px solid var(--lab-disabled-border);background:var(--lab-disabled-bg);"
            "color:var(--lab-disabled-text)") in cards
    assert "color:var(--lab-body)" in gates
    assert "Not in export" in gates


def test_level_style_takes_the_palette_gray_and_keeps_session_colors() -> None:
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    assert charts.level_style("pdh") == {"color": theme.COLORS["muted"], "dash": "solid"}
    assert charts.level_style("prev_ny_low") == {"color": theme.COLORS["muted"], "dash": "dot"}
    assert charts.level_style("not a level") == {"color": theme.COLORS["muted"], "dash": "dot"}
    assert charts.level_style("asia_high") == {"color": "#2E9990", "dash": "dash"}
    with _dark_palette() as dark:
        assert charts.level_style("pdh")["color"] == dark["muted"]
        assert charts.level_style("asia_high")["color"] == "#2E9990"  # semantic, unchanged
    assert charts.level_style("pdh")["color"] == theme.COLORS["muted"]  # resolver restored


def test_result_figures_take_the_dark_ground_when_the_palette_says_dark() -> None:
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    section = {"calibration": [{"mean_p": 0.4, "actual": 0.5, "n": 3}]}
    light = charts.build_calibration_figure(section)
    assert light.layout.plot_bgcolor == theme.COLORS["chart_ground"]
    assert light.layout.font.color == theme.COLORS["body"]
    assert light.layout.xaxis.gridcolor == theme.COLORS["grid"]
    assert light.data[0].line.color == theme.COLORS["muted"]  # the "perfect" diagonal
    with _dark_palette() as dark:
        fig = charts.build_calibration_figure(section)
        assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
        assert fig.layout.plot_bgcolor == dark["chart_ground"]
        assert fig.layout.font.color == dark["body"]
        assert fig.layout.xaxis.gridcolor == dark["grid"]
        assert fig.layout.yaxis.gridcolor == dark["grid"]
        assert fig.data[0].line.color == dark["muted"]
        assert fig.data[1].line.color == "#4C78A8"  # the model series keeps its own color
        hist = charts.build_r_histogram_figure({"bins": [-1, 0, 1], "counts": [1, 2, 3]})
        assert hist.layout.plot_bgcolor == dark["chart_ground"]
        equity = charts.build_equity_figure(
            {"equity": {"usd": {"equity": [0.0, 1.0], "timestamps": [0, 1]}}})
        assert equity.layout.yaxis2.gridcolor == dark["grid"]  # both subplot rows
        bins = pd.DataFrame({"mean_probability": [0.25, 0.75], "observed_rate": [0.0, 1.0],
                             "count": [1, 1]})
        assert charts.build_reliability_figure(bins).data[0].line.color == dark["muted"]
        coverage = charts.build_coverage_figure([{"threshold": 0.5, "coverage_count": 1,
                                                  "coverage_fraction": 0.5, "net_r_sum": 1.0}])
        assert coverage.layout.xaxis2.gridcolor == dark["grid"]


def test_replay_figure_grounds_and_neutral_lines_follow_the_palette() -> None:
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    t0, t1 = _ts("2026-01-13 04:30"), _ts("2026-01-13 04:41")
    minute = pd.Timedelta(minutes=1)
    bars = pd.DataFrame({
        "timeframe_ticks": [60, 60], "open_ts_utc": [t0, t1],
        "close_ts_utc": [t0 + minute, t1 + minute], "open_ticks": [398, 402],
        "high_ticks": [404, 406], "low_ticks": [396, 400], "close_ticks": [402, 404],
    })
    overlays = charts.trade_overlays(_entry_capture(long=True))

    def build():
        return charts.build_replay_figure(day="2026-01-13", bars=bars, levels=pd.DataFrame(),
                                          zones=pd.DataFrame(), overlays=overlays, markers=[])

    with _dark_palette() as dark:
        fig = build()
    assert fig.layout.plot_bgcolor == dark["chart_ground"]
    assert fig.layout.font.color == dark["body"]
    assert fig.layout.xaxis.gridcolor == dark["grid"]
    by_name = {trace.name: trace for trace in fig.data}
    assert by_name["MFE"].line.color == dark["muted"]
    assert by_name["dropped candidates"].marker.color == dark["muted"]
    assert by_name["entries"].marker.color == "#2CA02C"  # the good/bad pair stays
    assert by_name["SL"].line.color == "#D62728"
    band_note = next(a for a in fig.layout.annotations if "(engine)" in (a.text or ""))
    assert band_note.font.color == dark["muted"]
    outcome = next(a for a in fig.layout.annotations if a.bgcolor)
    assert outcome.bgcolor == theme.rgba("panel", 0.75, "dark")
    light = build()
    assert light.layout.plot_bgcolor == theme.COLORS["chart_ground"]
    light_outcome = next(a for a in light.layout.annotations if a.bgcolor)
    assert light_outcome.bgcolor == "rgba(255,255,255,0.75)"
    assert light_outcome.text == outcome.text  # colors only: the same words in both themes
    assert len(light.data) == len(fig.data)
