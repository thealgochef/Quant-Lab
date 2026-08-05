"""AppTest coverage for the verifier tab's SETUP selection mode.

Mirrors the repo's AppTest pattern (``test_ifvg_lab_tab.py``): providers are
monkeypatched on the process-shared module, the app function re-imports the
module and renders the real section. No real artifacts on disk are required.
Candidate-mode behavior is covered by ``test_ifvg_verifier_tab.py`` and must
stay untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_verifier_tab as tab  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (  # noqa: E402
    ArtifactPairRef,
    VerifierBundleRef,
)
from alpha_lab.agents.data_infra.ifvg.setup_verifier_provider import (  # noqa: E402
    SetupEvidence,
)

_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64
_HEX_D = "d" * 64
_SETUP_LESS = "setup-candidateless-0001"
_SETUP_FULL = "setup-with-candidates-0002"


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


_START = _ts("2026-01-07T10:00:00Z")


def _entry() -> dict:
    return {
        "profile_name": "profile",
        "v2_artifact_id": _HEX_A,
        "v2_manifest_payload_sha256": _HEX_A,
        "v3_artifact_id": _HEX_B,
        "v3_manifest_payload_sha256": _HEX_B,
    }


def _bundle() -> VerifierBundleRef:
    return VerifierBundleRef(
        profile_name="profile",
        v2_dataset_id=_HEX_A,
        v2_manifest_hash=_HEX_A,
        v3_dataset_id=_HEX_B,
        v3_manifest_hash=_HEX_B,
        fsm_audit_artifact_id=_HEX_D,
        fsm_audit_manifest_hash=_HEX_D,
        replay_chart_artifact_id=_HEX_C,
        replay_chart_manifest_hash=_HEX_C,
    )


def _setups_frame() -> pd.DataFrame:
    base = {
        "activation_ts_utc": _START,
        "terminal_ts_utc": _ts("2026-01-07T11:30:00Z"),
        "display_end_ts_utc": _ts("2026-01-07T11:40:00Z"),
        "htf_tf_seconds": 14400,
        "parent_tf_seconds": 300,
        "session_doc_at_activation": "london",
        "q40_exposed": True,
        "conflict_flag": False,
        "structural_suppression_flag": False,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "setup_id": _SETUP_LESS,
                "terminal_reason": "invalidated_htf_filled",
                "phase_at_death": "S1",
                "candidate_less": True,
                "candidate_ids": "[]",
                "candidate_count": 0,
                "parentless": True,
                "parentless_interval_count": 1,
            },
            {
                **base,
                "setup_id": _SETUP_FULL,
                "terminal_reason": "slot_freed",
                "phase_at_death": "S6",
                "candidate_less": False,
                "candidate_ids": '["cand-000000000001"]',
                "candidate_count": 1,
                "parentless": False,
                "parentless_interval_count": 0,
            },
        ]
    )


def _bars_1m(minutes: int = 100) -> pd.DataFrame:
    rows = []
    for index in range(minutes):
        rows.append(
            {
                "bar_id": f"60s:2026-01-07:{index}",
                "close_ts_utc": _START + pd.Timedelta(minutes=index + 1),
                "open_ticks": 103000 + index,
                "high_ticks": 103050 + index,
                "low_ticks": 102950 + index,
                "close_ticks": 103020 + index,
                "volume": 10,
                "trade_count": 4,
            }
        )
    return pd.DataFrame(rows)


def _bars_tf() -> pd.DataFrame:
    rows = []
    for timeframe in (300, 14400):
        for index in range(max(1, (100 * 60) // timeframe)):
            open_ts = _START + pd.Timedelta(seconds=timeframe * index)
            rows.append(
                {
                    "bar_id": f"{timeframe}s:2026-01-07:{index}",
                    "timeframe_seconds": timeframe,
                    "logical_open_ts_utc": open_ts,
                    "logical_close_ts_utc": open_ts + pd.Timedelta(seconds=timeframe),
                    "open_ticks": 103000,
                    "high_ticks": 103100,
                    "low_ticks": 102900,
                    "close_ticks": 103040,
                    "volume": 100,
                    "trade_count": 40,
                }
            )
    return pd.DataFrame(rows)


def _fake_ctx() -> SimpleNamespace:
    pair_ref = ArtifactPairRef(
        profile_name="profile",
        v2_dataset_id=_HEX_A,
        v2_manifest_hash=_HEX_A,
        v3_dataset_id=_HEX_B,
        v3_manifest_hash=_HEX_B,
    )
    base = SimpleNamespace(
        pair_ref=pair_ref,
        bars_1m=_bars_1m(),
        replay=SimpleNamespace(artifact_id=_HEX_C, manifest={}, bars_tf=_bars_tf()),
    )
    return SimpleNamespace(
        base=base,
        bundle=_bundle(),
        fsm_audit=SimpleNamespace(artifact_id=_HEX_D),
        setup_ranges=_setups_frame(),
    )


def _make_evidence(setup_id: str, stage: str | None) -> SetupEvidence:
    frame = _setups_frame()
    range_row = frame.loc[frame["setup_id"] == setup_id].iloc[0].to_dict()
    hidden = 0 if stage in (None, "terminal") else 3
    events = pd.DataFrame(
        [
            {
                "event_kind": "htf_tap",
                "stage": "htf_tap",
                "ts_utc": _ts("2026-01-07T10:05:00Z"),
                "fvg_id": "14400s:2026-01-06:1",
                "selected": True,
                "drop_reason": None,
            },
            {
                "event_kind": "parent_window_opened",
                "stage": "activation",
                "ts_utc": _START,
                "fvg_id": None,
                "selected": None,
                "drop_reason": None,
            },
        ]
    )
    deaths = pd.DataFrame(
        [
            {
                "envelope_ts_utc": _ts("2026-01-07T11:30:00Z"),
                "death_ts_utc": _ts("2026-01-07T11:30:00Z"),
                "death_reason": str(range_row["terminal_reason"]),
                "phase": str(range_row["phase_at_death"]),
                "setup_terminated": True,
                "fill_depth_ticks": 30,
                "prior_reached_ticks": 103012,
                "new_reached_ticks": 103042,
                "parent_clocks": "{}",
                "remaining_window_bars_by_tf": "{}",
                "open_window_timeframes": "[]",
            }
        ]
    )
    empty = pd.DataFrame()
    return SetupEvidence(
        setup_id=setup_id,
        stage=stage,
        range_row=range_row,
        events=events,
        tap_candidates=empty,
        parent_candidates=empty,
        opposing=empty,
        lock=empty,
        inversion=empty,
        fill_events=empty,
        slot_deaths=deaths,
        window_events=empty,
        parentless_intervals=empty,
        entry_causality=empty,
        terminal=deaths.iloc[0].to_dict(),
        gating_report={
            "stage": stage,
            "hidden_events": hidden,
            "total_events": hidden + len(events),
        },
    )


@pytest.fixture()
def setup_mode_app(monkeypatch):
    """AppTest in setup mode with monkeypatched providers; returns (at, log)."""
    apptest = pytest.importorskip("streamlit.testing.v1")
    ctx = _fake_ctx()
    log: dict = {"stages": [], "reviews": []}

    monkeypatch.setattr(tab, "_discover_setup_bundle", lambda pair_ref: _bundle())
    monkeypatch.setattr(tab, "_cached_setup_context", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(
        tab, "_cached_setup_frame", lambda *args, **kwargs: _setups_frame()
    )

    def _fake_evidence(replay_chart_artifact_id, setup_id, stage, _ctx):
        log["stages"].append(stage)
        return _make_evidence(setup_id, stage)

    monkeypatch.setattr(tab, "_cached_setup_evidence", _fake_evidence)
    monkeypatch.setattr(tab, "list_reviews", lambda **kwargs: pd.DataFrame())

    def _fake_append_review(**kwargs):
        log["reviews"].append(kwargs)
        return kwargs

    monkeypatch.setattr(tab, "append_review", _fake_append_review)
    monkeypatch.setattr(tab, "_APPTEST_ENTRY", _entry(), raising=False)

    def _app() -> None:
        # AppTest re-executes this function's SOURCE in a fresh namespace, so
        # import the (already monkeypatched, process-shared) module inside.
        import ifvg_verifier_tab
        import streamlit as st

        ifvg_verifier_tab.render_verifier_section(
            st, None, ifvg_verifier_tab._APPTEST_ENTRY
        )

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    at.radio(key=f"{tab._STATE_PREFIX}selection_mode").set_value("setup").run()
    assert not at.exception
    return at, log


def test_setup_mode_renders_with_default_terminal_stage(setup_mode_app) -> None:
    at, log = setup_mode_app
    # the setup selectbox exists and defaults to the first (candidate-less) row
    picker = at.selectbox(key=tab._SETUP_KEY)
    assert picker.value == _SETUP_LESS
    # default stage is the full "terminal" view and is passed to the provider
    assert at.selectbox(key=f"{tab._STATE_PREFIX}setup_stage").value == "terminal"
    assert log["stages"][-1] == "terminal"
    # no gating warning at the terminal stage
    assert not any("stage gate" in str(w.value) for w in at.warning)
    # the ordered event log and setup summary are rendered
    assert len(at.dataframe) >= 1
    assert any("Terminal reason" in str(m.label) for m in at.metric)


def test_present_but_empty_filters_are_visible(setup_mode_app) -> None:
    at, _log = setup_mode_app
    # conflict + structural-suppression filters render even with zero rows
    assert at.checkbox(key=f"{tab._STATE_PREFIX}setup_conflict") is not None
    assert at.checkbox(key=f"{tab._STATE_PREFIX}setup_suppression") is not None
    captions = [str(c.value) for c in at.caption]
    honest = [c for c in captions if tab._PRESENT_BUT_EMPTY in c]
    assert len(honest) == 2, captions


def test_all_setup_filters_render(setup_mode_app) -> None:
    at, _log = setup_mode_app
    for key in (
        "setup_candidate_less",
        "setup_q40",
        "setup_parentless",
    ):
        assert at.selectbox(key=f"{tab._STATE_PREFIX}{key}") is not None
    for key in ("setup_terminal_reason", "setup_phase", "setup_htf_tf",
                "setup_parent_tf", "setup_session"):
        assert at.multiselect(key=f"{tab._STATE_PREFIX}{key}") is not None
    # separate activation AND display-end range bounds
    for key in (
        "setup_activation_start",
        "setup_activation_end",
        "setup_display_end_start",
        "setup_display_end_end",
    ):
        assert at.text_input(key=f"{tab._STATE_PREFIX}{key}") is not None


def test_stage_select_passes_through_and_reports_hidden_events(
    setup_mode_app,
) -> None:
    at, log = setup_mode_app
    at.selectbox(key=f"{tab._STATE_PREFIX}setup_stage").select("activation").run()
    assert not at.exception
    assert log["stages"][-1] == "activation"
    warnings = [str(w.value) for w in at.warning]
    assert any("3 of 5 events" in text and "stage gate" in text for text in warnings)


def test_candidate_less_setup_shows_nothing_model_related(setup_mode_app) -> None:
    at, _log = setup_mode_app
    assert at.selectbox(key=tab._SETUP_KEY).value == _SETUP_LESS
    texts = [str(c.value) for c in at.caption] + [str(m.value) for m in at.markdown]
    assert not any("Model tier" in text or "model tier" in text for text in texts)
    # the candidate-less warning is explicit
    assert any("Candidate-less setup" in str(w.value) for w in at.warning)


def test_candidate_full_setup_keeps_model_semantics_candidate_scoped(
    setup_mode_app,
) -> None:
    at, _log = setup_mode_app
    at.selectbox(key=tab._SETUP_KEY).set_value(_SETUP_FULL).run()
    assert not at.exception
    captions = [str(c.value) for c in at.caption]
    assert any(
        "Model tier statuses are only defined for candidates at decision time"
        in text
        for text in captions
    )
    assert not any("Candidate-less setup" in str(w.value) for w in at.warning)


def test_review_ledger_write_uses_setup_kwargs(setup_mode_app) -> None:
    at, log = setup_mode_app
    at.text_input(key=f"{tab._STATE_PREFIX}setup_review_reviewer").input("reviewer-1")
    at.button(key=f"{tab._STATE_PREFIX}setup_review_save").click().run()
    assert not at.exception
    assert log["reviews"], "append_review must be invoked"
    record = log["reviews"][-1]
    assert record["setup_id"] == _SETUP_LESS
    assert record["fsm_audit_artifact_id"] == _HEX_D
    assert record["candidate_id"] == ""
    assert record["decision_id"] is None and record["trade_id"] is None
    assert record["replay_chart_artifact_id"] == _HEX_C
    assert record["verdicts"]["overall_verdict"] in tab.REVIEW_VERDICTS
    assert record["pair_ref"] == {
        "profile_name": "profile",
        "v2_dataset_id": _HEX_A,
        "v2_manifest_hash": _HEX_A,
        "v3_dataset_id": _HEX_B,
        "v3_manifest_hash": _HEX_B,
    }


def test_candidate_mode_stays_default_and_untouched(monkeypatch) -> None:
    """The radio defaults to candidate mode; without a v1 artifact the section
    falls back exactly as before (info + build hint), never the setup path."""
    apptest = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(tab, "read_replay_chart_catalog", lambda path: {})
    monkeypatch.setattr(tab, "_APPTEST_ENTRY", _entry(), raising=False)

    def _app() -> None:
        import ifvg_verifier_tab
        import streamlit as st

        ifvg_verifier_tab.render_verifier_section(
            st, None, ifvg_verifier_tab._APPTEST_ENTRY
        )

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.radio(key=f"{tab._STATE_PREFIX}selection_mode").value == "candidate"
    assert any("No replay-chart artifact" in str(i.value) for i in at.info)
