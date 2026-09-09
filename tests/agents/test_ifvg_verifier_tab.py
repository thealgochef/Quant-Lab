"""Contract tests for the verifier tab module (no live Streamlit runtime)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_verifier_tab as tab  # noqa: E402

#: The forbidden-control inventory from the restored Lab: no verifier control
#: may offer sealed access or destructive/promoting actions.
_FORBIDDEN = ("delete", "sealed", "recapture", "promote", "unlock")

_BUTTON_LABEL = re.compile(
    r"(?:st_module|st|nav\d|first|second|pick)\.button\(\s*\n?\s*\"([^\"]+)\""
)
_SOURCE = Path(tab.__file__).read_text(encoding="utf-8")


def test_no_forbidden_control_labels() -> None:
    labels = _BUTTON_LABEL.findall(_SOURCE)
    assert labels, "expected button labels in the verifier tab source"
    for label in labels:
        lowered = label.lower()
        for token in _FORBIDDEN:
            assert token not in lowered, label


def test_no_allow_sealed_parameter_anywhere() -> None:
    assert "allow_sealed" not in _SOURCE


def test_queue_jump_accepts_only_exact_id_kinds() -> None:
    # setup_id is the fourth exact kind since ifvg_prop_robust_config_search_v1
    # R2 (funnel-delta exact-setup drill-through); time/nearest kinds remain
    # structurally impossible.
    for kind in ("candidate_id", "decision_id", "trade_id", "setup_id"):
        tab.queue_jump(kind, "some-exact-id")
        assert tab.st.session_state.pop(tab._PENDING_JUMP_KEY) == (
            kind,
            "some-exact-id",
        )
    with pytest.raises(ValueError, match="jump kind"):
        tab.queue_jump("nearest_time", "2026-01-07T10:00:00Z")
    with pytest.raises(ValueError, match="jump kind"):
        tab.queue_jump("fuzzy_geometry", "whatever")


def test_setup_jump_routes_to_setup_mode_exact_resolver() -> None:
    tab.queue_jump("setup_id", "exact-setup-id")

    class _Stub:
        pass

    tab._route_pending_setup_jump(_Stub)
    assert tab._PENDING_JUMP_KEY not in tab.st.session_state
    assert tab.st.session_state.pop(f"{tab._STATE_PREFIX}selection_mode") == "setup"
    assert (
        tab.st.session_state.pop(f"{tab._STATE_PREFIX}setup_jump")
        == "exact-setup-id"
    )
    # a queued candidate jump is untouched by the setup router
    tab.queue_jump("candidate_id", "exact-candidate")
    tab._route_pending_setup_jump(_Stub)
    assert tab.st.session_state.pop(tab._PENDING_JUMP_KEY) == (
        "candidate_id",
        "exact-candidate",
    )


def test_sanitize_error_redacts_paths_and_secrets() -> None:
    message = tab._sanitize_error(
        ValueError(r"C:\Users\someone\secret.txt failed with token: abc123")
    )
    assert "someone" not in message
    assert "abc123" not in message


def test_sanitize_select_synchronizes_stale_state() -> None:
    class _State(dict):
        pass

    class _Stub:
        session_state = _State({"key": "gone"})

    tab._sanitize_select(_Stub, "key", ("a", "b"))
    assert _Stub.session_state["key"] == "a"
    _Stub.session_state["key"] = "b"
    tab._sanitize_select(_Stub, "key", ("a", "b"))
    assert _Stub.session_state["key"] == "b"
    tab._sanitize_select(_Stub, "key", ())
    assert "key" not in _Stub.session_state


def test_stage_order_matches_provider() -> None:
    from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import STAGE_ORDER

    assert tuple(tab._STAGE_ORDER) == tuple(stage.value for stage in STAGE_ORDER)
