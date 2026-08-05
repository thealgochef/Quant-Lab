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
    with pytest.raises(ValueError, match="jump kind"):
        tab.queue_jump("setup_id", "some-setup")
    with pytest.raises(ValueError, match="jump kind"):
        tab.queue_jump("nearest_time", "2026-01-07T10:00:00Z")


def test_sanitize_error_redacts_paths_and_secrets() -> None:
    message = tab._sanitize_error(
        ValueError(r"C:\Users\someone\secret.txt failed with token: abc123")
    )
    assert "someone" not in message
    assert "abc123" not in message


def test_sanitize_select_drops_stale_state() -> None:
    class _State(dict):
        pass

    class _Stub:
        session_state = _State({"key": "gone"})

    tab._sanitize_select(_Stub, "key", ("a", "b"))
    assert "key" not in _Stub.session_state
    _Stub.session_state["key"] = "a"
    tab._sanitize_select(_Stub, "key", ("a", "b"))
    assert _Stub.session_state["key"] == "a"


def test_stage_order_matches_provider() -> None:
    from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import STAGE_ORDER

    assert tuple(tab._STAGE_ORDER) == tuple(stage.value for stage in STAGE_ORDER)
