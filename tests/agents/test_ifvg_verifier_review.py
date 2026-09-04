"""UI-2 (owner Q3; plan F-06): the reviewer verdict surface of the visual
verifier — a case opens UNREVIEWED (nothing preselected, no ledger row),
nothing persists without the explicit Save Review, the owner-approved labels
map onto the preserved ledger keys, widgets never carry a verdict across
cases, and existing v1 rows render with their labels.

The review section is rendered directly over fake context / evidence rows
(the candidate-mode chart machinery is covered elsewhere).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

apptest = pytest.importorskip("streamlit.testing.v1")

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_verifier_tab as tab  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.review_vocabulary import (  # noqa: E402
    UNREVIEWED_LABEL,
    verdict_options,
)

_CANDIDATE_A = "cand-000000000001"
_CANDIDATE_B = "cand-000000000002"


def _app() -> None:
    from types import SimpleNamespace

    import ifvg_verifier_tab
    import pandas as pd
    import streamlit as st

    ctx = SimpleNamespace(
        replay=SimpleNamespace(artifact_id="c" * 64),
        pair_ref=SimpleNamespace(as_dict=lambda: {"profile_name": "profile"}),
    )
    evidence = SimpleNamespace(candidate_id=ifvg_verifier_tab._APPTEST_CANDIDATE)
    row = pd.Series({"decision_id": "dec-1", "trade_id": None})
    ifvg_verifier_tab._render_review_section(st, ctx, evidence, row)


@pytest.fixture()
def review_app(monkeypatch):
    log: dict = {"reviews": [], "listed": pd.DataFrame()}
    monkeypatch.setattr(tab, "list_reviews", lambda **kwargs: log["listed"])

    def _fake_append(**kwargs):
        log["reviews"].append(kwargs)
        return {"review_id": "r" * 32, **kwargs}

    monkeypatch.setattr(tab, "append_review", _fake_append)
    monkeypatch.setattr(tab, "export_csv", lambda **kwargs: "")
    monkeypatch.setattr(tab, "_APPTEST_CANDIDATE", _CANDIDATE_A, raising=False)
    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    return at, log


def _overall(at, candidate: str):
    return at.selectbox(key=f"{tab._STATE_PREFIX}review_overall_{candidate}")


def _save(at, candidate: str):
    return at.button(key=f"{tab._STATE_PREFIX}review_save_{candidate}")


def test_review_opens_unreviewed_and_persists_nothing_without_save(review_app) -> None:
    at, log = review_app
    overall = _overall(at, _CANDIDATE_A)
    assert overall.value == UNREVIEWED_LABEL
    assert tuple(overall.options) == verdict_options()
    assert _save(at, _CANDIDATE_A).disabled  # no verdict, no reviewer
    assert log["reviews"] == []
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption)
    assert "Unsaved" in text
    # a reviewer name and notes alone persist nothing
    at.text_input(key=f"{tab._STATE_PREFIX}review_reviewer_{_CANDIDATE_A}").input("r-1").run()
    at.text_area(key=f"{tab._STATE_PREFIX}review_notes_{_CANDIDATE_A}").input("notes").run()
    assert _save(at, _CANDIDATE_A).disabled
    assert log["reviews"] == []
    # the definitions of every verdict are visible
    captions = " ".join(str(c.value) for c in at.caption)
    assert "not yet asserting incorrectness" in captions
    assert "does not apply to this case" in captions
    _overall(at, _CANDIDATE_A).set_value("Needs investigation").run()
    save = _save(at, _CANDIDATE_A)
    assert not save.disabled
    save.click().run()
    assert not at.exception
    assert len(log["reviews"]) == 1
    record = log["reviews"][0]
    assert record["verdicts"] == {"overall_verdict": "questionable"}  # the owner-approved mapping
    assert record["candidate_id"] == _CANDIDATE_A
    assert record["reviewer"] == "r-1"
    assert record["notes"] == "notes"
    success = " ".join(str(s.value) for s in at.success)
    assert "Saved" in success


def test_detail_verdicts_and_not_applicable_map_onto_the_ledger_keys(review_app) -> None:
    at, log = review_app
    at.text_input(key=f"{tab._STATE_PREFIX}review_reviewer_{_CANDIDATE_A}").input("r-2").run()
    _overall(at, _CANDIDATE_A).set_value("Unclear").run()
    at.selectbox(key=f"{tab._STATE_PREFIX}review_entry_verdict_{_CANDIDATE_A}").set_value(
        "Not applicable"
    ).run()
    _save(at, _CANDIDATE_A).click().run()
    assert not at.exception
    record = log["reviews"][-1]
    assert record["verdicts"] == {
        "overall_verdict": "insufficient_evidence",
        "entry_verdict": "not_applicable",
    }


def test_review_widgets_never_carry_a_verdict_across_cases(review_app, monkeypatch) -> None:
    at, _log = review_app
    _overall(at, _CANDIDATE_A).set_value("Correct").run()
    monkeypatch.setattr(tab, "_APPTEST_CANDIDATE", _CANDIDATE_B, raising=False)
    at.run()
    assert not at.exception
    assert _overall(at, _CANDIDATE_B).value == UNREVIEWED_LABEL
    assert _save(at, _CANDIDATE_B).disabled


def test_existing_v1_rows_render_with_the_owner_labels(review_app) -> None:
    at, log = review_app
    log["listed"] = pd.DataFrame(
        [
            {
                "reviewed_at": "2026-09-01T00:00:00+00:00",
                "reviewer": "legacy",
                "overall_verdict": "questionable",
                "tags": ["late_entry"],
                "notes": "v1 row",
            }
        ]
    )
    at.run()
    assert not at.exception
    frames = " ".join(frame.value.to_string() for frame in at.dataframe)
    assert "Needs investigation" in frames  # the label, never a bare key on display
    assert "questionable" not in frames.replace("Needs investigation", "")
    assert _overall(at, _CANDIDATE_A).value == UNREVIEWED_LABEL  # prior rows preselect nothing
