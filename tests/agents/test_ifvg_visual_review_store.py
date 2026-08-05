"""Tests for the ``ifvg_visual_review_v1`` human audit ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.visual_review_store import (
    VISUAL_REVIEW_LEDGER,
    VisualReviewError,
    append_review,
    export_csv,
    list_reviews,
)

_PAIR_REF = {
    "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
    "v2_dataset_id": "a" * 64,
    "v2_manifest_hash": "b" * 64,
    "v3_dataset_id": "c" * 64,
    "v3_manifest_hash": "d" * 64,
}


def _append(tmp_path: Path, **overrides):
    payload = {
        "repo_root": tmp_path,
        "replay_chart_artifact_id": "e" * 64,
        "pair_ref": _PAIR_REF,
        "candidate_id": "cand-1",
        "decision_id": None,
        "trade_id": None,
        "reviewer": "gonzalo",
        "verdicts": {"overall_verdict": "correct"},
        "tags": ["chart_matches_strategy"],
        "notes": "looks right",
    }
    payload.update(overrides)
    return append_review(**payload)


def test_append_and_list_roundtrip(tmp_path: Path) -> None:
    record = _append(tmp_path)
    assert record["overall_verdict"] == "correct"
    frame = list_reviews(repo_root=tmp_path)
    assert len(frame) == 1
    assert frame.iloc[0]["candidate_id"] == "cand-1"
    assert frame.iloc[0]["tags"] == ["chart_matches_strategy"]
    filtered = list_reviews(repo_root=tmp_path, candidate_id="cand-2")
    assert filtered.empty


def test_ledger_is_append_only_and_isolated(tmp_path: Path) -> None:
    """Writes land ONLY under data/ifvg_visual_review — nothing else changes."""
    decoy = tmp_path / "data" / "ifvg_datasets" / "v2" / ("f" * 64)
    decoy.mkdir(parents=True)
    (decoy / "manifest.json").write_text("{}", encoding="utf-8")
    before = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    _append(tmp_path)
    _append(tmp_path, candidate_id="cand-2")
    after = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    created = set(after) - set(before)
    assert created == {tmp_path / VISUAL_REVIEW_LEDGER}
    assert (decoy / "manifest.json").read_text(encoding="utf-8") == "{}"
    lines = (tmp_path / VISUAL_REVIEW_LEDGER).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_invalid_records_are_refused(tmp_path: Path) -> None:
    with pytest.raises(VisualReviewError, match="reviewer"):
        _append(tmp_path, reviewer="  ")
    with pytest.raises(VisualReviewError, match="verdict"):
        _append(tmp_path, verdicts={"overall_verdict": "great"})
    with pytest.raises(VisualReviewError, match="overall"):
        _append(tmp_path, verdicts={"entry_verdict": "correct"})
    with pytest.raises(VisualReviewError, match="tags"):
        _append(tmp_path, tags=["nice_trade"])
    assert list_reviews(repo_root=tmp_path).empty


def test_export_csv(tmp_path: Path) -> None:
    assert export_csv(repo_root=tmp_path) == ""
    _append(tmp_path)
    payload = export_csv(repo_root=tmp_path)
    assert "candidate_id" in payload.splitlines()[0]
    assert "cand-1" in payload
