"""Feature-and-model setup listing does not repeat verification (repair R6).

Before the repair, listing N saved children re-verified every membership and
neutrality envelope once per child (N x M loads) and the whole listing ran
again on every Streamlit rerun. These tests pin the scoped fix: one verified
pass per listing call, identical results and error order, and one listing per
store state while the page is reused.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.search import research_runs  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search import research_subject as subject  # noqa: E402


def _family(root: Path, name: str, ids: list[str]) -> None:
    for artifact in ids:
        folder = root / name / artifact
        folder.mkdir(parents=True)
        (folder / "envelope.json").write_text("{}", encoding="utf-8")


def test_a_listing_verifies_each_envelope_once_and_keeps_error_order(monkeypatch, tmp_path):
    _family(tmp_path, "memberships", ["m1", "m2", "m3"])
    loads: list[str] = []

    def fake_load(root, store, artifact_id, cls):
        loads.append(artifact_id)
        if artifact_id == "m2":
            raise ValueError("m2 failed verification")
        return SimpleNamespace(payload=artifact_id)

    monkeypatch.setattr(subject, "load_verified_envelope", fake_load)
    cache: dict = {}
    for _child in range(5):  # five children in one listing
        seen = []
        with pytest.raises(ValueError, match="m2 failed"):
            for envelope in subject._verified_family(tmp_path, "memberships", object, cache):
                seen.append(envelope.payload)
        assert seen == ["m1"]  # the same position fails for every child
    assert loads == ["m1", "m2", "m3"]  # verified once for the whole listing
    loads.clear()
    with pytest.raises(ValueError):
        list(subject._verified_family(tmp_path, "memberships", object, None))
    assert loads == ["m1", "m2"]  # without a cache: today's lazy, fresh verified reads


def test_each_listing_owns_one_cache_shared_by_its_children(monkeypatch, tmp_path):
    _family(tmp_path, "core_replays", ["c1", "c2", "c3"])
    caches = []

    def fake_bind(root, core_id, *, family_cache=None):
        caches.append(family_cache)
        raise ValueError("unavailable in this test")

    monkeypatch.setattr(subject, "bind_research_subject", fake_bind)  # imported per call
    rows = research_runs.list_source_subjects(tmp_path)
    assert len(rows) == 3 and all("unavailable" in str(r).lower() for r in rows)
    assert all(c is caches[0] for c in caches) and isinstance(caches[0], dict)
    research_runs.list_source_subjects(tmp_path)
    assert caches[3] is not caches[0]  # never reused across listing calls


def _app():
    import ifvg_research_pipeline
    import streamlit as st

    ifvg_research_pipeline.render_research_configuration(
        st, roots=ifvg_research_pipeline._TEST_ROOTS)


def test_the_listing_runs_once_per_store_state_not_on_every_rerun(monkeypatch, tmp_path):
    import ifvg_research_pipeline as ui

    calls: list[Path] = []
    api = SimpleNamespace(list_source_subjects=lambda root: calls.append(Path(root)) or [])
    monkeypatch.setattr(ui, "_research_api", lambda: api)
    store = tmp_path / "store"
    monkeypatch.setattr(ui, "_TEST_ROOTS", {"store_root": store,
                                            "pipeline_state_root": tmp_path / "state"},
                        raising=False)
    at = AppTest.from_function(_app, default_timeout=60).run()
    at.run()
    at.run()
    assert not at.exception, at.exception
    assert len(calls) == 1
    folder = store / "core_replays" / ("c" * 64)  # a new saved child appears
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text("{}", encoding="utf-8")
    at.run()
    assert len(calls) == 2  # the changed store is listed again, never served stale


def test_the_store_signature_changes_when_an_envelope_is_replaced(tmp_path):
    """Review finding: a replaced envelope.json alone also invalidates the listing."""

    import ifvg_research_pipeline as pipeline

    store = tmp_path / "store"
    folder = store / "charters" / ("a" * 64)
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text("{}", encoding="utf-8")
    (folder / "envelope.json").write_text("{}", encoding="utf-8")
    first = pipeline._source_store_signature(store)
    (folder / "envelope.json").write_text('{"replaced": true}', encoding="utf-8")
    second = pipeline._source_store_signature(store)
    assert second != first
    (folder / "manifest.json").write_text('{"replaced": true}', encoding="utf-8")
    assert pipeline._source_store_signature(store) != second
    assert pipeline._source_store_signature(tmp_path / "other")[0] != first[0]
