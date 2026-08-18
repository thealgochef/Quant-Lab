"""Immutable store + concurrency-safe catalog suites (TEST_MATRIX §3.1, P0-18)."""

from __future__ import annotations

import json
import threading

import pytest

from alpha_lab.agents.data_infra.ifvg.search.catalog import (
    CATALOG_EVENTS_FILENAME,
    append_catalog_event,
    read_catalog_events,
    rebuild_catalog_index,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    SearchCharterEnvelope,
    _example_charter_payload,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SearchStoreError,
    has_envelope,
    load_verified_envelope,
    save_envelope_immutable,
    save_or_reuse_envelope,
)


def _charter() -> SearchCharterEnvelope:
    return SearchCharterEnvelope.from_payload(_example_charter_payload())


def test_store_save_reload_assert_roundtrip(tmp_path) -> None:
    envelope = _charter()
    destination = save_envelope_immutable(tmp_path, "charters", envelope)
    assert destination.name == envelope.search_id
    reloaded = load_verified_envelope(
        tmp_path, "charters", envelope.search_id, SearchCharterEnvelope
    )
    assert reloaded.model_dump(mode="json") == envelope.model_dump(mode="json")


def test_store_overwrite_refusal_and_verified_reuse(tmp_path) -> None:
    envelope = _charter()
    save_envelope_immutable(tmp_path, "charters", envelope)
    with pytest.raises(FileExistsError):
        save_envelope_immutable(tmp_path, "charters", envelope)
    reused, was_reused = save_or_reuse_envelope(tmp_path, "charters", envelope)
    assert was_reused is True
    assert reused.search_id == envelope.search_id


def test_store_detects_tampered_artifacts(tmp_path) -> None:
    envelope = _charter()
    destination = save_envelope_immutable(tmp_path, "charters", envelope)
    payload_file = destination / "envelope.json"
    tampered = json.loads(payload_file.read_text(encoding="utf-8"))
    tampered["payload"]["seed"] = 999
    payload_file.write_text(json.dumps(tampered, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(SearchStoreError):
        load_verified_envelope(tmp_path, "charters", envelope.search_id, SearchCharterEnvelope)


def test_store_exact_id_resolution_never_lists(tmp_path) -> None:
    envelope = _charter()
    save_envelope_immutable(tmp_path, "charters", envelope)
    assert has_envelope(tmp_path, "charters", envelope.search_id)
    assert not has_envelope(tmp_path, "charters", "0" * 64)
    with pytest.raises(SearchStoreError, match="unknown search store"):
        has_envelope(tmp_path, "not_a_store", envelope.search_id)
    with pytest.raises(SearchStoreError):
        has_envelope(tmp_path, "charters", "../escape")
    # non-64-hex ids (incl. drive-relative / ADS-shaped keys) are refused
    for bad in ("C:evil", "abc", "A" * 64, "0" * 63, "0" * 64 + ":ads"):
        with pytest.raises(SearchStoreError):
            has_envelope(tmp_path, "charters", bad)


def test_concurrent_catalog_publishers_lose_nothing(tmp_path) -> None:
    writers = 8
    events_per_writer = 25
    errors: list[Exception] = []

    def _publish(writer: int) -> None:
        try:
            for index in range(events_per_writer):
                append_catalog_event(
                    tmp_path,
                    kind="note",
                    artifact_id=f"artifact-{writer}",
                    payload=f"note-{writer}-{index}",
                )
        except Exception as error:  # pragma: no cover - failure evidence
            errors.append(error)

    threads = [threading.Thread(target=_publish, args=(w,)) for w in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    events, torn = read_catalog_events(tmp_path)
    assert torn == 0
    assert len(events) == writers * events_per_writer
    index = rebuild_catalog_index(events)
    assert len(index["entries"]) == writers


def test_torn_final_line_is_recovered_on_rebuild(tmp_path) -> None:
    append_catalog_event(tmp_path, kind="display_name", artifact_id="a1", payload="First")
    append_catalog_event(tmp_path, kind="star", artifact_id="a1", payload=True)
    events_path = tmp_path / CATALOG_EVENTS_FILENAME
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id": "torn", "ts": 1, "kind": "note", "artifact')
    events, torn = read_catalog_events(tmp_path)
    assert torn == 1
    assert len(events) == 2
    index = rebuild_catalog_index(events)
    assert index["entries"]["a1"]["display_name"] == "First"
    assert index["entries"]["a1"]["starred"] is True
    # a malformed INTERIOR line is corruption, not recoverable
    lines = events_path.read_text(encoding="utf-8").splitlines()
    lines.insert(0, "{broken")
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="interior"):
        read_catalog_events(tmp_path)


def test_rebuild_is_deterministic_and_folds_in_order(tmp_path) -> None:
    append_catalog_event(tmp_path, kind="display_name", artifact_id="a1", payload="First")
    append_catalog_event(tmp_path, kind="display_name", artifact_id="a1", payload="Second")
    append_catalog_event(tmp_path, kind="archive", artifact_id="a2", payload=True)
    events, _ = read_catalog_events(tmp_path)
    index_a = rebuild_catalog_index(events)
    index_b = rebuild_catalog_index(events)
    assert index_a == index_b
    assert index_a["entries"]["a1"]["display_name"] == "Second"
    assert index_a["entries"]["a2"]["archived"] is True
    # rebuild against a manifest listing keeps unmatched events visible
    partial = rebuild_catalog_index(events, manifests=("a1",))
    assert partial["unmatched_artifact_ids"] == ("a2",)


def test_stale_orphan_lock_is_broken_after_threshold(tmp_path) -> None:
    import os
    import time as _time

    from alpha_lab.agents.data_infra.ifvg.search.catalog import _LOCK_FILENAME

    lock = tmp_path / _LOCK_FILENAME
    tmp_path.mkdir(exist_ok=True)
    lock.write_text("99999", encoding="utf-8")
    old = _time.time() - 3600
    os.utime(lock, (old, old))
    record = append_catalog_event(
        tmp_path,
        kind="note",
        artifact_id="a1",
        payload="recovered",
        timeout_seconds=0.2,
        stale_lock_seconds=30.0,
    )
    assert record["payload"] == "recovered"
    events, torn = read_catalog_events(tmp_path)
    assert torn == 0 and len(events) == 1
    # a FRESH lock (live holder) still times out rather than being broken
    lock.write_text("99999", encoding="utf-8")
    from alpha_lab.agents.data_infra.ifvg.search.catalog import CatalogLockTimeoutError

    with pytest.raises(CatalogLockTimeoutError):
        append_catalog_event(
            tmp_path,
            kind="note",
            artifact_id="a1",
            payload="blocked",
            timeout_seconds=0.2,
            stale_lock_seconds=3600.0,
        )


def test_catalog_never_accepts_research_metrics_kinds(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown catalog event kind"):
        append_catalog_event(tmp_path, kind="net_expectancy", artifact_id="a1", payload=1.0)  # type: ignore[arg-type]
