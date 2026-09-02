"""HARDENING-BACKEND §4.2 (F-12) — the immutable supersession-record chain.

Records are content-addressed store entries; the mandatory head commits to
the whole chain from the namespace's genesis anchor; publication is atomic
in the plan's four steps (an orphan record has no authority; a head only
ever names an already verified record); rollback, deletion, forgery and
divergent replays are refused; the witness rule refuses a missing, shorter,
or different current head.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.search import supersession_chain as chain_module
from alpha_lab.agents.data_infra.ifvg.search.store import (
    envelope_destination,
    save_envelope_immutable,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    StoreNamespaceError,
    SupersessionHeadWitness,
    chain_head_digest,
    initialize_test_namespace,
    read_supersession_head,
    supersession_head_path,
    write_supersession_head_atomic,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    OWNER_DECISION_SUPERSESSION_STORE,
    OwnerDecisionSupersessionEnvelope,
    OwnerDecisionSupersessionPayload,
    assert_head_witness_current,
    current_supersession_head_witness,
    load_supersession_records,
    publish_supersession,
    verify_chain_structure,
)

_AT = "2026-09-01T00:00:00+00:00"


def _publish(root: Path, superseded: str, replacement: str, **overrides):
    kwargs = dict(
        superseded_decision_id=superseded,
        replacement_decision_id=replacement,
        reason="test supersession",
        effective_at=_AT,
        owner_evidence_ref=replacement,
    )
    kwargs.update(overrides)
    return publish_supersession(root, **kwargs)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    initialize_test_namespace(store)
    return store


def test_records_are_immutable_entries_and_the_head_commits_to_the_chain(root: Path) -> None:
    namespace_id = current_supersession_head_witness(root).store_namespace_id
    genesis = current_supersession_head_witness(root).head_sha256
    first = _publish(root, "a" * 64, "b" * 64)
    second = _publish(root, "b" * 64, "c" * 64, effective_at="2026-09-01T01:00:00+00:00")
    # every record is a manifest-verified store entry, never a log line
    for record in (first, second):
        destination = envelope_destination(
            root, OWNER_DECISION_SUPERSESSION_STORE, record.supersession_record_id
        )
        assert (destination / "manifest.json").exists()
    assert second.payload.prior_head_record_id == first.supersession_record_id
    assert second.payload.prior_line_count == 1
    expected_head = chain_head_digest(
        chain_head_digest(genesis, first.supersession_record_id), second.supersession_record_id
    )
    head = read_supersession_head(root, store_namespace_id=namespace_id)
    assert head["line_count"] == 2 and head["head_sha256"] == expected_head
    assert head["record_id"] == second.supersession_record_id
    records = load_supersession_records(root)
    assert [(r.line_number, r.superseded_artifact_id[0], r.replacement_artifact_id[0])
            for r in records] == [(1, "a", "b"), (2, "b", "c")]
    assert records[-1].head_sha256 == expected_head
    witness = current_supersession_head_witness(root)
    assert witness == SupersessionHeadWitness(
        store_namespace_id=namespace_id, line_count=2, head_sha256=expected_head
    )
    assert_head_witness_current(root, witness)
    # no supersession log file exists anywhere under the owner-decision store
    assert not list((root / "owner_decisions").glob("*.jsonl"))


def test_identical_replay_reuses_and_a_divergent_replay_is_refused(root: Path) -> None:
    first = _publish(root, "a" * 64, "b" * 64)
    again = _publish(root, "a" * 64, "b" * 64)
    assert again.supersession_record_id == first.supersession_record_id
    assert current_supersession_head_witness(root).line_count == 1
    with pytest.raises(StoreNamespaceError) as divergent:
        _publish(root, "a" * 64, "b" * 64, reason="a different reason")
    assert divergent.value.reason == "supersession_divergent_replay"
    assert current_supersession_head_witness(root).line_count == 1


def test_an_orphan_record_has_no_authority(root: Path) -> None:
    """A record published without the head advancing (a failure before step
    4) is invisible to the chain and to every witness."""

    before = current_supersession_head_witness(root)
    payload = OwnerDecisionSupersessionPayload(
        store_namespace_id=before.store_namespace_id,
        superseded_decision_id="a" * 64,
        replacement_decision_id="b" * 64,
        prior_head_record_id=None,
        prior_head_sha256=before.head_sha256,
        prior_line_count=0,
        reason="orphan",
        effective_at=_AT,
        owner_evidence_ref="b" * 64,
    )
    orphan = OwnerDecisionSupersessionEnvelope.from_payload(payload)
    save_envelope_immutable(root, OWNER_DECISION_SUPERSESSION_STORE, orphan)
    assert load_supersession_records(root) == ()
    assert current_supersession_head_witness(root) == before
    assert_head_witness_current(root, before)


def test_crash_before_the_head_update_is_repaired_by_an_idempotent_retry(
    root: Path, monkeypatch
) -> None:
    before = current_supersession_head_witness(root)
    calls = {"n": 0}
    real_write = chain_module.write_supersession_head_atomic

    def _crash_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash after the record published, before the head")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(chain_module, "write_supersession_head_atomic", _crash_once)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _publish(root, "a" * 64, "b" * 64)
    # the record exists (immutable, verified) but carries no authority yet
    entries = [p for p in (root / OWNER_DECISION_SUPERSESSION_STORE).iterdir() if p.is_dir()]
    assert len(entries) == 1
    assert current_supersession_head_witness(root) == before
    # the retry reuses the identical record and advances the head — one record
    retried = _publish(root, "a" * 64, "b" * 64)
    assert retried.supersession_record_id == entries[0].name
    assert current_supersession_head_witness(root).line_count == 1
    entries = [p for p in (root / OWNER_DECISION_SUPERSESSION_STORE).iterdir() if p.is_dir()]
    assert len(entries) == 1
    # the lock never survived either attempt
    assert not (root / "owner_decisions" / "SUPERSESSIONS.lock").exists()


def test_rollback_deletion_and_forgery_are_refused(root: Path) -> None:
    namespace_id = current_supersession_head_witness(root).store_namespace_id
    first = _publish(root, "a" * 64, "b" * 64)
    second = _publish(root, "b" * 64, "c" * 64, effective_at="2026-09-01T01:00:00+00:00")
    witness = current_supersession_head_witness(root)
    head_path = supersession_head_path(root)
    saved = head_path.read_text(encoding="utf-8")
    # (i) rolling the head back to line 1: shorter than the witnessed head
    write_supersession_head_atomic(
        root,
        store_namespace_id=namespace_id,
        record_id=first.supersession_record_id,
        line_count=1,
        head_sha256=chain_head_digest(first.payload.prior_head_sha256,
                                      first.supersession_record_id),
    )
    assert current_supersession_head_witness(root).line_count == 1  # structurally valid...
    with pytest.raises(StoreNamespaceError) as shorter:
        assert_head_witness_current(root, witness)  # ...but refused by the witness
    assert shorter.value.reason == "supersession_head_shorter_than_witness"
    # (ii) the head deleted: corruption, never "no supersessions"
    head_path.unlink()
    with pytest.raises(StoreNamespaceError) as missing:
        load_supersession_records(root)
    assert missing.value.reason == "supersession_head_missing"
    # (iii) a forged head digest breaks the chain
    write_supersession_head_atomic(
        root,
        store_namespace_id=namespace_id,
        record_id=second.supersession_record_id,
        line_count=2,
        head_sha256="9" * 64,
    )
    with pytest.raises(StoreNamespaceError) as forged:
        verify_chain_structure(root)
    assert forged.value.reason == "supersession_chain_broken"
    # (iv) a head naming a record that does not exist
    write_supersession_head_atomic(
        root, store_namespace_id=namespace_id, record_id="8" * 64, line_count=3,
        head_sha256="7" * 64,
    )
    with pytest.raises(StoreNamespaceError) as unverifiable:
        verify_chain_structure(root)
    assert unverifiable.value.reason == "supersession_record_unverifiable"
    # (v) a rewritten record fails the store manifest verification
    head_path.write_text(saved, encoding="utf-8")
    assert_head_witness_current(root, witness)
    destination = envelope_destination(
        root, OWNER_DECISION_SUPERSESSION_STORE, second.supersession_record_id
    )
    envelope_file = destination / "envelope.json"
    document = json.loads(envelope_file.read_text(encoding="utf-8"))
    document["payload"]["reason"] = "rewritten"
    envelope_file.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(StoreNamespaceError) as rewritten:
        load_supersession_records(root)
    assert rewritten.value.reason == "supersession_record_unverifiable"
    # (vi) a head that belongs to another namespace
    envelope_file.write_text(
        json.dumps(second.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_supersession_head_atomic(
        root, store_namespace_id="6" * 64, record_id=second.supersession_record_id,
        line_count=2, head_sha256=witness.head_sha256,
    )
    with pytest.raises(StoreNamespaceError) as other:
        current_supersession_head_witness(root)
    assert other.value.reason == "supersession_head_namespace_mismatch"


def test_witness_refuses_missing_shorter_or_different_heads(root: Path, tmp_path: Path) -> None:
    genesis_witness = current_supersession_head_witness(root)
    assert_head_witness_current(root, genesis_witness)
    _publish(root, "a" * 64, "b" * 64)
    current = current_supersession_head_witness(root)
    # a stale (older) witness differs from the current head
    with pytest.raises(StoreNamespaceError) as stale:
        assert_head_witness_current(root, genesis_witness)
    assert stale.value.reason == "supersession_head_witness_mismatch"
    # a witness of another namespace
    other = tmp_path / "other"
    initialize_test_namespace(other)
    with pytest.raises(StoreNamespaceError) as foreign:
        assert_head_witness_current(other, current)
    assert foreign.value.reason == "supersession_head_witness_mismatch"
    # a longer witness than the current head (the store was rolled back)
    longer = current.model_copy(update={"line_count": 5})
    with pytest.raises(StoreNamespaceError) as shorter:
        assert_head_witness_current(root, longer)
    assert shorter.value.reason == "supersession_head_shorter_than_witness"
    # not a witness at all
    with pytest.raises(StoreNamespaceError):
        assert_head_witness_current(root, None)  # type: ignore[arg-type]
    # the store relocated: the witness still verifies (authority is semantic)
    import shutil

    moved = tmp_path / "relocated" / "search_test" / "v1"
    shutil.copytree(root, moved)
    assert_head_witness_current(moved, current)


def test_the_transition_verifier_runs_inside_the_lock_before_any_write(root: Path) -> None:
    seen: list = []

    def _verify(namespace):
        seen.append(namespace.store_namespace_id)
        raise PermissionError("transition refused by the caller")

    with pytest.raises(PermissionError, match="transition refused"):
        _publish(root, "a" * 64, "b" * 64, verify_transition=_verify)
    assert seen == [current_supersession_head_witness(root).store_namespace_id]
    assert load_supersession_records(root) == ()
    assert not list((root / OWNER_DECISION_SUPERSESSION_STORE).iterdir()) if (
        root / OWNER_DECISION_SUPERSESSION_STORE
    ).exists() else True
    assert not (root / "owner_decisions" / "SUPERSESSIONS.lock").exists()
