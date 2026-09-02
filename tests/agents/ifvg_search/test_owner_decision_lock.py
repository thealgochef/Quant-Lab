"""HARDENING-BACKEND §4.3 (F-13) — the liveness-aware owner-decision lock.

A slow LIVE holder is never reclaimed (age alone is not evidence); a dead
pid, an exited pid, and PID reuse (a different process-start token) are
reclaimed once the heartbeat timed out; another host, a malformed body and
an unassessable holder are never reclaimed; the writer re-verifies its
token before every publication and a lost lock aborts before the head
moves; release unlinks only the writer's own token.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.search import supersession_chain as chain_module
from alpha_lab.agents.data_infra.ifvg.search.owner_decision_lock import (
    OWNER_DECISION_LOCK_FILE,
    OwnerDecisionLock,
    OwnerDecisionLockError,
    current_process_start_token,
    process_liveness,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    OWNER_DECISION_STORE,
    initialize_test_namespace,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    current_supersession_head_witness,
    load_supersession_records,
    publish_supersession,
)

_OLD = "2026-01-01T00:00:00+00:00"
_DEAD_PID = 999_999_999  # no such process on any supported platform


def _lock_path(root: Path) -> Path:
    return root / OWNER_DECISION_STORE / OWNER_DECISION_LOCK_FILE


def _write_lock(
    root: Path,
    *,
    pid,
    token: str = "0" * 32,
    start_token=None,
    host: str | None = None,
    heartbeat_at: str = _OLD,
) -> Path:
    path = _lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "lock_schema_version": 1,
                "pid": pid,
                "process_start_token": start_token,
                "lock_token": token,
                "host": platform.node() if host is None else host,
                "created_at": _OLD,
                "heartbeat_at": heartbeat_at,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    initialize_test_namespace(store)
    return store


def test_liveness_verdicts_are_demonstrable_or_unknown() -> None:
    mine = current_process_start_token()
    assert process_liveness(os.getpid(), mine, platform.node()) == "alive"
    assert process_liveness(_DEAD_PID, None, platform.node()) == "dead"
    assert process_liveness(os.getpid(), None, "some-other-host") == "unknown"
    assert process_liveness(None, None, platform.node()) == "unknown"
    if mine is not None:
        # PID reuse: the recorded start token differs from the live process's
        assert process_liveness(os.getpid(), "not-this-process", platform.node()) == "dead"


def test_a_slow_live_holder_is_never_reclaimed(root: Path) -> None:
    """The heartbeat is ancient but the holder (this process) is ALIVE."""

    _write_lock(root, pid=os.getpid(), start_token=current_process_start_token())
    lock = OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0)
    with pytest.raises(OwnerDecisionLockError) as held:
        lock.acquire()
    assert held.value.reason == "lock_held_by_live_holder"
    assert _lock_path(root).exists()
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == "0" * 32


def test_a_dead_pid_is_reclaimed_only_after_the_heartbeat_timeout(root: Path) -> None:
    from datetime import UTC, datetime

    # fresh heartbeat + dead pid: not yet reclaimable (age is the first gate)
    _write_lock(root, pid=_DEAD_PID, heartbeat_at=datetime.now(UTC).isoformat())
    lock = OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=60.0)
    with pytest.raises(OwnerDecisionLockError) as fresh:
        lock.acquire()
    assert fresh.value.reason == "lock_held_by_live_holder"
    # stale heartbeat + dead pid: demonstrably dead → reclaimed
    _write_lock(root, pid=_DEAD_PID)
    with OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0) as reclaimed:
        assert reclaimed.reclaimed_from is not None
        assert reclaimed.reclaimed_from.pid == _DEAD_PID
        body = json.loads(_lock_path(root).read_text(encoding="utf-8"))
        assert body["lock_token"] == reclaimed.token and body["pid"] == os.getpid()
    assert not _lock_path(root).exists()


def test_pid_reuse_is_detected_by_the_process_start_token(root: Path) -> None:
    if current_process_start_token() is None:
        pytest.skip("the platform exposes no process-start token")
    _write_lock(root, pid=os.getpid(), start_token="a-process-that-exited")
    with OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0) as lock:
        assert lock.reclaimed_from is not None
        assert lock.reclaimed_from.process_start_token == "a-process-that-exited"


def test_other_host_and_malformed_bodies_are_never_reclaimed(root: Path) -> None:
    _write_lock(root, pid=_DEAD_PID, host="another-host")
    with pytest.raises(OwnerDecisionLockError) as other:
        OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0).acquire()
    assert other.value.reason == "lock_holder_liveness_unknown"
    path = _lock_path(root)
    path.write_text("not json at all", encoding="utf-8")
    stale = 1_600_000_000.0
    os.utime(path, (stale, stale))
    with pytest.raises(OwnerDecisionLockError) as malformed:
        OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0).acquire()
    assert malformed.value.reason == "lock_body_malformed"
    assert path.exists()


def test_heartbeat_refresh_keeps_the_token_and_release_unlinks_only_own_token(
    root: Path,
) -> None:
    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    lock.acquire()
    first = json.loads(_lock_path(root).read_text(encoding="utf-8"))
    lock.refresh()
    second = json.loads(_lock_path(root).read_text(encoding="utf-8"))
    assert second["lock_token"] == first["lock_token"] == lock.token
    assert second["created_at"] == first["created_at"]
    assert second["heartbeat_at"] >= first["heartbeat_at"]
    # a thief replaces the body: refresh/verify refuse, release leaves it alone
    _write_lock(root, pid=os.getpid(), token="f" * 32, start_token=current_process_start_token())
    with pytest.raises(OwnerDecisionLockError) as lost:
        lock.verify_held()
    assert lost.value.reason == "lock_lost"
    with pytest.raises(OwnerDecisionLockError):
        lock.refresh()
    lock.release()
    assert _lock_path(root).exists()
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == "f" * 32
    _lock_path(root).unlink()


def test_a_lost_lock_aborts_the_writer_before_the_head_moves(root: Path, monkeypatch) -> None:
    """The token is re-verified after the immutable record write and before
    the head publication: a stolen lock leaves an orphan record (no
    authority) and the head untouched."""

    before = current_supersession_head_witness(root)
    real_save = chain_module.save_or_reuse_envelope

    def _steal_then_save(*args, **kwargs):
        stored = real_save(*args, **kwargs)
        _write_lock(
            root, pid=os.getpid(), token="e" * 32, start_token=current_process_start_token()
        )
        return stored

    monkeypatch.setattr(chain_module, "save_or_reuse_envelope", _steal_then_save)
    with pytest.raises(OwnerDecisionLockError) as lost:
        publish_supersession(
            root,
            superseded_decision_id="a" * 64,
            replacement_decision_id="b" * 64,
            reason="stolen",
            effective_at=_OLD,
            owner_evidence_ref="b" * 64,
        )
    assert lost.value.reason == "lock_lost"
    assert current_supersession_head_witness(root) == before
    assert load_supersession_records(root) == ()
    # the thief's lock was not unlinked by the losing writer
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == "e" * 32


def test_crash_recovery_reclaims_a_dead_writers_lock_and_publishes(root: Path) -> None:
    _write_lock(root, pid=_DEAD_PID)
    record = publish_supersession(
        root,
        superseded_decision_id="a" * 64,
        replacement_decision_id="b" * 64,
        reason="after crash",
        effective_at=_OLD,
        owner_evidence_ref="b" * 64,
        lock=OwnerDecisionLock(root, wait_seconds=0.5, heartbeat_timeout_seconds=1.0),
    )
    assert [r.supersession_record_id for r in load_supersession_records(root)] == [
        record.supersession_record_id
    ]
    assert not _lock_path(root).exists()
