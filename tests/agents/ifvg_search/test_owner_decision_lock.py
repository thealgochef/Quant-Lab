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
import sys
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


# ── HARDENING-BACKEND-FIX §4.1 — token-safe stale-lock reclamation ───────────


def _fresh_live_lock(root: Path, token: str) -> None:
    from datetime import UTC, datetime

    _write_lock(
        root,
        pid=os.getpid(),
        token=token,
        start_token=current_process_start_token(),
        heartbeat_at=datetime.now(UTC).isoformat(),
    )


def test_two_stale_reclaimers_cannot_delete_the_winner(root: Path) -> None:
    """HB-FIX-01 (real concurrency). Two reclaimers both observe the same
    stale dead-holder body ``S`` OUTSIDE the reclaim mutex. The first
    reclaims ``S`` under the mutex and creates its live lock ``A``; the
    second — holding only its stale verdict for ``S`` — re-reads under the
    mutex, finds ``A`` (a different body / token), and must not unlink it:
    it times out with ``lock_held_by_live_holder`` while ``A`` survives."""

    import threading

    _write_lock(root, pid=_DEAD_PID, token="5" * 32)
    both_observed = threading.Barrier(2, timeout=10)
    winner_acquired = threading.Event()
    loser_finished = threading.Event()
    outcomes: dict[str, object] = {}

    class _Reclaimer(OwnerDecisionLock):
        def __init__(self, name: str) -> None:
            super().__init__(
                root, wait_seconds=1.5, heartbeat_timeout_seconds=1.0, poll_seconds=0.01
            )
            self.name = name
            self._synced = False

        def _observe_stale(self):
            observed, reason = super()._observe_stale()
            if observed is not None and not self._synced:
                self._synced = True
                both_observed.wait()  # both now hold a stale verdict for S
                if self.name == "second":
                    # the first reclaimer replaces S with its live lock A first
                    assert winner_acquired.wait(10)
            return observed, reason

    def _run(name: str) -> None:
        lock = _Reclaimer(name)
        try:
            lock.acquire()
        except OwnerDecisionLockError as error:
            outcomes[name] = error.reason
            outcomes[name + "_reclaimed"] = lock.reclaimed_from
            loser_finished.set()
            return
        except BaseException as error:  # noqa: BLE001 — surfaced by the assertions
            outcomes[name] = repr(error)
            loser_finished.set()
            return
        outcomes[name] = "acquired"
        outcomes[name + "_token"] = lock.token
        outcomes[name + "_reclaimed"] = lock.reclaimed_from
        winner_acquired.set()
        # hold A until the second reclaimer has given up, then release
        loser_finished.wait(10)
        outcomes["body_while_held"] = json.loads(_lock_path(root).read_text(encoding="utf-8"))
        lock.release()

    threads = [threading.Thread(target=_run, args=(name,)) for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert outcomes["first"] == "acquired", outcomes
    assert outcomes["second"] == "lock_held_by_live_holder", outcomes
    # the winner reclaimed S; the loser reclaimed nothing
    assert outcomes["first_reclaimed"].lock_token == "5" * 32
    assert outcomes["second_reclaimed"] is None
    # A survived the stale verdict of the loser and was released only by its owner
    assert outcomes["body_while_held"]["lock_token"] == outcomes["first_token"]
    assert not _lock_path(root).exists()


def test_many_reclaimers_never_overlap(root: Path) -> None:
    """Real concurrency: several writers contend over a stale dead-holder
    lock; every one eventually acquires, exactly one holds at a time, and
    the reclaimed body is reported by exactly one of them."""

    import threading
    import time

    _write_lock(root, pid=_DEAD_PID, token="5" * 32)
    start = threading.Barrier(4, timeout=10)
    holders = {"now": 0, "max": 0}
    guard = threading.Lock()
    reclaimed: list[str] = []
    failures: list[str] = []

    def _run() -> None:
        start.wait()
        lock = OwnerDecisionLock(
            root, wait_seconds=10.0, heartbeat_timeout_seconds=1.0, poll_seconds=0.005
        )
        try:
            with lock:
                with guard:
                    holders["now"] += 1
                    holders["max"] = max(holders["max"], holders["now"])
                if lock.reclaimed_from is not None:
                    reclaimed.append(lock.reclaimed_from.lock_token)
                assert lock.verify_held().lock_token == lock.token
                time.sleep(0.03)
                with guard:
                    holders["now"] -= 1
        except BaseException as error:  # noqa: BLE001
            failures.append(repr(error))

    threads = [threading.Thread(target=_run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert failures == []
    assert holders["max"] == 1
    assert reclaimed == ["5" * 32]
    assert not _lock_path(root).exists()


def test_reclaimer_that_observed_old_token_cannot_unlink_new_live_token(
    root: Path, monkeypatch
) -> None:
    """A stale verdict reached before the mutex grants no authority: between
    the observation of dead holder S and the reclaim under the mutex, a LIVE
    holder A (this process, fresh heartbeat) replaces the file. A is never
    unlinked; the waiter times out against the live holder."""

    _write_lock(root, pid=_DEAD_PID, token="5" * 32)
    lock = OwnerDecisionLock(
        root, wait_seconds=0.5, heartbeat_timeout_seconds=1.0, poll_seconds=0.01
    )
    real_reclaim = lock._reclaim
    swapped = {"done": False}

    def _swap_then_reclaim(observed, *, deadline):
        if not swapped["done"]:
            swapped["done"] = True
            assert observed.lock_token == "5" * 32
            _fresh_live_lock(root, "a" * 32)
        return real_reclaim(observed, deadline=deadline)

    monkeypatch.setattr(lock, "_reclaim", _swap_then_reclaim)
    with pytest.raises(OwnerDecisionLockError) as held:
        lock.acquire()
    assert held.value.reason == "lock_held_by_live_holder"
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == "a" * 32
    assert lock.reclaimed_from is None
    # ...and a DIFFERENT dead holder that replaced S is not unlinked by the
    # stale verdict either: it is observed afresh and reclaimed lawfully on
    # the next pass (the reclaimed body is the one that was actually present)
    _write_lock(root, pid=_DEAD_PID, token="5" * 32)
    second = OwnerDecisionLock(
        root, wait_seconds=2.0, heartbeat_timeout_seconds=1.0, poll_seconds=0.01
    )
    real_second = second._reclaim
    passes = {"n": 0}

    def _replace_with_other_dead_holder(observed, *, deadline):
        passes["n"] += 1
        if passes["n"] == 1:
            _write_lock(root, pid=_DEAD_PID, token="6" * 32)
        return real_second(observed, deadline=deadline)

    monkeypatch.setattr(second, "_reclaim", _replace_with_other_dead_holder)
    with second:
        assert second.reclaimed_from is not None
        assert second.reclaimed_from.lock_token == "6" * 32
        assert passes["n"] == 2  # the first pass refused, the second reclaimed
    assert not _lock_path(root).exists()


def test_unknown_liveness_remains_non_reclaimable(root: Path, monkeypatch) -> None:
    """Another host is ``unknown`` under the mutex exactly as before; and a
    verdict that flips from ``dead`` (outside the mutex) to ``unknown``
    (re-evaluated under the mutex) reclaims nothing."""

    from alpha_lab.agents.data_infra.ifvg.search import owner_decision_lock as lock_module

    _write_lock(root, pid=_DEAD_PID, host="another-host", token="7" * 32)
    with pytest.raises(OwnerDecisionLockError) as other:
        OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0).acquire()
    assert other.value.reason == "lock_holder_liveness_unknown"
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == "7" * 32
    # the flip: the unlocked pre-check says dead, the re-evaluation under the
    # mutex says unknown → no unlink
    _write_lock(root, pid=_DEAD_PID, token="8" * 32)
    real = lock_module.process_liveness
    calls = {"n": 0}

    def _flip(pid, start_token, host):
        calls["n"] += 1
        return real(pid, start_token, host) if calls["n"] == 1 else "unknown"

    monkeypatch.setattr(lock_module, "process_liveness", _flip)
    with pytest.raises(OwnerDecisionLockError) as refused:
        OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0).acquire()
    assert refused.value.reason == "lock_holder_liveness_unknown"
    assert calls["n"] >= 2
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == "8" * 32


def test_persistent_read_error_is_typed_not_absence(root: Path, monkeypatch) -> None:
    from alpha_lab.agents.data_infra.ifvg.search import owner_decision_lock as lock_module

    _write_lock(root, pid=_DEAD_PID, token="9" * 32)
    monkeypatch.setattr(lock_module, "_FS_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(lock_module, "_FS_RETRY_DELAY_SECONDS", 0.001)
    real_read_raw = OwnerDecisionLock._read_raw
    calls = {"n": 0}

    def _always_failing(self):
        calls["n"] += 1
        raise PermissionError("sharing violation (simulated, persistent)")

    monkeypatch.setattr(OwnerDecisionLock, "_read_raw", _always_failing)
    lock = OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0)
    with pytest.raises(OwnerDecisionLockError) as failed:
        lock._read_state()
    assert failed.value.reason == "lock_read_failed"
    assert calls["n"] == 3  # retried, then typed
    with pytest.raises(OwnerDecisionLockError) as acquire_failed:
        lock.acquire()
    assert acquire_failed.value.reason == "lock_read_failed"
    # the stale lock was never treated as absent (never unlinked)
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == "9" * 32
    # a TRANSIENT failure is retried through to the body
    flaky = {"n": 0}

    def _flaky(self):
        flaky["n"] += 1
        if flaky["n"] == 1:
            raise PermissionError("sharing violation (simulated, transient)")
        return real_read_raw(self)

    monkeypatch.setattr(OwnerDecisionLock, "_read_raw", _flaky)
    state, body = lock._read_state()
    assert state == "present" and body is not None and body.lock_token == "9" * 32
    # absent and malformed remain distinguishable states
    monkeypatch.setattr(OwnerDecisionLock, "_read_raw", real_read_raw)
    _lock_path(root).write_text("not json", encoding="utf-8")
    assert lock._read_state() == ("malformed", None)
    _lock_path(root).unlink()
    assert lock._read_state() == ("absent", None)


def test_release_read_error_does_not_silently_leave_success(root: Path, monkeypatch) -> None:
    from alpha_lab.agents.data_infra.ifvg.search import owner_decision_lock as lock_module

    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    lock.acquire()
    monkeypatch.setattr(lock_module, "_FS_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(lock_module, "_FS_RETRY_DELAY_SECONDS", 0.001)
    real_read_raw = OwnerDecisionLock._read_raw
    monkeypatch.setattr(
        OwnerDecisionLock,
        "_read_raw",
        lambda self: (_ for _ in ()).throw(PermissionError("read failure (simulated)")),
    )
    with pytest.raises(OwnerDecisionLockError) as failed:
        lock.release()
    assert failed.value.reason == "lock_release_failed"
    # the lock still carries our token (nothing was removed blindly)
    monkeypatch.setattr(OwnerDecisionLock, "_read_raw", real_read_raw)
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == lock.token
    lock.release()
    assert not _lock_path(root).exists()
    # a held lock that VANISHED is typed too (never a silent success)...
    vanished = OwnerDecisionLock(root, wait_seconds=0.3)
    vanished.acquire()
    _lock_path(root).unlink()
    with pytest.raises(OwnerDecisionLockError) as gone:
        vanished.release()
    assert gone.value.reason == "lock_release_failed"
    # ...and so is a held lock whose body became unreadable/malformed
    malformed = OwnerDecisionLock(root, wait_seconds=0.3)
    malformed.acquire()
    _lock_path(root).write_text("{", encoding="utf-8")
    with pytest.raises(OwnerDecisionLockError) as bad:
        malformed.release()
    assert bad.value.reason == "lock_release_failed"
    assert _lock_path(root).exists()  # a body we cannot verify is never removed
    _lock_path(root).unlink()


def test_failed_body_write_cleans_partial_exclusive_lock(root: Path, monkeypatch) -> None:
    from alpha_lab.agents.data_infra.ifvg.search import owner_decision_lock as lock_module

    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    monkeypatch.setattr(
        lock_module.os,
        "fsync",
        lambda _handle: (_ for _ in ()).throw(OSError("disk full (simulated)")),
    )
    with pytest.raises(OwnerDecisionLockError) as failed:
        lock.acquire()
    assert failed.value.reason == "lock_create_failed"
    assert "removed" in str(failed.value)
    # no partial (token-less) lock file and no quarantine file survive
    assert not _lock_path(root).exists()
    assert not list((root / OWNER_DECISION_STORE).glob(".SUPERSESSIONS.lock.partial-*"))
    monkeypatch.undo()
    with OwnerDecisionLock(root, wait_seconds=0.3) as again:
        assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == again.token
    assert not _lock_path(root).exists()


# ── HARDENING-BACKEND-FIX focused review round (RA-01, RA-02, RA-04) ─────────


def test_release_of_a_held_lock_replaced_by_another_writer_is_typed(root: Path) -> None:
    """Review RA-01: a lock this writer HELD that now carries another writer's
    token is the one state that means a publication may have overlapped — a
    typed ``lock_release_failed``, never a silent success; the foreign lock
    is left untouched."""

    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    lock.acquire()
    path = _lock_path(root)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["lock_token"] = "f" * 32  # a non-conforming actor replaced the live lock
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(OwnerDecisionLockError) as replaced:
        lock.release()
    assert replaced.value.reason == "lock_release_failed"
    assert "another writer" in str(replaced.value)
    assert json.loads(path.read_text(encoding="utf-8"))["lock_token"] == "f" * 32  # untouched
    assert lock._held is False
    # no longer held: a second release is silent and still leaves the foreign lock alone
    lock.release()
    assert path.exists()
    path.unlink()


def test_mutex_persistent_io_error_is_not_reported_as_contention(root: Path, monkeypatch) -> None:
    """Review RA-02: only the platform's contention errno means "another
    holder"; EBADF / EIO on the mutex file is the typed ``FileMutexError``,
    surfaced by the lock as ``lock_mutex_failed`` and by namespace
    initialization as ``store_namespace_initialization_failed`` — never a live
    holder, never a busy initializer."""

    import errno

    from alpha_lab.agents.data_infra.ifvg.search import file_mutex as mutex_module
    from alpha_lab.agents.data_infra.ifvg.search.file_mutex import (
        ExclusiveFileMutex,
        FileMutexError,
        FileMutexTimeoutError,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import StoreNamespaceError

    def _failing(*_args, **_kwargs):
        raise OSError(errno.EBADF, "bad file descriptor (simulated, persistent)")

    if sys.platform == "win32":
        monkeypatch.setattr(mutex_module.msvcrt, "locking", _failing)
    else:  # pragma: win32 no cover
        monkeypatch.setattr(mutex_module.fcntl, "flock", _failing)
    mutex = ExclusiveFileMutex(root / "probe.mutex", wait_seconds=0.2)
    with pytest.raises(FileMutexError) as failed:
        mutex.acquire()
    assert failed.value.errno == errno.EBADF
    assert not isinstance(failed.value, FileMutexTimeoutError)
    assert not mutex.held
    # the lock: a stale dead holder cannot be assessed under a failing mutex — typed,
    # never "held by a live holder", and the stale lock is never unlinked
    _write_lock(root, pid=_DEAD_PID, token="9" * 32)
    lock = OwnerDecisionLock(root, wait_seconds=0.5, heartbeat_timeout_seconds=1.0)
    with pytest.raises(OwnerDecisionLockError) as typed:
        lock.acquire()
    assert typed.value.reason == "lock_mutex_failed"
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == "9" * 32
    # namespace initialization: a failing init mutex is not a busy initializer
    with pytest.raises(StoreNamespaceError) as init_failed:
        initialize_test_namespace(root / "fresh-store")
    assert init_failed.value.reason == "store_namespace_initialization_failed"
    monkeypatch.undo()
    # contention is still contention: a held mutex times out as before
    holder = ExclusiveFileMutex(root / "probe.mutex", wait_seconds=0.2)
    holder.acquire()
    try:
        with pytest.raises(FileMutexTimeoutError):
            ExclusiveFileMutex(root / "probe.mutex", wait_seconds=0.1).acquire()
    finally:
        holder.release()
    _lock_path(root).unlink()


def test_read_failure_after_exclusive_creation_is_typed_and_removes_nothing(
    root: Path, monkeypatch
) -> None:
    """HARDENING-BACKEND-FIX.1 §1 (supersedes review RA-04's discard): a
    persistent read failure right after this writer's own exclusive creation
    is the typed ``lock_read_failed`` and acquires nothing — and because the
    readback failed, ownership of the file is NOT proven, so the fresh lock
    is left in place (never removed blindly). It carries this writer's live
    body: another writer waits on it as a live holder (fail closed) and it
    becomes reclaimable by liveness once this process exits."""

    from alpha_lab.agents.data_infra.ifvg.search import owner_decision_lock as lock_module

    monkeypatch.setattr(lock_module, "_FS_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(lock_module, "_FS_RETRY_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(
        OwnerDecisionLock,
        "_read_raw",
        lambda self: (_ for _ in ()).throw(PermissionError("read failure (simulated)")),
    )
    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    with pytest.raises(OwnerDecisionLockError) as failed:
        lock.acquire()
    assert failed.value.reason == "lock_read_failed"
    assert "left in place" in str(failed.value)
    assert lock._held is False
    monkeypatch.undo()
    # the fresh lock survives untouched, carrying this writer's token
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == lock.token
    assert not list((root / OWNER_DECISION_STORE).glob(".SUPERSESSIONS.lock.partial-*"))
    # fail closed: another writer sees a live holder (this process, fresh heartbeat)
    with pytest.raises(OwnerDecisionLockError) as held:
        OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0).acquire()
    assert held.value.reason == "lock_held_by_live_holder"
    assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == lock.token
    _lock_path(root).unlink()  # operator intervention
