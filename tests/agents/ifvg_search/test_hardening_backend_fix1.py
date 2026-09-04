"""HARDENING-BACKEND-FIX.1 — the four remaining backend corrections.

1. Strict lock-body validation and the post-create ownership proof
   (``owner_decision_lock.py``): a malformed body is never coerced and never
   reclaimed; acquisition is proven by the exact persisted readback; a file
   is cleaned up only when ownership is proven.
2. Recoverable namespace initialization (``store_namespace.py`` + CLI): the
   real initializer never generates an instance id — the first
   initialization of a real store requires an explicit one; the disposable
   test helper generates its id BEFORE the real initializer runs.
3. Native candidate-id validation before coercion
   (``regime_oos_assignment.candidate_as_of_frame``).
4. (The capacity gate is closed by the formal benchmark artifact, not here.)
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import AvailabilityStage
from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
    assert_native_candidate_ids,
    candidate_as_of_frame,
)
from alpha_lab.agents.data_infra.ifvg.search import owner_decision_lock as lock_module
from alpha_lab.agents.data_infra.ifvg.search import store_namespace as namespace_module
from alpha_lab.agents.data_infra.ifvg.search.owner_decision_lock import (
    LOCK_BODY_FIELDS,
    LOCK_PID_MAX,
    LOCK_SCHEMA_VERSION,
    OWNER_DECISION_LOCK_FILE,
    LockBody,
    OwnerDecisionLock,
    OwnerDecisionLockError,
    current_process_start_token,
    process_liveness,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    OWNER_DECISION_STORE,
    STORE_NAMESPACE_FAILURE_REASONS,
    STORE_NAMESPACE_FILE,
    StoreNamespaceError,
    initialize_store_namespace,
    initialize_test_namespace,
    load_store_namespace,
    namespace_class_of,
    supersession_head_path,
)

_OLD = "2026-01-01T00:00:00+00:00"
_DEAD_PID = 999_999_999  # no such process on any supported platform
_STALE_MTIME = 1_600_000_000.0  # a malformed body falls back to the file's age: stale


def _lock_path(root: Path) -> Path:
    return root / OWNER_DECISION_STORE / OWNER_DECISION_LOCK_FILE


def _valid_body(**overrides) -> dict:
    """A stale DEAD-holder body that a lawful reclaimer WOULD reclaim — every
    refusal below is therefore attributable to the one field it corrupts."""

    body = {
        "lock_schema_version": LOCK_SCHEMA_VERSION,
        "pid": _DEAD_PID,
        "process_start_token": None,
        "lock_token": "0" * 32,
        "host": platform.node(),
        "created_at": _OLD,
        "heartbeat_at": _OLD,
    }
    body.update(overrides)
    return body


def _write_stale(root: Path, document: dict | None, *, raw: bytes | None = None) -> Path:
    path = _lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw if raw is not None else json.dumps(document).encode("utf-8"))
    os.utime(path, (_STALE_MTIME, _STALE_MTIME))
    return path


def _assert_malformed_and_non_reclaimable(
    root: Path, document: dict | None = None, *, raw: bytes | None = None
) -> None:
    text = raw.decode("utf-8", errors="replace") if raw is not None else json.dumps(document)
    assert LockBody.parse(text) is None
    path = _write_stale(root, document, raw=raw)
    before = path.read_bytes()
    lock = OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0)
    with pytest.raises(OwnerDecisionLockError) as refused:
        lock.acquire()
    assert refused.value.reason == "lock_body_malformed"
    assert path.read_bytes() == before  # untouched
    assert lock.reclaimed_from is None and lock._held is False
    path.unlink()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    initialize_test_namespace(store)
    return store


# ── 1. strict lock-body validation ───────────────────────────────────────────


def test_control_the_valid_stale_dead_holder_body_is_reclaimed(root: Path) -> None:
    body = LockBody.parse(json.dumps(_valid_body()))
    assert body is not None and body.pid == _DEAD_PID and body.lock_token == "0" * 32
    # a `Z` suffix and the maximum pid are valid (aware ISO-8601; in range)
    assert LockBody.parse(json.dumps(_valid_body(heartbeat_at="2026-01-01T00:00:00Z"))) is not None
    assert LockBody.parse(json.dumps(_valid_body(pid=LOCK_PID_MAX))) is not None
    assert set(_valid_body()) == LOCK_BODY_FIELDS
    path = _write_stale(root, _valid_body())
    with OwnerDecisionLock(root, wait_seconds=0.5, heartbeat_timeout_seconds=1.0) as lock:
        assert lock.reclaimed_from is not None and lock.reclaimed_from.pid == _DEAD_PID
    assert not path.exists()


@pytest.mark.parametrize("version", [2, 0, -1, "1", 1.0, True, None])
def test_wrong_lock_schema_is_malformed_and_non_reclaimable(root: Path, version) -> None:
    _assert_malformed_and_non_reclaimable(root, _valid_body(lock_schema_version=version))


def test_missing_or_extra_lock_fields_are_malformed(root: Path) -> None:
    missing = _valid_body()
    del missing["lock_schema_version"]
    _assert_malformed_and_non_reclaimable(root, missing)
    _assert_malformed_and_non_reclaimable(root, _valid_body(extra="field"))
    _assert_malformed_and_non_reclaimable(root, raw=b"[]")
    _assert_malformed_and_non_reclaimable(root, raw=b"\xff\xfe{")  # not UTF-8
    # review FIX.1-R1: a pathologically nested body is malformed, not an
    # untyped RecursionError escaping the lock
    _assert_malformed_and_non_reclaimable(root, raw=b"[" * 200_000)


@pytest.mark.parametrize(
    "pid",
    [
        "123",
        str(_DEAD_PID),
        float(_DEAD_PID),
        1.5,
        True,
        False,
        0,
        -1,
        -_DEAD_PID,
        LOCK_PID_MAX + 1,
        None,
    ],
)
def test_string_float_bool_zero_negative_and_out_of_range_pids_are_refused(root: Path, pid) -> None:
    _assert_malformed_and_non_reclaimable(root, _valid_body(pid=pid))


@pytest.mark.parametrize(
    "token",
    ["0" * 31, "0" * 33, "A" * 32, "g" * 32, "0" * 32 + "\n", 0, None, ["0" * 32]],
)
def test_invalid_token_shape_is_refused(root: Path, token) -> None:
    _assert_malformed_and_non_reclaimable(root, _valid_body(lock_token=token))


@pytest.mark.parametrize("field", ["created_at", "heartbeat_at"])
@pytest.mark.parametrize(
    "stamp",
    ["2026-01-01T00:00:00", "2026-01-01", "not-a-time", "", 1_700_000_000, 1.0, None],
)
def test_naive_or_malformed_timestamps_are_refused(root: Path, field, stamp) -> None:
    _assert_malformed_and_non_reclaimable(root, _valid_body(**{field: stamp}))


@pytest.mark.parametrize("field", ["host", "process_start_token"])
@pytest.mark.parametrize("value", [123, 1.5, True, ["h"], {"host": "h"}])
def test_host_and_start_token_must_be_strings_or_null(root: Path, field, value) -> None:
    _assert_malformed_and_non_reclaimable(root, _valid_body(**{field: value}))
    assert LockBody.parse(json.dumps(_valid_body(**{field: None}))) is not None
    assert LockBody.parse(json.dumps(_valid_body(**{field: "text"}))) is not None


# ── 1b. Win32 liveness: a FAILED exit-code query is unknown, never dead ──────

_WIN32 = sys.platform == "win32"


def _patch_kernel32_exit_query(monkeypatch, query) -> None:
    """Route the lock module's ``ctypes.WinDLL("kernel32", …)`` through a proxy
    whose ``GetExitCodeProcess`` is ``query``; every other export
    (``OpenProcess``, ``GetProcessTimes``, ``CloseHandle`` …) is the REAL
    kernel32, so the probe reaches the real live process and only the
    exit-code query is controlled."""

    import ctypes

    real_windll = ctypes.WinDLL
    real_kernel32 = real_windll("kernel32", use_last_error=True)

    class _Kernel32Proxy:
        GetExitCodeProcess = staticmethod(query)

        def __getattr__(self, name):
            return getattr(real_kernel32, name)

    proxy = _Kernel32Proxy()

    def _windll(name, *args, **kwargs):
        if str(name).lower().split(".")[0] == "kernel32":
            return proxy
        return real_windll(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "WinDLL", _windll)


@pytest.mark.skipif(not _WIN32, reason="the Win32 liveness reader")
def test_win32_exit_code_query_failure_is_unknown(monkeypatch) -> None:
    """``GetExitCodeProcess`` returning FALSE means the process state was NOT
    determined: ``unknown`` (never reclaimable) — not ``dead``. A query that
    SUCCEEDS with an exit code other than STILL_ACTIVE is the demonstrable
    death that remains ``dead``: the two conditions are distinguished."""

    from alpha_lab.agents.data_infra.ifvg.search.owner_decision_lock import (
        _win32_process_times,
    )

    pid = os.getpid()
    # control: the real query on this live process
    liveness, token = _win32_process_times(pid)
    assert liveness == "alive" and token is not None
    calls = {"n": 0}

    def _query_fails(_handle, _exit_code_pointer):
        calls["n"] += 1
        return 0  # FALSE: the query itself failed; nothing was determined

    _patch_kernel32_exit_query(monkeypatch, _query_fails)
    assert _win32_process_times(pid) == ("unknown", None)
    assert calls["n"] == 1
    assert process_liveness(pid, token, platform.node()) == "unknown"
    # the recorded start token is not consulted: nothing was determined, so
    # even a "different process" token cannot turn the verdict into dead
    assert process_liveness(pid, "not-this-process", platform.node()) == "unknown"
    # the writer's OWN start token never goes through the exit-code query
    assert current_process_start_token() == token
    monkeypatch.undo()

    def _query_succeeds_exited(_handle, exit_code_pointer):
        exit_code_pointer._obj.value = 0  # the query SUCCEEDED: exit code 0
        return 1

    _patch_kernel32_exit_query(monkeypatch, _query_succeeds_exited)
    assert _win32_process_times(pid) == ("dead", None)
    assert process_liveness(pid, token, platform.node()) == "dead"


@pytest.mark.skipif(not _WIN32, reason="the Win32 liveness reader")
def test_stale_lock_is_not_reclaimed_when_win32_exit_query_fails(root: Path, monkeypatch) -> None:
    """The failure scenario: an OLD heartbeat, a holder that is still ALIVE
    (this process), ``OpenProcess`` succeeds and ``GetExitCodeProcess`` fails.
    Before the correction the reader answered ``dead`` and the stale-lock
    evaluator authorized reclamation of a LIVE writer's lock. Now the lock
    file remains present, acquisition does not succeed, and the typed result
    is ``lock_holder_liveness_unknown``."""

    _write_stale(
        root,
        _valid_body(
            pid=os.getpid(),
            process_start_token=current_process_start_token(),
            lock_token="b" * 32,
        ),
    )
    before = _lock_path(root).read_bytes()
    _patch_kernel32_exit_query(monkeypatch, lambda _handle, _exit_code_pointer: 0)
    lock = OwnerDecisionLock(
        root, wait_seconds=0.4, heartbeat_timeout_seconds=1.0, poll_seconds=0.01
    )
    with pytest.raises(OwnerDecisionLockError) as refused:
        lock.acquire()
    assert refused.value.reason == "lock_holder_liveness_unknown"
    assert lock.reclaimed_from is None and lock._held is False
    assert _lock_path(root).exists()
    assert _lock_path(root).read_bytes() == before
    assert json.loads(before)["lock_token"] == "b" * 32
    monkeypatch.undo()
    # control: with the real query the same stale body is a LIVE holder (this
    # process is alive) — still never reclaimed, still typed
    with pytest.raises(OwnerDecisionLockError) as live:
        OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0).acquire()
    assert live.value.reason == "lock_held_by_live_holder"
    assert _lock_path(root).read_bytes() == before
    _lock_path(root).unlink()


# ── 1. the post-create ownership proof ───────────────────────────────────────


def _patch_post_create_readback(monkeypatch, lock: OwnerDecisionLock, reaction) -> dict:
    """Intercept ONLY the readback that follows this lock's own exclusive
    creation (the file exists and carries this writer's bytes); every other
    read goes through the real seam."""

    real = OwnerDecisionLock._read_raw
    calls = {"intercepted": 0}

    def _read(self):
        if self is lock and self._created_bytes is not None and calls["intercepted"] == 0:
            calls["intercepted"] += 1
            return reaction(self)
        return real(self)

    monkeypatch.setattr(OwnerDecisionLock, "_read_raw", _read)
    return calls


def test_a_successful_acquire_returns_the_exactly_persisted_body(root: Path) -> None:
    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    body = lock.acquire()
    persisted = _lock_path(root).read_bytes()
    assert persisted == lock._created_bytes
    assert LockBody.parse(persisted.decode("utf-8")) == body
    assert body.lock_token == lock.token and body.pid == os.getpid()
    assert lock._held is True
    lock.release()
    assert not _lock_path(root).exists()


def test_missing_post_create_readback_does_not_acquire(root: Path, monkeypatch) -> None:
    lock = OwnerDecisionLock(root, wait_seconds=0.3)

    def _vanish(self):
        self.path.unlink()  # another party removed the fresh lock
        raise FileNotFoundError(str(self.path))

    calls = _patch_post_create_readback(monkeypatch, lock, _vanish)
    with pytest.raises(OwnerDecisionLockError) as lost:
        lock.acquire()
    assert lost.value.reason == "lock_lost"
    assert calls["intercepted"] == 1
    assert lock._held is False
    assert not _lock_path(root).exists()
    monkeypatch.undo()
    with OwnerDecisionLock(root, wait_seconds=0.3) as again:
        assert json.loads(_lock_path(root).read_text(encoding="utf-8"))["lock_token"] == again.token
    assert not _lock_path(root).exists()


def test_malformed_post_create_readback_does_not_acquire_and_removes_nothing(
    root: Path, monkeypatch
) -> None:
    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    calls = _patch_post_create_readback(monkeypatch, lock, lambda self: b"{")
    with pytest.raises(OwnerDecisionLockError) as malformed:
        lock.acquire()
    assert malformed.value.reason == "lock_body_malformed"
    assert "left untouched" in str(malformed.value)
    assert calls["intercepted"] == 1
    assert lock._held is False
    monkeypatch.undo()
    # ownership was not proven: the file was left exactly as written
    assert _lock_path(root).read_bytes() == lock._created_bytes
    assert not list((root / OWNER_DECISION_STORE).glob(".SUPERSESSIONS.lock.partial-*"))
    # fail closed: it carries this writer's live body, so another writer waits
    with pytest.raises(OwnerDecisionLockError) as held:
        OwnerDecisionLock(root, wait_seconds=0.3, heartbeat_timeout_seconds=1.0).acquire()
    assert held.value.reason == "lock_held_by_live_holder"
    _lock_path(root).unlink()  # operator intervention


def test_foreign_token_post_create_readback_does_not_acquire(root: Path, monkeypatch) -> None:
    foreign = json.dumps(
        _valid_body(
            pid=os.getpid(),
            lock_token="f" * 32,
            heartbeat_at=datetime.now(UTC).isoformat(),
        )
    ).encode("utf-8")
    lock = OwnerDecisionLock(root, wait_seconds=0.3)

    def _replaced_by_a_foreign_writer(self):
        self.path.write_bytes(foreign)
        return foreign

    calls = _patch_post_create_readback(monkeypatch, lock, _replaced_by_a_foreign_writer)
    with pytest.raises(OwnerDecisionLockError) as lost:
        lock.acquire()
    assert lost.value.reason == "lock_lost"
    assert calls["intercepted"] == 1
    assert lock._held is False
    monkeypatch.undo()
    # the foreign lock is never unlinked by the writer that lost
    assert _lock_path(root).read_bytes() == foreign
    _lock_path(root).unlink()
    # our token but ALTERED bytes (a valid JSON re-serialization) is not an
    # exact persisted readback either
    lock2 = OwnerDecisionLock(root, wait_seconds=0.3)
    _patch_post_create_readback(monkeypatch, lock2, lambda self: b" " + self._created_bytes)
    with pytest.raises(OwnerDecisionLockError) as altered:
        lock2.acquire()
    assert altered.value.reason == "lock_lost"
    assert lock2._held is False
    monkeypatch.undo()
    assert _lock_path(root).read_bytes() == lock2._created_bytes  # untouched
    _lock_path(root).unlink()


def test_unreadable_post_create_readback_is_typed_and_leaves_the_lock(
    root: Path, monkeypatch
) -> None:
    monkeypatch.setattr(lock_module, "_FS_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(lock_module, "_FS_RETRY_DELAY_SECONDS", 0.001)
    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    real = OwnerDecisionLock._read_raw

    def _unreadable(self):
        if self is lock and self._created_bytes is not None:
            raise PermissionError("sharing violation (simulated, persistent)")
        return real(self)

    monkeypatch.setattr(OwnerDecisionLock, "_read_raw", _unreadable)
    with pytest.raises(OwnerDecisionLockError) as failed:
        lock.acquire()
    assert failed.value.reason == "lock_read_failed"
    assert "left in place" in str(failed.value)
    assert lock._held is False
    monkeypatch.undo()
    assert _lock_path(root).read_bytes() == lock._created_bytes
    _lock_path(root).unlink()


def test_partial_write_cleanup_requires_the_exact_prefix_proof(root: Path, monkeypatch) -> None:
    """A failed body write after exclusive creation removes the partial file
    ONLY when the persisted bytes are exactly a prefix of what this writer
    wrote (ownership proven); a foreign readback or a readback failure
    leaves the file in place, typed."""

    monkeypatch.setattr(lock_module, "_FS_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(lock_module, "_FS_RETRY_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(
        lock_module.os,
        "fsync",
        lambda _handle: (_ for _ in ()).throw(OSError("disk full (simulated)")),
    )
    # (a) proven: the persisted bytes are this writer's (a prefix) → removed
    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    with pytest.raises(OwnerDecisionLockError) as proven:
        lock.acquire()
    assert proven.value.reason == "lock_create_failed" and "removed" in str(proven.value)
    assert not _lock_path(root).exists()
    # (b) unproven: the readback is not this writer's bytes → left in place
    real = OwnerDecisionLock._read_raw
    monkeypatch.setattr(OwnerDecisionLock, "_read_raw", lambda self: b'{"foreign": true}')
    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    with pytest.raises(OwnerDecisionLockError) as foreign:
        lock.acquire()
    assert foreign.value.reason == "lock_create_failed" and "left in place" in str(foreign.value)
    assert _lock_path(root).exists()
    _lock_path(root).unlink()
    # (c) unproven: the readback fails persistently → left in place
    monkeypatch.setattr(
        OwnerDecisionLock,
        "_read_raw",
        lambda self: (_ for _ in ()).throw(PermissionError("read failure (simulated)")),
    )
    lock = OwnerDecisionLock(root, wait_seconds=0.3)
    with pytest.raises(OwnerDecisionLockError) as unreadable:
        lock.acquire()
    assert unreadable.value.reason == "lock_create_failed"
    assert "left in place" in str(unreadable.value)
    assert _lock_path(root).exists()
    monkeypatch.setattr(OwnerDecisionLock, "_read_raw", real)
    assert not list((root / OWNER_DECISION_STORE).glob(".SUPERSESSIONS.lock.partial-*"))
    _lock_path(root).unlink()


# ── 2. recoverable namespace initialization ──────────────────────────────────


def _nothing_published(root: Path) -> None:
    assert namespace_class_of(root) is None
    assert not (root / STORE_NAMESPACE_FILE).exists()
    assert not supersession_head_path(root).exists()
    assert not list(root.glob(".STORE_NAMESPACE.json.tmp-*"))
    assert not list((root / OWNER_DECISION_STORE).glob(".SUPERSESSIONS.head.tmp-*"))


@pytest.mark.parametrize("namespace_class", ["research", "test"])
def test_first_real_initialization_without_store_instance_id_is_refused(
    tmp_path: Path, namespace_class
) -> None:
    assert "store_instance_id_required" in STORE_NAMESPACE_FAILURE_REASONS
    root = tmp_path / namespace_class
    with pytest.raises(StoreNamespaceError) as refused:
        initialize_store_namespace(root, namespace_class=namespace_class)
    assert refused.value.reason == "store_instance_id_required"
    _nothing_published(root)


@pytest.mark.parametrize("bad", ["A" * 32, "0" * 31, "0" * 33, b"0" * 32, 0, "0" * 32 + " "])
def test_a_malformed_explicit_instance_id_is_refused_before_anything_is_published(
    tmp_path: Path, bad
) -> None:
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="store_instance_id must be a native string"):
        initialize_store_namespace(root, namespace_class="test", store_instance_id=bad)
    _nothing_published(root)


def test_explicit_id_crash_recovery_and_idempotent_reuse_still_pass(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "store"
    instance = "d1" * 16
    real_write = namespace_module._atomic_write_text

    def _crash_before_namespace(path: Path, text: str) -> None:
        if path.name == STORE_NAMESPACE_FILE:
            raise RuntimeError("simulated crash between the two publications")
        real_write(path, text)

    monkeypatch.setattr(namespace_module, "_atomic_write_text", _crash_before_namespace)
    with pytest.raises(RuntimeError, match="simulated crash"):
        initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    monkeypatch.setattr(namespace_module, "_atomic_write_text", real_write)
    head_bytes = supersession_head_path(root).read_bytes()
    # a class-only request cannot recover it (no id to reproduce the pair)
    with pytest.raises(StoreNamespaceError) as class_only:
        initialize_store_namespace(root, namespace_class="test")
    assert class_only.value.reason == "incomplete_store_namespace_initialization"
    # the identical explicit request recovers EXACTLY
    recovered = initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    reference = initialize_store_namespace(
        tmp_path / "reference", namespace_class="test", store_instance_id=instance
    )
    assert recovered.store_namespace_id == reference.store_namespace_id
    assert supersession_head_path(root).read_bytes() == head_bytes
    namespace_bytes = (root / STORE_NAMESPACE_FILE).read_bytes()
    assert namespace_bytes == (tmp_path / "reference" / STORE_NAMESPACE_FILE).read_bytes()
    # idempotent reuse: the explicit replay and the class-only replay rewrite nothing
    for kwargs in ({"store_instance_id": instance}, {}):
        again = initialize_store_namespace(root, namespace_class="test", **kwargs)
        assert again.store_namespace_id == recovered.store_namespace_id
        assert (root / STORE_NAMESPACE_FILE).read_bytes() == namespace_bytes
        assert supersession_head_path(root).read_bytes() == head_bytes
    with pytest.raises(StoreNamespaceError) as divergent:
        initialize_store_namespace(root, namespace_class="test", store_instance_id="d2" * 16)
    assert divergent.value.reason == "store_namespace_divergent"


def test_the_disposable_test_helper_generates_its_id_before_the_real_initializer(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict] = []
    real = namespace_module.initialize_store_namespace

    def _recording(root, **kwargs):
        calls.append(dict(kwargs))
        return real(root, **kwargs)

    monkeypatch.setattr(namespace_module, "initialize_store_namespace", _recording)
    root = tmp_path / "disposable"
    first = initialize_test_namespace(root)
    assert calls[-1]["namespace_class"] == "test"
    generated = calls[-1]["store_instance_id"]
    assert isinstance(generated, str) and len(generated) == 32
    assert int(generated, 16) >= 0 and generated == generated.lower()
    assert first.payload.store_instance_id == generated
    # a second disposable store gets its own id (never a shared / derived one)
    other = initialize_test_namespace(tmp_path / "other")
    assert other.payload.store_instance_id != generated
    # idempotent on a marked test store: a class-only replay, the same envelope
    again = initialize_test_namespace(root)
    assert calls[-1] == {"namespace_class": "test"}
    assert again.store_namespace_id == first.store_namespace_id
    assert load_store_namespace(root).payload.store_instance_id == generated
    # a marked research root is not silently reused as a test store
    research = tmp_path / "research"
    initialize_store_namespace(research, namespace_class="research", store_instance_id="e1" * 16)
    with pytest.raises(StoreNamespaceError) as divergent:
        initialize_test_namespace(research)
    assert divergent.value.reason == "store_namespace_divergent"
    assert load_store_namespace(research).payload.namespace_class == "research"


def test_concurrent_initialization_remains_serialized_and_fails_closed(tmp_path: Path) -> None:
    # (i) class-only initializers on an unmarked root all refuse; nothing published
    root = tmp_path / "unmarked"
    start = threading.Barrier(4, timeout=10)
    outcomes: list[str] = []

    def _class_only() -> None:
        start.wait()
        try:
            initialize_store_namespace(root, namespace_class="research")
            outcomes.append("initialized")
        except StoreNamespaceError as error:
            outcomes.append(error.reason)
        except BaseException as error:  # noqa: BLE001
            outcomes.append(repr(error))

    threads = [threading.Thread(target=_class_only) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert outcomes == ["store_instance_id_required"] * 4
    _nothing_published(root)
    # (ii) identical explicit initializers converge on one coherent pair, and a
    # divergent instance loses closed
    root2 = tmp_path / "explicit"
    start2 = threading.Barrier(5, timeout=10)
    results: dict[str, object] = {}

    def _explicit(name: str, instance: str) -> None:
        start2.wait()
        try:
            results[name] = initialize_store_namespace(
                root2, namespace_class="test", store_instance_id=instance
            ).store_namespace_id
        except StoreNamespaceError as error:
            results[name] = error.reason
        except BaseException as error:  # noqa: BLE001
            results[name] = repr(error)

    threads = [
        threading.Thread(target=_explicit, args=(f"same{i}", "f1" * 16)) for i in range(4)
    ] + [threading.Thread(target=_explicit, args=("other", "f2" * 16))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    same = {results[f"same{i}"] for i in range(4)}
    if results["other"] == "store_namespace_divergent":
        assert len(same) == 1 and load_store_namespace(root2).payload.store_instance_id == (
            "f1" * 16
        )
    else:  # the divergent writer won the mutex first: the four identical ones refuse
        assert same == {"store_namespace_divergent"}
        assert load_store_namespace(root2).payload.store_instance_id == "f2" * 16
    assert load_store_namespace(root2).store_namespace_id in {results["other"], *same}


def test_cli_first_initialization_requires_the_explicit_instance_id(tmp_path: Path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ifvg_store_namespace", Path("scripts/ifvg_store_namespace.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    root = tmp_path / "cli_store"
    common = ["init", "--store-root", str(root), "--namespace-class", "research"]
    assert module.main(common) == 0
    intent = json.loads(capsys.readouterr().out)
    assert intent["store_instance_id_required"] is True and intent["store_instance_id"] is None
    assert "record" in intent["note"]
    assert module.main([*common, "--confirm"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "store_instance_id_required"
    _nothing_published(root)
    # review FIX.1-R2: a malformed explicit id is a typed CLI refusal, never a traceback
    for malformed in ("C3" * 16, "c3" * 15, "c3" * 16 + "0"):
        assert module.main([*common, "--store-instance-id", malformed, "--confirm"]) == 2
        assert json.loads(capsys.readouterr().out)["reason"] == "store_instance_id_malformed"
        _nothing_published(root)
    instance = "c3" * 16
    assert module.main([*common, "--store-instance-id", instance]) == 0
    assert json.loads(capsys.readouterr().out)["store_instance_id_required"] is False
    assert module.main([*common, "--store-instance-id", instance, "--confirm"]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["status"] == "initialized"
    assert initialized["store_instance_id"] == instance
    assert module.main([*common, "--confirm"]) == 0  # class-only replay of a marked store
    assert (
        json.loads(capsys.readouterr().out)["store_namespace_id"]
        == (initialized["store_namespace_id"])
    )


# ── 3. native candidate-id validation before coercion ────────────────────────


def _frame(candidate_ids) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": pd.Series(candidate_ids, dtype=object),
            "entry_ts_utc": ["2026-01-13T15:00:00Z"] * len(candidate_ids),
        }
    )


@pytest.mark.parametrize(
    "bad",
    [
        1,
        0,
        1.5,
        True,
        False,
        None,
        float("nan"),
        pd.NA,
        pd.NaT,
        np.int64(7),
        np.float64(7.0),
        np.bool_(True),
        np.str_("np-string"),
        b"bytes",
        "",
        "   ",
        "\t\n",
        {"candidate_id": "a"},
        ["a"],
        ("a",),
    ],
)
def test_candidate_as_of_rejects_non_native_or_blank_ids_before_coercion(bad) -> None:
    frame = _frame([bad, "ok"])
    with pytest.raises(ValueError, match="candidate_id must be a (native|non-blank) string"):
        candidate_as_of_frame(frame, stage=AvailabilityStage.ENTRY_DECISION)
    with pytest.raises(ValueError, match="candidate_id must be a (native|non-blank) string"):
        assert_native_candidate_ids(frame["candidate_id"])


def test_candidate_as_of_native_check_precedes_astype_and_the_duplicate_check() -> None:
    # `1` and `"1"` would collapse into a duplicate after astype(str): the
    # native refusal fires first (no coercion ever happened)
    with pytest.raises(ValueError, match="native string.*row 0 carries int 1"):
        candidate_as_of_frame(_frame([1, "1"]), stage=AvailabilityStage.ENTRY_DECISION)
    # `None` / NaN would silently become the strings "None" / "nan"
    with pytest.raises(ValueError, match="row 1 carries NoneType"):
        candidate_as_of_frame(_frame(["a", None]), stage=AvailabilityStage.ENTRY_DECISION)
    # a genuine duplicate of valid ids still refuses as before
    with pytest.raises(ValueError, match="repeats"):
        candidate_as_of_frame(_frame(["a", "a"]), stage=AvailabilityStage.ENTRY_DECISION)


def test_candidate_as_of_valid_ids_keep_the_canonical_output() -> None:
    ids = ["kc_0001", "kc_0000", "z", "a b", "1"]
    for dtype in (object, "string"):
        frame = pd.DataFrame(
            {
                "candidate_id": pd.Series(ids, dtype=dtype),
                "entry_ts_utc": ["2026-01-13T15:00:00Z", "2026-01-13T15:05:00Z", None, "", "x"],
                "tap_ts_utc": ["2026-01-13T14:00:00Z"] * 5,
            }
        )
        out = candidate_as_of_frame(frame, stage=AvailabilityStage.ENTRY_DECISION)
        # the pre-FIX.1 construction (astype(str) over the same ids), unchanged
        expected = pd.DataFrame(
            {
                "candidate_id": frame["candidate_id"].astype(str).to_numpy(),
                "as_of_ts_utc": frame["entry_ts_utc"].to_numpy(),
            }
        )
        pd.testing.assert_frame_equal(out, expected)
        assert out["candidate_id"].tolist() == ids
        assert all(type(value) is str for value in out["candidate_id"].tolist())
        assert list(out.columns) == ["candidate_id", "as_of_ts_utc"]
        tap = candidate_as_of_frame(frame, stage=AvailabilityStage.HTF_TAP)
        assert tap["candidate_id"].tolist() == ids
