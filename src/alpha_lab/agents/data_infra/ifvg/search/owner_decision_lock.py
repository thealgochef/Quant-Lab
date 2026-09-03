"""Liveness-aware owner-decision lock (HARDENING-BACKEND §4.3; F-13).

R6.1 reclaimed the supersession lock by AGE alone: a slow live holder could
be reclaimed and overlapped by a second writer. This lock records ``pid``,
a process-start token (the process creation time where the platform exposes
it), a random lock token, host, creation time and a heartbeat:

* **Reclaim** only when the heartbeat exceeds the timeout AND the recorded
  process is demonstrably not alive on THIS host — ``dead`` from
  :func:`process_liveness`: no such pid, an exited pid, or a live pid whose
  start token differs from the recorded one (PID reuse). A holder on another
  host, a malformed body, or a platform that cannot assess liveness is
  ``unknown`` and is NEVER reclaimed — the waiter times out with a typed
  reason for operator intervention.
* **Token-safe reclamation** (HARDENING-BACKEND-FIX §4.1). A stale verdict
  reached OUTSIDE the reclaim mutex grants no authority to unlink: two
  reclaimers that both inspected stale body ``S`` could otherwise race —
  the first replaces ``S`` with its live lock ``A`` and the second unlinks
  ``A``. Reclamation therefore runs under a dedicated cross-process mutex
  (:class:`~.file_mutex.ExclusiveFileMutex`; ``msvcrt`` / ``fcntl``): the
  body is re-read under the mutex, heartbeat age / host / pid liveness /
  start token are re-evaluated, and the file is unlinked ONLY when the
  still-current body is byte-for-byte the same dead holder that was
  observed (same ``lock_token``, same body). Ordinary acquisition stays
  ``O_CREAT | O_EXCL``; the mutex guards nothing else.
* **Typed reads**: the lock distinguishes ``absent`` / ``malformed`` /
  persistent read I/O failure (``lock_read_failed``) — a read failure is
  never converted into absence.
* **Heartbeat** — :meth:`OwnerDecisionLock.refresh` rewrites the body
  (atomically) before long verification and before commit.
* **Token re-verification** — :meth:`OwnerDecisionLock.verify_held` reads
  the body back and refuses to continue when the token is not ours (a lost
  lock aborts the writer BEFORE publication).
* **Release** unlinks only its own token and raises a typed
  ``lock_release_failed`` when it cannot read or verify its own lock.
* **Partial creation**: when ``O_EXCL`` creation succeeds but the body
  write / fsync fails, the partial file is removed (or quarantined) before
  the typed ``lock_create_failed`` is raised.

No new dependency: the Win32 liveness reader uses ``ctypes`` (``OpenProcess``
/ ``GetExitCodeProcess`` / ``GetProcessTimes``); POSIX uses ``os.kill(pid, 0)``
plus ``/proc/<pid>/stat`` where present.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .file_mutex import ExclusiveFileMutex, FileMutexError, FileMutexTimeoutError
from .store_namespace import OWNER_DECISION_STORE

__all__ = [
    "OWNER_DECISION_LOCK_FILE",
    "OWNER_DECISION_RECLAIM_MUTEX_FILE",
    "LOCK_FAILURE_REASONS",
    "OwnerDecisionLockError",
    "LockBody",
    "current_process_start_token",
    "process_liveness",
    "OwnerDecisionLock",
]

OWNER_DECISION_LOCK_FILE = "SUPERSESSIONS.lock"
#: The reclaim mutex lives beside the lock; it carries no body and no token
#: and is never unlinked (see ``file_mutex``).
OWNER_DECISION_RECLAIM_MUTEX_FILE = "SUPERSESSIONS.reclaim.mutex"
LOCK_SCHEMA_VERSION = 1
LOCK_FAILURE_REASONS: tuple[str, ...] = (
    "lock_held_by_live_holder",
    "lock_holder_liveness_unknown",
    "lock_lost",
    "lock_body_malformed",
    "lock_release_failed",
    "lock_refresh_failed",
    # HARDENING-BACKEND-FIX §4.1: a persistent read I/O failure is typed,
    # never absence; a failed body write after exclusive creation is typed
    # and leaves no partial lock behind
    "lock_read_failed",
    "lock_create_failed",
    # review RA-02: a persistent NON-contention failure of the reclaim mutex file
    # is typed — never reported as a live holder
    "lock_mutex_failed",
)
#: Adversarial RA-02: Windows refuses ``unlink`` / ``replace`` while ANY other
#: handle has the lock body open (a polling waiter's read is enough). Every
#: filesystem step of the lock retries a transient ``OSError`` with a short
#: backoff; a PERSISTENT failure is a typed error, never silence.
_FS_RETRY_ATTEMPTS = 40
_FS_RETRY_DELAY_SECONDS = 0.025
Liveness = Literal["alive", "dead", "unknown"]
ReadState = Literal["absent", "malformed", "present"]

_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87
_ERROR_ACCESS_DENIED = 5
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class OwnerDecisionLockError(TimeoutError):
    """The lock could not be acquired / held (typed ``reason``)."""

    def __init__(self, reason: str, message: str) -> None:
        if reason not in LOCK_FAILURE_REASONS:
            raise ValueError(f"unregistered lock failure reason {reason!r}")
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _retry_fs(operation, *, attempts: int | None = None, delay: float | None = None):
    """Run a filesystem operation, retrying transient ``OSError``s (sharing
    violations) with a short backoff; the last error propagates."""

    attempts = _FS_RETRY_ATTEMPTS if attempts is None else attempts
    delay = _FS_RETRY_DELAY_SECONDS if delay is None else delay
    last: OSError | None = None
    for _ in range(max(1, int(attempts))):
        try:
            return operation()
        except FileNotFoundError:
            raise
        except OSError as error:
            last = error
            time.sleep(delay)
    if last is not None:
        raise last
    return None


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


# ── process liveness ────────────────────────────────────────────────────────


def _win32_process_times(pid: int | None) -> tuple[Liveness, str | None]:
    """(liveness, creation-time token) of ``pid`` on Windows via ctypes."""

    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    if pid is None:
        handle = kernel32.GetCurrentProcess()
        own = True
    else:
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        own = False
        if not handle:
            code = ctypes.get_last_error()
            if code == _ERROR_INVALID_PARAMETER:
                return "dead", None
            if code == _ERROR_ACCESS_DENIED:
                return "alive", None  # exists; token unavailable
            return "unknown", None
    try:
        exit_code = wintypes.DWORD()
        if not own and (
            not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            or exit_code.value != _STILL_ACTIVE
        ):
            return "dead", None
        creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exit_), ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return "alive", None
        token = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        return "alive", f"win32:{token}"
    finally:
        if not own:
            kernel32.CloseHandle(handle)


def _posix_start_token(pid: int) -> str | None:
    stat = Path(f"/proc/{pid}/stat")
    try:
        text = stat.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # field 22 (1-based) is starttime; the comm field may contain spaces, so
    # split after the closing parenthesis
    tail = text.rsplit(")", 1)[-1].split()
    try:
        return f"proc:{tail[19]}"
    except IndexError:
        return None


def _posix_liveness(pid: int) -> tuple[Liveness, str | None]:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return "dead", None
    except PermissionError:
        return "alive", _posix_start_token(int(pid))
    except OSError:
        return "unknown", None
    return "alive", _posix_start_token(int(pid))


def current_process_start_token() -> str | None:
    """A token that changes when THIS pid is reused by a new process
    (creation time), or ``None`` where the platform cannot provide one."""

    try:
        if sys.platform == "win32":
            return _win32_process_times(None)[1]
        return _posix_start_token(os.getpid())
    except Exception:  # noqa: BLE001 — a liveness helper never raises into the writer
        return None


def process_liveness(pid: int | None, start_token: str | None, host: str | None) -> Liveness:
    """``dead`` only when demonstrable on this host: no such pid, an exited
    pid, or a live pid whose start token differs from the recorded one (PID
    reuse). Another host, a missing pid, or an unassessable platform is
    ``unknown``."""

    if pid is None or host is None or host != platform.node():
        return "unknown"
    try:
        if sys.platform == "win32":
            liveness, token = _win32_process_times(int(pid))
        else:
            liveness, token = _posix_liveness(int(pid))
    except Exception:  # noqa: BLE001
        return "unknown"
    if liveness != "alive":
        return liveness
    if start_token is not None and token is not None and token != start_token:
        return "dead"  # the recorded process is gone; the pid was reused
    return "alive"


# ── the lock ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LockBody:
    lock_schema_version: int
    pid: int | None
    process_start_token: str | None
    lock_token: str
    host: str | None
    created_at: str
    heartbeat_at: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True) + "\n"

    @classmethod
    def parse(cls, raw: str) -> LockBody | None:
        try:
            document = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(document, dict):
            return None
        try:
            return cls(
                lock_schema_version=int(document["lock_schema_version"]),
                pid=None if document.get("pid") is None else int(document["pid"]),
                process_start_token=document.get("process_start_token"),
                lock_token=str(document["lock_token"]),
                host=document.get("host"),
                created_at=str(document["created_at"]),
                heartbeat_at=str(document["heartbeat_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


class OwnerDecisionLock:
    """Exclusive owner-decision writer lock (context manager)."""

    def __init__(
        self,
        root: Path,
        *,
        wait_seconds: float = 30.0,
        heartbeat_timeout_seconds: float = 60.0,
        poll_seconds: float = 0.05,
    ) -> None:
        self.path = Path(root) / OWNER_DECISION_STORE / OWNER_DECISION_LOCK_FILE
        self.reclaim_mutex_path = self.path.with_name(OWNER_DECISION_RECLAIM_MUTEX_FILE)
        self.wait_seconds = float(wait_seconds)
        self.heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.token = uuid.uuid4().hex
        self._held = False
        self._reclaimed: LockBody | None = None

    # -- body helpers ------------------------------------------------------

    def _body(self, *, created_at: str | None = None) -> LockBody:
        now = _now_iso()
        return LockBody(
            lock_schema_version=LOCK_SCHEMA_VERSION,
            pid=os.getpid(),
            process_start_token=current_process_start_token(),
            lock_token=self.token,
            host=platform.node(),
            created_at=created_at or now,
            heartbeat_at=now,
        )

    def _read_raw(self) -> str:
        """One raw read of the lock file (the seam every retry goes through)."""

        return self.path.read_text(encoding="utf-8")

    def _read_state(self) -> tuple[ReadState, LockBody | None]:
        """``absent`` / ``malformed`` / ``present`` (with the parsed body). A
        transient ``OSError`` (RA-02 sharing violation) is retried; a
        PERSISTENT read failure is the typed ``lock_read_failed`` — never
        absence, never a malformed body."""

        last: OSError | None = None
        for _ in range(max(1, int(_FS_RETRY_ATTEMPTS))):
            try:
                raw = self._read_raw()
            except FileNotFoundError:
                return "absent", None
            except OSError as error:
                last = error
                time.sleep(_FS_RETRY_DELAY_SECONDS)
                continue
            body = LockBody.parse(raw)
            return ("present", body) if body is not None else ("malformed", None)
        raise OwnerDecisionLockError(
            "lock_read_failed",
            f"the owner-decision lock body could not be read after retries ({last}); a "
            "persistent read failure is never treated as an absent lock",
        )

    def _read(self) -> LockBody | None:
        """The parsed body, or ``None`` for an absent / malformed lock (a
        persistent read failure raises ``lock_read_failed``)."""

        return self._read_state()[1]

    def _heartbeat_age(self, body: LockBody | None) -> float | None:
        if body is not None:
            stamp = _parse_iso(body.heartbeat_at)
            if stamp is not None:
                return (datetime.now(UTC) - stamp).total_seconds()
        try:
            return time.time() - self.path.stat().st_mtime
        except OSError:
            return None

    # -- acquire / refresh / verify / release --------------------------------

    def _try_create(self) -> bool:
        """Ordinary ``O_CREAT | O_EXCL`` acquisition. When creation succeeds
        but the body cannot be written / fsynced, the partial file is removed
        (or quarantined) BEFORE the typed ``lock_create_failed`` is raised —
        a token-less lock file never survives."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            data = self._body().to_json().encode("utf-8")
            written = 0
            while written < len(data):
                written += os.write(handle, data[written:])
            os.fsync(handle)
        except OSError as error:
            with contextlib.suppress(OSError):
                os.close(handle)
            disposition = self._discard_partial_lock()
            raise OwnerDecisionLockError(
                "lock_create_failed",
                f"the owner-decision lock body could not be written after exclusive creation "
                f"({error}); the partial lock file was {disposition}",
            ) from error
        os.close(handle)
        return True

    def _discard_partial_lock(self) -> str:
        try:
            _retry_fs(lambda: os.unlink(self.path))
            return "removed"
        except FileNotFoundError:
            return "removed"
        except OSError:
            pass
        quarantine = self.path.with_name(f".{self.path.name}.partial-{uuid.uuid4().hex}")
        try:
            _retry_fs(lambda: os.replace(self.path, quarantine))
            return f"quarantined as {quarantine.name}"
        except OSError:
            return "NOT removed (operator intervention required)"

    def _evaluate(self, state: ReadState, body: LockBody | None) -> tuple[bool, str]:
        """Whether the present lock may be reclaimed, with the reason it may
        not. ``absent`` is reported as such (the caller retries creation)."""

        if state == "absent":
            return False, "absent"
        age = self._heartbeat_age(body)
        if age is None:
            return False, "lock_holder_liveness_unknown"
        if age <= self.heartbeat_timeout_seconds:
            return False, "lock_held_by_live_holder"
        if body is None:
            return False, "lock_body_malformed"
        liveness = process_liveness(body.pid, body.process_start_token, body.host)
        if liveness == "dead":
            return True, "dead"
        if liveness == "alive":
            return False, "lock_held_by_live_holder"
        return False, "lock_holder_liveness_unknown"

    def _observe_stale(self) -> tuple[LockBody | None, str]:
        """The UNLOCKED pre-check: the body that APPEARS reclaimable (or
        ``None``) with the reason. This observation grants no authority to
        unlink — only :meth:`_reclaim` (under the reclaim mutex) does."""

        state, body = self._read_state()
        reclaimable, reason = self._evaluate(state, body)
        return (body if reclaimable else None), reason

    def _reclaim(self, observed: LockBody, *, deadline: float) -> tuple[bool, str]:
        """Token-safe reclamation under the dedicated reclaim mutex: re-read
        the ordinary lock, re-evaluate heartbeat age / host / pid liveness /
        start token, and unlink ONLY when the still-current body is exactly
        the same dead holder that was observed (same ``lock_token``, same
        body). Returns ``(reclaimed, reason)``."""

        wait = max(0.0, deadline - time.monotonic())
        mutex = ExclusiveFileMutex(
            self.reclaim_mutex_path, wait_seconds=wait, poll_seconds=self.poll_seconds
        )
        try:
            with mutex:
                state, current = self._read_state()
                reclaimable, reason = self._evaluate(state, current)
                if not reclaimable:
                    return False, reason
                if current != observed:
                    # a different body now holds the file (a new live lock, a
                    # refreshed heartbeat, or another dead holder): the stale
                    # verdict for `observed` grants nothing; the loop observes
                    # the current body afresh
                    return False, "lock_held_by_live_holder"
                try:
                    _retry_fs(lambda: os.unlink(self.path))
                except FileNotFoundError:
                    return False, "absent"
                except OSError:
                    # a persistent sharing violation while reclaiming: keep
                    # waiting (contention), never a silent false success
                    return False, "lock_held_by_live_holder"
                self._reclaimed = current
                return True, "dead"
        except FileMutexTimeoutError:
            return False, "lock_held_by_live_holder"
        except FileMutexError as error:
            raise OwnerDecisionLockError(
                "lock_mutex_failed",
                f"the reclaim mutex failed persistently ({error}); the stale lock cannot be "
                "assessed under the mutex — never reported as a live holder; operator "
                "intervention required",
            ) from error

    def acquire(self) -> LockBody:
        deadline = time.monotonic() + self.wait_seconds
        last_reason = "lock_held_by_live_holder"
        while True:
            if self._try_create():
                try:
                    body = self._read()
                except OwnerDecisionLockError as error:
                    # review RA-04: the file was created by THIS writer microseconds
                    # ago — a persistent read failure must not leave a live,
                    # non-reclaimable orphan behind for the life of this process
                    disposition = self._discard_partial_lock()
                    raise OwnerDecisionLockError(
                        error.reason,
                        f"{error}; the freshly created lock was {disposition} so that the "
                        "store is not denied for the life of this process",
                    ) from error
                self._held = True
                return body if body is not None and body.lock_token == self.token else self._body()
            observed, last_reason = self._observe_stale()
            if observed is not None:
                reclaimed, last_reason = self._reclaim(observed, deadline=deadline)
                if reclaimed:
                    continue
            if last_reason == "absent":
                continue  # the lock vanished between the probes: retry creation now
            if time.monotonic() > deadline:
                detail = (
                    "the owner-decision lock is held by a writer whose heartbeat is current "
                    "or whose process is alive"
                    if last_reason == "lock_held_by_live_holder"
                    else (
                        "the owner-decision lock body is malformed; its holder cannot be "
                        "assessed — operator intervention required"
                        if last_reason == "lock_body_malformed"
                        else "the owner-decision lock holder's liveness cannot be assessed "
                        "(another host or an unassessable platform) — never reclaimed by age "
                        "alone; operator intervention required"
                    )
                )
                raise OwnerDecisionLockError(last_reason, detail)
            time.sleep(self.poll_seconds)

    def refresh(self) -> None:
        """Rewrite the heartbeat (atomically) — refuses when the lock is lost."""

        current = self.verify_held()
        temporary = self.path.with_name(f".{self.path.name}.hb-{uuid.uuid4().hex}")
        temporary.write_text(
            self._body(created_at=current.created_at).to_json(), encoding="utf-8"
        )
        try:
            _retry_fs(lambda: os.replace(temporary, self.path))
        except OSError as error:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise OwnerDecisionLockError(
                "lock_refresh_failed",
                f"the heartbeat could not be published after retries ({error}); the writer "
                "must abort before publication",
            ) from error

    def verify_held(self) -> LockBody:
        """The lock body, which MUST carry our token (else ``lock_lost``); a
        persistent read failure is ``lock_read_failed`` (never a silent pass)."""

        body = self._read()
        if body is None or body.lock_token != self.token:
            self._held = False
            raise OwnerDecisionLockError(
                "lock_lost",
                "the owner-decision lock no longer carries this writer's token; aborting "
                "before publication",
            )
        return body

    def release(self) -> None:
        """Unlink ONLY our own token. A persistent failure to read, verify or
        unlink our own lock is the typed ``lock_release_failed`` (RA-02 /
        HARDENING-BACKEND-FIX §4.1: never a silently orphaned or silently
        vanished lock); a lock that carries another writer's token is left
        untouched."""

        was_held = self._held
        try:
            state, body = self._read_state()
        except OwnerDecisionLockError as error:
            raise OwnerDecisionLockError(
                "lock_release_failed",
                f"the owner-decision lock could not be read to verify ownership before "
                f"release ({error}); the lock may still carry this writer's token — operator "
                "intervention required",
            ) from error
        if state == "absent":
            self._held = False
            if was_held:
                raise OwnerDecisionLockError(
                    "lock_release_failed",
                    "the owner-decision lock file is absent although this writer held the "
                    "lock: another party removed it and a publication may have overlapped",
                )
            return
        if state == "malformed":
            self._held = False
            if was_held:
                raise OwnerDecisionLockError(
                    "lock_release_failed",
                    "the owner-decision lock body is malformed; this writer cannot verify "
                    "its own ownership before release — operator intervention required",
                )
            return
        if body is None or body.lock_token != self.token:
            self._held = False
            if was_held:
                # review RA-01: a lock this writer HELD now carries another writer's
                # token — the one state that means a publication may have
                # overlapped; typed, never a silent success (the foreign lock is
                # left untouched)
                raise OwnerDecisionLockError(
                    "lock_release_failed",
                    "the owner-decision lock carries another writer's token although this "
                    "writer held the lock: it was replaced by a non-conforming actor and a "
                    "publication may have overlapped — operator intervention required",
                )
            return  # never remove a lock this writer does not own
        try:
            _retry_fs(lambda: os.unlink(self.path))
        except FileNotFoundError:
            self._held = False
            if was_held:
                raise OwnerDecisionLockError(
                    "lock_release_failed",
                    "the owner-decision lock vanished between ownership verification and "
                    "release: another party removed this writer's lock",
                ) from None
            return
        except OSError as error:
            raise OwnerDecisionLockError(
                "lock_release_failed",
                f"the owner-decision lock could not be released after retries ({error}); "
                "the lock file still carries this writer's token — operator intervention "
                "required (a live holder is never reclaimed by age)",
            ) from error
        self._held = False

    @property
    def reclaimed_from(self) -> LockBody | None:
        """The dead holder's body when this acquisition reclaimed a lock."""

        return self._reclaimed

    def __enter__(self) -> OwnerDecisionLock:
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
