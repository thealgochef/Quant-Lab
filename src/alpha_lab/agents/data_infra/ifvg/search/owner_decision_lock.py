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
* **Heartbeat** — :meth:`OwnerDecisionLock.refresh` rewrites the body
  (atomically) before long verification and before commit.
* **Token re-verification** — :meth:`OwnerDecisionLock.verify_held` reads
  the body back and refuses to continue when the token is not ours (a lost
  lock aborts the writer BEFORE publication).
* **Release** unlinks only its own token.

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

from .store_namespace import OWNER_DECISION_STORE

__all__ = [
    "OWNER_DECISION_LOCK_FILE",
    "LOCK_FAILURE_REASONS",
    "OwnerDecisionLockError",
    "LockBody",
    "current_process_start_token",
    "process_liveness",
    "OwnerDecisionLock",
]

OWNER_DECISION_LOCK_FILE = "SUPERSESSIONS.lock"
LOCK_SCHEMA_VERSION = 1
LOCK_FAILURE_REASONS: tuple[str, ...] = (
    "lock_held_by_live_holder",
    "lock_holder_liveness_unknown",
    "lock_lost",
    "lock_body_malformed",
    "lock_release_failed",
    "lock_refresh_failed",
)
#: Adversarial RA-02: Windows refuses ``unlink`` / ``replace`` while ANY other
#: handle has the lock body open (a polling waiter's read is enough). Every
#: filesystem step of the lock retries a transient ``OSError`` with a short
#: backoff; a PERSISTENT failure is a typed error, never silence.
_FS_RETRY_ATTEMPTS = 40
_FS_RETRY_DELAY_SECONDS = 0.025
Liveness = Literal["alive", "dead", "unknown"]

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


def _retry_fs(operation, *, attempts: int = _FS_RETRY_ATTEMPTS,
              delay: float = _FS_RETRY_DELAY_SECONDS):
    """Run a filesystem operation, retrying transient ``OSError``s (sharing
    violations) with a short backoff; the last error propagates."""

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

    def _read(self) -> LockBody | None:
        for attempt in range(2):  # RA-02: one retry on a transient sharing violation
            try:
                raw = self.path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
            except OSError:
                if attempt == 0:
                    time.sleep(_FS_RETRY_DELAY_SECONDS)
                    continue
                return None
            return LockBody.parse(raw)
        return None

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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            os.write(handle, self._body().to_json().encode("utf-8"))
        finally:
            os.close(handle)
        return True

    def _reclaimable(self) -> tuple[bool, str]:
        """Whether the present lock may be reclaimed, with the reason it may not."""

        body = self._read()
        age = self._heartbeat_age(body)
        if age is None:
            return False, "lock_holder_liveness_unknown"
        if age <= self.heartbeat_timeout_seconds:
            return False, "lock_held_by_live_holder"
        if body is None:
            return False, "lock_body_malformed"
        liveness = process_liveness(body.pid, body.process_start_token, body.host)
        if liveness == "dead":
            self._reclaimed = body
            return True, "dead"
        if liveness == "alive":
            return False, "lock_held_by_live_holder"
        return False, "lock_holder_liveness_unknown"

    def acquire(self) -> LockBody:
        deadline = time.monotonic() + self.wait_seconds
        last_reason = "lock_held_by_live_holder"
        while True:
            if self._try_create():
                self._held = True
                return self._read() or self._body()
            reclaimable, last_reason = self._reclaimable()
            if reclaimable:
                try:
                    _retry_fs(lambda: os.unlink(self.path))
                except FileNotFoundError:
                    pass
                except OSError:
                    # a persistent sharing violation while reclaiming: keep
                    # waiting (contention), never a silent false success
                    last_reason = "lock_held_by_live_holder"
                continue
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
        """The lock body, which MUST carry our token (else ``lock_lost``)."""

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
        """Unlink ONLY our own token; a persistent failure to unlink is a typed
        error (RA-02: never a silently orphaned lock owned by a live pid)."""

        body = self._read()
        self._held = False
        if body is None or body.lock_token != self.token:
            return
        try:
            _retry_fs(lambda: os.unlink(self.path))
        except FileNotFoundError:
            return
        except OSError as error:
            raise OwnerDecisionLockError(
                "lock_release_failed",
                f"the owner-decision lock could not be released after retries ({error}); "
                "the lock file still carries this writer's token — operator intervention "
                "required (a live holder is never reclaimed by age)",
            ) from error

    @property
    def reclaimed_from(self) -> LockBody | None:
        """The dead holder's body when this acquisition reclaimed a lock."""

        return self._reclaimed

    def __enter__(self) -> OwnerDecisionLock:
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
