"""Private cross-platform exclusive file mutex (HARDENING-BACKEND-FIX §4).

Standard library only — ``msvcrt.locking`` byte-range locks on Windows,
``fcntl.flock`` on POSIX — and used ONLY to serialize two short critical
sections that the ordinary ``O_CREAT | O_EXCL`` lock file cannot protect by
itself:

* stale owner-decision-lock reclamation (§4.1: two reclaimers that both
  inspected the same stale body must not both act on it — the second could
  otherwise unlink the live lock the first just acquired);
* one-time store-namespace initialization (§4.2: the namespace envelope and
  the genesis head are published as one coherent pair).

The mutex file carries NO authority: it holds no body and no token, nothing
reads it, and it is never unlinked (unlinking a mutex file would race every
later opener onto a different file object). Byte-range / ``flock`` locks are
per open handle, so the mutex serializes threads of one process as well as
separate processes on the same host.
"""

from __future__ import annotations

import errno
import os
import sys
import time
from pathlib import Path

__all__ = ["FileMutexError", "FileMutexTimeoutError", "ExclusiveFileMutex"]

if sys.platform == "win32":
    import msvcrt
else:  # pragma: win32 no cover
    import fcntl


#: Review RA-02: ONLY the platform's contention errno means "another holder" —
#: ``msvcrt.locking(LK_NBLCK)`` fails with EACCES (EDEADLOCK for the blocking
#: variant), ``fcntl.flock(LOCK_NB)`` with EWOULDBLOCK / EAGAIN. Every other
#: ``OSError`` (EBADF, EIO, ENOSPC …) is a persistent failure of the mutex file
#: itself: typed, never reported as contention (a live holder / a busy
#: initializer that does not exist).
_CONTENTION_ERRNOS: frozenset[int] = frozenset(
    {errno.EACCES, getattr(errno, "EDEADLOCK", 36), getattr(errno, "EDEADLK", 36)}
    if sys.platform == "win32"
    else {errno.EWOULDBLOCK, errno.EAGAIN}
)


class FileMutexError(OSError):
    """The mutex file itself failed with a NON-contention error (typed; a
    caller must never report it as a live holder or a busy initializer)."""


class FileMutexTimeoutError(TimeoutError):
    """The mutex could not be acquired within the wait window."""


class ExclusiveFileMutex:
    """``with ExclusiveFileMutex(path, wait_seconds=...):`` — an exclusive,
    host-local, cross-process critical section keyed by ``path``."""

    def __init__(self, path: Path, *, wait_seconds: float = 30.0, poll_seconds: float = 0.01):
        self.path = Path(path)
        self.wait_seconds = float(wait_seconds)
        self.poll_seconds = float(poll_seconds)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def _try_lock(self, fd: int) -> bool:
        try:
            if sys.platform == "win32":
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:  # pragma: win32 no cover
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in _CONTENTION_ERRNOS:
                return False
            raise FileMutexError(
                error.errno,
                f"the file mutex {self.path.name!r} failed with a non-contention error "
                f"({error}); a persistent mutex failure, not a live holder",
            ) from error
        return True

    def _unlock(self, fd: int) -> None:
        if sys.platform == "win32":
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:  # pragma: win32 no cover
            fcntl.flock(fd, fcntl.LOCK_UN)

    def acquire(self) -> None:
        if self._fd is not None:
            raise RuntimeError("the file mutex is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if sys.platform == "win32":
            flags |= os.O_NOINHERIT
        else:  # pragma: win32 no cover
            flags |= getattr(os, "O_CLOEXEC", 0)
        fd = os.open(str(self.path), flags, 0o644)
        deadline = time.monotonic() + self.wait_seconds
        try:
            while not self._try_lock(fd):
                if time.monotonic() > deadline:
                    raise FileMutexTimeoutError(
                        f"the file mutex {self.path.name!r} was not acquired within "
                        f"{self.wait_seconds:g}s"
                    )
                time.sleep(self.poll_seconds)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            self._unlock(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> ExclusiveFileMutex:
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
