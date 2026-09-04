"""Concurrency-safe mutable catalog: lock-guarded append-only event log plus a
deterministic, rebuildable index (§8; revision P0-18).

The catalog holds ONLY mutable display annotations (display names, notes,
stars, archive flags). Research metrics and identities live in immutable
artifacts; the catalog can be deleted and rebuilt from its event log at any
time. Appends are serialized by an ``O_CREAT|O_EXCL`` lock file, so concurrent
child publishers never lose writes, and the rebuilder recovers from a torn
final line.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "CATALOG_EVENTS_FILENAME",
    "CatalogLockTimeoutError",
    "CatalogEvent",
    "append_catalog_event",
    "read_catalog_events",
    "rebuild_catalog_index",
]

CATALOG_EVENTS_FILENAME = "catalog_events.jsonl"
_LOCK_FILENAME = "catalog_events.lock"

CatalogEventKind = Literal["display_name", "note", "star", "archive", "purpose"]


class CatalogLockTimeoutError(TimeoutError):
    """The catalog append lock could not be acquired in time."""


class CatalogEvent(dict):
    """One appended event: kept dict-shaped for exact JSONL round-tripping."""


def _lock_path(root: Path) -> Path:
    return Path(root) / _LOCK_FILENAME


def _events_path(root: Path) -> Path:
    return Path(root) / CATALOG_EVENTS_FILENAME


def _try_break_stale_lock(lock: Path, stale_seconds: float) -> bool:
    """Remove an orphaned lock (crashed writer) once it is provably stale."""

    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return False  # lock vanished or is in transition — treat as contention
    if age < stale_seconds:
        return False
    try:
        os.unlink(lock)
    except OSError:
        return False  # another waiter broke it first, or it is delete-pending
    return True


def append_catalog_event(
    root: Path,
    *,
    kind: CatalogEventKind,
    artifact_id: str,
    payload: Any,
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.02,
    stale_lock_seconds: float = 30.0,
) -> dict:
    """Append one event under the exclusive lock; returns the appended record.

    Contention handling is Windows-truthful: after a holder unlinks the lock,
    a concurrent ``O_CREAT|O_EXCL`` open can observe the delete-pending file
    as ``PermissionError`` rather than ``FileExistsError`` — both are
    contention, never data loss (P0-18). An orphaned lock from a crashed
    writer is broken once its mtime is older than ``stale_lock_seconds``.
    """

    if kind not in ("display_name", "note", "star", "archive", "purpose"):
        raise ValueError(f"unknown catalog event kind {kind!r}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "event_id": uuid.uuid4().hex,
        "ts": time.time_ns(),
        "kind": kind,
        "artifact_id": str(artifact_id),
        "payload": payload,
        "writer_pid": os.getpid(),
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    deadline = time.monotonic() + timeout_seconds
    lock = _lock_path(root)
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError, OSError):
            # FileExistsError: a live holder. PermissionError/OSError: the
            # just-unlinked lock in delete-pending state (Windows) — retry.
            if time.monotonic() >= deadline:
                if _try_break_stale_lock(lock, stale_lock_seconds):
                    deadline = time.monotonic() + timeout_seconds
                    continue
                raise CatalogLockTimeoutError(
                    f"catalog lock {lock.name} not released within {timeout_seconds}s"
                ) from None
            time.sleep(poll_seconds)
    try:
        os.write(handle, str(os.getpid()).encode("ascii"))
        with _events_path(root).open("a", encoding="utf-8") as events:
            events.write(line + "\n")
            events.flush()
            os.fsync(events.fileno())
    finally:
        os.close(handle)
        with contextlib.suppress(OSError):  # lock broken by a stale-lock waiter
            os.unlink(lock)
    return record


def read_catalog_events(root: Path) -> tuple[tuple[dict, ...], int]:
    """All well-formed events in file order plus the count of torn lines.

    Only a torn FINAL line (a crash mid-append) is recoverable; a malformed
    interior line indicates corruption and raises.
    """

    path = _events_path(root)
    if not path.exists():
        return (), 0
    raw_lines = path.read_text(encoding="utf-8").split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    events: list[dict] = []
    torn = 0
    for index, line in enumerate(raw_lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(raw_lines) - 1:
                torn = 1
                break
            raise ValueError(
                f"catalog event log has a malformed interior line at {index}"
            ) from None
    return tuple(events), torn


def rebuild_catalog_index(
    events: Iterable[dict],
    manifests: Iterable[str] = (),
) -> dict:
    """Deterministically fold the event log into the read model.

    ``manifests`` is the set of known immutable artifact ids (from store
    manifests); events for unknown artifacts are retained under
    ``unmatched_artifact_ids`` rather than dropped, so a rebuild against a
    partial manifest listing loses nothing.
    """

    known = set(manifests)
    entries: dict[str, dict] = {}
    unmatched: set[str] = set()
    for event in events:
        artifact_id = str(event["artifact_id"])
        entry = entries.setdefault(
            artifact_id,
            {
                "artifact_id": artifact_id,
                "display_name": None,
                "note": None,
                "starred": False,
                "archived": False,
                "purpose": None,
                "last_event_id": None,
            },
        )
        kind = event["kind"]
        if kind == "display_name":
            entry["display_name"] = event["payload"]
        elif kind == "note":
            entry["note"] = event["payload"]
        elif kind == "star":
            entry["starred"] = bool(event["payload"])
        elif kind == "archive":
            entry["archived"] = bool(event["payload"])
        elif kind == "purpose":
            # UI-1 (plan §5.6): the MUTABLE presentation-only run-purpose
            # annotation of a frozen charter / pipeline — never an identity
            entry["purpose"] = event["payload"]
        else:
            raise ValueError(f"unknown catalog event kind {kind!r}")
        entry["last_event_id"] = event["event_id"]
        if known and artifact_id not in known:
            unmatched.add(artifact_id)
    return {
        "index_schema_version": 1,
        "entries": {key: entries[key] for key in sorted(entries)},
        "unmatched_artifact_ids": tuple(sorted(unmatched)),
    }
