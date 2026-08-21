"""Disk-persisted wizard drafts for the study workspace (R4; FUX §7, §29).

Drafts are the ONE deliberately mutable authoring surface: JSON files under
``data/ifvg_study_drafts/<draft_id>/draft.json``, written atomically with the
repository's tmp-then-``os.replace`` idiom (``preparation.py`` house style).
They carry no research identity — a draft becomes research-bearing only when
the wizard freezes it into an immutable ``SearchCharterEnvelope`` via the R1
charter store, at which point the draft is marked ``frozen`` and refuses
every further mutation (FUX §15: the draft remains only as historical
provenance). ``Clone as New Search`` deep-copies any draft — frozen or not —
into a new mutable draft; the original is untouched (FUX-WIZ-003).

This module is Streamlit-free and fully unit-testable.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

__all__ = [
    "DRAFT_SCHEMA_VERSION",
    "STUDY_DRAFT_ROOT",
    "STEP_KEYS",
    "StudyDraft",
    "DraftError",
    "DraftFrozenError",
    "DraftNotFoundError",
    "new_draft",
    "save_draft",
    "load_draft",
    "list_drafts",
    "discard_draft",
    "clone_draft",
    "mark_frozen",
]

DRAFT_SCHEMA_VERSION = 1

#: The mutable draft namespace (a sibling of the job/state roots — never
#: inside the immutable search store).
STUDY_DRAFT_ROOT = Path("data/ifvg_study_drafts")

_DRAFT_FILE = "draft.json"

#: One payload mapping per wizard step, keyed in step order (FUX §7).
STEP_KEYS: tuple[str, ...] = (
    "objective",
    "baseline",
    "search_space",
    "prop_contracts",
    "risk_policies",
    "benchmarks",
    "validation",
    "review",
)


class DraftError(ValueError):
    """A draft operation violated the draft policy."""


class DraftFrozenError(DraftError):
    """The stored draft is frozen; mutation and discard are refused."""


class DraftNotFoundError(DraftError):
    """No draft exists under the given id."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class StudyDraft:
    """A mutable wizard draft. NOT an identity-bearing contract."""

    draft_id: str
    display_name: str
    mode_id: str
    step_index: int = 0
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    status: str = "draft"  # "draft" | "frozen"
    frozen_search_id: str | None = None
    cloned_from: str | None = None
    created_at_utc: str = ""
    updated_at_utc: str = ""
    schema_version: int = DRAFT_SCHEMA_VERSION

    def step_payload(self, step_key: str) -> dict[str, Any]:
        if step_key not in STEP_KEYS:
            raise DraftError(f"unknown wizard step {step_key!r}")
        return self.steps.setdefault(step_key, {})


def new_draft(
    mode_id: str,
    *,
    display_name: str = "",
    now_fn: Callable[[], str] = _utc_now,
) -> StudyDraft:
    now = now_fn()
    return StudyDraft(
        draft_id=uuid4().hex,
        display_name=display_name or f"Untitled study ({now[:10]})",
        mode_id=mode_id,
        created_at_utc=now,
        updated_at_utc=now,
    )


#: uuid4().hex shape — the ONLY accepted draft-id form (path-safe by
#: construction; Windows drive-relative components and device names cannot
#: match).
_DRAFT_ID = re.compile(r"^[0-9a-f]{32}$")

_LOCK_FILE = "draft.lock"
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.02
_STALE_LOCK_SECONDS = 30.0


def _draft_path(root: Path, draft_id: str) -> Path:
    if not _DRAFT_ID.fullmatch(draft_id or ""):
        raise DraftError("draft id must be a 32-hex identifier")
    return Path(root) / draft_id / _DRAFT_FILE


@contextlib.contextmanager
def _draft_lock(root: Path, draft_id: str):
    """O_EXCL per-draft lock (the repo's catalog-lock idiom): every
    check-then-write transition (save / freeze / discard) holds it, so a
    freeze can never be silently overwritten by a concurrent save."""

    lock = _draft_path(root, draft_id).with_name(_LOCK_FILE)
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(handle)
            break
        except FileExistsError:
            with contextlib.suppress(OSError):
                if time.time() - lock.stat().st_mtime > _STALE_LOCK_SECONDS:
                    lock.unlink()
                    continue
            if time.monotonic() > deadline:
                raise DraftError(
                    f"draft {draft_id[:12]}… is locked by another writer"
                ) from None
            time.sleep(_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def save_draft(
    root: Path,
    draft: StudyDraft,
    *,
    now_fn: Callable[[], str] = _utc_now,
) -> Path:
    """Atomically persist ``draft``; refuses to mutate a frozen record.

    The freeze transition itself goes through :func:`mark_frozen` (which
    performs the one permitted final write). The frozen check and the write
    happen under the per-draft lock, so a concurrent freeze cannot be
    silently overwritten.
    """

    path = _draft_path(root, draft.draft_id)
    if draft.status == "frozen":
        raise DraftFrozenError(
            "freezing goes through mark_frozen(), never a plain save"
        )
    with _draft_lock(root, draft.draft_id):
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("status") == "frozen":
                raise DraftFrozenError(
                    "this draft is frozen historical provenance; use "
                    "'Clone as New Search' to start a mutable copy"
                )
        draft.updated_at_utc = now_fn()
        _write_json_atomic(path, asdict(draft))
    return path


def load_draft(root: Path, draft_id: str) -> StudyDraft:
    path = _draft_path(root, draft_id)
    if not path.exists():
        raise DraftNotFoundError(f"no draft {draft_id[:12]}…")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("draft_id") != draft_id:
        raise DraftError(
            "draft file identity does not match its directory; refusing the "
            "record"
        )
    known = {f for f in StudyDraft.__dataclass_fields__}
    return StudyDraft(**{k: v for k, v in payload.items() if k in known})


def list_drafts(root: Path) -> tuple[StudyDraft, ...]:
    """Every persisted draft, newest-updated first (mutable namespace only)."""

    root = Path(root)
    if not root.exists():
        return ()
    drafts: list[StudyDraft] = []
    for child in sorted(root.iterdir()):
        if (child / _DRAFT_FILE).exists():
            try:
                drafts.append(load_draft(root, child.name))
            except (DraftError, json.JSONDecodeError, TypeError):
                continue  # unreadable drafts never break the listing
    drafts.sort(key=lambda draft: draft.updated_at_utc, reverse=True)
    return tuple(drafts)


def discard_draft(root: Path, draft_id: str) -> None:
    """Delete a mutable draft; frozen provenance refuses (FUX §29)."""

    with _draft_lock(root, draft_id):
        draft = load_draft(root, draft_id)
        if draft.status == "frozen":
            raise DraftFrozenError(
                "frozen drafts are historical provenance and cannot be "
                "discarded"
            )
        path = _draft_path(root, draft_id)
        path.unlink()
    with contextlib.suppress(OSError):
        _draft_path(root, draft_id).with_name(_LOCK_FILE).unlink()
    with contextlib.suppress(OSError):
        (Path(root) / draft_id).rmdir()  # leftovers keep the dir; never force


def clone_draft(
    source: StudyDraft,
    *,
    now_fn: Callable[[], str] = _utc_now,
) -> StudyDraft:
    """Deep-copy any draft (frozen included) into a new mutable draft."""

    now = now_fn()
    return StudyDraft(
        draft_id=uuid4().hex,
        display_name=f"{source.display_name} (clone)",
        mode_id=source.mode_id,
        step_index=source.step_index,
        steps=copy.deepcopy(source.steps),
        status="draft",
        frozen_search_id=None,
        cloned_from=source.draft_id,
        created_at_utc=now,
        updated_at_utc=now,
    )


def mark_frozen(
    root: Path,
    draft: StudyDraft,
    *,
    search_id: str,
    now_fn: Callable[[], str] = _utc_now,
) -> StudyDraft:
    """The one permitted final write: stamp the frozen charter linkage.

    After this, the on-disk record refuses every further save/discard; the
    charter itself lives in the immutable R1 store under ``search_id``.
    Runs under the per-draft lock, so a second freezer (or a racing save)
    observes the stored frozen status and refuses.
    """

    if draft.status == "frozen":
        raise DraftFrozenError("draft is already frozen")
    with _draft_lock(root, draft.draft_id):
        path = _draft_path(root, draft.draft_id)
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("status") == "frozen":
                raise DraftFrozenError("draft is already frozen")
        draft.status = "frozen"
        draft.frozen_search_id = search_id
        draft.updated_at_utc = now_fn()
        _write_json_atomic(path, asdict(draft))
    return draft
