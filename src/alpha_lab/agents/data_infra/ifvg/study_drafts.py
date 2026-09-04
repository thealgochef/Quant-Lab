"""Disk-persisted wizard drafts for the study workspace (R4; FUX §7, §29;
UI-2 lifecycle per owner Q2).

Drafts are the ONE deliberately mutable authoring surface: JSON files under
``data/ifvg_study_drafts/<draft_id>/draft.json``, written atomically with the
repository's tmp-then-``os.replace`` idiom (``preparation.py`` house style).
They carry no research identity — a draft becomes research-bearing only when
the wizard freezes it into an immutable ``SearchCharterEnvelope`` via the R1
charter store, at which point the draft is marked ``frozen`` and refuses
every further mutation (FUX §15: the draft remains only as historical
provenance). ``Clone as New Search`` deep-copies any draft — frozen or not —
into a new mutable draft; the original is untouched (FUX-WIZ-003).

UI-2 (owner Q2): a new draft lives in the SESSION until the first explicit
Save Draft or the first valid Next (this module writes a file only when
asked); *Archive* is the normal reversible action (archived drafts are hidden
from the default listing and restorable); *permanent delete* exists only for
drafts that were never frozen and never launched, only from the archived
view, and only with the exact typed draft name; frozen drafts are provenance
and never deletable; ``discard_draft`` (the R4 hard delete) is retired and
deletes nothing; the empty untitled step-0 files of the R4 era are archived
by a one-time bulk action — never deleted.

This module is Streamlit-free and fully unit-testable.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

__all__ = [
    "DRAFT_SCHEMA_VERSION",
    "STUDY_DRAFT_ROOT",
    "STEP_KEYS",
    "UNTITLED_PREFIX",
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
    "archive_draft",
    "restore_draft",
    "delete_draft_permanently",
    "is_empty_untitled_draft",
    "bulk_archive_empty_untitled_drafts",
    "find_duplicate_drafts",
    "proposed_draft_name",
]

#: UI-2: schema 2 adds the ADDITIVE lifecycle fields (``archived``,
#: ``archived_at_utc``) and the exact-restore ``current_step_key``; schema-1
#: files load unchanged (unknown keys are filtered, absent keys default).
DRAFT_SCHEMA_VERSION = 2

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

#: The R4 default display-name prefix of a draft nobody named.
UNTITLED_PREFIX = "Untitled study ("


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
    #: UI-1 (plan §5.6): the MUTABLE presentation-only run-purpose annotation
    #: (``RunPurposeAnnotation.to_dict()``); never part of a charter or
    #: pipeline identity. Absent on legacy drafts — the wizard derives a
    #: purpose only when the legacy scope is unambiguous, else the draft is
    #: ``purpose_unresolved`` until the owner confirms it.
    purpose_annotation: dict[str, Any] | None = None
    #: UI-2 (owner Q2): the reversible archive flag (hidden from the default
    #: listing; restorable; never a delete) and its instant.
    archived: bool = False
    archived_at_utc: str | None = None
    #: UI-2 (plan §5.4): the flow step key the draft is on — exact restore
    #: under a goal-conditional flow (``step_index`` stays as the legacy
    #: eight-step position for schema-1 readers).
    current_step_key: str | None = None

    def step_payload(self, step_key: str) -> dict[str, Any]:
        if step_key not in STEP_KEYS:
            raise DraftError(f"unknown wizard step {step_key!r}")
        return self.steps.setdefault(step_key, {})

    @property
    def never_frozen(self) -> bool:
        """True only for a draft that was never frozen (and therefore never
        launched): the sole class owner Q2 admits to permanent deletion."""

        return self.status != "frozen" and not self.frozen_search_id


def new_draft(
    mode_id: str,
    *,
    display_name: str = "",
    now_fn: Callable[[], str] = _utc_now,
) -> StudyDraft:
    now = now_fn()
    return StudyDraft(
        draft_id=uuid4().hex,
        display_name=display_name or f"{UNTITLED_PREFIX}{now[:10]})",
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
    check-then-write transition (save / freeze / archive / restore / delete)
    holds it, so a freeze can never be silently overwritten by a concurrent
    save."""

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


def _stored_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
        stored = _stored_record(path)
        if stored is not None and stored.get("status") == "frozen":
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


def list_drafts(root: Path, *, include_archived: bool = False) -> tuple[StudyDraft, ...]:
    """Every persisted draft, newest-updated first (mutable namespace only).
    Archived drafts are hidden unless ``include_archived`` (owner Q2)."""

    root = Path(root)
    if not root.exists():
        return ()
    drafts: list[StudyDraft] = []
    for child in sorted(root.iterdir()):
        if (child / _DRAFT_FILE).exists():
            try:
                draft = load_draft(root, child.name)
            except (DraftError, json.JSONDecodeError, TypeError):
                continue  # unreadable drafts never break the listing
            if draft.archived and not include_archived:
                continue
            drafts.append(draft)
    drafts.sort(key=lambda draft: draft.updated_at_utc, reverse=True)
    return tuple(drafts)


def discard_draft(root: Path, draft_id: str) -> None:
    """RETIRED (UI-2, owner Q2): the R4 hard delete deletes nothing any more.

    Archive the draft (``archive_draft``); a never-frozen archived draft may
    then be deleted permanently from the archived view with its exact typed
    name (``delete_draft_permanently``).
    """

    raise DraftError(
        "discard_draft is retired (UI-2, owner Q2): archive the draft, then delete it "
        "permanently from the Archived view with its exact name — nothing was deleted"
    )


def clone_draft(
    source: StudyDraft,
    *,
    now_fn: Callable[[], str] = _utc_now,
) -> StudyDraft:
    """Deep-copy any draft (frozen or archived included) into a new mutable,
    LIVE draft."""

    now = now_fn()
    annotation = copy.deepcopy(source.purpose_annotation)
    if isinstance(annotation, dict):
        annotation = {**annotation, "derivation": "cloned", "updated_at": now}
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
        purpose_annotation=annotation,
        archived=False,
        archived_at_utc=None,
        current_step_key=source.current_step_key,
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
        stored = _stored_record(path)
        if stored is not None and stored.get("status") == "frozen":
            raise DraftFrozenError("draft is already frozen")
        draft.status = "frozen"
        draft.frozen_search_id = search_id
        draft.updated_at_utc = now_fn()
        _write_json_atomic(path, asdict(draft))
    return draft


# ─────────────────────────────────────────────────────────────────────────────
# UI-2 lifecycle (owner Q2): archive / restore / typed permanent delete
# ─────────────────────────────────────────────────────────────────────────────


def _load_under_lock(root: Path, draft_id: str) -> tuple[Path, StudyDraft]:
    path = _draft_path(root, draft_id)
    if not path.exists():
        raise DraftNotFoundError(f"no draft {draft_id[:12]}…")
    return path, load_draft(root, draft_id)


def archive_draft(
    root: Path, draft_id: str, *, now_fn: Callable[[], str] = _utc_now
) -> StudyDraft:
    """The normal reversible action: hide the draft from the default listing.
    Frozen drafts are historical provenance and are refused (they are listed
    with their runs, never as mutable drafts)."""

    with _draft_lock(root, draft_id):
        path, draft = _load_under_lock(root, draft_id)
        if draft.status == "frozen":
            raise DraftFrozenError(
                "frozen drafts are historical provenance and are never archived here; the "
                "run carries the catalog archive flag"
            )
        if draft.archived:
            raise DraftError("this draft is already archived")
        now = now_fn()
        draft.archived = True
        draft.archived_at_utc = now
        draft.updated_at_utc = now
        _write_json_atomic(path, asdict(draft))
    return draft


def restore_draft(
    root: Path, draft_id: str, *, now_fn: Callable[[], str] = _utc_now
) -> StudyDraft:
    """Undo an archive: the draft returns to the default listing unchanged."""

    with _draft_lock(root, draft_id):
        path, draft = _load_under_lock(root, draft_id)
        if not draft.archived:
            raise DraftError("this draft is not archived")
        draft.archived = False
        draft.archived_at_utc = None
        draft.updated_at_utc = now_fn()
        _write_json_atomic(path, asdict(draft))
    return draft


def delete_draft_permanently(root: Path, draft_id: str, *, confirm_name: str) -> None:
    """Owner Q2: permanent deletion ONLY for a draft that was never frozen and
    never launched, ONLY once archived (the Archived / Advanced view), and
    ONLY with the exact typed display name. Every other draft is refused;
    nothing is ever deleted implicitly."""

    with _draft_lock(root, draft_id):
        path, draft = _load_under_lock(root, draft_id)
        if not draft.never_frozen:
            raise DraftFrozenError(
                "a draft that was frozen or launched is historical provenance and is never "
                "deletable (catalog archive flag only)"
            )
        if not draft.archived:
            raise DraftError(
                "permanent deletion is available only from the Archived view — archive it "
                "first (the reversible action)"
            )
        if str(confirm_name) != draft.display_name:
            raise DraftError(
                "type the exact draft name to delete it permanently (case and whitespace "
                "exact); nothing was deleted"
            )
        path.unlink()
    with contextlib.suppress(OSError):
        _draft_path(root, draft_id).with_name(_LOCK_FILE).unlink()
    with contextlib.suppress(OSError):
        (Path(root) / draft_id).rmdir()  # leftovers keep the dir; never force


def is_empty_untitled_draft(draft: StudyDraft) -> bool:
    """The R4-era artifact owner Q2 names: a never-named, never-annotated,
    never-frozen, never-advanced draft whose every step payload is empty."""

    return (
        draft.never_frozen
        and not draft.archived
        and draft.display_name.startswith(UNTITLED_PREFIX)
        and draft.purpose_annotation is None
        and int(draft.step_index) == 0
        and all(not payload for payload in draft.steps.values())
    )


def bulk_archive_empty_untitled_drafts(
    root: Path, *, now_fn: Callable[[], str] = _utc_now
) -> tuple[str, ...]:
    """The one-time migration: archive (never delete) every empty untitled
    step-0 draft; returns the archived ids. Named, advanced, annotated,
    frozen and already-archived drafts are untouched. Idempotent."""

    archived: list[str] = []
    for draft in list_drafts(Path(root)):
        if is_empty_untitled_draft(draft):
            archive_draft(root, draft.draft_id, now_fn=now_fn)
            archived.append(draft.draft_id)
    return tuple(archived)


def find_duplicate_drafts(
    drafts: Iterable[StudyDraft],
    *,
    mode_id: str,
    question_id: str | None,
    purpose: str | None,
    baseline_profile_name: str | None,
    exclude_draft_id: str | None = None,
) -> tuple[StudyDraft, ...]:
    """Persisted LIVE drafts with the same mode, research question, purpose
    annotation and baseline profile — the wizard warns before a second
    identical draft is persisted (owner Q2). Never a block."""

    found: list[StudyDraft] = []
    for draft in drafts:
        if draft.draft_id == exclude_draft_id or draft.archived or draft.status == "frozen":
            continue
        if draft.mode_id != mode_id:
            continue
        objective = draft.steps.get("objective") or {}
        if str(objective.get("question_id") or "") != str(question_id or ""):
            continue
        annotation = draft.purpose_annotation or {}
        if str(annotation.get("purpose") or "") != str(purpose or ""):
            continue
        baseline = draft.steps.get("baseline") or {}
        if str(baseline.get("baseline_profile_name") or "") != str(baseline_profile_name or ""):
            continue
        found.append(draft)
    return tuple(found)


def proposed_draft_name(
    goal_label: str, *, baseline_profile_name: str | None, day: str
) -> str:
    """Owner Q2's default proposal ``<goal> — <baseline short name> — <date>``
    (the baseline part is omitted while no baseline is selected)."""

    parts = [str(goal_label).strip()]
    if baseline_profile_name:
        short = str(baseline_profile_name)
        for prefix in ("ifvg_v2_", "ifvg_"):
            if short.startswith(prefix):
                short = short[len(prefix) :]
                break
        parts.append(short.replace("_", " "))
    parts.append(str(day))
    return " — ".join(parts)
