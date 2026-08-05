"""``ifvg_visual_review_v1`` — the human audit ledger for the visual verifier.

A separate, append-only annotation layer so chart-review conclusions live with
the lab.  It never touches immutable artifacts: every write lands under
``data/ifvg_visual_review/`` and nowhere else.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "VISUAL_REVIEW_ROOT",
    "VISUAL_REVIEW_LEDGER",
    "REVIEW_VERDICTS",
    "REVIEW_TAGS",
    "VisualReviewError",
    "append_review",
    "list_reviews",
    "export_csv",
]

VISUAL_REVIEW_ROOT = Path("data/ifvg_visual_review")
VISUAL_REVIEW_LEDGER = VISUAL_REVIEW_ROOT / "review_v1.jsonl"

REVIEW_VERDICTS = ("correct", "incorrect", "questionable", "insufficient_evidence")
REVIEW_TAGS = (
    "wrong_htf_fvg",
    "wrong_parent",
    "wrong_opposing_gap",
    "wrong_inversion",
    "late_entry",
    "bad_stop_window",
    "lookahead_suspected",
    "chart_matches_strategy",
    # setup-level review tags (additive; ifvg_fsm_audit_v1 evidence)
    "premature_htf_fill_death",
    "over_restrictive_guard_suspected",
    "parent_window_too_short",
    "missed_viable_parent",
    "wrong_slot_death_evidence",
    "parentless_interval_mismatch",
    "fill_depth_disputed",
)
_VERDICT_FIELDS = (
    "overall_verdict",
    "fvg_geometry_verdict",
    "lifecycle_verdict",
    "entry_verdict",
    "stop_verdict",
    "outcome_verdict",
    # setup-level verdict fields (additive; optional on candidate reviews)
    "htf_verdict",
    "parent_verdict",
    "opposing_verdict",
    "inversion_verdict",
    "fill_verdict",
)


class VisualReviewError(ValueError):
    pass


def append_review(
    *,
    repo_root: Path,
    replay_chart_artifact_id: str,
    pair_ref: dict[str, str],
    candidate_id: str,
    decision_id: str | None,
    trade_id: str | None,
    reviewer: str,
    verdicts: dict[str, str],
    tags: list[str],
    notes: str,
    setup_id: str | None = None,
    fsm_audit_artifact_id: str | None = None,
) -> dict[str, Any]:
    """Append one review record; the ledger is the only writable location.

    ``setup_id``/``fsm_audit_artifact_id`` are the additive setup-level
    review keys; a setup-level review passes ``candidate_id=""`` for a
    candidate-less setup and MUST carry both new keys."""
    if not reviewer.strip():
        raise VisualReviewError("a reviewer name is required")
    if not str(candidate_id).strip() and setup_id is None:
        raise VisualReviewError(
            "a review requires a candidate_id or a setup_id"
        )
    if setup_id is not None and not str(setup_id).strip():
        raise VisualReviewError("setup_id must be non-empty when provided")
    if setup_id is not None and fsm_audit_artifact_id is None:
        raise VisualReviewError(
            "a setup-level review must pin its fsm_audit_artifact_id"
        )
    for name, value in verdicts.items():
        if name not in _VERDICT_FIELDS:
            raise VisualReviewError(f"unknown verdict field: {name}")
        if value not in REVIEW_VERDICTS:
            raise VisualReviewError(f"unknown verdict value: {value}")
    if "overall_verdict" not in verdicts:
        raise VisualReviewError("an overall verdict is required")
    unknown_tags = sorted(set(tags) - set(REVIEW_TAGS))
    if unknown_tags:
        raise VisualReviewError(f"unknown tags: {unknown_tags}")

    record = {
        "review_id": uuid.uuid4().hex,
        "schema": "ifvg_visual_review_v1",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "replay_chart_artifact_id": replay_chart_artifact_id,
        "pair_ref": dict(sorted(pair_ref.items())),
        "candidate_id": candidate_id,
        "decision_id": decision_id,
        "trade_id": trade_id,
        "setup_id": setup_id,
        "fsm_audit_artifact_id": fsm_audit_artifact_id,
        "reviewer": reviewer.strip(),
        **{name: verdicts.get(name) for name in _VERDICT_FIELDS},
        "tags": sorted(set(tags)),
        "notes": notes.strip(),
    }
    ledger = Path(repo_root) / VISUAL_REVIEW_LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def list_reviews(
    *,
    repo_root: Path,
    candidate_id: str | None = None,
    replay_chart_artifact_id: str | None = None,
    setup_id: str | None = None,
) -> pd.DataFrame:
    ledger = Path(repo_root) / VISUAL_REVIEW_LEDGER
    if not ledger.is_file():
        return pd.DataFrame()
    records = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise VisualReviewError("review ledger has a corrupt line") from error
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    if candidate_id is not None:
        frame = frame[frame["candidate_id"].astype(str) == candidate_id]
    if setup_id is not None and "setup_id" in frame.columns:
        frame = frame[frame["setup_id"].astype(str) == setup_id]
    if replay_chart_artifact_id is not None:
        frame = frame[
            frame["replay_chart_artifact_id"].astype(str) == replay_chart_artifact_id
        ]
    return frame.sort_values("reviewed_at", kind="mergesort").reset_index(drop=True)


def export_csv(*, repo_root: Path) -> str:
    frame = list_reviews(repo_root=repo_root)
    if frame.empty:
        return ""
    flat = frame.copy()
    if "tags" in flat.columns:
        flat["tags"] = flat["tags"].map(
            lambda value: "|".join(value) if isinstance(value, list) else value
        )
    if "pair_ref" in flat.columns:
        flat["pair_ref"] = flat["pair_ref"].map(json.dumps)
    return flat.to_csv(index=False)
