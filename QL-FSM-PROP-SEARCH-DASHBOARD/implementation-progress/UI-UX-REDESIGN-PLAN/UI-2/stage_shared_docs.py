"""Stage the three shared docs as HEAD + UI-2 lane transforms ONLY.

Same mechanism as ``../UI-1/stage_shared_docs.py`` (user hunks never enter the
release commit; ``--apply-worktree`` post-commit replays the same transforms so
the surviving worktree diff is the user's pre-existing hunks only). HEAD for
UI-2 is the UI-1 release head (``f1827e8``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
UI-2 additions (Phase 2 of the UI/UX redesign over the UI-1 release head; UI-3 … UI-6 follow;
acceptance still transitively blocked by R1):

- **Verification Center** (`scripts/ifvg_verification_center.py`): six state-driven sections —
  purpose / readiness (every typed state on one sticky card), the logical trading-day fixture
  (the shortlist re-validated through its contract; logical days and physical partitions as
  separate tables; the owner's PROVISIONAL window recorded in the mutable center root — the
  document keeps `owner_selection = NOT PERFORMED`), the seed lane (the unsigned packet prepared
  in-app; the owner's registration and the seed job as exact external CLI commands whose
  `--receipt-out` receipts are picked up by exact id; the authorization, the run receipt and the
  seed snapshot verified through the backend's own loaders), the final authorization (the unsigned
  packet after a verified seed; the owner's completed reference validated typed, never persisted
  here), review / run (the exact-baseline charter frozen from the SIGNED reference — its content
  hash is the 21/R-5 decision artifact, never a run id — the pipeline spec, the registered
  `VerificationRunEnvelope`, the §6.1 preflight and, only when it passes, the exact bounded-run
  command) and the monitor of the resolved seed / verification stages only. No spawn seam, no
  Publish route; nothing signs, produces a seed, launches or registers the program allowlist.
- **Presentation package**: `flows` — the goal-derived conditional flows (plan §5.4; skipped
  steps carry a visible reason and contribute nothing; a selected prop objective keeps the
  contract step and blocks there; the Validation step stays in every research flow; exact restore
  by the stored step key); `review_vocabulary` — the owner-approved verdict labels over the
  preserved `ifvg_visual_review_v1` keys plus the additive `not_applicable`; `Unreviewed` is a
  UI state only.
- **Drafts** (`study_drafts.py`, schema 2 — additive): session-only until the first Save Draft or
  the first valid Next, then autosave with a visible chip; archive / restore; permanent delete
  only for never-frozen archived drafts with the exact typed name; the one-time bulk archive of
  the empty untitled drafts; duplicate detection; `discard_draft` retired.
- **Providers** (`study_providers.py`): the shortlist / inventory / center-record / seed-state /
  signed-reference / run-registration / preflight / monitor read models;
  `verification_bundle_from_signed_ref` (run-independent).
- **Seed CLI** (`scripts/ifvg_seed_production.py`): `register-authorization`, `--receipt-out`.
- **UI**: History (archive / restore / typed delete, bulk archive, read-only purpose / store
  filters, the run archive flag); the verifier review form (per-case keys, `Unreviewed`,
  definitions, explicit Save Review, `Unsaved` / `Saved`); the wizard's goal-derived steps and
  the seven additive §31 states.
- **Unchanged**: Strategy-Core, the fixed M0–M3 lane, every immutable artifact identity, the
  exact-ID loading rule, S11, MBP-1's `research_only_offline` boundary, the backend contracts.
"""

README_OLD = """runtime truth, direction-aware colorscales and evaluated-gate banners —
`ifvg/presentation/{run_purpose,charter_satisfiability,status_vocabulary}.py`)
implementations are complete;"""
README_NEW = """runtime truth, direction-aware colorscales and evaluated-gate banners —
`ifvg/presentation/{run_purpose,charter_satisfiability,status_vocabulary}.py`),
and UI-2 (Phase 2: the complete Verification Center — fixture, seed,
final authorization, review / run, monitor — with no spawn seam, the
goal-derived conditional flows, session-only drafts with archive /
restore / typed delete, the explicit reviewer verdicts and the seed CLI
receipt seams — `ifvg/presentation/{flows,review_vocabulary}.py`,
`scripts/ifvg_verification_center.py`)
implementations are complete;"""

YAML_OLD = (
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)
YAML_NEW = (
    '    UI_2_verification_center_flows_drafts_reviews: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
    'ui_phase: "2 of 6 (UI-UX-REDESIGN-PLAN revision 2)", '
    'surfaces: "scripts/ifvg_verification_center.py (six sections; no spawn seam; no Publish), '
    "ifvg/presentation/{flows,review_vocabulary}.py, study_drafts schema 2, History lifecycle, "
    'verifier review form", '
    'contracts_consumed: "VerificationWindowShortlist, SeedProductionAuthorization / Run, '
    "SeedSnapshot, VerificationAuthorizationRef, VerificationRunEnvelope, "
    'preflight_bounded_verification", '
    'external_steps: "seed registration and job, owner signature, bounded run — exact CLI '
    'commands with --receipt-out pickup", '
    'findings_closed: "F-06 F-08 F-12(part) F-13(workflow)", '
    'browser_acceptance: "open (UI-6)", '
    'owner_actions_performed: "none (no signing, no seed production, no real verification run)"}\n'
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [(README_OLD, README_NEW)],
    "docs/pipeline_state.yaml": [(YAML_OLD, YAML_NEW)],
}


def _git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], input=data, capture_output=True, check=True)
    return result.stdout


def _transformed(head: str, transforms: list[tuple[str | None, str]]) -> str:
    staged = head
    for old, new in transforms:
        if old is None:
            staged = staged.rstrip("\n") + "\n" + new
        else:
            if old not in staged:
                raise SystemExit(f"transform anchor missing at HEAD: {old[:60]!r}")
            staged = staged.replace(old, new)
    return staged


def main() -> int:
    apply_worktree = "--apply-worktree" in sys.argv[1:]
    dry_run = "--dry-run" in sys.argv[1:]
    for path, transforms in TRANSFORMS.items():
        if apply_worktree:
            worktree = Path(path).read_text(encoding="utf-8")
            updated = worktree
            for old, new in transforms:
                if old is None:
                    if new not in updated:
                        updated = updated.rstrip("\n") + "\n" + new
                elif old in updated:
                    updated = updated.replace(old, new)
                elif new not in updated:
                    raise SystemExit(f"worktree anchor missing: {path} {old[:60]!r}")
            Path(path).write_text(updated, encoding="utf-8", newline="\n")
            print(f"worktree {path} updated with the UI-2 lane transforms")
            continue
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = _transformed(head, transforms)
        if dry_run:
            print(f"dry-run {path}: {len(head)} -> {len(staged)} chars")
            continue
        blob = (
            _git(
                "hash-object",
                "-w",
                "--stdin",
                "--path",
                path,
                data=staged.encode("utf-8"),
            )
            .decode("ascii")
            .strip()
        )
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + UI-2 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
