"""Stage the three shared docs as HEAD + UI-1 lane transforms ONLY.

Same mechanism as ``../../HARDENING-BACKEND-FIX/stage_shared_docs.py`` (user
hunks never enter the release commit; ``--apply-worktree`` post-commit
replays the same transforms so the surviving worktree diff is the user's
pre-existing hunks only). HEAD for UI-1 is the HARDENING-BACKEND-FIX.1
release head (``ffcf39b``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_OLD = """  M0–M3 Context Research panel delegates verbatim; a namespace selector
  switches between the research (`search/v1`) and verification
  (`search_test/v1`) stores with truthful badging.
"""
ARCH_NEW = """  M0–M3 Context Research panel delegates verbatim; the store a draft
  freezes into derives from its run purpose (UI-1 — the R4 namespace
  selector is gone; the store's verified `store_namespace_id` is displayed
  read-only and a local path never defines authority).
"""

ARCH_APPEND = """
UI-1 additions (Phase 1 of the owner-approved UI/UX redesign,
`QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/UI-UX-REDESIGN-PLAN/IMPLEMENTATION_PLAN.md`
revision 2, over the HARDENING-BACKEND-FIX.1 release head; UI-2 … UI-6 follow; acceptance
still transitively blocked by R1):

- **Presentation package** (`ifvg/presentation/`): `run_purpose` — the presentation-only
  `RunPurpose` (Implementation Verification → `verification_5d` in the `test` namespace;
  Development Research / Full Authorized Development → `full_authorized_development` in the
  `research` namespace; `RunScope` values unchanged), the `EvidenceClass` (a synthetic fixture
  is confined to Implementation Verification), the mutable non-semantic `RunPurposeAnnotation`
  (draft field + catalog `purpose` event), `resolve_draft_purpose` (stored → unambiguous legacy
  derivation → `purpose_unresolved`), `namespace_state_for_store` (the id from the VERIFIED
  envelope only) and `resolve_purpose` (the freeze verdict); `charter_satisfiability` — the
  named-rule report refused BEFORE freeze (FSM ≥ 1 challenger; Evaluate exactly one profile;
  Compare exactly one challenger configuration; a selected prop objective requires a verified
  contract and is never rewritten; Prop ≥ 1 / Universal ≥ 2 firms; real verification is the
  exact baseline with verification gates only); `status_vocabulary` — the thirteen `UiStatus`
  values (glyph + word + color token; PASS only for an evaluated true) with additive adapters.
- **Validator** (`ifvg/search/charter.py`): the identity-bearing subset fail-closed (FSM ≥ 2
  profiles, single ≤ 2, Universal ≥ 2 firms; a real charter's prop objective requires a firm
  contract — never a silent rewrite).
- **Providers** (`ifvg/study_providers.py`): `resolve_store_namespace`,
  `artifact_scope_for_charter`, the typed `AuthorizationReadiness` for the
  `VerificationAuthorizationRef` (namespace + current head through the complete authority-chain
  proof, profile, allowlist) and for the owner bundle (catalogued verified owner-decision
  artifacts per required key), bundles assembled only from `ready`, and run listings annotated
  with each run's own store (exact-id located), namespace class and scope.
- **UI** (`scripts/ifvg_study_tab.py`, `ifvg_study_wizard.py`, `ifvg_pipeline_tab.py`,
  `ifvg_results_tab.py`, `ifvg_active_runs_tab.py`, `ifvg_results_charts.py`,
  `ifvg_lab_tab.py`): the `Start` task cards and the `Verify Implementation` readiness surface;
  the goal card on every step; the evidence-class choice and typed readiness on Validation; the
  frozen warmup prefix read-only with field-level logical-day validation; no worker control
  (`sequential_children_v1 · effective workers 1`); the satisfiability card on Review; the
  registered executor resolved before any spawn (`runner_unavailable`) and the launch reported
  only after persisted state exists (`launch_not_started`); gates recorded with the verified
  namespace id + state digest and activation bound to both; direction-aware heatmap / firm-matrix
  colorscales; the reconciliation banner derived from evaluated gates; the additive §31 states
  (`no_runs`, `not_selected`, `artifact_missing` / `artifact_corrupt`, `purpose_unresolved`,
  `authorization_not_ready`, …); the fifth research question `Evaluate one configuration`.
- **Unchanged**: Strategy-Core, the fixed M0–M3 lane, every immutable artifact identity, the
  exact-ID loading rule, S11, MBP-1's `research_only_offline` boundary, the backend contracts.
"""

README_OLD = """authority-chain proof at every real seam)
implementations are complete;"""
README_NEW = """authority-chain proof at every real seam), and UI-1 (Phase 1 of the
owner-approved UI/UX redesign: the presentation-only run purpose that
derives scope / namespace / evidence / authorization class, the removal
of the namespace selector, `Start` and `Verify Implementation` routes,
charter satisfiability before freeze, typed authorization readiness, the
honest launch outcome, namespace-bound publication, the sequential-V1
runtime truth, direction-aware colorscales and evaluated-gate banners —
`ifvg/presentation/{run_purpose,charter_satisfiability,status_vocabulary}.py`)
implementations are complete;"""

YAML_OLD = (
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)
YAML_NEW = (
    '    UI_1_semantic_purpose_truth: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
    'ui_phase: "1 of 6 (UI-UX-REDESIGN-PLAN revision 2)", '
    'presentation_package: "ifvg/presentation/{run_purpose,charter_satisfiability,'
    'status_vocabulary}.py", '
    'routes: "Start | Verify Implementation | New Study | Active Runs | Results | History | '
    'Context Research (namespace radio removed; namespace derived from the run purpose)", '
    'contracts_consumed: "StoreNamespaceEnvelope, SupersessionHeadWitness, '
    "VerificationAuthorizationRef, OwnerAuthorizationBundle, runner registry, "
    'SUPPORTED_CHILD_WORKERS=1 / sequential_children_v1", '
    'findings_closed: "F-01 F-02 F-03 F-04 F-05 F-07(minimal) F-08(part) F-13(part)", '
    'browser_acceptance: "open (UI-6)", '
    'owner_actions_performed: "none (no signing, no seed production, no real verification run)"}\n'
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(ARCH_OLD, ARCH_NEW), (None, ARCH_APPEND)],
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
            print(f"worktree {path} updated with the UI-1 lane transforms")
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
        print(f"staged {path} = HEAD + UI-1 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
