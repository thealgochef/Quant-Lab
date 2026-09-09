"""Stage the three shared docs as HEAD + R5 lane transforms ONLY.

The user's pre-existing uncommitted hunks never enter the release commit
(kickoff §1.4): each staged blob is rebuilt from the HEAD content plus exactly
the transforms the R5 working-tree edit applied, hashed with
``git hash-object -w`` and planted via ``git update-index --cacheinfo``.
Run from the repo root AFTER ``git add`` of the normal (non-shared) files and
BEFORE ``git commit``. Idempotent. Pass ``--apply-worktree`` AFTER the commit
to apply the SAME transforms to the worktree copies (so the surviving diff is
the user hunks only — the R4 gotcha). Same mechanism as
``../R4/stage_shared_docs.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
R5 additions (pipeline runner + MBP-1 contract readiness + supervised model
ladder; capability-scoped operator availability per V3 P1-5):

- **Pipeline contracts + runner** (`ifvg/search/pipeline.py`): the CS §7
  split — `PipelineSemanticSpecPayload` (research-bearing fields only;
  canonical-order 16-stage plans with dependency closure) hashes to
  `pipeline_semantic_id`; `ExecutionAttemptIdentity` (workers/host/
  timestamps/retry reason) is deliberately NOT an identity envelope. The
  runner composes the study-lane primitives (`enumerate_children`, the
  reuse/neutrality/publication semantics, the costed-evaluation cache, the
  extracted `merge_prop_vectors` — ONE implementation of ALL-legs
  feasibility + worst-firm merge — and `build_frontier`); every stage
  executor is idempotent and store-reusing, and a stage whose
  freshly-minted `PipelineStageResultEnvelope` id equals the prior
  attempt's is marked REUSED (reuse proven by identity). S11 terminal-
  blocks with the exact registered reason; S15 persists the immutable
  `PipelineResultEnvelope` with `prepared_not_published` operational
  state; publication is verify-then-activate and verification scope can
  never activate a research catalog entry.
- **Carried-seam closures**: S02 persists per-child lineage evidence
  (uniqueness reports + serialized key projections as stage sidecars) and
  S14 builds + persists deterministic insight panels
  (`InsightPanelEnvelope`, `insights` store) and baseline↔challenger
  `ComparisonResultEnvelope`s (`search_results` store) that the UI now
  consumes (DEV-R4-7/DEV-R4-16); S12/S13 run the additively extended
  `alpha_lab.propsim.search_bridge` (scenario/bootstrap/stress mode
  bridging, DEV-R3-11) and persist real `AccountPolicySetEnvelope`s +
  `AccountSimulationEnvelope`s with the trader-UI sidecars (DEV-R4-17).
- **Supervised ML lane** (`ifvg/ml/`): the registry-gated ladder
  (`model_protocols` · `logistic_model` · `supervised_ladder`) runs the
  prevalence reference + fold-local logistic + the existing CatBoost
  protocol on IDENTICAL rows/folds (identical `oos_row_id` sets asserted
  before any delta; `paired_cell_delta_report` under the fixed 10k/seed-7
  day-block bootstrap); every preprocessing statistic is fold-fitted;
  fitted artifacts persist portably (manifest-relative refs + checksums,
  relocation-proof). `calibration_policies` / `decision_policies` are
  registry pairs with distinct logical keys and resolved envelope ids;
  only the diagnostic baselines are executable, every execution-affecting
  decision policy requires an owner-ratified `RejectedCandidatePolicy`,
  and `ProhibitedSelectionError` refuses enumeration outside a ratified
  charter. Core `drift_monitoring` exports report builders only.
- **Bundle feature views** (`ifvg/features/bundle_feature_view.py`):
  available-blocks-only views over the immutable candidate view;
  `IFVG_ORDER_FLOW_MBP1_V1` stays `planned` and every bundle carrying it
  refuses (no baseline-vs-MBP-1 study is constructible before R5B); a
  bundle maps onto the supervised ladder only when its resolved feature
  set equals a frozen tier exactly.
- **Real executors + pipeline shim** (`ifvg/search/executors.py` ·
  `runner_registry.py` · `scripts/ifvg_pipeline_job.py`): the registry now
  names the real baseline-verification search/pipeline executors — their
  factories fail closed at CONSTRUCTION without the owner's persisted
  `VerificationRunEnvelope` (registration unblocks the launch surface,
  never the data) — and full-development charters keep NO registered
  entry (the operator full run stays a separate authorized action). The
  detached pipeline shim mirrors the search shim (registry-gated worker,
  atomic status, safe-boundary cancel, `publish-gates`/`activate` CLI).
- **Full Pipeline Run surface** (`scripts/ifvg_pipeline_tab.py`, session
  namespace `ifvg_pipeline_v1_*`): the complete §30 workflow — Configure
  (capability-scoped stage plans; planned/blocked feature + model entries
  visible-disabled), Preview (exact counts/estimates/reuse/new-artifact
  disclosure), Launch (one scanned `_spawn_pipeline_job` seam; the exact
  §15 typed full-scope confirmation), Monitor (all 16 stages glyph+word
  incl. `not required`; `pipeline_semantic_id` + execution-attempt
  history; the supervised-ladder panel with planned rungs and the exact
  S11 blocked reason), Resume/Retry (operational clone; research-bearing
  change = new semantic id), Publish (gates checklist first; activation
  refused for verification scope).
"""

README_DECISIONS_OLD = """Decisions D-039 through D-043 + D-045's
trader-workspace half (D-044 and D-045's pipeline half reserved for
R5–R6)."""
README_DECISIONS_NEW = """Decisions D-039 through D-045 (D-044's
supervised-ladder half and D-045's pipeline half landed with R5; D-044's
KMeans regime half lands with R6)."""

README_OLD = """`ifvg/study_{status,presentation,drafts,providers}.py`) implementations
are complete;"""
README_NEW = """`ifvg/study_{status,presentation,drafts,providers}.py`), and R5 (pipeline
runner + MBP-1 contract readiness + supervised model ladder:
`ifvg/search/pipeline.py` with the semantic/attempt split and 16-stage
executors, `ifvg/ml/` ladder + decision/calibration registries + core
drift builders, `ifvg/features/bundle_feature_view.py`, the real
baseline-verification executors in `ifvg/search/executors.py`, and the
Full Pipeline Run surface `scripts/ifvg_pipeline_tab.py` +
`ifvg_pipeline_job.py`) implementations are complete;"""

YAML_OLD = (
    '    R5_pipeline_mbp1_readiness_ladder: {implementation_status: "not_started", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)
YAML_NEW = (
    '    R5_pipeline_mbp1_readiness_ladder: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [
        (README_DECISIONS_OLD, README_DECISIONS_NEW),
        (README_OLD, README_NEW),
    ],
    "docs/pipeline_state.yaml": [(YAML_OLD, YAML_NEW)],
}


def _git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], input=data, capture_output=True, check=True
    )
    return result.stdout


def _transformed(head: str, transforms: list[tuple[str | None, str]]) -> str:
    staged = head
    for old, new in transforms:
        if old is None:
            staged = staged.rstrip("\n") + "\n" + new
        else:
            if old not in staged:
                raise SystemExit("transform anchor missing at HEAD")
            staged = staged.replace(old, new)
    return staged


def main() -> int:
    apply_worktree = "--apply-worktree" in sys.argv[1:]
    for path, transforms in TRANSFORMS.items():
        if apply_worktree:
            # post-commit: replay the SAME transforms onto the worktree copy
            # so the surviving diff is the user's pre-existing hunks only
            worktree = Path(path).read_text(encoding="utf-8")
            updated = worktree
            for old, new in transforms:
                if old is None:
                    if new not in updated:
                        updated = updated.rstrip("\n") + "\n" + new
                elif old in updated:
                    updated = updated.replace(old, new)
            Path(path).write_text(updated, encoding="utf-8", newline="\n")
            print(f"worktree {path} updated with the R5 lane transforms")
            continue
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = _transformed(head, transforms)
        blob = _git(
            "hash-object", "-w", "--stdin", "--path", path,
            data=staged.encode("utf-8"),
        ).decode("ascii").strip()
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + R5 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
