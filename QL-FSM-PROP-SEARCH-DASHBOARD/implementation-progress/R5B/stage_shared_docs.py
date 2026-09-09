"""Stage the three shared docs as HEAD + R5B lane transforms ONLY.

The user's pre-existing uncommitted hunks never enter the release commit
(kickoff §1.4): each staged blob is rebuilt from the HEAD content plus exactly
the transforms the R5B working-tree edit applied, hashed with
``git hash-object -w`` and planted via ``git update-index --cacheinfo``.
Run from the repo root AFTER ``git add`` of the normal (non-shared) files and
BEFORE ``git commit``. Idempotent. Pass ``--apply-worktree`` AFTER the commit
to apply the SAME transforms to the worktree copies (so the surviving diff is
the user hunks only — the R4 gotcha). Same mechanism as
``../R5/stage_shared_docs.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
R5B additions (offline MBP-1 feature activation — research-only, owner
ruling R-6; the versioned registry event of revision P1-5):

- **Activation as the published registry** (`ifvg/features/feature_blocks.py`):
  `FEATURE_BLOCK_REGISTRY` IS `with_activated_block` applied to the exported
  R5-era planned state (`PRE_ACTIVATION_*`) — `IFVG_ORDER_FLOW_MBP1_V1` at
  `block_version=2`, status available, its first
  `resolved_feature_block_id` minted from the real Arrow schema hashes, the
  frozen 9-window registry, `ifvg_order_flow_mbp1_formula_v1`, and
  `mbp1_feature_materializer_v1`; the block-registry hash changed and every
  ORDER_FLOW bundle (B2/B3) gained a new resolved id, while B1/B4 kept
  theirs and regime/execution-liquidity bundles keep refusing. The
  research-only boundary rides the DEFINITION
  (`expected_computation_path="offline_research_feature_materialization_v1"`,
  `can_affect_execution=False`): no live model feature, execution gate, or
  Trade-Lab serving use without a later Strategy-Core formula/parity
  contract and a separately approved sequential model-gated replay.
- **Exact schemas + immutable evidence** (`mbp1_arrow_schemas.py` ·
  `mbp1_source_artifact.py`): four pinned Arrow schemas with canonical
  field/type hashes (raw Databento mbp-1 retaining `ts_recv` — the
  materializer decodes it directly, no Strategy-Core change; the normalized
  working schema with the deterministic `source_ordinal` final tie-break;
  the 76-metric feature table with per-window validity/missing-reason
  evidence; the stage-window evidence table). The content-addressed
  `Mbp1SourceArtifact` freezes per-partition hashes, first/last order keys,
  sequence-gap intervals (vendor sequence resets are NOT gaps), and
  gap-adjusted day coverage; synthetic fixtures persist canonical event
  bytes as manifest-hashed sidecars, real artifacts reference partitions by
  hash; real reads run authorize-before-path through the verification
  policy family and `legacy_verified_replay_source` provenance is refused
  outright (guards re-run at R5B).
- **Point-in-time stage windows** (`mbp1_stage_windows.py` ·
  `mbp1_feature_materializer.py`): the candidate row's own five anchors
  (`tap/lock/armed/inversion/entry_ts_utc`) become `COMPLETED_BAR_BOUNDARY`
  cutoffs (`completed_bar_boundary_exclusive_v1`); admission runs on the
  complete `(ts_event, ts_recv, sequence, source_ordinal)` key —
  `PRE_TRIGGER_EXCLUSIVE` `<` / `POST_TRIGGER_INCLUSIVE` `<=` on exact
  keys, strict `ts_event <` for timestamp-only evidence with EVERY
  same-timestamp event excluded and the window typed
  `same_timestamp_order_unavailable` when any tie exists; a missing lower
  anchor refuses rather than widening; no `+inf` bound exists anywhere
  (source-scanned). The offline materializer preserves every candidate row
  under the deterministic typed-missing precedence (no partition → coverage
  → outside coverage → same-ts ambiguity → sequence gap → roll boundary →
  minimum events); formula-edge NaNs stay VALID; batch/repeat runs are
  byte-identical; the feature artifact's recipe identity (source artifact ×
  resolved block × anchor hash × cutoff policy × schema hashes) is separate
  from the post-materialization table hashes on the envelope.
- **Exact joins + bound views** (`mbp1_feature_join.py` ·
  `bundle_feature_view.py`): one-to-one on `candidate_id` only (duplicates
  refuse; no nearest-time or row-order fallback exists — source-scanned);
  MBP-1-bearing bundle views REQUIRE the materialized frame AND its
  artifact id, and the view payload pins `mbp1_feature_artifact_id`, so the
  same view+bundle over different evidence can never share one identity.
- **Controlled Baseline vs Baseline+MBP-1 study**
  (`ml/controlled_feature_study.py` + the bundle-parametrized ladder in
  `ml/supervised_ladder.py`, DECISIONS_TAKEN #41 arrival): the challenger
  runs against its OWN base bundle on identical rows/labels/folds —
  prevalence + logistic rungs (the CatBoost fold runner is tier-locked in
  the frozen M0–M3 lane and refuses with that exact reason); cross-arm row
  identity plus a numerically identical prevalence reference are asserted
  before the paired Brier delta (fixed 10k/seed-7 day-block bootstrap); the
  persisted study pins every input identity and carries the permanent
  `research_only_offline` stamp. Pipeline wiring: S00 refuses MBP-1 plans
  without the `mbp1_evidence_source` seam (order-flow evidence is never
  fabricated), S05 materializes + immutably persists the
  source/feature/coverage artifacts and joins them into the bundle view,
  S09 runs + persists the controlled study (readiness blocks CatBoost×MBP-1
  plans before launch), and the new `mbp1_source_artifacts` /
  `mbp1_feature_artifacts` / `mbp1_coverage_reports` /
  `controlled_feature_studies` stores follow the manifest protocol.
- **Dashboard** (`scripts/ifvg_mbp1_panels.py` + the pipeline surface): the
  MBP-1 Order Flow panel (availability with pre/post-activation registry
  hashes, the frozen window registry, exact-ID coverage/missingness
  evidence, per-candidate stage-window drill-down, and the Baseline vs
  Baseline+MBP-1 comparison) under the persistent `research_only_offline`
  badge; Configure restricts MBP-1-bearing bundles to the logistic
  protocol. The search job shim now passes the worker's `--store-root` to
  runner-entry factories (DEV-R5-10 closure). The five-day REAL
  control-flow verification of the materializer remains blocked on the
  owner's `VerificationAuthorizationRef`, exactly like every real half
  since R1.
"""

ARCH_R5_BULLET_OLD = """- **Bundle feature views** (`ifvg/features/bundle_feature_view.py`):
  available-blocks-only views over the immutable candidate view;
  `IFVG_ORDER_FLOW_MBP1_V1` stays `planned` and every bundle carrying it
  refuses (no baseline-vs-MBP-1 study is constructible before R5B); a
  bundle maps onto the supervised ladder only when its resolved feature
  set equals a frozen tier exactly."""
ARCH_R5_BULLET_NEW = """- **Bundle feature views** (`ifvg/features/bundle_feature_view.py`):
  available-blocks-only views over the immutable candidate view; through
  R5, `IFVG_ORDER_FLOW_MBP1_V1` stayed `planned` and every bundle carrying
  it refused (no baseline-vs-MBP-1 study was constructible before the R5B
  activation below); a bundle maps onto the tier-frozen supervised ladder
  only when its resolved feature set equals a frozen tier exactly."""

README_DECISIONS_OLD = """Decisions D-039 through D-045 (D-044's
supervised-ladder half and D-045's pipeline half landed with R5; D-044's
KMeans regime half lands with R6)."""
README_DECISIONS_NEW = """Decisions D-039 through D-046 (D-044's
supervised-ladder half and D-045's pipeline half landed with R5; the
MBP-1 activation D-046 landed with R5B; D-044's KMeans regime half lands
with R6)."""

README_OLD = """Full Pipeline Run surface `scripts/ifvg_pipeline_tab.py` +
`ifvg_pipeline_job.py`) implementations are complete;"""
README_NEW = """Full Pipeline Run surface `scripts/ifvg_pipeline_tab.py` +
`ifvg_pipeline_job.py`), and R5B (offline MBP-1 feature activation,
research-only: `ifvg/features/mbp1_{arrow_schemas,source_artifact,
stage_windows,feature_materializer,coverage,feature_join}.py`, the
versioned `IFVG_ORDER_FLOW_MBP1_V1` activation in `feature_blocks.py`,
the controlled Baseline vs Baseline+MBP-1 study
`ifvg/ml/controlled_feature_study.py`, and the MBP-1 dashboard panels
`scripts/ifvg_mbp1_panels.py`) implementations are complete;"""

YAML_OLD = (
    '    R5B_mbp1_offline_activation: {implementation_status: "not_started", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)
YAML_NEW = (
    '    R5B_mbp1_offline_activation: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [
        (ARCH_R5_BULLET_OLD, ARCH_R5_BULLET_NEW),
        (None, ARCH_APPEND),
    ],
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
            print(f"worktree {path} updated with the R5B lane transforms")
            continue
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = _transformed(head, transforms)
        blob = _git(
            "hash-object", "-w", "--stdin", "--path", path,
            data=staged.encode("utf-8"),
        ).decode("ascii").strip()
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + R5B lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
