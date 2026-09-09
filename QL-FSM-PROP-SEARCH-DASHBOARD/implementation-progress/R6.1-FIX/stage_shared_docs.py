"""Stage the three shared docs as HEAD + R6.1-FIX lane transforms ONLY.

Same mechanism as ``../R6.1/stage_shared_docs.py`` (user hunks never enter
the release commit; ``--apply-worktree`` post-commit replays the same
transforms so the surviving diff is the user's pre-existing hunks only).
HEAD for R6.1-FIX is the R6.1 commit (``6c0b60a``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
R6.1-FIX additions (the compact correction of R6.1 after independent review —
plan `QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md`
revision 3, Phase 1; findings F-01…F-10D):

- **Verified assignment evidence (F-01/F-02/F-05)** (`ifvg/ml/regime_contracts.py`,
  `regime_store.py`, `regime_oos_assignment.py`, `regime_executor.py`): the per-fit
  assignment sidecar is serialized under the ENFORCED `FIT_ASSIGNMENT_SCHEMA`
  (exactly `RegimeAssignmentColumns`; hash `FIT_ASSIGNMENT_SCHEMA_HASH`) and every
  table satisfies `validate_assignment_rows` (kinds `fit` / `descriptive` /
  `model_facing`: a valid row carries the complete, self-consistent value set —
  64-hex fit id, fold, lawful partition, local id in `[0, k)`, `k` finite
  distances with `assigned_distance == distances[local] == min` and
  `assignment_margin == d2 − d1 ≥ 0`, the canonical id except on the
  model-facing kind — an invalid row keeps its linkage key and carries no output
  and one registered reason; no optional-column fallback survives). `load_regime_fit_assignments` returns
  `VerifiedFitAssignments` (envelope, artifact, frame, sidecar SHA-256, schema
  hash) and `persist_regime_fit` reuses an existing fit only when the candidate
  assignment bytes equal the stored sidecar byte-for-byte. The executor
  exact-loads every fit it persisted and builds the descriptive OOS artifact
  from those verified frames only; `RegimeOosAssignmentPayload` binds
  `regime_fit_assignment_refs` (`FitAssignmentRef` per fit, sorted; the
  `regime_fit_ids` projection is validated against them), `resolved_cluster_count`,
  `candidate_as_of_stage` (the anchor the hashed as-of instants came from; an
  unparseable non-null anchor is a hard error) and a `consulted_assignments_hash`
  over EVERY consulted value (fit, row, fold,
  partition, local id, canonical id, distance vector, assigned distance, margin,
  validity, reason); formula `regime_oos_assignment_v2`. `RegimeAssignmentEvidenceRef`
  pins `assignment_table_sha256` + `assignment_schema_hash` of the verified artifact.
- **Fold-feature source identity (F-03)** (`ifvg/ml/regime_fold_features.py`,
  `regime_supervised_stage.py`): `FoldFitRef` binds `assignments_sidecar_sha256`
  + `assignment_schema_hash` whenever a fit is present (null together only for an
  absent fit); `build_regime_fold_features(fit_assignments=…)` consumes
  `VerifiedFitAssignments` only (an in-memory run frame is refused by type) and
  S09b passes the executor's verified evidence; the loaders re-check every ref
  against the store by exact id; `validate_fold_feature_rows` holds on build, load
  and the ladder seam.
- **Candidate as-of policy (F-04)**: a candidate whose stage anchor is null is
  PRESERVED as `candidate_as_of_missing` (registered in
  `PANEL_ASSIGNMENT_MISSING_REASONS`, hence in the fold-feature vocabulary); an
  unparseable non-null instant stays a hard error; the as-of source hash
  represents the null deterministically.
- **Thin-regime accounting (F-09) + normalized frame (F-10B)**
  (`ifvg/ml/regime_stratified_strategy.py`, `regime_stratified_contracts.py`,
  `search/strategy_metrics.py`): the executed trades are validated + normalized
  ONCE and that projection (`normalized_executed_trades`) drives every join,
  stratum, computation and the binding `executed_trade_table_sha256`;
  `RegimeNetRAccounting` (formula `regime_net_r_accounting_v1`, basis
  `all_valid_assigned_trades_v1`) sums `per_trade_net_r` over EVERY valid assigned
  trade — thin regimes included — with `abs_net_r_share_by_regime`, signed
  contribution fractions, `unassigned_net_r`, `assigned_regime_count`,
  zero-denominator reasons and the `works_only_in_regime` claim (true; FALSE when
  the assigned side refutes it; null with `incomplete_assignment_accounting` only
  when unassigned trades prevent a supported claim), every derived value
  recomputed by the contract's validator; the reportability floor governs
  interval/reportability metrics only. The stratification service re-verifies
  each child's frame against its persisted executed-trade table and binds the
  artifact's projection hash (`executed_trade_table_artifact_sha256`).
- **Exact label identity (F-08)** (`ifvg/ml/comparison_rows.py`,
  `controlled_feature_study.py`, S07): `label_artifact_content_id` binds the
  registered label policy and EVERY consumed label / economic column
  (`LABEL_CONSUMED_COLUMNS`); S07 mints it; `ControlledFeatureStudyPayload.label_artifact_id`
  is mandatory with `label_identity_source ∈ {label_artifact, content_hash_unpersisted}`
  — a helper run without the exact artifact carries the full consumed-column hash
  and can never be saved, compared as an immutable study, or promoted; every
  helper path (the ladder, the CatBoost bundle rung, the logistic rung) defaults
  to that hash and the ladder run is stamped `label_identity_source`.
- **Immutable executed-trade table (F-06)** (`ifvg/search/executed_trade_table.py`,
  store `executed_trade_tables`): the EXACT ordered 42-column Arrow projection
  `EXECUTED_TRADE_TABLE_SCHEMA_V1` (`core_executed_trade_exact_v1`; typed as the v2
  capture types it) is declared, never inferred; the identity derives from the
  core replay (`executed_trade_table_id_for`), so S02/S14 exact-load it without a
  listing; the envelope binds the projection SHA-256, the raw core table hash the
  neutrality report hashes, row count and byte size. S02 persists it after a fresh
  completion and after verified reproduction (byte-for-byte against the persisted
  table — projection bytes AND raw core-table hash; the costed-evaluation
  reproduction remains the fallback for a child without a table; nothing
  verifiable → typed `executed_trade_table_unavailable`), exact-loads it back and
  computes EVERY costed evaluation from the loaded projection inside per-child
  containment (one identity, one byte content); S14 iterates the charter's child
  set (never a prior attempt's report record), verified-loads every gated child's
  table, binds `executed_trade_table_id` into the report body +
  `source_metric_refs`, and records `children_evidence` / typed `children_skipped`
  (`child_not_completed_or_reused`, `strategy_gates_not_passed`,
  `executed_trade_table_unavailable`).
- **Fail-closed prior-stage sidecars (F-07)** (`ifvg/search/store.py`,
  `pipeline.py`, `pipeline_regime.py`, `ml/regime_report_stage.py`): the typed
  probe contract — `probe_sidecar` / `has_sidecar` / `load_optional_sidecar_bytes`
  / `load_json_sidecar`; `sidecar_not_produced_for_path` is the ONLY optional
  absence; `store_entry_missing` (the entry directory does not exist),
  `manifest_missing_for_existing_entry`, `malformed_manifest`,
  `manifest_hash_mismatch`, `envelope_identity_mismatch`,
  `sidecar_missing_but_manifest_declares_it`, `sidecar_hash_mismatch`,
  `malformed_sidecar`, `unexpected_io_error` are typed `SidecarLoadError`s that
  propagate — raised at the detection point by `load_verified_envelope`,
  `load_sidecar_bytes` and `has_envelope` (corrupt is never absent). The prop-vector / account-simulation /
  lineage / regime-report / S09c-record recoveries use it; S15 records
  `reload_failures` (store/id → sanitized reason) in the state file's publication
  block and immutably as `PipelineResultPayload.reload_failure_reasons`, and
  reloads every executed-trade table and stratified report the S14 record names;
  a halted or cancelled attempt marks every later planned stage PENDING, resets
  the publication block, and `activate_pipeline_result` re-derives the gates from
  the latest attempt (no stale terminal status is carried forward).
- **Production correctness (F-10A/C/D)**: the five pipeline wiring checks raise
  `PipelineWiringError` (typed, survives `python -O`); the fold-feature builder's
  two asserts became typed errors; `Mbp1PartitionEvidence` requires FULL scope
  equality with the gap manifest and a positive completeness claim requires the
  compilation report's `verified_partition_refs` to EQUAL the complete partition
  content refs (`partition_content_refs=None` is refused for a positive claim);
  `FrozenContract.model_copy` refuses a raw string for a scalar enum-typed field
  and a non-member element in a sequence-of-enum field.
- **Unchanged (golden-tested)**: `resolved_regime_protocol_id`, the R6 golden
  `regime_fit_id`, the frozen M0 CatBoost hash, `core_replay_id`,
  `account_simulation_id`, `feature_block_registry_hash`
  (`tests/agents/ifvg_search/test_r61_fix_goldens.py`). Re-minted (synthetic only):
  OOS-assignment, fold-feature, bundle-path ladder / study, stratified-report,
  controlled-study, label-artifact and pipeline-result identities. Verified reuse:
  zero replay for non-stratified runs; one verified reproduction per reused child
  (projection bytes + core-table hash) when stratified reports are requested.
"""

README_OLD = """the D15 prop-event
detail `alpha_lab/propsim/event_detail.py`, per-fold stability + grain
transition policies, and the regime surfaces of the pipeline tab)
implementations are complete;"""
README_NEW = """the D15 prop-event
detail `alpha_lab/propsim/event_detail.py`, per-fold stability + grain
transition policies, and the regime surfaces of the pipeline tab), and
R6.1-FIX (verified per-fit assignment evidence with an enforced sidecar
schema, fold-feature source refs, the typed `candidate_as_of_missing`
reason, raw thin-regime net-R accounting, the exact label artifact id, the
immutable `executed_trade_tables` store consumed by S02/S14, the typed
sidecar probe with fail-closed prior-stage recoveries, and the
`PipelineWiringError` / MBP-1 scope-equality / enum-copy corrections)
implementations are complete;"""

YAML_OLD = (
    '    R6_1_regime_correction: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
)
YAML_NEW = (
    '    R6_1_FIX_compact_correction: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
    'new_store: "executed_trade_tables", '
    'contracts: "FIT_ASSIGNMENT_SCHEMA, FitAssignmentRef, VerifiedFitAssignments, '
    "RegimeNetRAccounting, ExecutedTradeTable, SidecarLoadError, PipelineWiringError, "
    'label_artifact_content_id", '
    'findings_closed: "F-01 F-02 F-03 F-04 F-05 F-06 F-07 F-08 F-09 F-10A F-10B F-10C F-10D", '
    'adversarial_round: "RA-01..RA-08 B-01..B-10 closed"}\n'
    '    R6_1_regime_correction: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
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
            Path(path).write_text(updated, encoding="utf-8", newline="\n")
            print(f"worktree {path} updated with the R6.1-FIX lane transforms")
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
        print(f"staged {path} = HEAD + R6.1-FIX lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
