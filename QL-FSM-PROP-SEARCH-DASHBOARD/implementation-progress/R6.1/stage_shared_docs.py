"""Stage the three shared docs as HEAD + R6.1 lane transforms ONLY.

Same mechanism as ``../R6/stage_shared_docs.py`` / ``../R5B.1/stage_shared_docs.py``
(user hunks never enter the release commit; ``--apply-worktree`` post-commit
replays the same transforms so the surviving diff is the user's pre-existing
hunks only). HEAD for R6.1 is the R5B.1 commit (``f3f9ac2``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
R6.1 additions (the regime-lane correction release — owner audit of R6,
planning decisions Q1–Q4, plan-review corrections 1–7, final contract-closure
rulings 1–8; `QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6.1-CORRECTION-PLAN-DOCS-FINAL/`):

- **Context-bar panel materializer + block** (`ifvg/features/context_bar_panel_{contract,materializer}.py`,
  `arrow_tables.py`): the seven registered `cbp_*` features (owner Q3) are
  materialized ONLY from a `VerifiedReplayChartArtifact` (bars re-read and
  rehashed against the manifest; the in-memory frame is never trusted) at the
  two owner-registered intervals (300 s, 900 s) over COMPLETED bars; N = 12
  prior same-trading-day bars, `MINIMUM_SOURCE_BARS = 13`, `STD_DDOF = 0`;
  every one of the 13 source bars must satisfy `observed_1m_count ==
  expected_1m_count` or ALL seven features are null with
  `source_bar_incomplete` and a manifest-listed validity sidecar names the
  offending bars; windows never cross 18:00 ET; the session OF A BAR is the
  session in force at the bar's final instant (`session_of_bar_final_instant_v1`).
  `IFVG_CONTEXT_BAR_PANEL_V1` is registered as a versioned event
  (`with_registered_block`; AVAILABLE, family `context_bar_panel`, stage
  `HTF_TAP`, `research_only_offline`, `offline_panel_materialization_v1`,
  `row_id` join, categorical `cbp_session_state`); bundle
  `BP0_CONTEXT_BAR_PANEL` (no base; a candidate view refuses it); the R5B.1
  state stays exported as `PRE_R6_1_*`; `feature_block_registry_hash()`
  moves. Protocol tightening: `panel_source_artifact_id` is a 64-hex verified
  artifact id, only registered intervals/as-of policies, and grain/bundle-key
  coherence (`assert_grain_bundle_coherent`: every input-bundle block joins on
  `row_id` for the panel grain and on `candidate_id` otherwise).
- **Fold schedules and fold-set artifacts** (`ifvg/fold_schedules.py`,
  `ifvg/ml/fold_set_artifact.py`): `FoldScheduleEnvelope.fold_schedule_id`
  hashes the grain-agnostic 40/5/5/2 windows over the authorized days;
  every fold set (candidate or panel) is a persisted artifact carrying the
  schedule id + its own row-population `fold_set_id` (the ONE legacy hash,
  now delegated to by the ladder and the controlled study); cross-grain
  studies require equal `fold_schedule_id` and per-fold windows, never equal
  `fold_set_id`; the label-free `build_candidate_folds_from_schedule` and the
  panel-native `build_context_bar_panel_folds` (embargo 2 days, purge by bar
  span, stamped floor 300) exist beside the unchanged labeled builder.
- **Two regime-assignment artifacts, two uses (D7)**: the DESCRIPTIVE
  `RegimeOosAssignmentArtifact` (`ifvg/ml/regime_oos_assignment.py`; candidate
  grain = the single OOS fold; panel grain = the normative PIT rule — same
  trading day, last completed bar `<=` as-of, `cbp_valid`, elapsed ≤ one
  interval, compatible partition; typed `panel_warmup` /
  `no_completed_panel_bar` / `panel_gap` / `panel_stale` / `coverage_gap`;
  never carried across 18:00 ET) feeds stratification only; the
  `RegimeFoldFeatureArtifact` (`ifvg/ml/regime_fold_features.py`; fit k → fold
  k only; fit-local `ctx_regime_<p12>_local_{distance_i,assigned_distance,
  margin,id}` — the local id is a model feature only under
  `hard_id_encoding = fit_local_categorical_v1`, default `none`; canonical
  alignment reporting-only) is the ONLY supervised feature source — the leakage test proves later folds cannot alter an
  earlier fold's bytes.
- **Verified observation seam + executor** (`ifvg/ml/regime_observation_source.py`,
  `regime_executor.py`): every regime input is a verified-loaded artifact
  (`bundle_feature_views` / `context_bar_panels` stores); `source_artifact_ids`
  come from the loaded envelope, never a caller string; `execute_regime_protocol`
  persists protocol → loads the fold set → loads observations → runs →
  persists fits by VERIFIED REUSE BY REPRODUCTION (an existing fit must
  reproduce this fit's transform and labels from its verified bytes; joblib
  bytes are never rewritten) → assessment → the descriptive OOS artifact.
  The panel-grain candidate as-of instants come from a verified-loaded
  bundle view (`candidate_as_of_source: RegimeObservationSourceRef`; a
  tuple / bare string / panel-kind ref is "not evidence") and
  `candidate_as_of_source_ref` is the LOADED envelope line
  (`bundle_feature_view:<id>`, pattern-checked).
  `run_regime_protocol` executes under `threadpool_limits(1)` so assignment
  tables are byte-reproducible (an execution-environment control, not a
  protocol field; fit ids unchanged — the R6 golden fit id holds;
  `threadpoolctl` is imported lazily with an explicit failure message).
- **Pipeline integration (D1/D4/D14)**: `PipelineSemanticSpecPayload.regime_study`
  (`ifvg/ml/regime_study.py::RegimeStudyRequest`; every semantic id moves)
  freezes algorithm, grain, bundle, inputs, k, winsorization, bootstrap
  budget, the requested comparison classes and — for `feature_only` /
  `cohort_model` only — the EXACT `regime_promotion_decision_id`,
  `owner_decision_artifact_id`, and `required_capability_assessment_id`;
  descriptive requests must omit them. Stages (`ifvg/search/pipeline_regime.py`):
  S04 records the chart ids (one replay chart per child); S05 persists every
  bundle view + frame and the regime observation (the panel materialized
  from `PipelineWiring.context_bar_source` for the chart selected by the
  declared policy `panel_chart_lowest_core_replay_id_v1`; the returned
  artifact must carry the requested id and the run's pair or S05 fails
  closed; the sidecar records the LOADED chart id) + the protocol (a
  model-bearing run verifies it against the frozen decision); S06 the input
  coverage + stamped floors; S08 the schedule derived ONCE from the
  candidate view's observed trading days (charter allowlist ∩ observed;
  `days_without_labels` typed) + fold-set artifact(s) built from those same
  days + the per-fold adequacy preview; S09a the executor (a
  model-bearing run refuses any assessment other than the frozen one); S09b
  the fold-local features and S09c the controlled regime study / cohort model
  (`ifvg/ml/regime_supervised_stage.py`) only for model-bearing requests — every
  fit of the run happens in S09; S10 derives the deterministic decisions
  (DESCRIPTIVE_ONLY or a typed BLOCKED state, then STRATIFICATION_READY iff
  coverage gates passed AND an OOS assignment exists AND descriptive classes
  were requested; `decided_at` = the evidence as-of instant, never the wall
  clock) or re-verifies the frozen authority; S14 builds the stratified
  reports from persisted artifacts only (the panel PIT assigner loads the
  OOS assignment → protocol → panel frame → every fit's assignment sidecar
  by exact id; the executed-trade tables are S02's this-run tables — no
  persisted trade artifact exists) and performs ZERO fitting (test-enforced
  on the descriptive, supervised-candidate and supervised-panel runs); the
  modeled classes are recorded under `delivered_by` (verified S09c ids),
  never as refusals; S15 reloads every regime artifact. A reused child whose
  executed-trade tables a stratified report needs is re-derived and its
  tables are adopted ONLY when they reproduce the persisted costed
  evaluation of THIS cost policy (otherwise "reproduction unverifiable":
  reused, no gates evaluated, no evaluation published, out of the
  stratified reports); S12/S13 record the exact persisted
  account-simulation ids in an attempt-invariant sidecar. Readiness
  verified-loads a model-bearing request's frozen authority from the store
  before any path — the tab's Preview and Launch handler do the same with
  the store root and run scope BEFORE any charter / spec envelope or spawn.
- **Verified owner-decision evidence (D5)** (`ifvg/search/owner_decisions.py`,
  store `owner_decisions`): the artifact binds the EXACT protocol and
  assessment and states decisions **25/28/29/30** (algorithm key + the exact
  pinned KMeans parameter snapshot/hash; grain / interval / stage; k;
  occupancy / rows / sample floors and the stability minimum) — every value
  re-verified against the registry, protocol, and assessment; supersession is
  store-owned and hash-chained (`SUPERSESSIONS.jsonl` + `SUPERSESSIONS.head`;
  the line precedes the replacement's publication; every line backed by a
  verified replacement; any edited / deleted / reordered / dangling line
  fails closed; provenance may never weaken); synthetic provenance is lawful
  in the `synthetic_fixture` run scope only, and that scope is confined to
  test namespaces (`assert_run_scope_lawful_for_root`, the P0-4 mirror:
  synthetic-provenance artifacts are refused at persist AND at load in a
  research root); `persist_regime_promotion` requires the artifact from
  FEATURE_ELIGIBLE onward (a bare 64-hex reference is not evidence),
  structurally requires passing coverage gates + an OOS assignment for
  STRATIFICATION_READY (D6; re-derived from the loaded assessment at the
  report gate, whose owner branch re-runs the full authorization under the
  run scope), refuses `MODEL_FEATURE` in V1, and derives `decided_at` from
  verified artifacts (no flag, no wall clock; the CLI's STRATIFICATION_READY
  reuses S10's exact decision).
  Draft proposals (`DRAFT_OWNER_DECISION_PROPOSALS/`) carry placeholders that
  cannot persist; `scripts/ifvg_regime_promotion.py` is the ratification CLI.
- **Status-gated block activation (D9)** (`ifvg/ml/regime_block_activation.py`):
  `IFVG_REGIME_CONTEXT_V1` activates as a pure versioned event bound to the
  exact frozen FEATURE_ELIGIBLE decision, owner artifact, assessment,
  protocol, and fold-feature artifact (`as_of_policy =
  fold_local_fit_partition_assignment_v1`; join keys candidate/fold); the
  module registries stay PLANNED at import; bundle `B7_CORE_REGIME`.
- **Stratification classes (§5.5)** (`ifvg/ml/regime_stratified_{contracts,strategy,prop,frontier}.py`,
  `regime_stratification_{gate,service}.py`, `regime_assignment_sources.py`):
  `cohort_descriptive` / `stratified_prop` / `stratified_frontier` need
  STRATIFICATION_READY; `feature_only` / `cohort_model` need FEATURE_ELIGIBLE
  and are S09c deliverables (`regime_controlled_study.py`,
  `regime_cohort_model.py`); reports bind the exact decision / assessment /
  fit ids, the descriptive OOS artifact, and the executed-trade table hash;
  `RegimeFilterRef` is consumed (one cohort per stratum); thin strata are
  typed at the stamped 20 trades / 60 training rows; the frontier view is
  never a selection input; prop events attribute by source trade, then PIT
  for historical no-trade events, never a synthetic clock
  (`source_trade_then_pit_v1`), ordered by the D15 `EVENT_TYPE_PRECEDENCE`
  and dated by the event's own `trading_day`; the `stratified_prop` report
  carries the bounded report-local `account_event_regime_summary.parquet`
  (schema v1; one row per simulation × path × regime stratum / typed reason
  × event type; keyed by the exact regime-assignment evidence; budget
  `account_event_regime_summary_budget_v1` = 5,000,000 rows / 256 MiB,
  typed refusal before publication; four envelope extras — never per-event
  JSON); "nothing here promotes".
- **Prop-event detail (D15)** (`alpha_lab/propsim/event_detail.py`):
  `event_detail_persistence_policy_id ∈ {none_v0, account_event_detail_by_path_parquet_v2}`
  + storage policy / schema version / budgets enter the account- and
  portfolio-simulation identities and `SimulationProtocol` (default `none_v0`;
  every simulation/charter/pipeline id moves — pre-acceptance evolution);
  under v2 literally streaming ZSTD Parquet partitions (one path block at
  a time through the store's sidecar-producer protocol — `ProducedSidecar` /
  `SidecarProducer`; memory = one block + a 32-byte-per-row event-id index)
  by `path_block_id = floor(path_ordinal / 250)`
  with exact event time / trading day / clock policy / total-order fields;
  budgets (10,000,000 rows; 2 GiB; block 250) fail BEFORE atomic publication;
  `none_v0` artifacts are never widened.
- **Bundle-aware CatBoost rung (D13)** (`ifvg/ml/catboost_bundle_model.py`,
  `comparison_rows.py`): `ifvg_context_catboost_bundle_v1` (AVAILABLE, kind
  `nonlinear_challenger_bundle`; frozen-lane parameters by value; native NaN
  + `MISSING_CATEGORY`; registry ∪ block-declared categoricals) runs on both
  arms of every controlled comparison; `comparison_row_id = hash{fold_schedule_id,
  candidate_fold_set_id, fold_index, candidate_id, label_artifact_id}` keys
  the identical-rows gate and every paired delta (the legacy `oos_row_id`
  stays for M0–M3 compatibility); the frozen M0–M3 CatBoost lane is byte- and
  identity-unchanged (golden protocol hash).
- **Stability / transitions (D10/D11)**: bootstrap on EVERY valid fold
  (per-fold seeds; fold 0 reproduces R6); the promotion gate
  `minimum_bootstrap_aligned_ami_mean` (0.5) applies to the protocol-wide
  minimum fold mean; candidate-event transitions reset on trading day /
  named session / a stamped 7200 s gap; panel transitions count consecutive
  completed bars within a trading day.
- **Evidence hygiene**: the two provider-key tests are hermetic
  (`monkeypatch.delenv`); the R6.1 browser manifest v2 binds screenshots to
  the final commit (`verify_browser_manifest.py`); each release folder
  carries a `git format-patch` source-review patch + sha256; two independent
  read-only adversarial reviews (contract fidelity; safety/access) — 28
  findings, every one dispositioned in
  `R6.1/ADVERSARIAL_REVIEW_RESOLUTION.md`.
- **Trust boundary (stated in-tree; DEV-R6-8)**: the manifest protocol verifies
  INTEGRITY, not authenticity — a store root is a trusted local directory;
  the UI never unpickles; signing / a non-pickle fit serialization remains a
  post-V1 hardening candidate. `IFVG_REGIME_CONTEXT_V1` reaches a predictive
  bundle only through the status-gated activation above; S11 stays blocked;
  no regime output reaches an execution surface in V1.
"""

README_OLD = """panel `scripts/ifvg_regime_panels.py` — kmeans_v1 only; the
GMM/minibatch/spectral/Nyström implementations are the post-V1
regime-expansion release) implementations are complete;"""
README_NEW = """panel `scripts/ifvg_regime_panels.py` — kmeans_v1 only; the
GMM/minibatch/spectral/Nyström implementations are the post-V1
regime-expansion release), and R6.1 (the regime-lane correction: the
5m/15m context-bar panel materializer + `IFVG_CONTEXT_BAR_PANEL_V1` /
`BP0_CONTEXT_BAR_PANEL`, fold schedules + fold-set artifacts, the verified
observation seam + executor, the descriptive OOS-assignment and fold-local
feature artifacts, the regime study inside the 16-stage pipeline
(`ifvg/ml/regime_study.py`, `ifvg/search/pipeline_regime.py`), verified
owner-decision evidence `ifvg/search/owner_decisions.py` + the
`ifvg_regime_promotion.py` CLI, the five stratification classes
`ifvg/ml/regime_strat*.py`, the bundle-aware CatBoost rung
`ifvg/ml/catboost_bundle_model.py` + D13 comparison rows, the D15 prop-event
detail `alpha_lab/propsim/event_detail.py`, per-fold stability + grain
transition policies, and the regime surfaces of the pipeline tab)
implementations are complete;"""

YAML_OLD = (
    '    R6_kmeans_regime_lane: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)
YAML_NEW = (
    '    R6_kmeans_regime_lane: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
    '    R6_1_regime_correction: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
    'regime_stores: "bundle_feature_views, context_bar_panels, fold_schedules, fold_sets, '
    "regime_oos_assignments, regime_fold_features, owner_decisions, "
    'regime_stratified_reports, regime_controlled_studies, regime_cohort_model_studies", '
    'owner_proposals: "DRAFT_OWNER_DECISION_PROPOSALS (25/28/29/30 per protocol; not authorizations)", '
    'adversarial_round: "28 findings (6 major / 4 medium / 18 minor) dispositioned; 0 open"}\n'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [(README_OLD, README_NEW)],
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
            worktree = Path(path).read_text(encoding="utf-8")
            updated = worktree
            for old, new in transforms:
                if old is None:
                    if new not in updated:
                        updated = updated.rstrip("\n") + "\n" + new
                elif old in updated:
                    updated = updated.replace(old, new)
            Path(path).write_text(updated, encoding="utf-8", newline="\n")
            print(f"worktree {path} updated with the R6.1 lane transforms")
            continue
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = _transformed(head, transforms)
        blob = _git(
            "hash-object", "-w", "--stdin", "--path", path,
            data=staged.encode("utf-8"),
        ).decode("ascii").strip()
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + R6.1 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
