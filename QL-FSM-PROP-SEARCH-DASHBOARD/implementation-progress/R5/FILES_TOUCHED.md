# R5 — Repository files touched

(Excludes the user's pre-existing dirty files where noted; the three shared
docs commit as HEAD + R5 lane transforms only via `stage_shared_docs.py`,
and `docs/ML_TRAINING_WORKBENCH.md` stays untouched/uncommitted.)

**Reconciliation (R5-FIX finding 9):** the R5 commit `dda40c9`
(`6b820ad..dda40c9`) touches exactly **50 files = 27 added + 23 modified**,
verified from `git diff --name-status`. The category counts below now sum
to that: 10 + 2 added src/scripts, 15 added test files, 12 + 2 modified
src/scripts, 5 modified test files, 4 modified docs. (The original
document's headings — "6 new source modules", "11 new tests", "Modified
source (12)" with 13 rows including two scripts, and GATE_SUMMARY's "6
modified test files" — did not reconcile; they are corrected here.)

## New source modules (10)

| File | Content |
|---|---|
| `src/.../ifvg/search/pipeline.py` | CS §7 pipeline contracts (semantic/attempt split, stage results, pipeline result), capability-scoped stage-plan readiness, the 16-stage runner (idempotent store-reusing executors; identity-proven REUSED; S11 blocked; verify-then-activate publication), publication gates + activation |
| `src/.../ifvg/search/executors.py` | Real runner-entry factories (baseline-verification search/pipeline wiring; fail-before-path at construction) + the canonical `synthetic_firm_specs()` |
| `src/.../ifvg/features/bundle_feature_view.py` | Available-blocks-only bundle feature views + `frozen_tier_for_bundle` (registered identity pair) |
| `src/.../ifvg/ml/__init__.py` · `model_protocols.py` · `logistic_model.py` · `supervised_ladder.py` · `calibration_policies.py` · `decision_policies.py` · `drift_monitoring.py` | The bounded §7B supervised lane (registries, ladder, paired deltas, portable fits, S11 text, drift builders) — 7 modules |

## New scripts (2)

| File | Content |
|---|---|
| `scripts/ifvg_pipeline_job.py` | Detached pipeline CLI shim (start/status/cancel/resume/worker/publish-gates/activate; registry-gated worker; one Popen) |
| `scripts/ifvg_pipeline_tab.py` | The §30 Full Pipeline Run surface (`render_pipeline_run`; `ifvg_pipeline_v1_*`; one scanned `_spawn_pipeline_job` seam) |

## Modified source (12)

| File | Change |
|---|---|
| `search/identities.py` | `registered_identity_pairs()` imports the new contract modules (ml decision policies, pipeline, bundle feature view) |
| `search/orchestrator.py` | `merge_prop_vectors` extracted (behavior-identical; `run_search` now calls it) — DEV-R5-3 |
| `search/insights.py` | `InsightPanelPayload/Envelope` + identity registration (S14 persistence) |
| `search/lineage.py` | `serialize_/deserialize_native_lineage_map` (S02 sidecar projection) |
| `search/store.py` | +2 store names (`pipeline_stage_results`, `pipeline_specs`) — DEV-R5-2 |
| `search/runner_registry.py` | +3 registered entries at R5 (real search/pipeline executors + synthetic pipeline wiring); `runner_entry_key_for_charter` maps real verification charters; new `pipeline_entry_key_for_charter`; refreshed refusal copy. *(R5-FIX later removed the tests.* values from the production map — see the R5-FIX section.)* |
| `study/comparison_contracts.py` | `ComparisonResultEnvelope` + identity registration (DEV-R4-16 persistence) |
| `study_status.py` | `PIPELINE_STAGE_TITLES` + stage-status presentations + `PIPELINE_STAGE_NOT_REQUIRED`; `pipeline_runner_planned` → `pipeline_no_runs`; `runner_executor_planned` copy updated to the operator-run boundary |
| `study_presentation.py` | `PipelineStageRow/PipelineProgress` + `derive_pipeline_stage_rows/derive_pipeline_progress` |
| `study_providers.py` | `list_pipeline_runs`, `load_comparison_results_for_search`, `load_ladder_diagnostics` |
| `search/failure.py` | `sanitize_failure_message` drops any drive-letter rooted path line (`[A-Za-z]:[\\/]`), not only `C:` — adversarial F9 |
| `src/alpha_lab/propsim/search_bridge.py` | Additive mode bridging + store writers (DEV-R3-11/DEV-R4-17); `policy_set_envelope()`; `persist_account_simulation` |

## Modified scripts (2)

| File | Change |
|---|---|
| `scripts/ifvg_ui_common.py` | `PIPELINE_STATE_PREFIX = "ifvg_pipeline_v1_"` |
| `scripts/ifvg_study_wizard.py` | Mode-5 charter-assembly mapping (DECISIONS_TAKEN #36); step-8 mode-5 seam now renders the pipeline surface (field-persist first) |

## New tests (15 files)

Package/fixture scaffolding (5):
`tests/agents/data_infra/{__init__,ifvg/__init__,ifvg/ml_fixtures/__init__}.py`,
`ml_fixtures/synthetic_supervised.py` (fixture 1),
`tests/agents/ifvg_search/pipeline_fixture.py`.

Test suites (10): `test_supervised_ladder.py`, `test_logistic_model.py`,
`test_ml_registries.py`, `test_drift_monitoring.py`;
`test_pipeline_contracts.py`, `test_pipeline_run.py`,
`test_pipeline_job_script.py` (hosts the synthetic pipeline entry),
`test_bundle_feature_view.py`; `tests/propsim/test_search_bridge_modes.py`;
`tests/agents/test_ifvg_pipeline_tab.py` (FUX-PIPE-001..006 AppTests).

## Modified tests (5)

`tests/agents/ifvg_search/test_orchestrator.py` (charter fixture gains the
`simulation_protocol` kwarg), `test_runner_registry.py` (R5 registry set +
fail-before-path factory rows), `test_study_status.py` (§31 set:
`pipeline_no_runs`), `tests/agents/test_ifvg_study_scans.py` (pipeline tab
in scope; second Popen seam; `ifvg_pipeline_v1_*` allowlist),
`tests/agents/test_ifvg_study_wizard.py` (mode-5 test flipped to the
operator workflow).

## Docs (4 modified; committed via staging where user-dirty)

`docs/DECISIONS.md` — D-044 (supervised half) + D-045 (pipeline half) +
updated reservation note (committed normally). Shared docs staged as HEAD +
R5 transforms with user hunks preserved: `ARCHITECTURE.md` (R5 additions
section), `docs/README.md` (decisions line + R5 sentence),
`docs/pipeline_state.yaml` (`R5_pipeline_mbp1_readiness_ladder → complete`).

## R5-FIX round (second commit on the branch)

Code (see `R5_FIX_REPORT.md` for the finding→change map and
`_r5fix_diff_stat.txt` for the raw stat):

| File | Change |
|---|---|
| `src/.../search/runner_registry.py` | Production map = src executors ONLY (no `tests.*`); guarded `register_development_runner_entries` + merged resolution (DECISIONS_TAKEN #43) |
| `src/.../search/pipeline.py` | `PipelineWiring.loaded_seed_snapshot_id_source` + S00 requires it for the real scope (`_loaded_seed_snapshot_id_for_real_scope`, #44); S09 zero-row parity copy "not evaluable"; S14 mints the typed `SearchDerivationComparisonSubject` |
| `src/.../search/executors.py` | `loaded_seed_snapshot_id_source(...)` — verified seed-snapshot load → content-derived id; wired into the real pipeline entry |
| `src/.../study/comparison_contracts.py` | `StudyCellComparisonSubject` / `SearchDerivationComparisonSubject` discriminated union replaces the bare `comparison_id` field (#45) |
| `src/.../ifvg/ml/supervised_ladder.py` | `parity["status"] = "held" / "not_evaluable"` |
| `scripts/ifvg_pipeline_tab.py` | Arrow-safe typed frames (`_ladder_frame` Int64/Float64/string + attempts/comparisons tables), `_parity_caption`, comparison Subject column, `width="stretch"` |
| `scripts/ifvg_active_runs_tab.py` · `ifvg_lab_tab.py` · `ifvg_results_compare.py` · `ifvg_results_tab.py` · `ifvg_study_wizard.py` · `ifvg_verifier_tab.py` | `use_container_width=True` → `width="stretch"` (73 occurrences across the 7 branch scripts incl. the pipeline tab; 0 remain in branch files) |
| `pyproject.toml` | `streamlit>=1.41` → `>=1.54` (the `width` API's verified floor) |
| `tests/agents/ifvg_search/conftest.py` | Registers the development (synthetic) runner entries |
| `tests/.../test_runner_registry.py` | Rewritten for production purity + guarded registration |
| `tests/.../test_comparison_contracts.py` · `test_child_replay.py` · `test_pipeline_run.py` · `test_supervised_ladder.py` · `tests/agents/test_ifvg_pipeline_tab.py` | New/updated coverage: typed subject, seed-source evidence, parity wording, Arrow-safety regression (`pa.Table.from_pandas` on the ladder frame) |

Evidence-side (untracked folder): `r5_smoke_app.py` rewritten
(content-addressed scratch, evidence-validated reuse, stale-key clearing,
`smoke_manifest.json`); this document, `DEVIATIONS.md`, `TEST_RESULTS.md`,
`GATE_SUMMARY.md`, `ADVERSARIAL_REVIEW_RESOLUTION.md` addendum,
`R5_FIX_REPORT.md`, `browser-smoke/` regenerated with `MANIFEST.json`,
`_r5fix_pytest.txt` / `_r5fix_ruff.txt` / `_r5fix_diff_stat.txt` /
`_smoke_server_r5fix.log` raw outputs; `../DECISIONS_TAKEN.md` §R5-FIX
(#43–#45) + #42 wording brought forward.

## Implementation-progress files

`PRE_R5_BASELINE.md`, `FILES_TOUCHED.md` (this), `DEVIATIONS.md`,
`TEST_RESULTS.md`, `ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md`,
`ADVERSARIAL_REVIEW_RESOLUTION.md`, `GATE_SUMMARY.md`, `R5_FIX_REPORT.md`,
`stage_shared_docs.py`, `r5_smoke_app.py`, `browser-smoke/`,
`../DECISIONS_TAKEN.md` §R5 (#34–#42) + §R5-FIX (#43–#45).
