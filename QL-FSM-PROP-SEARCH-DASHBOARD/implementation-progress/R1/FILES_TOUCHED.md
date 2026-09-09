# R1 — Files Touched

Status of every repo file below: **staged and committed at the R1 release
checkpoint** (the exact commit hash is recorded in `GATE_SUMMARY.md`; until
that checkpoint the files exist uncommitted on the feature branch).

## New repository source files

```text
src/alpha_lab/agents/data_infra/ifvg/search/__init__.py
src/alpha_lab/agents/data_infra/ifvg/search/identities.py
src/alpha_lab/agents/data_infra/ifvg/search/axis_registry.py
src/alpha_lab/agents/data_infra/ifvg/search/authorization.py
src/alpha_lab/agents/data_infra/ifvg/search/charter.py
src/alpha_lab/agents/data_infra/ifvg/search/failure.py
src/alpha_lab/agents/data_infra/ifvg/search/store.py
src/alpha_lab/agents/data_infra/ifvg/search/catalog.py
src/alpha_lab/agents/data_infra/ifvg/search/verification.py
src/alpha_lab/agents/data_infra/ifvg/search/child_replay.py
src/alpha_lab/agents/data_infra/ifvg/study/__init__.py
src/alpha_lab/agents/data_infra/ifvg/study/dimension_contracts.py
src/alpha_lab/agents/data_infra/ifvg/study/cohort.py
src/alpha_lab/agents/data_infra/ifvg/study/study_cell.py
src/alpha_lab/agents/data_infra/ifvg/study/computation_path.py
src/alpha_lab/agents/data_infra/ifvg/study/delta_outputs.py
src/alpha_lab/agents/data_infra/ifvg/study/comparison_contracts.py
src/alpha_lab/agents/data_infra/ifvg/features/__init__.py
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_source_contract.py
src/alpha_lab/agents/data_infra/ifvg/features/feature_blocks.py
src/alpha_lab/agents/data_infra/ifvg/features/feature_bundles.py
```

## New test files

```text
tests/agents/ifvg_search/__init__.py
tests/agents/ifvg_search/conftest.py
tests/agents/ifvg_search/test_identities.py
tests/agents/ifvg_search/test_axis_registry.py
tests/agents/ifvg_search/test_authorization.py
tests/agents/ifvg_search/test_charter.py
tests/agents/ifvg_search/test_store_catalog.py
tests/agents/ifvg_search/test_verification.py
tests/agents/ifvg_search/test_child_replay.py
tests/agents/ifvg_search/test_study_cell.py
tests/agents/ifvg_search/test_computation_path.py
tests/agents/ifvg_search/test_comparison_contracts.py
tests/agents/ifvg_search/test_feature_blocks.py
```

## Modified repository files (per the plan's R1 modified list)

```text
src/alpha_lab/agents/data_infra/ifvg/development_access.py   (+VerificationReplayPolicy — third trusted class)
src/alpha_lab/agents/data_infra/ifvg/data_access.py          (require_fixed_exploration_allowlist: third trusted branch)
src/alpha_lab/agents/data_infra/ifvg/dataset.py              (table_content_hash promotion; ChainStart +
                                                              start_after_artifact/audit_capture_mode/
                                                              final_day_exhausts_dataset params;
                                                              V2CaptureResult.audit_frames/end_seed)
docs/DECISIONS.md                                            (D-039, D-040, D-042, D-043 + reservation note)
ARCHITECTURE.md                                              (§2d search-lane section appended)
docs/README.md                                               (search-lane pointer section appended)
docs/pipeline_state.yaml                                     (ifvg_prop_robust_config_search_v1 block appended)
```

Shared-file staging note: `ARCHITECTURE.md`, `docs/README.md`, and
`docs/pipeline_state.yaml` carried pre-existing, user-owned, uncommitted
modifications at baseline. The release commit stages HEAD content + this
implementation's appended sections ONLY (blob built via `git hash-object` /
`update-index`); the user's uncommitted hunks remain in the working tree,
uncommitted and byte-identical (preserved in `PRE_EXISTING_DIFF.patch`).
`docs/ML_TRAINING_WORKBENCH.md` (pre-existing modification) is untouched by
this implementation and is NOT part of the release commit.

## Never modified (verified)

All M0–M3 lane modules, all existing `src/alpha_lab/propsim/` modules, all of
Strategy-Core, all of Trade-Lab, all existing immutable artifacts/catalogs
under `data/`, all pre-existing tests.

## Implementation-progress files produced (NOT committed to the repo)

```text
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/PRE_IMPLEMENTATION_BASELINE.md
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/PRE_EXISTING_DIFF.patch
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/BASELINE_PYTEST.txt
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/R1_PYTEST_FULL.txt
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/COVERAGE_MATRIX_PROPOSED.json
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/WINDOW_COVERAGE_SCAN.json
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/DRAFT_VERIFICATION_AUTHORIZATION.md
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/FILES_TOUCHED.md
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/DEVIATIONS.md
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/TEST_RESULTS.md
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/ACCESS_SAFETY_EVIDENCE.md
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/ADVERSARIAL_REVIEW.md
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/ADVERSARIAL_REVIEW_RESOLUTION.md
QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/GATE_SUMMARY.md
```
