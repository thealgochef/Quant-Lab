# R6.1-FIX — Test Results

All commands run from the repo root on Python 3.13.1 / pytest 9.0.2
(scikit-learn 1.7.0, catboost 1.2.10, pyarrow 20.0.0, pandas 2.3.1,
numpy 2.3.1, pydantic 2.12.5, ruff 0.15.2 — unchanged; no new package
installed or declared).

## Release-final full suites (the working tree that became commit `0c8d528`)

The R6.1-FIX commit is path-scoped to the file list in `FILES_TOUCHED.md`;
every R6.1-FIX source / test / script file was present in the working tree
for both runs (the untracked new modules included, intent-added for the
hygiene check), so the runs below exercise exactly the committed code —
AFTER the adversarial-fix round. The only worktree content outside the
commit is the user-owned doc hunks (`ARCHITECTURE.md`, `docs/README.md`,
`docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md` — no test reads
them) and the pre-existing untracked local data / report files.

```text
python -m pytest -q -p no:cacheprovider                                   (environment AS-IS)
→ 1993 passed, 435 warnings in 1348.98s (0:22:28)            exit=0

env -u POLYGON_API_KEY -u DATABENTO_API_KEY python -m pytest -q -p no:cacheprovider
→ 1993 passed, 435 warnings in 1507.77s (0:25:07)            exit=0
```

(raw: `_final_pytest.txt` / `_final_pytest_keys_cleared.txt`; each records
HEAD at the time of the run (`6c0b60a`, the tree that became `0c8d528`), the
presence of the two provider keys, the start / finish instants and the
pytest exit code)

**Both runs: 0 failed.** The provider-key tests are hermetic since R6.1, so
the as-is and keys-cleared counts coincide.

**Baseline → final: 1931 (R6.1, both environments) → 1993 passed (+62 net
new tests), 0 failures in either environment.**

### Warnings

435 per run = **429 third-party** (the scikit-learn / SciPy L-BFGS-B
`disp` / `iprint` `DeprecationWarning` emitted by every logistic fit —
`sklearn/linear_model/_logistic.py:456`; the count grew with the new
ladder / pipeline tests) + **6 project-owned** (5 × `dataset.py:813`
pandas concat `FutureWarning` via `test_child_audit_companion` /
`test_ifvg_fsm_audit_contracts`; 1 × `tests/agents/test_ifvg_context_experiment_engine.py:260`
`FutureWarning`). R6.1 recorded 386 = 378 third-party + 8 project-owned: the
two project-owned Pydantic serialization warnings (enum `.value` strings in
`model_copy`, F-10D) are gone; the remaining six belong to Phase 2 (F-18,
warning policy). No warning filter was added.

## Sequence

| Run | Result | Raw |
|---|---|---|
| Pre-R6.1-FIX baseline (= R6.1 release-final, tree `70e972a0…`) | 1931 passed, 0 failed, as-is and keys cleared | `../R6.1/_final_pytest*.txt` |
| Golden identities alone (mixed worktree, before the round) | 1 passed | `_goldens_pytest.txt` |
| Midpoint (WS A–E landed, BEFORE the adversarial round) | **1967 passed, 1 failed** (the `test_regime_service.py` panel-PIT frame lacking the F-05 value columns — a fixture update), 412 warnings, 22:05 | `_midpoint_pytest.txt` |
| Release-final, as-is (the `0c8d528` content) | **1993 passed**, 0 failed, 435 warnings, 22:28 | `_final_pytest.txt` |
| Release-final, keys cleared (the `0c8d528` content) | **1993 passed**, 0 failed, 435 warnings, 25:07 | `_final_pytest_keys_cleared.txt` |

Midpoint → final: +25 tests from the adversarial-fix round (the three-run
provenance test, the manifest-less entry tests, the tampered S09 record, the
publication-block reset / activation re-derivation, the drifted runner, the
disagreeing neutrality hash, the real artifact tamper, the typed-reason
loader test, the parametrized projection-equality test, the stratification
refusals (caller frame / impossible accounting / refuted claim), the
assignment linkage + arithmetic + protocol-column + strict as-of tests, and
the seven enum-guard / label-identity tests) + the fixed midpoint failure.

## Targeted suites (post-round states, mixed worktree before the commit)

| Suite | Count |
|---|---|
| `tests/agents/ifvg_search/test_pipeline_evidence_integrity.py` (S02 persists / S14 verified-loads; reused children never vanish; tamper refused; typed skips; prior-attempt tamper → PENDING downstream; lineage tamper; S15 reload failures; wiring gaps; provenance-independent costed evaluations across run orders; manifest-less table entry; `_delivered_by_s09c` tamper; publication reset + activation re-derivation; drifted runner; disagreeing neutrality hash; real regime-artifact tamper + failing publication gates) | **18 passed** |
| `tests/agents/ifvg_search/test_pipeline_regime.py` (16 terminal states; second-attempt reuse with `replay_invocations == 1` per reused child under stratified reporting; S02 reproduction over the persisted table incl. the parked-table branch; S14 zero fitting ×3; the supervised E2Es) | **19 passed** (4:56) |
| `tests/agents/ifvg_search/test_pipeline_run.py` (`replay_invocations == 0` for every reused child of the non-stratified double run) | 19 passed |
| `tests/agents/ifvg_search/test_store_sidecar_probe.py` + `test_executed_trade_table.py` | 4 + 5 passed |
| Regression sweeps by the pipeline workstream: identities / regime_store / fold_schedules / child_replay / mbp1_coverage_evidence / orchestrator / pipeline_job_script; pipeline tab + owner_decisions + propsim | **83 passed**; **203 passed** |
| `tests/agents/data_infra/ifvg/test_regime_stratification_evidence.py` + `test_regime_stratification.py` | 7 + 15 passed |
| `test_regime_assignment_evidence.py` (12) + `test_regime_oos_assignment.py` + `test_regime_service.py` + `test_regime_store.py` + `test_regime_fold_feature_evidence.py` + `test_regime_fold_features.py` + goldens | **66 passed** |
| `test_r61_fix_review_fixes.py` (7) + `test_label_identity.py` + `test_supervised_ladder.py` + `test_controlled_feature_study.py` + `test_logistic_model.py` + `test_catboost_bundle_model.py` + `test_identities.py` + `test_mbp1_materializer.py` + goldens | **99 passed** |

## Lint and hygiene (the committed content)

```text
ruff check src tests scripts   → All checks passed!            (exit 0)
git diff --check               → clean (tracked files)          (exit 0)
git add -N <11 new files>; git diff --check → clean             (exit 0)
grep '^\s*assert ' over every touched src module → none
```

(raw: `_ruff_and_diffcheck.txt`)

## Golden identities

`tests/agents/ifvg_search/test_r61_fix_goldens.py` pins the seven identities
of `PRE_R6_1_FIX_BASELINE.md` (`resolved_regime_protocol_id`, the R6 golden
fold-0 `regime_fit_id`, the frozen M0 CatBoost `resolved_hash`,
`core_replay_id`, `account_simulation_id`, `feature_block_registry_hash`,
the `B0_CORE` bundle id) — green alone before the round
(`_goldens_pytest.txt`), inside every workstream's final run, and inside
both release-final suites.

## Double-run reuse (plan §3.10)

`test_second_attempt_with_different_workers_shares_every_semantic_identity`
(non-stratified): identical stage-result ids, every non-S11 stage REUSED,
`replay_invocations == 0` for every reused child.
`test_second_attempt_reuses_every_regime_stage` (stratified reporting
requested): identical stage-result ids, every non-S11 stage REUSED, exactly
ONE verified reproduction per reused child (projection bytes AND raw
core-table hash must reproduce; `replay_invocations == 1`) — DEV-R6.1-FIX-20.

## Post-commit shared-doc verification

`stage_shared_docs.py --apply-worktree` replayed the three R6.1-FIX lane
transforms into the worktree; the surviving worktree diff on the four
user-owned files (`ARCHITECTURE.md`, `docs/README.md`,
`docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md`) is
content-identical to `../R1/PRE_EXISTING_DIFF.patch` — the diff-of-diffs
with the git `index` header lines removed is EMPTY
(`_surviving_shared_doc_diff.patch`, 4 files, 12,348 bytes).

## Source-review evidence

`R6.1-FIX.patch` = `git format-patch --stdout 6c0b60a..0c8d528` (479,595
bytes); sha256 in `R6.1-FIX.patch.sha256`
(`f5c46e43ba88d5c202707f448b299434016ffd44b7712fc74b8be9effef376b0`).
`R6.1-FIX.bundle` = `git bundle create … 179a2c9..feature/ifvg-prop-robust-config-search-v1`
(the R5B.1 → R6.1 → R6.1-FIX chain; prerequisite R6 `179a2c9`; 506,631
bytes; `git bundle verify` OK); sha256 in `R6.1-FIX.bundle.sha256`
(`cd38ce5ba7e0d839e42ece0568c9bc0e78d61c98a2b79336cc7e7bff9591ae8e`).
