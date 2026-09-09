# R6 — Files Touched

Reconciled against `git status --short` before the release commit
(HEAD = `7f018f5`, R5B), including the adversarial-fix round.

## New source modules (7)

| File | Content |
|---|---|
| `src/.../ifvg/ml/regime_contracts.py` | the V3 P1-3 four-way split (protocol/fit/assessment/promotion) + P1-B panel-grain validation; the STRUCTURAL promotion + role ladders (one step, chained `previous_decision_ref`, ISO-8601 `decided_at`, 64-hex owner ratification from FEATURE_ELIGIBLE, execution-side roles unrepresentable with the S11 reason); `pinned_parameters_hash` in the protocol identity; verified non-empty `source_artifact_ids` + `training_feature_matrix_hash` in the fit identity; the sidecar-name whitelist; default-on registry-derived `assert_no_regime_leakage`; `REGIME_PROPOSED_DEFAULTS` (+ the stamped decision-row minimum) and `sample_adequacy_minimum`; four identity pairs registered |
| `src/.../ifvg/ml/regime_algorithms.py` | the algorithm registry — `kmeans_v1` implemented with the exact §4 pinned params; five post-V1 entries planned with fail-closed refusals, per-entry `initialization_policy` and `executable_policies`, `pinned_parameters_hash`, and `assert_protocol_executable` (planned algorithms AND planned policy values refuse before any fit); nested pins immutable; V1-boundary import invariants |
| `src/.../ifvg/ml/regime_preprocessing.py` | the fixed fold-local pipeline (median imputer + indicators → optional train-fitted winsorizer → standard scaler); the fold-only fit API; unique observation keys (`candidate_id`/`row_id`); all-missing training rows excluded and recorded; canonical parameter payload + hash; the training-matrix hash |
| `src/.../ifvg/ml/regime_service.py` | bundle + leakage input permission, protocol executability, per-fold fit/assign (vectorized; distances to every centroid, margin d2−d1, `observation_ts_utc`), typed-null row preservation, the capability assessment with the stamped sample-adequacy/occupancy/rows gates, and the OOS-only PIT panel→candidate assignment |
| `src/.../ifvg/ml/regime_alignment.py` | `centroid_min_distance_hungarian_v1` reporting-only alignment over the scaled input-feature space with the EXACT ascending-local-id tie-break; geometry-ranked canonical ids; top-magnitude semantic descriptors |
| `src/.../ifvg/ml/regime_diagnostics.py` | stability builders: seeded bootstrap aligned-AMI + per-cluster agreement (reference fold training rows only), the OOS-timeline temporal persistence + transition matrix (within-fold, by observation timestamp), fold-to-fold recurrence, centroid separation, descriptive silhouette; `oos_regime_timeline` |
| `src/.../ifvg/ml/regime_store.py` | dual-format persistence (canonical JSON + joblib) with assignment↔fit binding, verify-before-publish, post-publish re-verification with withdrawal, manifest-relative refs; the structural `persist_regime_promotion` (assessment + previous-decision verification); the no-unpickle UI loader `load_regime_fit_assignments`; the four regime stores |

## New scripts (1)

`scripts/ifvg_regime_panels.py` — the Regime Lane panel (FUX §35 R6):
algorithm registry with planned-disabled entries + the mandatory spectral
warning, proposal-stamp table, exact-ID model card (grain identity incl.
panel fields, input bundle, fixed-k stamp, OOS/alignment policies,
coverage + per-fold coverage, fit identities, nominal-id occupancy,
stability incl. per-cluster agreement and centroid profiles, transition
matrix, insufficient-sample blocked state), the exact-fit-id assignment /
stratification / timeline view, and the promotion-decision role/status
view. Read-only (source-scanned: no button/form/toggle/select).

## Modified source (3)

| File | Change |
|---|---|
| `src/.../ifvg/search/store.py` | four new store names (`regime_protocols`, `regime_fits`, `regime_assessments`, `regime_promotions`); `load_sidecar_bytes` hardened (manifest-hash verification, sidecar-name whitelist, hash of the returned bytes — review S6); `hashlib` hoisted to module level |
| `src/.../ifvg/search/identities.py` | audit-enumeration import list gains `ml.regime_contracts` |
| `scripts/ifvg_pipeline_tab.py` | the "Regime Lane (V1 KMeans, development)" expander mounted beside the MBP-1 panel (+4 lines; no spawn path) |

## New fixtures (2) and tests (3 files + AppTests + scan list)

`tests/agents/data_infra/ifvg/ml_fixtures/synthetic_clusters.py`
(fixture 2 — known clusters, fold-input-shaped),
`synthetic_oos_assignment.py` (fixture 4, KMeans arm — injected
all-missing rows); `test_regime_contracts.py` (14),
`test_regime_service.py` (20), `test_regime_store.py` (9);
`tests/agents/test_ifvg_pipeline_tab.py` gains the 8 R6 panel tests
(registry/stamps, model card, blocked state, panel-grain identity,
assignment + stratification view, promotion view, bogus id, no-control
scan); `tests/agents/test_ifvg_study_scans.py::_UI_SCRIPTS` gains
`ifvg_mbp1_panels.py` and `ifvg_regime_panels.py` (review S5).

## Docs

- `docs/DECISIONS.md` — D-044's KMeans regime half landed + reservation
  note updated (normal commit; promotion wording corrected after the
  round).
- `ARCHITECTURE.md` / `docs/README.md` / `docs/pipeline_state.yaml` — R6
  lane transforms via `R6/stage_shared_docs.py` (same user-hunk isolation
  as every release: the commit carries HEAD + the R6 transforms only; the
  post-commit `--apply-worktree` replay leaves the user's pre-existing
  hunks as the sole surviving worktree diff — the owner should be aware
  the committed files intentionally differ from the worktree).
- `docs/ML_TRAINING_WORKBENCH.md` — untouched, uncommitted (user-owned).

## Implementation-progress files (this folder)

`PRE_R6_BASELINE.md`, `FILES_TOUCHED.md` (this), `DEVIATIONS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md`,
`ADVERSARIAL_REVIEW_RESOLUTION.md`, `TEST_RESULTS.md`, `GATE_SUMMARY.md`,
`stage_shared_docs.py`, `r6_smoke_app.py` (shell-only smoke harness,
`%TEMP%`-only writes), `browser-smoke/` (screenshots + `MANIFEST.json`),
raw outputs `_baseline_pytest.txt` (= the R5B release-final run),
`_midpoint_pytest.txt`, `_final_pytest.txt`, `_smoke_server.log`,
`_surviving_shared_doc_diff.patch`; `../DECISIONS_TAKEN.md` entries 54–66.

## Never modified

All M0–M3 lane modules, all propsim modules, Strategy-Core, Trade-Lab,
every existing immutable artifact/catalog, the plan package. No new
package dependency (scikit-learn/scipy/joblib all pre-existing).
