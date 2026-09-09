# R5B — Files Touched

Reconciled against `git status --short` before the release commit
(including the adversarial-review fix round).

## New source modules (7)

| File | Content |
|---|---|
| `src/.../ifvg/features/mbp1_arrow_schemas.py` | four pinned Arrow schemas + canonical schema hashes; price-scale policy + instrument tick map; per-window evidence-column helpers; deep-book guard at import |
| `src/.../ifvg/features/mbp1_source_artifact.py` | immutable content-addressed MBP-1 source/coverage artifact (deliverable 1); normalization (source_ordinal, tick scaling, four-part-key sort); sequence-gap/coverage computation; authorize-before-path real reads; legacy-provenance refusal; save/reload/reuse with event-byte sidecars |
| `src/.../ifvg/features/mbp1_stage_windows.py` | `StageEvidenceCutoff` admission on the complete order key (deliverables 3/4); exact-key `<`/`<=` per `WindowTriggerSemantics`; strict timestamp-only cutoffs with same-ts ambiguity; completed-bar cutoffs from the candidate row's five anchors; window construction that can never widen |
| `src/.../ifvg/features/mbp1_feature_materializer.py` | the offline materializer (deliverable 5): 76 registered metrics (snapshot + CKS-OFI transition formulas), typed-missing precedence, row preservation, deterministic batch/repeat, `Mbp1FeatureArtifactPayload/Envelope` (recipe identity + post-materialization table hashes as envelope facts), immutable persistence |
| `src/.../ifvg/features/mbp1_coverage.py` | coverage/validity reports (deliverables 6/9): per-day source coverage, per-window reason counts, per-feature non-null fractions, permanent `research_only_offline` stamp, content-addressed persistence |
| `src/.../ifvg/features/mbp1_feature_join.py` | exact one-to-one candidate joins (deliverable 7): duplicate refusal, typed nulls with `stage_outside_coverage` for absent rows, no nearest-time/row-order fallback |
| `src/.../ifvg/ml/controlled_feature_study.py` | the controlled Baseline vs Baseline+MBP-1 study (deliverable 10): challenger-vs-own-base arms, cross-arm identity assertions, paired Brier delta, `ControlledFeatureStudyPayload/Envelope` + detail sidecar, persistence |

## New scripts (1)

| File | Content |
|---|---|
| `scripts/ifvg_mbp1_panels.py` | the MBP-1 Order Flow dashboard (deliverable 11; FUX §35 R5B): availability + pre/post-activation registry hashes, frozen window registry, exact-ID coverage/missingness evidence, per-candidate stage-window drill-down, Baseline vs Baseline+MBP-1 comparison, persistent `research_only_offline` badge |

## Modified source (6)

| File | Change |
|---|---|
| `src/.../ifvg/features/feature_blocks.py` | the activation event published as the registry (deliverable 8): `with_activated_block` applied at module level over the exported `PRE_ACTIVATION_*` state; `mbp1_activation_resolution_payload()` (real schema hashes, window registry, formula/materializer versions); definition-level research boundary (`MBP1_RESEARCH_BOUNDARY_PATH`); import-time activation invariants |
| `src/.../ifvg/features/bundle_feature_view.py` | MBP-1 join integration: `ofl_*` columns come from the exact materialized frame; `mbp1_feature_artifact_id` binding on the view payload; inert-evidence-claim refusal; `mbp1_block_keys_in_bundle` |
| `src/.../ifvg/features/mbp1_source_contract.py` | added `MBP1_FORMULA_VERSION` / `MBP1_MATERIALIZER_VERSION` / `MIN_DAY_COVERAGE_FRACTION` constants (leaf-module home; additive) |
| `src/.../ifvg/ml/supervised_ladder.py` | bundle-parametrized ladders (DECISIONS_TAKEN #41 arrival): tier XOR (bundle_features + bundle_ref); typed `feature_source` in the run + ladder id; `CATBOOST_BUNDLE_REFUSAL` fail-closed |
| `src/.../ifvg/search/pipeline.py` | `PipelineWiring.mbp1_evidence_source` seam + S00 refusal; S05 materialize→persist→join with the `__mbp1_evidence__` sidecar; S09 controlled-study path (`_run_controlled_mbp1_stage`); S10 study diagnostics; readiness CatBoost×MBP-1 launch block; `_mbp1_bearing_bundles` helper |
| `src/.../ifvg/search/store.py` | four new store names: `mbp1_source_artifacts`, `mbp1_feature_artifacts`, `mbp1_coverage_reports`, `controlled_feature_studies` |
| `src/.../ifvg/search/identities.py` | `registered_identity_pairs()` import list gains the four new contract modules (audit-enumeration completeness; additive) |
| `src/.../ifvg/study_providers.py` | `mbp1_stage_evidence_defaults` — the R5B panel's exact-ID auto-fill provider with the store-integrity note (review S6 + the FUX raw-override scan) |

## Modified scripts (2)

| File | Change |
|---|---|
| `scripts/ifvg_pipeline_tab.py` | Configure: MBP-1-bearing bundle selection → `research_only_offline` badge + logistic-only model options + tier-lock caption; `_mbp1_default_ids` exact-ID auto-fill from persisted stage evidence; the MBP-1 Order Flow expander mounted on the pipeline surface |
| `scripts/ifvg_search_job.py` | DEV-R5-10 closure: `factory(charter, store_root=<worker --store-root>)` |

## New tests (5 files)

`tests/agents/ifvg_search/mbp1_fixture.py` (shared synthetic events/anchors),
`test_mbp1_schemas_and_source.py` (15), `test_mbp1_stage_windows.py` (10),
`test_mbp1_materializer.py` (18),
`tests/agents/data_infra/ifvg/test_controlled_feature_study.py` (8).

## Modified tests (8 files)

`pipeline_fixture.py` (+`build_mbp1_evidence`, `mbp1=True` mode),
`test_bundle_feature_view.py` (R5B verified-evidence/join/typed-null/
cohort-misalignment/inert-claim tests),
`test_feature_blocks.py` (activation-event replay proof; R5B active-state assertions),
`test_pipeline_contracts.py` (MBP-1 plan launchable under logistic; CatBoost tier-lock block),
`test_pipeline_run.py` (4 MBP-1 E2E tests incl. the S00 seam refusal),
`test_search_job_script.py` (factory accepts store_root),
`test_ifvg_pipeline_tab.py` (Configure R5B flips + 5 MBP-1 panel AppTests),
`test_identities.py` (type-aware placeholder for boolean envelope extras).

## Docs

- `docs/DECISIONS.md` — D-046 (the R5B activation decision) + reservation-note update (normal commit; not user-dirty).
- `ARCHITECTURE.md` / `docs/README.md` / `docs/pipeline_state.yaml` — R5B lane transforms committed via `R5B/stage_shared_docs.py` (HEAD + lane transforms only; the user's pre-existing hunks never enter the commit; post-commit `--apply-worktree` replays the transforms so the surviving diff is the user hunks only). ARCHITECTURE additionally corrects the R5-era bullet that said the block "stays planned" (tense fix — now truthful history).
- `docs/ML_TRAINING_WORKBENCH.md` — untouched, uncommitted (user-owned).

## Never modified

All M0–M3 lane modules, all existing propsim modules, Strategy-Core,
Trade-Lab, all existing immutable artifacts and catalogs, the plan package.
