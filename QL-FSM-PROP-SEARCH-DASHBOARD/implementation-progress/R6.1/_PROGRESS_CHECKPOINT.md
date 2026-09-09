# R6.1 correction release — PROGRESS CHECKPOINT (resume-from-here file)

Purpose: the R6.1 implementation phase has hit the context limit twice.
This file is the single place that records WHERE the work is, so any
reset resumes from here without re-deriving state from the tree.
Update it at every milestone (a milestone = a commit, a completed
workstream, a finished evidence document, or a started long run).

Authority: `../R6.1-CORRECTION-PLAN-DOCS-FINAL/R6.1_IMPLEMENTATION_PLAN.md`
(revision 3, owner-approved) — two release-scoped commits on
`feature/ifvg-prop-robust-config-search-v1`: **R5B.1 first, then R6.1**.
Process: `../../IMPLEMENTATION-START/` kickoff rules (tests-first ->
implementation -> ruff -> `git diff --check` -> two independent read-only
adversarial reviewers -> fixes -> evidence folder -> release-scoped commit;
no push, no merge; `acceptance_status: transitively_blocked_by_R1`).

## Timeline of resets

| Reset | When | State at reset |
|---|---|---|
| 1 | 2026-08-28 (earlier) | R5B.1 code authored + first internal review fixes; R6.1 workstreams H / A(part) / B / C / D(part) / F(part) authored |
| 2 | 2026-08-28 ~22:33 | R5B.1 adversarial round re-run + resolved; `R5B.1/{PRE_R5B_1_BASELINE,ADVERSARIAL_REVIEW,ADVERSARIAL_REVIEW_RESOLUTION,DEVIATIONS}.md` + `stage_shared_docs.py` written; shared docs STAGED (index) via `stage_shared_docs.py`; NOT committed |

## Phase 1 — R5B.1 closure (status: DONE 23:20 — commit f3f9ac2)

- [x] R5B.1 code complete; lane suites green in the mixed worktree (177 passed:
      `test_mbp1_coverage_evidence` 18 / `test_mbp1_materializer` 26 /
      `test_mbp1_schemas_and_source` 22 / `test_feature_blocks` 11 /
      `test_identities` 19 / `test_bundle_feature_view` 9 / `test_pipeline_run` 19 /
      `test_controlled_feature_study` 7 / `test_ifvg_pipeline_tab` 32 / `test_ifvg_study_scans` 14)
- [x] Adversarial round + resolution written (`R5B.1/ADVERSARIAL_REVIEW*.md`)
- [x] `DEVIATIONS.md`, `PRE_R5B_1_BASELINE.md` written; DECISIONS_TAKEN #67-#75 written
- [x] Shared docs staged (HEAD + R5B.1 transforms only) — verified in the index 22:37
- [x] `FILES_TOUCHED.md` (written 22:45)
- [x] Verification: index tree `b9fa5b84…` (dangling commit `c9349f6`); detached-worktree runs = location artifacts (47 real-data skips, 2 launch-spawn failures that pass in the main tree) — kept as `_detached_worktree_pytest_*.txt`; MAIN-TREE run with R6.1 files stashed (`git stash` = `R6.1 worktree files (temporarily removed …)`): AS-IS **2 failed (env pair), 1756 passed, 9:40** (`_final_pytest.txt`); keys-cleared run re-launched 23:07 (first attempt collected my new test files -> parked them in `scratchpad/parked/`); ruff + diff-check clean (`_ruff_and_diffcheck.txt`)
- [x] Real commit **`f3f9ac2`** (tree `b9fa5b84…`, parent `179a2c9`) — NOT pushed
- [x] Post-commit `--apply-worktree` replayed; `_surviving_shared_doc_diff.patch` IDENTICAL to R1/PRE_EXISTING_DIFF.patch modulo index lines
- [x] `R5B.1.patch` + `.sha256` (517b548c…)
- [x] `ACCESS_SAFETY_EVIDENCE.md` (written 23:10)
- [x] `TEST_RESULTS.md` (as-is 1756 passed + 2 env; keys cleared 1758 passed), `GATE_SUMMARY.md` written; verification worktrees removed; stash popped; parked files restored

### R5B.1 path list (the ONLY paths in the R5B.1 commit; everything else in the worktree is R6.1)

```
src/alpha_lab/agents/data_infra/ifvg/features/feature_blocks.py            (M — with_reresolved_block, PRE_R5B1_*, v2 envelope ONLY; the panel block registration is R6.1 and lands AFTER this commit)
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_arrow_schemas.py        (M)
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_coverage.py             (M)
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_coverage_evidence.py    (A)
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_coverage_diagnostic.py  (A)
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_feature_materializer.py (M)
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_source_artifact.py      (M)
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_source_contract.py      (M)
src/alpha_lab/agents/data_infra/ifvg/search/identities.py                  (M)
src/alpha_lab/agents/data_infra/ifvg/search/pipeline.py                    (M)
src/alpha_lab/agents/data_infra/ifvg/search/store.py                       (M)
scripts/ifvg_mbp1_panels.py                                                (M)
scripts/ifvg_mbp1_coverage_diagnostic.py                                   (A)
tests/agents/ifvg_search/mbp1_fixture.py                                   (M)
tests/agents/ifvg_search/pipeline_fixture.py                               (M)
tests/agents/ifvg_search/test_feature_blocks.py                            (M)
tests/agents/ifvg_search/test_identities.py                                (M)
tests/agents/ifvg_search/test_mbp1_coverage_evidence.py                    (A)
tests/agents/ifvg_search/test_mbp1_materializer.py                         (M)
tests/agents/ifvg_search/test_mbp1_schemas_and_source.py                   (M)
tests/agents/test_ifvg_pipeline_tab.py                                     (M)
docs/DECISIONS.md                                                          (M — D-048 + reservation note)
ARCHITECTURE.md / docs/README.md / docs/pipeline_state.yaml                (staged via R5B.1/stage_shared_docs.py — never `git add` these)
```

NOT R5B.1 (R6.1 — leave in the worktree): `bundle_feature_view.py`,
`controlled_feature_study.py`, `supervised_ladder.py` (fold-hash delegation),
`regime_diagnostics/service/contracts` diffs, `fold_schedules.py`,
`ml/fold_set_artifact.py`, `features/arrow_tables.py`, `features/context_bar_panel_*.py`,
`search/owner_decisions.py`, `ml/regime_{oos_assignment,sample_adequacy,observation_source,executor}.py`,
`tests/.../test_{fold_schedules,regime_oos_assignment,regime_sample_adequacy,regime_service,regime_contracts}.py`,
`tests/agents/ifvg_search/test_context_bar_panel.py`, `ml_fixtures/synthetic_context_panel.py`,
`docs/ML_TRAINING_WORKBENCH.md` (user-owned, never committed).

### Parked / prepared while the runs execute (RESTORE after `git stash pop`)

- `scratchpad/parked/ml_fixtures/synthetic_observation_source.py` -> `tests/agents/data_infra/ifvg/ml_fixtures/`
- `scratchpad/parked/data_infra_ifvg/test_regime_observation_source.py` -> `tests/agents/data_infra/ifvg/`
- `scratchpad/parked/ifvg_search/test_owner_decisions.py` -> `tests/agents/ifvg_search/`
- `scratchpad/patch_r61_foundation.py` — applies: store names; regime_store (reuse-by-reproduction fits,
  verified owner evidence for FEATURE_ELIGIBLE+, `run_scope`); feature_blocks panel registration
  (`with_registered_block`, `IFVG_CONTEXT_BAR_PANEL_V1`, `PRE_R6_1_*`); feature_bundles `BP0_CONTEXT_BAR_PANEL`;
  regime_service grain/bundle join-key coherence. Run it, then: migrate the R6 panel tests
  (`test_regime_oos_assignment.py` / `test_regime_service.py` `_panel_protocol` use B0 + M0 names -> must use
  BP0 + `cbp_*`), migrate `test_regime_store.py` test (f) to the synthetic owner decision (`run_scope="synthetic_fixture"`),
  add R6.1 panel-block tests to `test_feature_blocks.py`, add identity-pair names to `test_identities.py`.
- ruff fixes already applied in the worktree BEFORE the stash (owner_decisions.py `OwnerDecisionRefusalError`
  rename + E501; regime_observation_source.py E501) — they are inside the stash.

## Phase 2 — R6.1 implementation (status: FOUNDATION DONE 23:40; E/G/J/K/L open)

Foundation patch applied (`scratchpad/patch_r61_foundation.py`), plus fixes: store names; regime_store
(reuse-by-reproduction fits with `return_reuse`; verified owner evidence for FEATURE_ELIGIBLE+ via
`run_scope`); panel block registration (`with_registered_block`, `IFVG_CONTEXT_BAR_PANEL_V1`, `PRE_R6_1_*`,
`BP0_CONTEXT_BAR_PANEL`); grain/bundle join-key coherence (`assert_grain_bundle_coherent`); the panel session
state = session of the bar's FINAL instant (`SESSION_CLASSIFICATION_INSTANT_POLICY`); `run_regime_protocol`
runs under `threadpool_limits(1)` for bit-reproducible assignment tables (DECISION to record: execution-env
control, not a protocol field; golden fit id unchanged); owner-decision nested values are `ImmutableMap`
(`freeze_decision_values` / `plain_decision_values`, `DecisionValue` type). R6 panel tests migrated to BP0 +
`cbp_*` inputs (test_regime_service / test_regime_oos_assignment / test_regime_contracts / test_ifvg_pipeline_tab
fixture); test_regime_store (f) migrated to the synthetic owner decision; R6.1 registration test appended to
test_feature_blocks.py. Suites green at 23:40: regime suites 52 passed; pipeline-tab regime AppTests 8 passed;
owner/store/identities 50 passed; panel/blocks/oos/service/… 144+ passed.

Forks launched 23:45 (briefs: `scratchpad/FORK_BRIEFS.md`): J (CatBoost bundle rung + D13), G1 (stratification
contracts/gate/sources/descriptive classes/service), G2 (fold-local features + controlled regime study + cohort
model + block activation D9), G3 (D15 prop-event sidecar). Main agent: E (`ml/regime_study.py`, pipeline
S05–S15 regime path, fixture, `test_pipeline_regime.py`), then UI/CLI, K, L, adversarial round, evidence, commit.


| WS | Plan §6 | Module(s) | State at 22:37 |
|---|---|---|---|
| H | stability/transitions | `ml/regime_diagnostics.py` (+479), `regime_contracts.py`, `regime_service.py`, `test_regime_service.py` (+7 tests) | present; suites green (68 passed across the R6.1 suites present) |
| A | panel contract + materializer | `features/context_bar_panel_contract.py`, `context_bar_panel_materializer.py`, `arrow_tables.py`, fixture `synthetic_context_panel.py`, `test_context_bar_panel.py` | modules present; **block registration `IFVG_CONTEXT_BAR_PANEL_V1` / `with_registered_block` / bundle `BP0_CONTEXT_BAR_PANEL` NOT yet in `feature_blocks.py` / `feature_bundles.py`** (deliberately deferred until after the R5B.1 commit) -> 3 failed + 7 errors in `test_context_bar_panel.py` for exactly that reason |
| B | fold schedules / fold-set artifact | `ifvg/fold_schedules.py`, `ml/fold_set_artifact.py`, `test_fold_schedules.py`; delegation in `supervised_ladder.py` / `controlled_feature_study.py` | present, green |
| C | OOS assignment + PIT + adequacy preview | `ml/regime_oos_assignment.py`, `ml/regime_sample_adequacy.py` + tests | present, green |
| D | verified observation seam + executor | `bundle_feature_view.py` (save/load/verify, `bundle_categorical_features`, `regime_fold_feature_artifact_id`), `ml/regime_observation_source.py`, `ml/regime_executor.py` | modules present, **ruff E501 x2 in `regime_observation_source.py`; no tests yet** (`test_regime_observation_source.py`, fixture `synthetic_observation_source.py` missing) |
| F | owner-decision evidence | `search/owner_decisions.py` (570 lines) | present, **ruff N818 + E501 x4; no `test_owner_decisions.py`; `regime_store.persist_regime_promotion` not yet updated; store name `owner_decisions` not yet in `store.py`** |
| E | pipeline integration + UI + CLI | `ml/regime_study.py`, `search/pipeline.py`, `scripts/ifvg_pipeline_tab.py`, `ifvg_regime_panels.py`, `study_providers.py`, `scripts/ifvg_regime_promotion.py`, pipeline fixture | **not started** |
| G | stratification (5 classes), fold features, cohort model, prop sidecar (D15) | `ml/regime_stratified_contracts.py`, `regime_stratification_gate.py`, `regime_assignment_sources.py`, `regime_fold_features.py`, `regime_stratified_{strategy,prop,frontier}.py`, `regime_controlled_study.py`, `regime_cohort_model.py`, `regime_stratification_service.py`, propsim event-detail | **not started** |
| J | CatBoost bundle rung + comparison rows | `ml/catboost_bundle_model.py`, `model_protocols.py`, `supervised_ladder.py` dispatch | **not started** |
| K | evidence hygiene | hermetic `delenv` tests, `verify_browser_manifest.py`, `r61_smoke_app.py`, MANIFEST v2, patches | **not started** |
| L | docs | `R6.1/stage_shared_docs.py`, D-047, DECISIONS_TAKEN, DEVIATIONS, ACCESS_SAFETY | **not started** |

Store names still to add (`search/store.py`): `context_bar_panels`, `fold_schedules`,
`fold_sets`, `regime_oos_assignments`, `regime_fold_features`, `owner_decisions`,
`regime_stratified_reports`, `bundle_feature_views` — check which already exist before adding.


## Phase 2 progress log

- 00:20 — E landed: `ml/regime_study.py` (RegimeStudyRequest, readiness, verify_frozen_authority, S10 decisions),
  `search/pipeline_regime.py` (S05–S15 regime bodies), `ml/regime_report_stage.py` (S14 reports over G1's service;
  prior-attempt report recovery for reused children), pipeline.py wired (payload field `regime_study`, validator,
  readiness with store_root/run_scope, wiring `context_bar_source`, context.regime, S04 chart ids, S05/S06/S08/
  S09 (ladder conditional)/S10/S14/S15), fixture shapes `build_pipeline_fixture(regime_study=...)`,
  `tests/agents/ifvg_search/test_pipeline_regime.py` (11 tests incl. §9.1 model-bearing binding + S14 zero fitting).
  Prop simulation ids reach S14 via `make_prop_simulator(on_simulation_persisted=)` + the S12/S13
  `account_simulations.json` sidecar (attempt-invariant; prior-attempt recovery).
- Forks done: G3 (D15 event detail — propsim `event_detail.py`, charter policy field; ids move per §7),
  G1 (stratification contracts/gate/sources/strategy/prop/frontier/service + 13 tests). Hub changes applied:
  identities enumeration (+fold_schedules, context_bar_panel_materializer, fold_set_artifact, regime_oos_assignment,
  regime_stratified_contracts, owner_decisions), test_identities names, stamps (decision 30 family) single-sourced.
- Forks still running: J (CatBoost bundle rung + D13), G2 (fold features / controlled regime study / cohort model /
  block activation). After they land: wire S09b/S09c (`ml/regime_supervised_stage.py` shim expected by
  `pipeline_regime._s09_supervised`), the supervised fixture shapes E2E, test_pipeline_regime supervised tests.
- Deviations to record (DEV-R6.1-*): S07 stays a plan prerequisite of S08/S09 (frozen STAGE_DEPENDENCIES) so
  descriptive studies carry derived labels unused by S09a (the label-free builder serves S08 when labels are absent);
  `threadpool_limits(1)` in run_regime_protocol; session-of-final-instant policy; RegimeStatusRefusalError /
  OwnerDecisionRefusalError names carry the Error suffix (ruff N818); 55-day regime fixture under the 3-day allowlist.
- 01:10 — J landed (comparison_rows.py, catboost_bundle_model.py, ladder/study/readiness wiring); G2 landed
  (regime_fold_features.py, regime_block_activation.py, regime_controlled_study.py, regime_cohort_model.py + tests);
  hub changes applied (B7_CORE_REGIME, store names, identity enumeration + names, stamps). `ml/regime_supervised_stage.py`
  (S09b/S09c shim) written; S02 verified reuse BY REPRODUCTION of reused children when stratified reports are requested
  (metrics must equal the persisted costed evaluation); S09 ladder calls bind the S08 schedule id + S07 label id (D13).
  test_pipeline_regime.py = 12 tests green incl. the model-bearing two-pass E2E (S09b/S09c/S10 frozen authority/S14).
  Hermetic delenv added to the two provider tests (K). Fork U (UI + CLI) running.
- NEXT: K (r61_smoke_app.py, verify_browser_manifest.py, MANIFEST v2, full-suite runs x2, patch), L (docs:
  R6.1/stage_shared_docs.py ARCH/README/YAML transforms, docs/DECISIONS.md D-047, DECISIONS_TAKEN #76+, DEVIATIONS,
  ACCESS_SAFETY, FILES_TOUCHED, TEST_RESULTS, GATE_SUMMARY, DRAFT_OWNER_DECISION_PROPOSALS x3), adversarial round
  (2 read-only reviewers as subagents), fixes, commit R6.1, smoke + manifest validate.
- 00:40 — Fork U landed (UI + CLI; 38 AppTests; CLI tests). Non-UI suites 748 passed. ruff + diff-check clean.
  Docs drafted: DEVIATIONS.md (DEV-R6.1-1..10), PRE_R6_1_BASELINE.md, FILES_TOUCHED.md, ACCESS_SAFETY_EVIDENCE.md,
  DECISIONS_TAKEN #76-#89, docs/DECISIONS.md D-047, R6.1/stage_shared_docs.py (dry-run ok), DRAFT_OWNER_DECISION_PROPOSALS
  (3 drafts), verify_browser_manifest.py (+ validator test), r61_smoke_app.py (harness; ?run=candidate|panel|supervised).
  Midpoint full suite launched (background -> _midpoint_pytest.txt). Adversarial reviewers launched (general-purpose,
  read-only): reports land in R6.1/_review_contract.md and R6.1/_review_safety.md.
- NEXT after reviews: merge reports into ADVERSARIAL_REVIEW.md; fix; ADVERSARIAL_REVIEW_RESOLUTION.md; final full suite x2
  (as-is + keys cleared -> _final_pytest*.txt); ruff/diff-check raw; stage docs (stage_shared_docs.py); path-scoped
  `git add` of the R6.1 file list (FILES_TOUCHED.md) + docs/DECISIONS.md; commit; --apply-worktree; surviving diff;
  R6.1.patch + sha256; smoke (streamlit headless, log captured) + browser screenshots + MANIFEST v2 + validate;
  TEST_RESULTS.md; GATE_SUMMARY.md; update CLAUDE.md? (no — not in scope).

## Reset 3 (2026-08-29 ~00:45) — resume record

State at reset: midpoint full suite DONE (`_midpoint_pytest.txt`: 1896 passed, 0 failed, 16:45);
both adversarial reports DONE (`_review_contract.md`: F1–F4 major, F5–F16 minor;
`_review_safety.md`: S1–S2 major, S3–S6 medium, S7–S12 minor). Nothing fixed yet. Scratchpad from
the previous session is gone (new session dir): FORK_BRIEFS.md / probe scripts are not needed.

Fix round plan (4 forks, disjoint file ownership; Edit-only on shared files):
- Fork A1 "pipeline seam": F1, S1, F13, F7, F14, F4, F11, S7, S4 — owns regime_executor, regime_oos_assignment,
  pipeline_regime, pipeline.py, regime_study, regime_fold_features (F11 constant), regime_supervised_stage,
  regime_report_stage (panel-assigner region only), pipeline_fixture, test_pipeline_regime, test_regime_oos_assignment,
  ifvg_pipeline_tab (S7), test_ifvg_pipeline_tab.
- Fork A2 "stratification": F3, F6, F9, F12(UI half) — owns regime_stratification_service, regime_stratified_*,
  regime_report_stage (StratificationInputs / build_reports call region), test_regime_stratification,
  ifvg_regime_panels, study_status, study_providers. Threads `run_scope` into `resolve_report_gate(run_scope=)`.
- Fork B "governance": S6 (first: gate gains kw-only `run_scope`), S2/F8, F2, S3, S5, F10+S11, S10 — owns
  owner_decisions, regime_store, regime_stratification_gate, regime_contracts, ifvg_regime_promotion.py,
  test_owner_decisions, test_regime_store, test_ifvg_regime_promotion_cli, NEW test_regime_stratification_gate.py.
  No new store names (hash-chained SUPERSESSIONS log instead).
- Fork C "D15 writer + hygiene": S12/F16(writer), F15, F5, S9 — owns propsim/event_detail, search_bridge,
  search/store.py (producer/streaming support), test_account_event_detail, regime_controlled_study,
  regime_cohort_model, test_regime_supervised_studies, regime_service (S9 lazy import only).
- Main agent after forks: F5 name mapping / F12 / F16 / S8 doc entries; ADVERSARIAL_REVIEW.md (verbatim merge);
  ADVERSARIAL_REVIEW_RESOLUTION.md; DEVIATIONS/DECISIONS_TAKEN/ACCESS_SAFETY updates; final suites x2; ruff/diff-check;
  stage docs; path-scoped commit; --apply-worktree; patch; smoke + MANIFEST v2; TEST_RESULTS; GATE_SUMMARY.

### 01:05 — fix-round forks KILLED mid-edit by the session rate limit (HTTP 429, "session limit resets 2:20am CT")

All four forks terminated ~10 minutes in (A1 at "executor rewrite (F1)"; B at "replace the lock/append/load/persist
block" of owner_decisions.py; A2 at "the contracts edits"; C with no recorded progress). Owner instruction at that
point: revert the session priority to normal, report, and DO NOT continue the plan yet.

Tree state (diagnostic, nothing fixed): 14 files carry fork edits (mtime > this file's 00:49:59 append):
regime_oos_assignment.py, regime_stratification_gate.py, regime_executor.py, ml_fixtures/synthetic_observation_source.py,
regime_stratified_contracts.py, test_regime_observation_source.py, regime_contracts.py, test_regime_oos_assignment.py,
search/pipeline.py, regime_store.py, search/pipeline_regime.py, pipeline_fixture.py, scripts/ifvg_regime_promotion.py,
search/owner_decisions.py. All compile and import. Targeted suites over the touched lane (owner_decisions, regime_store,
regime_oos_assignment, regime_observation_source, promotion CLI, regime_stratification, regime_contracts,
pipeline_regime, pipeline_run): **12 failed, 96 passed** — half-applied fixes: the new `decided_at >= approved_at`
refusal (F10/S11) vs the old "not yet effective" expectation; the new D6 refusal at persistence (F2) vs
`test_regime_store.py` test (e) still persisting STRATIFICATION_READY over failing coverage; the CLI scope test;
the S6 gate now calling `assert_owner_decision_authorizes` (2 gate tests + the service test); 6 pipeline-regime E2E
failures with S14 `failed` (gate/executor half-rewrite). Docs already done this session: ADVERSARIAL_REVIEW.md (verbatim
merge), ACCESS_SAFETY_EVIDENCE.md + DEVIATIONS.md header corrections (S8/F16). Fork C's mandates (S12/F16 writer, F15,
F5, S9) are untouched. Resume = re-brief the four workstreams (sequentially or fewer in parallel to stay under the
session budget) from the reviews + this record; each must first read the half-applied state of its files.

- 01:12 — owner raised the session priority; the four forks RELAUNCHED with "reconcile the half-applied state first" briefs (A1/A2/B/C).
- 01:27 — Fork B DONE (S6, S2/F8, F2, S3 [namespace guard; `run_scope` NOT in the decision payload], S5, F10+S11, S10 all FIXED;
  owned suites 53 passed; ruff clean; no identity moved; new test file `test_regime_stratification_gate.py`; hash-chained
  `SUPERSESSIONS.jsonl` + `SUPERSESSIONS.head`; `MODEL_FEATURE_PROMOTION_REFUSAL` constant; `--decided-at` removed).
  Deviations to record: run_scope-not-in-payload; decided_at>=approved_at transitive; log+head deletion undetectable
  (DEV-R6-8); MODEL_FEATURE representable-but-unpersistable; CLI stratification_ready = reuse of S10. A1/A2/C still running.
- 01:33 — Fork C DONE (S12/F16 writer: store sidecar-PRODUCER protocol `ProducedSidecar`/`SidecarProducer`/`write_produced_sidecar`,
  `save_envelope_immutable(sidecar_producer=)`, streaming one-path-block-at-a-time D15 writer with byte-identical output;
  F15 `FrozenTree`/`FrozenJson` deep-immutable summaries, study ids unchanged (golden probe); F5 test renames + bundle ladder
  on the controlled study; S9 lazy threadpoolctl import + `THREADPOOLCTL_MISSING_MESSAGE`). No identity moved; ruff clean.
  Deviations: threadpoolctl undeclared (lazy import); writer memory = one block + 32 B/row id index; §9.1 name mapping.
  A1/A2 still running.
- 01:37 — Fork A2 DONE (F3: `account_event_regime_summary.parquet` sidecar on stratified_prop reports, schema
  `ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA` v1, budget `account_event_regime_summary_budget_v1` 5,000,000 rows / 268,435,456 B,
  four envelope extras `account_event_regime_summary_{sha256,bytes,rows,schema_hash}`, `EventRegimeSummaryBudgetError`
  before publication; F6 `StratificationInputs.delivered_by` verified by exact-id reload → `StratificationOutcome.delivered_by`;
  F9 D15 precedence + own trading_day; F12 no UI change needed; `run_scope` threaded). Stratified-report ids move.
  INTEGRATION REQUEST for me: `scripts/ifvg_pipeline_tab.py` Monitor regime expander (~1441-1465) must render
  `delivered_by` rows and keep the empty state for `refusals` only. A1 still running (panel_supervised fixture in progress).
- 02:55 — Fork A1 DONE (F1 verified `candidate_as_of_source` ref + `CANDIDATE_AS_OF_SOURCE_REF_PATTERN`; S1/F13 S05 chart
  binding + `panel_chart_lowest_core_replay_id_v1`; F7 S08 schedule from the candidate view's observed days; F14 persisted-only
  panel assigner [trade tables = S02 this-run tables → DEV]; F4 zero-fitting parametrized over 3 shapes + panel supervised
  E2E + S07 refusal test; F11 one hard-id vocabulary `fit_local_categorical_v1`; S7 UI readiness with store_root/run_scope;
  S4 S02 adopts re-derived tables only by verified reproduction). test_pipeline_regime 19 passed; tab 60 passed; ruff clean.
  Fixture panel bars now plant regimes per day (panel ids re-mint). ALL FORKS DONE. Next: A2's tab integration
  (delivered_by rows), docs for A1, RESOLUTION.md, ruff/diff-check, full suite x2, commit, patch, smoke, results.
- 03:15 — ALL fork docs applied (DEVIATIONS DEV-3/7/8 amended + DEV-11..18; DECISIONS_TAKEN #82/#85/#86/#88 amended + #90-#100; ACCESS_SAFETY; FILES_TOUCHED; ADVERSARIAL_REVIEW_RESOLUTION.md written; stage_shared_docs.py + D-047 updated). Tab integration (delivered_by rows) done. `_ruff_and_diffcheck.txt` clean (one EOF blank line fixed). Full suites RUNNING in the background (as-is → keys cleared; started 08:11Z) → `_final_pytest*.txt`. NEXT: stage docs, path-scoped commit, --apply-worktree + surviving diff, patch + sha256, smoke + MANIFEST v2 + validate, drafts re-mint check, TEST_RESULTS, GATE_SUMMARY.
- 03:50 — FULL SUITES GREEN: as-is 1931 passed / 0 failed (17:57); keys cleared 1931 passed / 0 failed (18:09) (`_final_pytest*.txt`). Commit sequence starting: stage docs → path-scoped add → commit.
- 03:55 — COMMIT `6c0b60a` (tree 70e972a0…, parent f3f9ac2): 87 files, +25,783/−566; not pushed, not merged. R6.1.patch + sha256 written. Next: --apply-worktree + surviving diff, smoke (streamlit :8611 headless, log → browser-smoke/_smoke_server.log), screenshots, MANIFEST v2, validate, drafts, TEST_RESULTS, GATE_SUMMARY.
- 04:15 — R6.1 CLOSED. Post-commit: --apply-worktree replayed; surviving diff IDENTICAL to R1/PRE_EXISTING_DIFF.patch
  modulo index lines; R6.1.patch (1,310,228 B, sha256 6a166afe…); browser smoke served candidate/panel/supervised from
  %TEMP%/ifvg_r61_smoke/d8dae9697cf5b3bb (26 marker lines, 0 tracebacks, 0 deprecations); 25 screenshots; MANIFEST.json v2
  built by build_browser_manifest.py and VALIDATED (OK, commit 6c0b60a, tree digest verified); drafts candidate + 300s
  regenerated over the smoke store (900s annotated); TEST_RESULTS.md, GATE_SUMMARY.md, FILES_TOUCHED.md reconciled
  (87 files: 46 A / 41 M). Not pushed, not merged. acceptance_status: transitively_blocked_by_R1.
