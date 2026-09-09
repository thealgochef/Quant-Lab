# UI-1 — Completion Report

**Feature:** `ifvg_prop_robust_config_search_v1` — IFVG Lab UI/UX redesign, **Phase 1 of 6**.
**Authority:** the owner's instruction of 2026-09-04 ("continue with the development plan") over
`../IMPLEMENTATION_PLAN.md` (revision 2; §9 "Phase 1 — Semantic correctness, backend-contract
handshake and truthfulness"), subordinate to `../../../FINAL-IMPLEMENTATION-PLAN-DOCS/`.
**Baseline:** `feature/ifvg-prop-robust-config-search-v1` @ `ffcf39b` (HARDENING-BACKEND-FIX.1
release head; `backend_dev_complete_for_ui = true`).
**Commit:** `f1827e8` (`f1827e81cf789286b748c0a1728e866f5944c139`) (release-scoped; not pushed, not merged).

```text
implementation_status: complete (Phase 1 of the UI/UX redesign)
ui_phase: UI-1 of UI-1 … UI-6
ui_acceptance: OPEN (browser / keyboard / viewport evidence is mandatory at UI-6; never waived)
formal_acceptance_status: transitively_blocked_by_R1
real_verification_run_completed: false
owner_actions_performed: none (no signing, no seed production, no real verification run)
```

## 1. What Phase 1 required (plan §9) → what landed

| Plan requirement | Landed as | Proof |
|---|---|---|
| One source of truth for purpose / scope / namespace (F-01, F-13, Q1) | `presentation/run_purpose.py`; the namespace radio removed; `Start` / `Verify Implementation` routes; the goal card; roots derived from the purpose; the verified `store_namespace_id` displayed read-only | `test_presentation_run_purpose.py`, `test_ifvg_study_tab.py::test_namespace_radio_is_gone`, `::test_start_cards_derive_purpose_namespace_and_create_annotated_drafts`, wizard `test_goal_card_shows_purpose_scope_namespace_and_evidence` |
| `RunPurposeAnnotation` round-trips; ambiguous legacy drafts are `purpose_unresolved` and cannot freeze | additive draft field + catalog `purpose` event; `resolve_draft_purpose` | `test_purpose_annotation_round_trips_and_is_non_semantic`, `test_ambiguous_legacy_drafts_become_purpose_unresolved`, wizard `test_ambiguous_legacy_purpose_blocks_freeze_until_confirmed`, `test_legacy_verification_draft_derives_its_purpose_unambiguously` |
| Real `verification_5d` refuses the synthetic marker; fully synthetic fixtures require it | `EvidenceClass`; authorization by the actual path in `_assemble_charter`; `resolve_purpose` refuses a synthetic research purpose | `test_synthetic_evidence_is_confined_to_implementation_verification`, satisfiability rule `synthetic_fixture_confined_to_verification`, `test_real_verification_over_the_exact_baseline_passes` |
| Typed authorization readiness (missing / stale / superseded / wrong namespace / head / profile / source / ready) | `study_providers.verification_authorization_readiness`, `owner_authorization_readiness`, `AUTHORIZATION_READINESS_STATUSES` | `test_study_providers_ui1.py::test_ui_authorization_states_match_backend_contracts`, `::test_owner_authorization_readiness_names_missing_decisions` |
| Namespace radio absent; local path never defines authority | `workspace_roots` / `roots_for_purpose`; `namespace_state_for_store` from the verified envelope only | `test_ui_uses_verified_store_namespace_contract`, `test_namespace_id_comes_only_from_the_verified_envelope` |
| Contradictory drafts fail before freeze (zero-axis search, baseline vs itself, Compare ≠ 1 challenger, prop objective without a verified contract, one-firm Universal, verification with research gates) | `presentation/charter_satisfiability.py` + `validate_charter` | `test_contradictory_drafts_fail_before_freeze[…]` (12 cases), `test_validate_charter_refuses_degenerate_charters`, `test_real_charter_prop_objective_requires_a_firm_contract_and_is_never_rewritten`, wizard `test_contradictory_drafts_fail_before_freeze[…]` |
| A challenger may carry multiple registered differences; every difference summarized | `challenger_configurations`; `summarize_challenger_differences`; the Review card | `test_one_challenger_may_carry_multiple_registered_differences` |
| Launch refuses an unregistered runner before spawn; success only after state exists | `resolve_registered_runner_entry` before `_spawn_*`; `_wait_for_search_state` / `_wait_for_pipeline_state`; `runner_unavailable` / `launch_not_started` | wizard `test_launch_refuses_unregistered_runner_before_spawn`, `test_launch_reports_started_only_after_state_exists`; pipeline `test_pipeline_launch_refuses_unregistered_runner_and_silent_worker` |
| Publication gates / activation bound to the same namespace, state hash and authorization | `run_publication_gates` records `gates_store_namespace_id` + `gates_state_sha256`; `activate_pipeline_result(expected_store_namespace_id)`; the UI cache key | `test_publish_gate_record_is_namespace_and_state_bound`, `test_publish_runs_gates_and_refuses_verification_activation` |
| Worker values above one are not offered; backend refusal rendered truthfully | no sliders / inputs; `validate_validation_step` refuses ≠ 1; `--max-workers 1`; Monitor shows the receipt's `execution_mode` / `effective_workers` | `test_worker_control_is_absent_or_fixed_to_one`, `test_validation_step_enforces_sequential_v1_and_development_dates`, `test_configure_renders_capability_scoped_fields`, `test_resume_retry_offers_operational_clone_and_new_attempt` |
| Minimize metrics use reversed colorscales | `direction_colorscale` (`RdYlGn_r` + "lower is better") | `test_heatmap_colorscale_follows_the_registered_direction[…]`, `test_firm_matrix_colorscale_follows_the_registered_direction[…]` |
| Reconciliation / access success derives only from evaluated evidence | `build_context_reconciliation_audit_report` folds evaluated flags (`None` when none); adapter statuses; the lab banner | `test_reconciliation_report_passed_is_derived_from_the_pair_reports`, `test_reconcile_banner_derives_from_evaluated_gates`, `test_unknown_access_evidence_is_never_pass` |
| §6.7 states: no-runs / not-selected / not-applicable never `artifact_unavailable`; missing vs corrupt | `study_status` additive states; Active Runs / Results / pipeline renderers | `test_every_section31_state_is_registered`, `test_empty_states_never_use_artifact_unavailable_for_no_runs`, active-runs `test_empty_state_when_no_jobs_exist` |
| F-08 part: development dates by the backend logical-day contract; no double acknowledgement | frozen warmup prefix read-only; `development_evidence_day_error`; the acknowledgement typed once (Review; the pipeline Launch for mode-5 drafts) | wizard `test_development_dates_are_validated_field_by_field`, `test_full_scope_requires_the_exact_typed_confirmation_and_readiness` |
| Docs in the same change | `FRONTEND_UX_CONTRACT.md` §§3.2, 7, 8.1, 14, 15, 30, 31, 35, 36; `docs/DECISIONS.md` D-053; `DECISIONS_TAKEN.md` #136; `ARCHITECTURE.md` / `docs/README.md` / `docs/pipeline_state.yaml` lane transforms | this folder, `stage_shared_docs.py` |

Owner Q1–Q4 as applied: Q1 (purpose derives the namespace; the radio removed; `RunScope`
unchanged) ✓; Q2 (drafts) — UI-2; Q3 (review vocabulary) — UI-2; Q4 (Evaluate vs Compare as two
tasks; FSM search separate; verification never a research result) ✓ (the fifth research
question, the satisfiability rules and the Start cards).

## 2. Gates

| Gate | Command / scope | Result |
|---|---|---|
| Red-first proofs | `_red_A_presentation.txt` (the presentation package absent → 3 collection errors), `_red_DE.txt` (providers symbols absent → collection error) | red as expected before implementation |
| Targeted suites (27 files) | `_targeted_pytest.txt` | **359 passed**, exit 0 |
| Complete suite, environment as-is, ONE invocation | `python -m pytest -q -p no:cacheprovider --junitxml=junit_full.xml` (`run_final_gates.sh` → `_final_pytest.txt`) | **2,363 passed, 0 failed** in 1,693.82 s (0:28:13), exit 0, over the FINAL tree (07:02Z → 07:30Z; both provider keys present). A first run over a tree edited mid-run failed one AppTest as a harness artifact and was superseded (`DEVIATIONS.md`, `TEST_RESULTS.md`) |
| Warnings as errors | pyproject `filterwarnings = error` (the one narrowly scoped third-party rule unchanged); the suites above ran under it | PASS — zero warnings-summary sections in the complete-suite log and in every targeted log (warnings are errors) |
| Ruff | `python -m ruff check src tests scripts` (`_ruff_and_diffcheck.txt`) | `All checks passed!` (exit 0) |
| `git diff --check` | over `src scripts tests docs QL-FSM-PROP-SEARCH-DASHBOARD` | clean (exit 0) |
| Capacity benchmark | not applicable — no benchmarked byte changed (the UI lane and the presentation package only; the event-detail writer / stratified summary untouched) | n/a |
| Browser / viewport / keyboard evidence | not captured in UI-1 (mandatory at UI-6; the plan leaves UI acceptance OPEN until then) | OPEN |

## 3. Preservation

| Constraint | Proof |
|---|---|
| Strategy-Core unchanged | not touched (no file under the Strategy-Core checkout modified by this release) |
| Fixed M0–M3 lane and identities | `data_access.py`, the capture / experiment modules untouched; `context_reporting.py` / `context_report_adapters.py` gained additive report keys only (the run identity `context_run_identity` is not derived from the reconciliation report's `passed`); the engine suite passed |
| Backend contracts | consumed read-only (`StoreNamespaceEnvelope`, `SupersessionHeadWitness`, `VerificationAuthorizationRef`, `OwnerAuthorizationBundle`, the runner registry, `SUPPORTED_CHILD_WORKERS`); the only backend edits are the fail-closed `validate_charter` tightening, the additive catalog kind, and the additive publication binding in `search/pipeline.py` — all named by the plan's §8 table |
| Immutable artifacts / `data/` | `find data -type f -newer <FIX.1 COMPLETION_REPORT>` → none |
| No launch, seed, signing, verification run | none executed (AppTests monkeypatch every spawn seam; the honest-launch tests write a fake state file into tmp roots) |
| User-owned hunks | the four docs and two data deletions remain the pre-existing worktree changes; none staged or committed |

## 4. What UI-2 starts from

The Verification Center readiness surface (`Verify Implementation`) and the typed readiness
providers exist; UI-2 builds the fixture shortlist → seed-production packet / ref / job / verified
seed → final verification packet / ref → review / run → filtered monitor flow over them, the
goal-conditional flows (`presentation/flows.py`), session-only drafts with archive / restore /
typed delete and the bulk archive of the empty untitled drafts (owner Q2), and the explicit
reviewer verdicts (owner Q3). The phase radio, the R4 disclosure labels and the registry-style
MBP-1 / regime panels remain until UI-6 / UI-3.

## 5. Packaging

| Item | Value |
|---|---|
| Commit | f1827e81cf789286b748c0a1728e866f5944c139 on `feature/ifvg-prop-robust-config-search-v1`; parent `ffcf39b`; neither pushed nor merged |
| Patch | `UI-1.patch` (`git format-patch -1`; 396,026 B) — sha256 `2e72d1fe3ebadfbfeb3330a85ada0bbdaa2415f87e98ace378de16a8dd39a214` (`UI-1.patch.sha256`, verified with `sha256sum -c`) |
| Bundle | `UI-1.bundle` (`ffcf39b..feature/ifvg-prop-robust-config-search-v1`; requires `ffcf39b`; 114,717 B; `git bundle verify` OK) — sha256 `cfc495acc455c8ad5dd84bf7f715731d2e1dffae798cfe7a0d20da2f59d35aa1` (`UI-1.bundle.sha256`, verified) |
| Evidence folder | this folder (see `FILES_TOUCHED.md`) |
