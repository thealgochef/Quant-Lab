# UI-2 — Completion Report

**Feature:** `ifvg_prop_robust_config_search_v1` — IFVG Lab UI/UX redesign, **Phase 2 of 6**.
**Authority:** the owner's instruction of 2026-09-04 ("UI-1 phase is completed. Please resume
with the next phase in the UI-UX-REDESIGN") over `../IMPLEMENTATION_PLAN.md` (revision 2; §9
"Phase 2 — Verification Center, conditional New Study, drafts and review"), subordinate to
`../../../FINAL-IMPLEMENTATION-PLAN-DOCS/`.
**Baseline:** `feature/ifvg-prop-robust-config-search-v1` @ `f1827e8` (UI-1 release head).
**Commit:** `eb85d09` (`eb85d09da3ab01aee91848962366ddb5c4ab2c9d`) (release-scoped; not pushed, not merged).

```text
implementation_status: complete (Phase 2 of the UI/UX redesign)
ui_phase: UI-2 of UI-1 … UI-6
ui_acceptance: OPEN (browser / keyboard / viewport evidence is mandatory at UI-6; never waived)
formal_acceptance_status: transitively_blocked_by_R1
real_verification_run_completed: false
owner_actions_performed: none (no signing, no seed production, no real verification run)
```

## 1. What Phase 2 required (plan §9) → what landed

| Plan requirement | Landed as | Proof |
|---|---|---|
| The complete Verification Center flow: logical-day shortlist → seed-production packet / ref / job / verified seed → final verification packet / ref → review / run → filtered monitor; exact command / packet path / refresh semantics where signing or execution is external | `scripts/ifvg_verification_center.py` (six sections) over the `study_providers` read models; the `Verify Implementation` route delegates to it | `test_ifvg_verification_center.py` (6), `test_study_providers_ui2.py` (10) |
| Verification Center never renders research gates, prop / risk / model controls or Publish | no such widget, heading or button; no spawn seam (scanned) | `test_verification_center_never_renders_research_steps_or_publish`, `test_process_launch_exists_only_in_the_designated_seams` |
| Logical trading days and physical partitions displayed separately | two tables (session bounds + coverage; td−1 / td partitions with kind and sha) | `test_logical_days_and_physical_partitions_are_displayed_separately`, `test_shortlist_loads_typed_and_separates_logical_days_from_partitions` |
| Seed production cannot launch without a validated `SeedProductionAuthorizationRef` | the seed job command is shown ONLY for a `verified` / `verified_envelope` authorization; `not_recorded` / `not_found` / mismatch states show the registration command only | `test_seed_production_cannot_launch_without_a_validated_authorization`, `test_seed_authorization_state_is_typed_by_exact_id` |
| Final verification packet cannot exist before a verified profile-matching seed | the packet action exists only when the exact-id, profile-bound loader verifies a seed continuous with the window | `test_final_verification_packet_cannot_exist_before_a_verified_seed`, `test_seed_snapshot_and_receipt_states_are_typed` |
| Real verification cannot launch without a validated final `VerificationAuthorizationRef` | the completed reference validates typed; the charter freezes from the SIGNED reference; the run is registered through `validate_verification_run`; the bounded-run command appears only after the §6.1 preflight passes | `test_real_verification_cannot_launch_without_a_validated_final_ref`, `test_signed_ref_validation_is_typed`, `test_register_verification_run_validates_before_persisting_and_preflight_passes` |
| Monitor lists only stages in the resolved seed / verification plan | the seed stage + `verification_stage_rows` (in-plan only); no Publish | `test_monitor_lists_only_the_planned_stages`, `test_verification_stage_rows_list_only_the_planned_stages` |
| Conditional flows skip irrelevant steps with visible reasons | `presentation/flows.py`; the goal card lists skipped steps with reasons; skipped payloads contribute nothing | `test_presentation_flows.py` (8), `test_conditional_flow_skips_steps_with_visible_reasons`, `test_wizard_shell_and_exact_step_restore` |
| A selected prop objective blocks rather than disappearing | the contract step stays in the flow; `validate_prop_step` blocks Next with the contract workflow; the objective echoed unchanged | `test_prop_objective_blocks_instead_of_disappearing`, `test_strategy_goals_skip_prop_steps_unless_a_prop_objective_is_selected` |
| Session-only draft writes no file until Save / valid Next; purpose annotation persists | session stash (`SESSION_DRAFT_KEY`); `_persist_now` is the one persistence seam; the annotation rides the session draft and the first save | `test_session_only_draft_writes_no_file_until_save_or_valid_next`, `test_first_valid_next_persists_a_session_draft`, `test_start_cards_derive_purpose_namespace_and_create_annotated_drafts` |
| Archive / restore / delete rules and the one-time bulk archive preserve evidence | `study_drafts` schema 2 lifecycle; History's archived view; the typed delete for never-frozen drafts only; frozen provenance refused; bulk archive never deletes | `test_study_drafts.py` (13), `test_history_archive_restore_and_typed_delete`, `test_history_bulk_archives_empty_untitled_drafts_and_deletes_nothing`, `test_owner_q2_delete_control_is_typed_archived_and_never_frozen` |
| Review opens Unreviewed, persists nothing without Save, owner label mappings, v1 rows valid | `presentation/review_vocabulary.py`; the shared review form (per-case keys; explicit Save Review; `Unsaved` / `Saved`); additive `not_applicable` | `test_ifvg_verifier_review.py` (4), `test_presentation_review_vocabulary.py` (2), `test_not_applicable_is_accepted_additively_and_v1_rows_stay_valid`, `test_review_ledger_write_uses_setup_kwargs` |
| `discard_draft` removed / hard-deprecated | retired: raises, deletes nothing; History no longer calls it | `test_discard_draft_is_retired_and_deletes_nothing` |
| Docs in the same change | contract §§3.2, 7, 11, 14.4, 27, 29, 31, 35, 36; `docs/DECISIONS.md` D-054; `DECISIONS_TAKEN.md` #137; the three shared docs via `stage_shared_docs.py` | this folder |

Owner Q1–Q4 as applied: Q1 (purpose derives the namespace) unchanged from UI-1 ✓; Q2 (drafts:
session-only, archive first, delete never-frozen only with the exact typed name, bulk archive of the
empty untitled files — never silently deleted) ✓; Q3 (ledger keys preserved; `Unreviewed` unsaved
only; the five labels; `not_applicable` additive under the current validator — no v2 schema) ✓;
Q4 (Evaluate / Compare as two tasks; FSM search separate; verification never a research result)
carried through the flows ✓.

## 2. Gates

| Gate | Command / scope | Result |
|---|---|---|
| Red-first proofs | `_red_A_pure.txt`, `_red_E_providers.txt`, `_red_G_wizard.txt`, `_red_HI_history_review.txt` | red as expected before implementation |
| Targeted suites (33 files) | `_targeted_pytest.txt` | **373 passed**, exit 0 (+ the 56-test addendum over the final tree after the vocabulary fix) |
| Complete suite, environment as-is, ONE invocation | `python -m pytest -q -p no:cacheprovider --junitxml=junit_full.xml` (`run_final_gates.sh` → `_final_pytest.txt`) | **2,416 passed, 0 failed** in 1,782.47 s (0:29:42), exit 0, over the FINAL tree (09:36Z → 10:06Z; both provider keys present). The first run over the pre-fix tree found one real omission (the seven UI-2 states lacked a `UiStatus` mapping) and was superseded after the fix — `DEVIATIONS.md`, `TEST_RESULTS.md` |
| Warnings as errors | pyproject `filterwarnings = error` (the one narrowly scoped third-party rule unchanged) | PASS — zero warnings-summary sections in every log |
| Ruff | `python -m ruff check src tests scripts` (`_ruff_and_diffcheck.txt`) | `All checks passed!` (exit 0) |
| `git diff --check` | over `src scripts tests docs QL-FSM-PROP-SEARCH-DASHBOARD` | clean (exit 0) |
| Capacity benchmark | not applicable — no benchmarked byte changed | n/a |
| Browser / viewport / keyboard evidence | not captured in UI-2 (mandatory at UI-6; the plan leaves UI acceptance OPEN until then) | OPEN |

## 3. Preservation

| Constraint | Proof |
|---|---|
| Strategy-Core unchanged | not touched |
| Fixed M0–M3 lane and identities | no capture / experiment / reporting module touched |
| Backend contracts | consumed read-only (`VerificationWindowShortlist`, `SeedProductionAuthorization` / `Run`, `SeedSnapshot`, `VerificationAuthorizationRef`, `VerificationRunEnvelope`, `preflight_bounded_verification`, the runner registry); the only backend-side edits are the seed CLI's operator seams (`register-authorization`, `--receipt-out`) and the providers' read models — no contract, identity or validator changed |
| No new spawn site | the center contains no `subprocess` use; the scan admits the two existing seams only |
| Immutable artifacts / `data/` | `find data -type f -newer <UI-1 COMPLETION_REPORT>` → none; every test store lives under `tmp_path` |
| No launch, seed, signing, verification run | none executed; the center's tests use synthetic-provenance authorizations, a hand-built receipt and test-fixture references |
| User-owned hunks | the four docs and two data deletions remain the pre-existing worktree changes (147 insertions / 4 deletions over the four docs); none staged or committed |

## 4. What UI-3 starts from

The Verification Center is complete for the owner's traversal once real packets exist; the goal
card, flows and drafts are in place; the review form is explicit. UI-3 lands the metric
registry, roll-ups, helper / glossary registries and `detail_levels`, then Context Research and
Results presentation and the MBP-1 / regime summary-first panels. The phase radio, the R4
disclosure labels and the verifier chart remain until UI-6 / UI-4.

## 5. Packaging

| Item | Value |
|---|---|
| Commit | eb85d09da3ab01aee91848962366ddb5c4ab2c9d on `feature/ifvg-prop-robust-config-search-v1`; parent `f1827e8`; 36 files (+6,945 / −458); neither pushed nor merged |
| Patch | `UI-2.patch` (`git format-patch -1`; 401,259 B) — sha256 `7022f61d85fc707aa153aad7ac22208b877ffab35e1cc09b5ded0552c73463fa` (`UI-2.patch.sha256`, verified with `sha256sum -c`) |
| Bundle | `UI-2.bundle` (`f1827e8..feature/ifvg-prop-robust-config-search-v1`; requires `f1827e8`; 115,576 B; `git bundle verify` OK) — sha256 `539f50f8dfc4de19a7d328150e785b102e2712061115df798116d7c8ba5c4cff` (`UI-2.bundle.sha256`, verified) |
| Evidence folder | this folder (see `FILES_TOUCHED.md`) |
