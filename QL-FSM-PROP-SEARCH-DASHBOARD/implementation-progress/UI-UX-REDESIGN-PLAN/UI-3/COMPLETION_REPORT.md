# UI-3 — Completion Report

**Feature:** `ifvg_prop_robust_config_search_v1` — IFVG Lab UI/UX redesign, **Phase 3 of 6**.
**Authority:** the owner's instruction of 2026-09-04 ("continue with the next phase. UI-2 is
completed. Resume") over `../IMPLEMENTATION_PLAN.md` (revision 2; §9 "Phase 3 — Shared
metric/helper/status system; Context Research, Results, MBP-1 and regime panels"), subordinate
to `../../../FINAL-IMPLEMENTATION-PLAN-DOCS/`.
**Baseline:** `feature/ifvg-prop-robust-config-search-v1` @ `eb85d09` (UI-2 release head).
**Commit:** `75a4c95` (`75a4c951ab2fc489653fb917e9369f15d036ddca`) (release-scoped; not pushed, not merged).

```text
implementation_status: complete (Phase 3 of the UI/UX redesign)
ui_phase: UI-3 of UI-1 … UI-6
ui_acceptance: OPEN (browser / keyboard / viewport evidence is mandatory at UI-6; never waived)
formal_acceptance_status: transitively_blocked_by_R1
real_verification_run_completed: false
owner_actions_performed: none (no signing, no seed production, no real verification run)
```

## 1. What Phase 3 required (plan §9) → what landed

| Plan requirement | Landed as | Proof |
|---|---|---|
| Registry coverage of displayed metric keys | `presentation/metric_registry.py` — one `MetricSpec` per key (the Results pickers and explorer columns, the wizard gate rows via `gate_metric_key`, the ladder columns, the Context Research keys, the regime and MBP-1 keys, the verifier execution metrics, the Data & Audit report fields); directionality asserted against `OBJECTIVE_DIRECTIONS` | `test_registry_covers_every_displayed_key`, `test_every_spec_is_complete_and_directions_follow_the_charter_registry` |
| Metric status rules | gate PASS / FAIL by direction; boundaries strict; the 0.5 AUC line and the calibration targets informational; adequacy minima inconclusive below; limits and measured zeros pass / fail; policy zeros informational; intervals crossing zero inconclusive | `test_metric_status_rules[…]` (31 cases), `test_interval_crossing_zero_is_inconclusive` |
| No unknown evidence shown PASS | `None` / non-numeric / `evaluated=False` → UNAVAILABLE everywhere; the reconciliation banner is green only for an EVALUATED persisted pass (a legacy `passed: True` without evaluated gates stays the UNAVAILABLE warning); the Data-integrity roll-up card reads the evaluated gates, the measured counters and the observed-vs-limit figures, and an unmeasured optional figure is omitted rather than fabricated | `test_missing_or_unevaluated_evidence_is_never_pass`, `test_reconcile_banner_derives_from_evaluated_gates` (UI-1), `test_decision_summary_orders_the_four_rollups_and_never_greens_unknown_evidence`, `test_prop_rollup_is_unavailable_without_simulations_never_pass`, the two panel `…_never_pass` tests |
| Deterministic roll-ups | `presentation/rollups.py` — FAIL → BLOCKED → INCONCLUSIVE → WARNING → PASS → INFORMATIONAL → UNAVAILABLE | `test_rollup_rule_order_is_fail_blocked_inconclusive_warning_pass_informational`, `test_rollups_are_deterministic_and_carry_the_section_label` |
| Coverage and net R on separate scales | `build_coverage_figure` over the adapter's threshold rows (row 1 coverage, row 2 net R); the reliability diagonal named | `test_coverage_figure_uses_separate_axes_and_reliability_names_the_diagonal` |
| MBP-1 / regime summary fields resolve from the selected run; manual ids Advanced-only | summary-first panels; the inputs (same keys) instantiated into the Advanced diagnostics container; facts resolved from the loaded artifacts | `test_mbp1_panel_leads_with_a_summary_resolved_from_the_selected_run`, `test_regime_panel_leads_with_a_summary_resolved_from_the_selected_run` (+ the two `…_never_pass` tests; the eleven existing panel tests unchanged and green) |
| Proposed / ratified / offline / planned statuses distinct | `labels.AvailabilityKind` + `AVAILABILITY_CHIPS` (distinct glyph + word) | `test_availability_chips_are_distinct_by_glyph_and_word`, `test_availability_derives_from_the_registries_not_from_wording` |
| Helper / glossary coverage with explicit `HELP_EXEMPTIONS` | 128 help entries; sixteen live navigation exemptions; the sixteen-term glossary; help on every widget of every UI script | `test_every_widget_has_help_or_a_registered_exemption`, `test_help_ids_exist_and_exemptions_are_live`, `test_glossary_defines_the_listed_acronyms`, `test_help_exemptions_are_explicit_and_justified` |
| Accessible names for every collapsed / hidden label | the wizard's gate inputs `Resolved value for <gate>`, History's `New display name`, the pipeline phase radio with help | `test_collapsed_labels_carry_accessible_names_and_help` |
| Context Research presentation (F-07 full, F-09) | the decision summary first, the sample-adequacy card, registry cards with references, the named diagonal, separate axes, fold chips, intervals, top-10 importance with fold stability, compatibility reasons in words, raw JSON under Technical identity & audit | `test_render_result_leads_with_the_decision_summary[…]`, `test_run_history_explains_incompatibility_in_words`, the pure `test_ifvg_context_presentation` tests |
| Results presentation | the Selected configuration block (roll-ups + registry cards against the charter's resolved gates; the prop vector's worst firm), registry captions on the pickers, the column guide, the three detail levels | `test_selected_configuration_rollups_and_registry_metric_cards`, `test_explorer_column_guide_and_metric_captions`, `test_presentation_results` |
| `_ladder_frame` | AUC numeric (Float64) + `AUC reason` string column + `Model` label; the registry caption | `test_ladder_frame_is_arrow_safe_with_nullable_dtypes` (updated), `test_ladder_panel_carries_registry_definitions` |
| Detail levels on every major screen | `ifvg_ui_common.detail_levels`; `disclosure_level` delegates; Context Research and Results use it | `test_detail_levels_render_the_three_tiers_over_the_persisted_vocabulary` |
| Docs in the same change | contract §§5.1, 5.6 (new), 17, 30.4, 32.1, 35, 36; `docs/DECISIONS.md` D-055; `DECISIONS_TAKEN.md` #138; the three shared docs via `stage_shared_docs.py` | this folder |

Acceptance criteria of the plan's Phase 3: every consequential metric / control / status is
interpretable (registry cards, chips, help) ✓; no arbitrary thresholds (every reference is a
registered code source) ✓; no registry dump or exact-ID paste field dominates the normal
workflow (summary-first panels; Advanced diagnostics) ✓; MBP-1 remains offline-only and the
regime roles are honest (the `research_only_offline` chip; the model-bearing path blocked until
the owner-ratified FEATURE_ELIGIBLE decision) ✓.

## 2. Gates

| Gate | Command / scope | Result |
|---|---|---|
| Red-first proofs | `_red_A_D_pure.txt`, `_red_E_L_common_scans.txt`, `_red_FG_context.txt`, `_red_H_results.txt`, `_red_IJK_panels.txt` | red as expected before implementation |
| Regression over the help patch | `_regression_L_part1.txt` / `_part2.txt` | the six expected label failures (updated in H) / 118 passed |
| Targeted suites (29 files; run 2 with the UI-1 truthfulness suite added: 30 files) | `_targeted_pytest.txt` / `_targeted_pytest_run2.txt` | **341 passed** / **355 passed**, exit 0 |
| Complete suite, environment as-is, ONE invocation | `python -m pytest -q -p no:cacheprovider --junitxml=junit_full.xml` (`run_final_gates.sh` → `_final_pytest.txt`) | **2500 passed in 1816.72s (0:30:16)**, exit 0 (2026-09-04T21:56:54Z → 2026-09-04T22:27:16Z) over the FINAL tree (both provider keys present). Run 1 over the pre-fix tree found one real defect (the reconciliation banner keyed on a roll-up that treated an unmeasured optional counter as unavailable) and was superseded after the fix — `DEVIATIONS.md` gate-run note, `TEST_RESULTS.md` |
| Warnings as errors | pyproject `filterwarnings = error` | PASS — no warnings-summary section in any log |
| Ruff | `python -m ruff check src tests scripts` (`_ruff_and_diffcheck.txt`) | `All checks passed!` (exit 0) |
| `git diff --check` | over `src scripts tests docs QL-FSM-PROP-SEARCH-DASHBOARD` | clean (exit 0) |
| Capacity benchmark | not applicable — no benchmarked byte changed | n/a |
| Browser / viewport / keyboard evidence | not captured in UI-3 (mandatory at UI-6; the plan leaves UI acceptance OPEN until then) | OPEN |

## 3. Preservation

| Constraint | Proof |
|---|---|
| Strategy-Core unchanged | not touched |
| Fixed M0–M3 lane and identities | `context_reporting.py`, `context_statistics.py` and the experiment service untouched; the adapters add keys only |
| Backend contracts | consumed read-only; no contract, identity or validator changed; `study_providers`, `study_drafts`, `study_presentation`, `study_status`, `search/*` untouched |
| No new spawn site | the scans admit the two existing seams only (`test_process_launch_exists_only_in_the_designated_seams` green) |
| Regime panel read-only rule | `test_regime_panel_exposes_no_control_that_promotes_launches_or_retrains` green over the restructured panel |
| Immutable artifacts / `data/` | `find data -type f -newer <UI-2 COMPLETION_REPORT>` → none; every test store lives under `tmp_path` |
| No launch, seed, signing, verification run | none executed |
| User-owned hunks | the four docs and two data deletions remain the pre-existing worktree changes; none staged or committed (the three shared docs staged as HEAD + the UI-3 lane transforms) |

## 4. What UI-4 starts from

The registries, roll-ups, help / glossary and detail levels exist for every screen; Context
Research and Results lead with decisions; the MBP-1 / regime panels are summary-first. UI-4
lands the Replay / Verifier redesign (the persistent case card, grouped controls with the PIT
scrubber, the dark chart with legend / timeline / table twins, the removal of the duplicate lower
inspectors and the setup-mode fallthrough) — the verifier's help entries exist and may be
reworded when the controls regroup. UI-5 lands Data & Audit over the registry rows that already
describe its report fields.

## 5. Packaging

| Item | Value |
|---|---|
| Commit | 75a4c951ab2fc489653fb917e9369f15d036ddca on `feature/ifvg-prop-robust-config-search-v1`; parent `eb85d09`; 35 files changed, 7638 insertions(+), 296 deletions(-); neither pushed nor merged |
| Patch | `UI-3.patch` (`git format-patch -1 --stdout`; 410,708 B) — sha256 `ffdfc42002f432a34a5429caec481bd133dea682bbe7c69b59845996f1629d1a` (`UI-3.patch.sha256`, verified with `sha256sum -c`) |
| Bundle | `UI-3.bundle` (`eb85d09..feature/ifvg-prop-robust-config-search-v1`; requires `eb85d09`; 108,508 B; `git bundle verify` OK) — sha256 `49f9c4626cf5eadf8ac14f821d9ae2526ec08ecba4beaa1c9665aa31ed24bc13` (`UI-3.bundle.sha256`, verified) |
| Evidence folder | this folder (see `FILES_TOUCHED.md`) |
