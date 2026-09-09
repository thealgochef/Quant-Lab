# UI-3 — Progress checkpoint (resume from here)

**Release:** UI-3 — Phase 3 of `../IMPLEMENTATION_PLAN.md` (revision 2): the shared metric /
helper / status system (metric metadata registry, section roll-ups, helper text + glossary,
human-label registry, detail levels), the Context Research and Results presentation, the
supervised-ladder frame, and the MBP-1 / regime summary-first panels (F-07 full, F-09, F-11 part).
**Authority:** the owner's instruction on 2026-09-04 ("continue with the next phase. UI-2 is
completed. Resume") over `../IMPLEMENTATION_PLAN.md` revision 2 (§9 Phase 3), subordinate to
`../../../FINAL-IMPLEMENTATION-PLAN-DOCS/`.
**Baseline:** `feature/ifvg-prop-robust-config-search-v1` @ `eb85d09` (UI-2 release head;
verified on resume: `UI-2.patch` / `UI-2.bundle` sha256 OK, `git bundle verify` OK, no `data/`
file newer than the UI-2 completion report).
**Rules carried:** one release-scoped commit per phase; never push / merge; the user's
pre-existing hunks in `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`,
`docs/ML_TRAINING_WORKBENCH.md` and the two `data/experiment/honest_edge/*.json` deletions never
enter a commit (stage release files by path; shared docs through `stage_shared_docs.py` in this
folder); acceptance uses isolated synthetic / test stores only — no owner signing, no seed
production, no real verification run, no new spawn site; never edit a tracked file while the
detached suite runs; the fixed M0–M3 computation lane (`context_reporting.py`,
`context_statistics.py`, the experiment service) is NOT modified — presentation reads the
persisted reports through the pure adapters only.

## Design decisions fixed before coding (see `DEVIATIONS.md`)

1. **One metric registry, no invented thresholds.** `presentation/metric_registry.py` holds one
   `MetricSpec` per displayed technical key; directionality from `OBJECTIVE_DIRECTIONS` where
   registered (asserted); every reference is a REGISTERED source in existing code; unevaluated /
   missing evidence is UNAVAILABLE, never PASS.
2. **Deterministic roll-ups.** `presentation/rollups.py`: FAIL → BLOCKED → INCONCLUSIVE →
   WARNING → PASS → INFORMATIONAL → UNAVAILABLE; nine sections; one sentence, main reason,
   inspect-next.
3. **Helper text is a registry, the scan is exhaustive, exemptions are explicit and live.**
   `presentation/help_registry.py`; the scan covers every widget of every UI script (the verifier
   and lab tabs included); sixteen navigation exemptions; the glossary's sixteen terms.
4. **Detail levels extend disclosure.** `ifvg_ui_common.detail_levels` over the unchanged
   persisted `DisclosureLevel` values; `disclosure_level` delegates (contract §5.1 amended).
5. **Summary-first panels without new controls.** Summary → Research details → Advanced
   diagnostics (the manual exact-id inputs, same keys, instantiated first into the lower
   container); the regime panel keeps its read-only rule; no second empty state.
6. **The ladder frame keeps AUC numeric** with a separate `AUC reason` column.
7. **Context Research reads the persisted reports only** through additive adapter keys;
   `context_reporting.py` untouched.

## Workstreams (Phase 3 scope, §9 of the plan)

| # | Workstream | Status | Evidence |
|---|---|---|---|
| A | `presentation/metric_registry.py` + tests | done | `_red_A_D_pure.txt` (red); registry 36 passed |
| B | `presentation/rollups.py` + tests | done | rollups 3 passed |
| C | `presentation/help_registry.py` + tests | done | help registry 4 passed |
| D | `presentation/labels.py` + tests | done | labels 3 passed |
| E | `ifvg_ui_common.py` primitives + tests | done | `_red_E_L_common_scans.txt` (red); ui_common ui3 3 passed |
| F | adapters' reference / calibration / fold / access fields; `build_coverage_figure` over threshold rows; `build_reliability_figure` + tests | done | `_red_FG_context.txt` (red); context presentation 12 passed; adapters 9 passed |
| G | `ifvg_lab_tab.py` Context Research + AppTests | done | lab tab 24 passed (with the F suite) |
| H | `ifvg_results_tab.py` + `presentation/results_presentation.py` + test updates | done | `_red_H_results.txt` (red); results 19 + pure 3 + compare 5 passed |
| I | `_ladder_frame` AUC / AUC reason + caption; help on every pipeline widget + test update | done | `_red_IJK_panels.txt` (red); pipeline tab 45 passed |
| J | `ifvg_mbp1_panels.py` summary-first + tests | done | pipeline tab 45 passed (2 new MBP-1 tests) |
| K | `ifvg_regime_panels.py` summary-first + tests | done | pipeline tab 45 passed (2 new regime tests) |
| L | help coverage over every UI script; accessible collapsed labels; the scans | done | help scans 12 passed; `_regression_L_part1.txt` / `_part2.txt` |
| M | Docs: contract §§5.1, 5.6 (new), 17, 30.4, 32.1, 35, 36; `docs/DECISIONS.md` D-055; `DECISIONS_TAKEN.md` #138; `stage_shared_docs.py` (dry-run OK); `DEVIATIONS.md`, `FILES_TOUCHED.md` | done | this folder |
| N | Gates: targeted (341 passed, `_targeted_pytest.txt`; run 2 with the UI-1 truthfulness suite 355 passed, `_targeted_pytest_run2.txt`); Ruff + diff-check (`_ruff_and_diffcheck.txt`); the complete suite as ONE detached invocation over the FINAL tree (run 1 caught one real defect — logs kept; run 2 **2,500 passed**, `_final_pytest.txt`, `junit_full.xml`); commit `75a4c95`; `UI-3.patch` / `UI-3.bundle` + `.sha256`; `TEST_RESULTS.md`, `COMPLETION_REPORT.md` | done | this folder |

## Log

- 2026-09-04 — Resume: UI-2 artifacts verified; plan §9 Phase 3 read in full; reconnaissance
  complete; folder opened; design decisions fixed.
- 2026-09-04 — A–D implemented tests-first (`_red_A_D_pure.txt`; 46 passed). E + the scans
  tests-first (`_red_E_L_common_scans.txt`); `ifvg_ui_common` rewritten; the help patch wired
  `help=` onto 102 widgets across twelve scripts (imports normalised with Ruff's isort fixer);
  regression over every touched script (`_regression_L_part1.txt`: the six expected Results
  label failures only; `_part2.txt`: 118 passed).
- 2026-09-04 — F/G tests-first (`_red_FG_context.txt`): `presentation/context_research.py`, the
  adapter fields, the chart builders, the lab tab renderer block (decision summary, sample
  adequacy, detail levels, registry cards, twins, raw JSON under audit, compatibility words,
  render tags) — 81 passed with the existing lab / adapter suites. H tests-first
  (`_red_H_results.txt`): `presentation/results_presentation.py`, the Results tab block, the
  test label updates — 27 passed. I/J/K tests-first (`_red_IJK_panels.txt`): the ladder frame,
  the MBP-1 rewrite, the regime restructure — pipeline tab 45 passed.
- 2026-09-04 21:17Z — Targeted suites (29 files) **341 passed** (`_targeted_pytest.txt`); Ruff
  clean; diff-check clean (`_ruff_and_diffcheck.txt`). Docs written (contract, D-055, #138,
  DEVIATIONS, FILES_TOUCHED, `stage_shared_docs.py` dry-run validated). Diff: 18 tracked files
  changed (+1,604 / −298) plus 14 new files. Tree FROZEN; the complete suite launched detached
  (`run_final_gates.sh` → `_final_pytest.txt`, `junit_full.xml`, `gates.done`).
- 2026-09-04 21:54Z — Complete-suite run 1 (21:22Z → 21:53Z): **2,499 passed / 1 failed** —
  `test_ifvg_ui1_truthfulness.py::test_reconcile_banner_derives_from_evaluated_gates` (a UI-1
  suite the targeted set had not included): the banner had been keyed on the Data-integrity
  roll-up, and an access audit without `denied_dates` (and any `None` optional performance
  figure) produced an UNAVAILABLE reading that made a fully evaluated section inconclusive, so
  a two-gate pass showed no success banner and a `passed: False` report no error banner. A real
  defect of the UI-3 change; logs kept as `_final_pytest_run1_integrity_rollup.txt` /
  `junit_full_run1_integrity_rollup.xml`. Fixed: the banner keys on the persisted derived flag
  AND `evaluated` (UI-1's contract; a legacy `passed: True` without evaluated gates stays a
  warning); unmeasured optional figures are omitted from the readings, never fabricated
  (`DEVIATIONS.md` item 7). Targeted run 2 with the UI-1 truthfulness suite added: **355 passed**
  (`_targeted_pytest_run2.txt`); Ruff / diff-check clean over the final tree
  (`_ruff_and_diffcheck.txt`). Tree FROZEN again; the complete suite relaunched detached as ONE
  invocation over the final tree (started 21:5xZ; see `gates.progress`).
- 2026-09-04 22:28Z — Complete-suite run 2 over the FINAL tree: **2,500 passed** (one
  invocation, 30:16, exit 0; 21:56:54Z → 22:27:16Z; both provider keys present). Release commit
  `75a4c95` (35 files, +7,638 / −296; parent `eb85d09`; not pushed, not merged); the shared docs
  staged as HEAD + the UI-3 transforms and replayed onto the worktree (the surviving diff is
  exactly the user's pre-existing hunks: the four docs and the two data deletions). `UI-3.patch`
  / `UI-3.bundle` + `.sha256` written and verified (`git bundle verify` OK). `TEST_RESULTS.md`,
  `COMPLETION_REPORT.md`, `DEVIATIONS.md` (gate-run note), `FILES_TOUCHED.md` final.
  **UI-3 closed.** Next release: UI-4 (the Replay / Verifier redesign — the persistent case
  card, grouped controls with the PIT stage scrubber, the dark chart with legend / timeline /
  table twins, the removal of the duplicate lower inspectors and the setup-mode fallthrough) —
  start from `../IMPLEMENTATION_PLAN.md` §9 Phase 4 over `75a4c95`; the verifier's help entries
  (`help_registry.py`, `verifier.*`) exist and may be reworded when the controls regroup.
