# UI-1 — Progress checkpoint (resume from here)

**Release:** UI-1 — Phase 1 of `../IMPLEMENTATION_PLAN.md` (revision 2): semantic purpose,
namespace and authorization truth; charter satisfiability; honest launch; truthful evidence.
**Authority:** the owner's instruction on 2026-09-04 ("the backend R6 phase and hardening is
completed. Now we move on to the UI design fix implementation … continue with the development
plan") over `../IMPLEMENTATION_PLAN.md` revision 2, subordinate to
`../../../FINAL-IMPLEMENTATION-PLAN-DOCS/`.
**Baseline:** `feature/ifvg-prop-robust-config-search-v1` @ `ffcf39b` (HARDENING-BACKEND-FIX.1
release head; `backend_dev_complete_for_ui = true`, `ui_implementation_may_begin = true`).
**Rules carried:** one release-scoped commit per phase; never push / merge; the user's
pre-existing hunks in `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`,
`docs/ML_TRAINING_WORKBENCH.md` and the two `data/experiment/honest_edge/*.json` deletions never
enter a commit (stage release files by path; shared docs through `stage_shared_docs.py` in this
folder); acceptance uses isolated synthetic / test stores only — no owner signing, no seed
production, no real verification run.

## Workstreams (Phase 1 scope, §9 of the plan)

| # | Workstream | Status | Evidence |
|---|---|---|---|
| A | `presentation/` package: `status_vocabulary`, `run_purpose`, `charter_satisfiability` (+ unit tests) | done | `_red_A_presentation.txt` (red), `_targeted_pytest.txt` |
| B | `study_status`: `Start` / `Verify Implementation` routes; §6.7 additive empty states | done | `_targeted_pytest.txt` |
| C | `study_drafts`: presentation-only `purpose_annotation` (additive); catalog `purpose` kind | done | `_targeted_pytest.txt` |
| D | `study_providers`: verified namespace resolution, typed `AuthorizationReadiness`, bundles from ready readiness, namespace-annotated run listings | done | `_red_DE.txt` (red), `_targeted_pytest.txt` |
| E | `search/charter.validate_charter`: fail-closed satisfiability | done | `_targeted_pytest.txt` |
| F | `ifvg_study_tab`: namespace radio removed; `Start` cards; `Verify Implementation` readiness surface; purpose-derived roots | done | `_targeted_pytest.txt` |
| G | `ifvg_study_wizard`: goal card, authorization by actual path, satisfiability card, honest launch, no worker control, backend-derived dates | done | `_targeted_pytest.txt` |
| H | `ifvg_pipeline_tab` + `search/pipeline`: purpose store, honest launch, no worker sliders, namespace- and state-bound gates / activation | done | `_targeted_pytest.txt` |
| I | `ifvg_results_charts`: direction-aware colorscales | done | `_targeted_pytest.txt` |
| J | reconciliation truth (`context_reporting`, adapters, lab tab) | done | `_targeted_pytest.txt` |
| K | Results / Active Runs / History: artifact-derived scope, per-run store roots, `no_runs` / `artifact_missing` | done | `_targeted_pytest.txt` |
| L | Docs: contract §§3.2, 7, 8.1, 14, 15, 30, 31, 35, 36; `docs/DECISIONS.md` D-053; `DECISIONS_TAKEN.md` #136; shared docs via `stage_shared_docs.py` (dry-run validated) | done (staging at commit) | `stage_shared_docs.py` |
| M | Gates: targeted 359 passed; Ruff clean; `git diff --check` clean; complete suite 2,363 passed as ONE invocation over the final tree; commit `f1827e8`; patch / bundle + sha256; `TEST_RESULTS.md`, `FILES_TOUCHED.md`, `COMPLETION_REPORT.md` | done | `_targeted_pytest.txt`, `_ruff_and_diffcheck.txt`, `_final_pytest.txt`, `junit_full.xml`, `UI-1.patch(.sha256)`, `UI-1.bundle(.sha256)` |

## Log

- 2026-09-04 — Plan read in full; backend release head verified (`ffcf39b`); reconnaissance of
  the R4–R6.1 UI, the backend contracts and the affected suites complete.
- 2026-09-04 — Workstreams A–K implemented tests-first (red logs kept for A and D/E); the
  reworked AppTests: study tab 9, wizard 31, pipeline tab 40, active runs 6, results 13, compare
  5, scans 14, truthfulness 14, adapters 9, lab tab 24 — all green; backend: charter 14,
  providers 12 + 4, pipeline_run 20 (incl. the new namespace/state binding), catalog 9,
  presentation 27 — all green. Consolidated targeted run: **359 passed** (`_targeted_pytest.txt`).
- 2026-09-04 — Docs: contract amended; D-053; #136; `stage_shared_docs.py` written and dry-run
  validated against HEAD anchors. `DEVIATIONS.md` written.
- 2026-09-04 06:32Z — Ruff clean, diff-check clean (`_ruff_and_diffcheck.txt`). Full suite
  launched detached (`run_final_gates.sh` → `_final_pytest.txt`, `junit_full.xml`, `gates.done`).
  **Next on resume:** if `gates.done` exists, read `_final_pytest.txt` (expect `exit=0`); then
  `python stage_shared_docs.py`, `git add` the release files by path (never the user hunks),
  commit `UI-1: semantic purpose, namespace and authorization truth; satisfiability; honest
  launch and evidence`, `python stage_shared_docs.py --apply-worktree`, write `TEST_RESULTS.md`,
  `FILES_TOUCHED.md`, `COMPLETION_REPORT.md`, `git format-patch -1` + bundle + sha256.
- 2026-09-04 07:30Z — Complete suite over the FINAL tree: **2,363 passed** (one invocation, 28 min; the first run over a mid-run-edited tree had one harness-artifact failure and was superseded — see `DEVIATIONS.md`). Release commit `f1827e81cf789286b748c0a1728e866f5944c139` (39 files, +6,031 / −562; parent `ffcf39b`; not pushed, not merged); the shared docs staged as HEAD + the UI-1 transforms and replayed onto the worktree (the surviving diff is exactly the user's pre-existing hunks: 147 insertions / 314 deletions over the six user-owned entries). `UI-1.patch` / `UI-1.bundle` + `.sha256` written and verified. **UI-1 closed.** Next release: UI-2 (the Verification Center flow, goal-conditional flows, session-only drafts with archive / restore / typed delete, explicit reviewer verdicts) — start from `../IMPLEMENTATION_PLAN.md` §9 Phase 2 over `f1827e8`.
