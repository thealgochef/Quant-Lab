# R4 — Repository files touched

## New source modules (src)

| File | Role |
|---|---|
| `src/alpha_lab/agents/data_infra/ifvg/study_status.py` | CS §13 presentation contracts: routes, disclosure, result scopes, status keys + glyph/label/help registry, §31 empty-state registry (14 states), exact required copy, forbidden display phrases |
| `src/alpha_lab/agents/data_infra/ifvg/study_presentation.py` | Pure presentation logic: axis grouping (7 FUX groups), classification labels, computation-path chips, wizard registries (5 modes / 4 questions / 6 templates) + validators, funnel/stage derivations w/ orchestrator sentinels, baseline-diff names, explorer presets, pagination, estimates, viewport classes |
| `src/alpha_lab/agents/data_infra/ifvg/study_drafts.py` | Mutable draft layer: atomic JSON persistence, exact-step restore, clone, freeze immutability (DEV-R4-1) |
| `src/alpha_lab/agents/data_infra/ifvg/study_providers.py` | Read-only UI providers: run listing (job root + catalog), exact-ID loads (charter/frontier/metrics/contract cards/prop vectors/account events), the lineage-gated `prepare_cross_profile_deltas` (DEV-R4-1) |
| `src/alpha_lab/agents/data_infra/ifvg/search/runner_registry.py` | Registry-gated runner entries (R2→R4 obligation): key→entry map, fail-closed refusals, charter→key mapping |

## Modified source modules (src)

| File | Change |
|---|---|
| `src/alpha_lab/agents/data_infra/ifvg/search/orchestrator.py` | Records the persisted frontier envelope id as `phase_notes["frontier_id"]` (DEV-R4-2; additive) |
| `src/alpha_lab/agents/data_infra/ifvg/study/contrasts.py` | Balanced two-way interaction evaluation (DEV-R2-5 closure, §7A.19.11); main-effect path refactored into helpers; dead `InteractionEvaluationUnavailableError` removed (DEV-R4-4) |

## New scripts

| File | Role |
|---|---|
| `scripts/ifvg_ui_common.py` | Shared helpers: sanitize_error/select, display_metric, status/dev/verification badges, disclosure, identity_block, cli_escape_hatch, render_empty_state, paginate_controls, result_scope_caption, queue_replay_drilldown |
| `scripts/ifvg_study_tab.py` | Experiments sub-navigation router + namespace selector + programmatic routing |
| `scripts/ifvg_study_wizard.py` | Eight-step five-mode wizard, drafts, freeze/validate/save, registry-gated detached launch |
| `scripts/ifvg_active_runs_tab.py` | Monitor: fragment+manual poll, phase checklist, keyboard funnel, child table/detail, safe cancel, CLI fallback |
| `scripts/ifvg_results_tab.py` | Results common frame, overview, frontier, heatmap, firm/survival/payout, explorer, Audit block; History (5 sections) |
| `scripts/ifvg_results_compare.py` | Dimension ribbon + four panels, deterministic insight panel, account timeline |
| `scripts/ifvg_results_charts.py` | Pure Plotly builders (funnel/delta/frontier/heatmap/matrix/survival/payout/timeline) with budgets + OmissionReports |

## Modified scripts / config

| File | Change |
|---|---|
| `scripts/ifvg_lab_tab.py` | Experiments branch delegates to `render_ifvg_study_tab` (Context Research passes the unchanged M0–M3 renderer) |
| `scripts/ifvg_search_job.py` | Registry gate on `--runner-entry`, new `--runner-entry-key`, new `resume` command |
| `pyproject.toml` | `streamlit>=1.41` floor |
| `.github/workflows/ci.yml` | `ruff check src tests scripts` |

## Shared docs (staged HEAD + lane transforms only)

- `ARCHITECTURE.md` (R4 lane append) · `docs/README.md` (R4 status text) ·
  `docs/pipeline_state.yaml` (`R4_trader_ui: complete`) — via
  `R4/stage_shared_docs.py`; user hunks byte-verified post-commit.
- `docs/DECISIONS.md` — D-045 trader-workspace half + reservation-note
  update (not user-dirty; committed normally).

## New tests

| File | Coverage |
|---|---|
| `tests/agents/ifvg_search/test_study_status.py` | CS §13 exactness, glyph/word/help, §31 registry, forbidden-wording scan |
| `tests/agents/ifvg_search/test_study_presentation.py` | Wizard registries, validators, grouping, funnel/stage mapping incl. sentinel pinning, names, presets, pagination, estimates, viewports |
| `tests/agents/ifvg_search/test_study_drafts.py` | Atomic persistence, exact restore, freeze immutability, clone, discard policy |
| `tests/agents/ifvg_search/test_study_providers.py` | Run listing, exact-ID loads, catalog∩store listing, event ordering, prop vectors, contract cards, lineage-gated deltas |
| `tests/agents/ifvg_search/test_runner_registry.py` | Registry resolution/refusal/immutability, charter→key fail-closed |
| `tests/agents/ifvg_search/study_ui_fixture.py` | Shared completed-search fixture (real 2×2 E2E + catalogued account sim + synthetic contract) |
| `tests/agents/test_ifvg_study_tab.py` | FUX-IA-001..003 + programmatic routing |
| `tests/agents/test_ifvg_study_wizard.py` | FUX-WIZ-001..012 (16 AppTests incl. both freeze directions + typed full-scope confirmation) |
| `tests/agents/test_ifvg_active_runs_tab.py` | FUX-MON-001..004 (funnel labels, exact columns/skipped copy, sanitized detail, safe cancel, CLI fallback) |
| `tests/agents/test_ifvg_results_tab.py` | FUX-RES-001..006 + FUX-HIST-001 (incl. exact no-pass copy) |
| `tests/agents/test_ifvg_results_compare.py` | FUX-RES-007..009 + FUX-DRILL-001 (ribbon/match_basis, suppression, 7 categories, timeline order + linked evidence, sanitized unresolved) |
| `tests/agents/test_ifvg_study_scans.py` | FUX-SAFE-001 / FUX-LABEL-001 source scans, launch-seam confinement, session-namespace scan, chart-builder units, fragment fallback (FUX-A11Y-004) |

## Modified tests

- `tests/agents/ifvg_search/test_search_job_script.py` (+registry-gate,
  key-resolution, resume tests)
- `tests/agents/ifvg_search/test_contrasts.py` (interaction refusal test →
  4 real interaction tests)

## Implementation-progress files produced

`PRE_R4_BASELINE.md`, `DEVIATIONS.md`, `FILES_TOUCHED.md` (this),
`ACCESS_SAFETY_EVIDENCE.md`, `TEST_RESULTS.md`, `ADVERSARIAL_REVIEW.md`,
`ADVERSARIAL_REVIEW_RESOLUTION.md`, `GATE_SUMMARY.md`,
`stage_shared_docs.py`, + `../DECISIONS_TAKEN.md` entries 26–33.
