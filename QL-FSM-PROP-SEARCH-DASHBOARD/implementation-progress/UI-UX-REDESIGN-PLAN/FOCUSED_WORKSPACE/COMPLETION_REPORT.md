# Focused IFVG workspace — implementation and acceptance record

Date: 2026-09-07. Implementation delivered; desktop browser flows reviewed.
Full live execution lifecycle acceptance remains limited by existing prerequisites.

## Delivered

- Selected-page Streamlit shell: IFVG Lab (default), ML Training, Dashboard
  Compatibility, Strategy Analysis, and startup-enabled Developer. The shared
  pipeline sidebar moved into Strategy Analysis.
- My studies unifies drafts, exact-linked search/pipeline results, progress and
  history. Normal lists exclude verification; legacy scope and corrupt progress
  remain unresolved/unavailable. Human names remain mutable annotations.
- Guided Evaluate/Compare/Search and additional research types reuse existing
  flows, defaults, validators, authorization and launch handlers. Session-only
  drafts save on explicit Save/valid Next; saved drafts autosave. Missing research
  executors/model labels remain blocking dependencies, not enabled capabilities.
- Concise strategy, prop, pipeline-model and context-model presentation uses the
  metric registry and compatibility gates. Detailed research tables omit technical
  identities and source references. Applicable prop survival/payout views and
  calibration charts have data-table alternatives.
- Typed Replay selection prevents parent fallthrough for setup-only, empty,
  missing-chart and unresolved exact-case evidence. Charts use sanitized presentation
  copies. Actual execution, hypothetical labels, model probability and saved review
  judgment remain distinct; reviews require explicit Save Review.
- Persisted-state progress, five-second refresh of selected running detail,
  safe cancellation, supported resume, clone/rename/archive/restore, eligibility
  checks before catalog addition, and Developer health summaries.

No scientific contracts, strategy calculations, cache identities or immutable
result formats were changed. Developer mode is presentation only and grants no
execution authority. Existing owner authorization, seed and real-verification
requirements remain in force.

## Automated verification

- Focused workspace integration/AppTests: **39 passed** (final focused rerun).
  Covers selected-page execution and Developer registration, all six guided flows,
  draft persistence, completed strategy metrics and detail exports, inconclusive
  model evidence, legacy/verification scope, corrupt progress, setup-only parent
  integration, empty filters, missing exact candidates, safe cancellation and
  draft rename/archive/restore/clone, nullable table values, and verified pipeline
  result adaptation with unavailable final-report evidence.
- Retained UI-1–UI-3 technical tests explicitly enter Developer presentation.
  Normal-workspace tests separately assert absence of technical renderers/fields.
- Focused plus retained wizard/pipeline/setup/help integration run: **141 passed**.
- Final workspace/Replay/help integration run: **91 passed**, including the
  additional Save Review regression: opening model probabilities preserves the
  selected candidate, decision, trade, chart and exact data-pair references.
- Existing local catalog: 12 unresolved legacy drafts and 9 completed context
  studies loaded without readiness errors. AppTest opened the actual landing
  page and a saved context result without exceptions or raw code/JSON blocks.
- `python -m ruff check src tests`: **passed**.
- New research modules: Ruff undefined/unused-name checks passed.
- Modified/new Python modules: compile check passed.
- Full `python -m pytest -q --tb=short --show-capture=no --durations=10`:
  **2,539 passed in 1,573.97 seconds (26:13)**. Collection preceded the additional
  Save Review regression, which passed in the final 91-test run above.
  This clean run supersedes an earlier run affected by AppTest source extraction
  while its test file was being edited; all collected test files remained unchanged
  throughout the clean run.

## Browser retry — desktop flows and actions

The subsequent in-app browser connection worked. The user explicitly removed
mobile responsiveness from acceptance and prioritized desktop screens and actions.
The main IFVG session used 1440×900; a separate Developer session used 1280×720.
Current screenshots and actual action evidence are recorded in the
[desktop flow report](../../../../reports/ifvg_browser_acceptance/20260907/DESKTOP_FLOW_REPORT.md).

Browser checks exercised draft save/resume/rename/archive/restore/clone, Evaluate
and Compare through review, Search axes, model-workflow prerequisites, blocked
prop setup, saved model results/details/comparison, exact supporting-opportunity
navigation, point-in-time keyboard replay, setup-only evidence, empty filters,
chart/table controls, explicit review saving and real CSV downloads. Developer
readiness/health and all app-shell workspaces were also opened.

The pass repaired unreachable draft management, technical source leakage in result
explanations, contradictory review save status, chart text contrast, a stale case
label after filtering, and unlabeled section verdicts in model results. Browser
retests confirmed the corrections. Final regression checks: **97 focused tests**
and **55 chart/provider tests** passed; **CI Ruff passed**. The full 2,539-test run
above preceded these browser corrections and was not repeated in this pass.

Six named browser-test drafts were archived. Two clearly labeled interface-only
Not applicable reviews remain in immutable review history. No research run,
verification run, authorization or scientific-result rewrite was performed.

The browser connection is no longer a blocker. Full live running/cancel/resume,
failed recovery, catalog addition and strategy/prop execution acceptance remains
open because the required authorization, verified contracts and model labels are
unavailable locally. Automated coverage is not claimed as browser acceptance.
See the linked report for exact coverage and additional unexecuted branches.
