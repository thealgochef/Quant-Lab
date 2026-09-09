# R4 — Test results

## Exact commands, exit codes, counts, timings

| Command | Result | Exit | Time |
|---|---|---|---|
| `python -m pytest -q` (full repo, PRE-R4 baseline at `a23893c`) | 1372 passed, 7 warnings | 0 | 7:27 |
| `python -m pytest -q` (full repo, post-implementation / pre-review checkpoint) | 1476 passed, 7 warnings | 0 | 7:15 |
| `python -m pytest -q` (full repo, **FINAL**, all adversarial findings resolved) | **1492 passed** (1372 baseline + 120 net new), 7 warnings (pre-existing pattern) | 0 | 6:41 |
| R4 lane block (**FINAL**): `pytest tests/agents/test_ifvg_study_tab.py test_ifvg_study_wizard.py test_ifvg_active_runs_tab.py test_ifvg_results_tab.py test_ifvg_results_compare.py test_ifvg_study_scans.py tests/agents/ifvg_search/ -q` | **338 passed**, 4 warnings | 0 | 9.7s |
| `python -m ruff check src tests scripts` (FINAL — note `scripts` now in scope, matching the widened CI) | All checks passed | 0 | <10s |
| `git diff --check` | clean (CRLF advisory warnings only — pre-existing user-owned files) | 0 | <1s |

## Suite composition (net-new/updated in R4)

| Suite | Tests | Covers |
|---|---|---|
| `tests/agents/ifvg_search/test_study_status.py` | 7 | CS §13 exactness, §31 registry (16 presentations), glyphs, exact copy, forbidden-wording |
| `tests/agents/ifvg_search/test_study_presentation.py` | 22 | wizard registries/validators (incl. interpretation gate + the four extra validators), grouping, funnel/stage mapping + orchestrator-sentinel pinning, names, presets, pagination, estimates, viewports |
| `tests/agents/ifvg_search/test_study_drafts.py` | 6 | atomic persistence, exact-step restore, freeze immutability + TOCTOU closure, hostile-id refusals, clone, discard |
| `tests/agents/ifvg_search/test_study_providers.py` | 11 | run listing, exact-ID loads, catalog∩store, envelope-ordered events, prop vectors, contract cards + launchable ladder, verification-authorization state (both directions), lineage-gated deltas |
| `tests/agents/ifvg_search/test_runner_registry.py` | 3 | registry resolution/refusal/immutability, charter→key fail-closed |
| `tests/agents/ifvg_search/test_search_job_script.py` | 10 (7 prior + 3 new) | shape+registry gate (import-bomb proof), key resolution, resume |
| `tests/agents/ifvg_search/test_contrasts.py` | 12 (8 prior + 4 net) | main effects + BALANCED two-way interactions (DiD, strata, determinism, refusals) |
| `tests/agents/test_ifvg_study_tab.py` | 6 AppTests | FUX-IA-001..003, single-route execution, delegation, programmatic routing |
| `tests/agents/test_ifvg_study_wizard.py` | 21 AppTests | FUX-WIZ-001..012 incl. Back-preserves, risk-step hierarchy, interpretation block, cross-draft purge, provenance refusal, both freeze directions, typed full-scope confirmation |
| `tests/agents/test_ifvg_active_runs_tab.py` | 6 AppTests | FUX-MON-001..004 incl. quoted §16.2 labels, membership/costed identities, safe cancel, CLI fallback |
| `tests/agents/test_ifvg_results_tab.py` | 13 AppTests | FUX-RES-001..006 + FUX-HIST-001 incl. exact no-pass copy (terminal-only), derived scope captions, true survival values, honest heatmap aggregation, 64-child pagination, artifact-unavailable (not gate-skip) |
| `tests/agents/test_ifvg_results_compare.py` | 5 AppTests | FUX-RES-007..009 + FUX-DRILL-001 incl. ribbon/match_basis, panel suppression, 7 insight categories, envelope-ordered timeline + linked evidence, sanitized unresolved |
| `tests/agents/test_ifvg_study_scans.py` | 14 | FUX-SAFE-001/LABEL-001 source scans (f-string-aware namespace scan, launch-seam confinement, override-editor, title scan), chart-builder units (budgets/omissions/glyphs/dll-line/phase markers/P10 prominence), sanitizer blind spots, fragment fallback (FUX-A11Y-004) |

## Browser smoke (live app; evidence in `browser-smoke/`)

`streamlit run …/R4/r4_smoke_app.py` (the IFVG Lab shell only) at 1440×900,
driven via Chrome automation against the REAL repository roots:

1. Shell + workspace (screenshot-…-0): top-level order Experiments · Replay
   / Verifier · Data & Audit; the five-route sub-nav; namespace selector;
   the exact development badge.
2. Five modes (…-1): the mode selectbox lists exactly Single Configuration
   / FSM Configuration Search / Prop Benchmark / Universal Prop Search /
   Full Pipeline Run.
3. Wizard step 1 (…-2): Step 1 of 8 + progress + full breadcrumb; four
   research questions; template with the resolved objective table
   (`proposed_protocol_default` stamp); Save Draft + Next.
4. Wizard step 2 (…-4): the real resolved baseline card
   (`ifvg_v2_doc_default_fresh_static_1r` · runnable · entry thesis ·
   direction · label family · FSM concurrency · development range) + View
   Technical Identity with copyable blocks.
5. Wizard step 3 (…-5): Staleness group axis cards — human label,
   technical key, "Search Axis · Requires New Sequential Replay" chips,
   market meaning, registered-values multiselect, Evidence & authorization
   expander.
6. Active Runs empty state (…-6): "Artifact unavailable" + sanitized
   explanation + next action + Audit disclosure + the exact CLI escape
   hatch; no local path anywhere.
7. History (…-7): all five sections in order; the draft row shows the
   exact step (3/8) with Open/Clone/confirm-gated Discard; Superseded
   notes the catalog archive flag; Legacy Read-Only carries the
   no-controls caption.
8. Context Research (…-8): the UNCHANGED M0–M3 panel renders inside the
   workspace over the real verified pair and immutable run history
   (FUX-IA-003 live).

The smoke draft was removed afterwards (`data/ifvg_study_drafts/` deleted;
`git status` over `data/` unchanged from the baseline). Full interactive
keyboard/viewport/screenshot QA at all four required sizes remains the
HARDENING gate (open by design at R4).
