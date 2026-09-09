# R5 — Test results (exact commands, counts, timings)

All commands run from the repo root on the R5 working tree
(post-implementation; the FINAL rows are re-run after the adversarial
resolutions and recorded below when they differ).

> **R5-FIX supersession note:** the rows below are the `dda40c9` record.
> The release-final evidence is the R5-FIX section at the bottom
> (commit `fb8062f`): full repo **1606 passed**, ruff clean, and a re-run
> browser smoke whose server log carries **zero tracebacks and zero
> deprecation warnings** (the original smoke's 17 Arrow tracebacks and 52
> `use_container_width` warnings are fixed, not waived). The
> browser-smoke section below describes the SUPERSEDED pre-fix captures,
> which now live in `browser-smoke/superseded-pre-fix-2026-08-21/`.

## Full-repo regression

| Command | Result | Exit | Timing |
|---|---|---|---|
| `python -m pytest -q` (full repo, PRE-R5 baseline at `6b820ad`) | 1492 passed, 7 warnings | 0 | 6:44 |
| `python -m pytest -q` (full repo, post-implementation checkpoint) | 1595 passed (+103 net new), 44 warnings | 0 | 7:37 |
| `python -m pytest -q` (full repo, **FINAL**, all adversarial findings resolved) | **1599 passed** (baseline 1492 + 107 net new), 46 warnings | 0 | 7:30 |
| `python -m ruff check src tests scripts` (**FINAL** — re-run after the resolutions; the two transient F401s the contract reviewer caught mid-session, M-4, are fixed) | All checks passed | 0 | <10 s |
| `git diff --check` (**FINAL**) | clean | 0 | <1 s |

Warning delta vs baseline: the sklearn L-BFGS `DeprecationWarning`
(library-internal; triggered by the new logistic/calibration fits — the
same warning family the pre-existing calibration path emits) plus one
pre-existing pandas concat `FutureWarning` pattern surfaced by an existing
suite. The R5 lanes themselves pass under
`-W error::FutureWarning` (verified: 32 pipeline/tab/bridge tests, 47 ML
tests).

## R5 lane blocks (representative, post-implementation)

| Command | Result | Timing |
|---|---|---|
| `python -m pytest tests/agents/data_infra -q` (ML lane: ladder, logistic, registries, drift, fixtures) | 47 passed | 48 s |
| `python -m pytest tests/agents/ifvg_search/test_pipeline_contracts.py test_pipeline_run.py -q` | 25 passed | ~5 s |
| `python -m pytest tests/agents/ifvg_search/test_pipeline_job_script.py test_runner_registry.py test_bundle_feature_view.py -q` | 15 passed | ~3 s |
| `python -m pytest tests/propsim/ -q` (incl. the new bridge-mode suite; R3 suites unchanged) | all passed | ~2 s |
| `python -m pytest tests/agents/test_ifvg_pipeline_tab.py -q` (FUX-PIPE-001..006 AppTests + fragment fallback + wizard integration) | 13 passed | ~3 s |
| `python -m pytest tests/agents/test_ifvg_study_wizard.py test_ifvg_study_tab.py test_ifvg_active_runs_tab.py tests/agents/ifvg_search/test_study_status.py tests/agents/test_ifvg_study_scans.py -q` | 54 passed | ~3 s |
| `python -m pytest tests/agents/ifvg_search/ tests/agents/data_infra/ tests/propsim/ -q` (combined backend lanes, mid-implementation checkpoint) | 461 passed | 60 s |

## Live browser smoke (`browser-smoke/`, port 8598, real completed run)

`r5_smoke_app.py` served the REAL `render_pipeline_run` over a completed
synthetic 16-stage run (scratch-dir stores; repo `data/` untouched —
`git status` byte-identical after):

1. `01_configure_no_draft_state.jpg` — dev badge, the six-phase radio, the
   honest no-draft Configure state with next-action copy.
2. `02_monitor_16_stages_s11_blocked.jpg` — run picker
   (`verification_5d · attempts 1 · prepared_not_published`), copyable
   `pipeline_semantic_id`, "16/16 planned stages terminal · attempt 1",
   the full stage table: **11 · Frozen Model-Gated Replays = ⛔ Blocked
   with the exact registered reason**; S12 "real AccountPolicySetEnvelopes
   persisted"; S13 "day_block_bootstrap; 32 paths, seed 42,
   day_block_bootstrap_h90_v1"; S14 "frontier … 4 insight panels … 3
   cross-profile comparison result(s) persisted"; S15
   "prepared_not_published — activation is a separate explicit action";
   execution-attempt history table.
3. `03_monitor_ladder_panel.jpg` — the supervised-ladder panel: three
   rungs at 0 OOS rows with `no_oos_predictions` /
   `insufficient_class_coverage` (the legitimate ≤5-day safe-failure
   shape) + the planned `ifvg_context_gam_v1` rung with its exact reason.
4. `04_publish_gates_activation_refused.jpg` — Publish phase: state
   caption, **all four publication gates ✓ pass**, and the activation
   control DISABLED with "verification-only artifacts can never activate
   a research catalog entry".

Hardening's 4-viewport/keyboard-only browser gate stays OPEN by design
(this smoke is desktop-1553×791 evidence, not a hardening waiver).

## Adversarial round

Two independent read-only reviewers (contract-fidelity; safety/identity) —
reports verbatim in `ADVERSARIAL_REVIEW.md`, dispositions in
`ADVERSARIAL_REVIEW_RESOLUTION.md`; FINAL suite results after resolutions
recorded in `GATE_SUMMARY.md`.

## R5-FIX round (commit `fb8062f`, 2026-08-21/22) — RELEASE-FINAL

| Command | Result | Exit | Timing |
|---|---|---|---|
| `python -m pytest -q` (full repo, **R5-FIX FINAL**; raw output `_r5fix_pytest.txt`) | **1606 passed** (R5 final 1599 + 7 net new: registry purity/guards ×2, typed comparison subject ×1, loaded-seed evidence ×2, Arrow-safe ladder frame + parity captions ×2), 46 warnings | 0 | 7:47 |
| `python -m ruff check src tests scripts` (raw output `_r5fix_ruff.txt`) | All checks passed | 0 | <10 s |
| `git diff --check` | clean | 0 | <1 s |
| Targeted lanes pre-commit: registry+comparison+ladder (35), pipeline/job-shim/child-replay (53), tab/wizard/scans AppTests (52), full `tests/agents` (1411) | all passed | 0 | — |

Raw diff evidence: `_r5fix_diff_stat.txt` (`git show --stat fb8062f` +
`git diff dda40c9..fb8062f --stat`: 20 files, +714/−171).

## R5-FIX live browser smoke (`browser-smoke/`, port 8598) — FINAL

Re-run AFTER the fix commit on the rewritten `r5_smoke_app.py`
(content-addressed scratch `%TEMP%/ifvg_r5_smoke/bf758c8e9eff0595`, keyed
by commit `fb8062f` + source-tree digest `23c8037f052e…` + the fixture's
`pipeline_semantic_id`; `reused_validated_evidence: false` — the 16-stage
run was rebuilt fresh, and reuse now additionally requires the persisted
state to be all-terminal AND the pipeline result envelope to reload
through the verifying store). Capture metadata, viewport, and every
stage's `stage_result_id`/output artifact ids: `browser-smoke/MANIFEST.json`.

Server log `_smoke_server_r5fix.log`: **0 `Traceback` · 0
`Please replace use_container_width` · 0 Arrow serialization-fallback
messages** — the log body is the startup banner only.

1. `01_configure_no_draft_state.jpg` — dev badge, six-phase radio, honest
   no-draft Configure state; the smoke header carries the manifest caption
   (commit `fb8062fe8512` · tree `23c8037f052e` · pipeline `0472b6cacdc9`
   · scratch key `bf758c8e9eff0595`), so the capture is self-dating.
2. `02a_monitor_run_header_stages_00_09.jpg` — run picker
   (`verification_5d · attempts 1 · prepared_not_published`), copyable
   `pipeline_semantic_id`, "16/16 planned stages terminal · attempt 1",
   §30.4 operational fields, stages 00–09 including **S09: "supervised
   ladder ran 3 rungs on tier M0; identical-rows parity not evaluable
   (0 OOS rows)"** (gate finding 5's copy, live).
3. `02_monitor_16_stages_s11_blocked.jpg` — stages 06–15: **11 · Frozen
   Model-Gated Replays = ⛔ Blocked with the exact registered reason**;
   S12 resolved policy-set envelopes; S13 bootstrap identifiers; S14
   "3 cross-profile comparison result(s) persisted"; S15
   `prepared_not_published`; typed attempts table.
4. `03_monitor_ladder_panel.jpg` — the supervised-ladder panel with the
   Arrow-safe typed frame: OOS rows Int64 (0 / None for the planned GAM
   rung — no placeholder strings in numeric columns), Brier/Brier skill
   Float64 None, `no_oos_predictions` / `insufficient_class_coverage`,
   and the caption **"Identical-rows parity not evaluable — 0 OOS
   rows…"** + the S11 caption.
5. `03b_monitor_comparisons_typed_subject.jpg` — persisted S14 comparison
   results with the new **Subject column =
   `search_cross_profile_derivation_v1`** (gate finding 6's typed
   reference rendering), Jaccard as nullable Float64 (1 / None), the
   truthful `not_comparable` trade kind.
6. `04_publish_gates_activation_refused.jpg` — Publish phase: state
   caption, **all four publication gates ✓ pass**, activation control
   DISABLED with "verification-only artifacts can never activate a
   research catalog entry".

Hardening's 4-viewport/keyboard-only browser gate stays OPEN by design
(this smoke is desktop-1553×791 evidence, not a hardening waiver).
