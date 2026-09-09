# Pre-Implementation Baseline — `ifvg_prop_robust_config_search_v1`

**Recorded:** 2026-08-17 (local), before any implementation change.
**Repository:** Claude-Quant-Lab (`C:\Users\gonza\Documents\Claude-Quant-Lab`, package `alpha_lab`, distribution `alpha-signal-lab` 0.1.0, installed editable).

## Git state

- **Current branch (pre-implementation):** `platform-refactor`
- **HEAD commit:** `58cbaa2c368abfe86cdde146d9c18450739ff211` (`chore(ifvg): catalog the retest480 variant replay-chart artifact (46d0d403)`)
- **Implementation branch created:** `feature/ifvg-prop-robust-config-search-v1` (from `58cbaa2`; branch creation touched no files)
- **Stash list:** empty.

### `git status --short` (exact, pre-implementation)

```text
 M ARCHITECTURE.md
 M docs/ML_TRAINING_WORKBENCH.md
 M docs/README.md
 M docs/pipeline_state.yaml
?? IFVG_CONTEXT_FEATURES_CODEX_PROMPT.md
?? IFVG_FINAL_REVIEW_REPORT_GENERATION_CODEX_PROMPT.md
?? IFVG_FUNNEL.md
?? IFVG_LABELS.md
?? IFVG_SEARCH_LEDGER.json
?? QL-FSM-PROP-SEARCH-DASHBOARD/
?? data/ifvg_datasets/context_pair_catalog_v1.json
?? data/ifvg_datasets/context_views/
?? data/ifvg_datasets/fsm_audit/
?? data/ifvg_datasets/replay_chart/
?? data/ifvg_datasets/v2/
?? data/ifvg_datasets/v3/
?? data/ifvg_experiments/
?? data/ifvg_preparation_jobs/
?? data/ifvg_profiles.json
?? data/ifvg_visual_review/
?? docs/IFVG_CONTEXT_CAPTURE_V3.md
?? docs/IFVG_PLUGIN_DESIGN_ict_amended_ml_research_revised.md
?? ifvg_funnel_2a40b18e0b273ee0.json
?? ifvg_search_runs/
?? reports/IFVG_WORK_COMPLETION_SUMMARY.docx
?? reports/ifvg_final_review/
?? reports/ifvg_tf_variants/
```

### Pre-existing modified tracked files — overlap analysis and separation plan

Four tracked files carry **pre-existing, user-owned, uncommitted modifications** (additive documentation of the IFVG v2 correctness replay lane, "Updated: 2026-07-31"): `ARCHITECTURE.md` (+74), `docs/ML_TRAINING_WORKBENCH.md` (+28, unrelated to this plan), `docs/README.md` (+5), `docs/pipeline_state.yaml` (+44). Three of these are files the final plan also requires this implementation to update (docs-in-same-change rule).

**Separation plan (no stash/reset/discard):** the pre-existing hunks are additive and disjoint from the sections this implementation adds. At each release commit, the shared files are staged as *HEAD content + this implementation's additions only*, via `git hash-object -w` + `git update-index --cacheinfo` of a purpose-built blob; the user's uncommitted hunks remain in the working tree, uncommitted, byte-identical. The full pre-existing diff is preserved at `implementation-progress/R1/PRE_EXISTING_DIFF.patch`. `docs/DECISIONS.md` is clean at baseline and needs no separation. If a release ever needs to edit a *line region* the user's hunks also touch, the stop condition in kickoff §1.4 applies instead.

## Sibling repositories (read-only for this implementation)

- **Strategy-Core:** `C:\Users\gonza\Documents\Strategy-Core`, branch `platform-refactor`, HEAD `a4e3303179ac6a1088aecaaa3482934cf1aec4d7`, working tree **clean**. Installed as **non-editable** pinned wheel `strategy-core 0.1.0`; pyproject pin: `strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@a4e3303179ac6a1088aecaaa3482934cf1aec4d7` — **pin == sibling checkout HEAD** (the `verification.py` pin gate precondition holds).
- **Trade-Lab:** `C:\Users\gonza\Documents\Trade-Lab`, branch `platform-refactor`, HEAD `f808b9fd284bc306769c509e27144ed7340a9985` (dirty with unrelated user work). Not touched by this implementation.

## Toolchain and dependency versions

- Python 3.13.1 · pytest 9.0.2 · ruff 0.15.2
- pandas 2.3.1 · numpy 2.3.1 · pyarrow 20.0.0 · pydantic 2.12.5 · duckdb 1.4.4 · scipy 1.16.0
- streamlit 1.54.0 (pyproject floor at baseline: >=1.38) · plotly 6.5.2
- catboost 1.2.10 · scikit-learn 1.7.0 · joblib 1.5.1

## Existing baseline checks

- **Ruff:** `ruff check src tests` → `All checks passed!` (exit 0).
- **pip check:** one pre-existing dependency complaint — `async-rithmic 1.5.9 has requirement protobuf<5,>=4.25.4, but you have protobuf 6.33.6`. Unrelated to this plan; **not repaired** (kickoff §1.6).
- **pytest:** full suite `python -m pytest -q` → **1089 passed, 0 failed, 3 warnings** in 394.59s, exit 0 (tail preserved in `BASELINE_PYTEST.txt`). No known pre-existing failures.

## Decision log state

`docs/DECISIONS.md` latest entry: **D-038** (38 `## D-0NN:` sections). D-039…D-045 are unallocated — matches the plan's reservation.

## Authoritative plan-package filenames (read in full before implementation)

`QL-FSM-PROP-SEARCH-DASHBOARD/FINAL-IMPLEMENTATION-PLAN-DOCS/`: `README.md`, `FINAL_CONSISTENCY_AUDIT.md`, `IMPLEMENTATION_PLAN.md`, `FRONTEND_UX_CONTRACT.md`, `FRONTEND_RETENTION_VERIFICATION.md`, `ARCHITECTURE_MAP.md`, `CONTRACTS_AND_SCHEMAS.md`, `DELTA_TAXONOMY.md`, `ML_REGIME_CONTRACT_PLAN.md`, `TEST_MATRIX.md`, `OWNER_DECISIONS.md`, `PHASED_DELIVERY.md`, `REVISION_CHANGELOG.md`, plus superseded-provenance `FSM-PLAN-DOCUMENT.md` (not authority) and `SHA256SUMS.txt`. Kickoff: `IMPLEMENTATION-START/UPDATED_IMPLEMENTATION_KICKOFF_PROMPT_FRONTEND_COMPLETE.md`. Note: the package README's reading-order row 10 names `IMPLEMENTATION_KICKOFF_PROMPT.md`, which does not exist in the folder; the updated kickoff prompt above is the operative kickoff document.

## Notes

- The implementation-progress tree (this folder) lives under the untracked `QL-FSM-PROP-SEARCH-DASHBOARD/` folder and is **not committed** to the repository; release gate summaries list repo files and progress files separately.
- No file outside `QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/` was created or modified while producing this baseline.
