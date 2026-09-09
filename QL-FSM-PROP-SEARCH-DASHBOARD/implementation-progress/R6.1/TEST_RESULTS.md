# R6.1 — Test Results

All commands run from the repo root on Python 3.13.1 / pytest 9.0.2
(scikit-learn 1.7.0, catboost 1.2.10, pyarrow 20.0.0, pandas 2.3.1,
numpy 2.3.1, pydantic 2.12.5, streamlit 1.54.0 — unchanged; no new package
installed or declared — `threadpoolctl` is imported lazily, DEV-R6.1-13).

## Release-final full suite (the working tree that became commit `6c0b60a`)

The R6.1 commit is path-scoped to the file list in `FILES_TOUCHED.md`; every
R6.1 source / test / script file was present in the working tree for both
runs (the untracked new modules included), so the runs below exercise exactly
the committed code. The only worktree content outside the commit is the
user-owned doc hunks (`ARCHITECTURE.md`, `docs/README.md`,
`docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md` — no test reads
them) and the pre-existing untracked local data / report files that enable
the real-data `skipif` tests.

```text
python -m pytest -q -p no:cacheprovider                                   (environment AS-IS)
→ 1931 passed, 386 warnings in 1077.32s (0:17:57)            exit=0

env -u POLYGON_API_KEY -u DATABENTO_API_KEY python -m pytest -q -p no:cacheprovider
→ 1931 passed, 386 warnings in 1089.03s (0:18:09)            exit=0
```

(raw: `_final_pytest.txt` / `_final_pytest_keys_cleared.txt`; each records
HEAD at the time of the run, the presence of the two provider keys, the
start / finish instants, and the pytest exit code)

**Both runs: 0 failed.** The two provider-key tests that failed as-is on
every earlier release (`test_connect_without_api_key_raises` /
`test_connect_without_key_raises`) are hermetic since R6.1 workstream K
(`monkeypatch.delenv`), so the as-is and keys-cleared counts coincide.

**Baseline → final: 1758 (R5B.1, keys cleared) → 1931 passed (+173 net new
tests), 0 failures in either environment.**

## Sequence

| Run | Result | Raw |
|---|---|---|
| Pre-R6.1 baseline (= R5B.1 release-final, tree `b9fa5b84…`) | 2 failed (env pair) / 1756 passed as-is; 1758 passed keys cleared | `../R5B.1/_final_pytest*.txt` |
| Midpoint (all workstreams landed, BEFORE the adversarial-fix round) | **1896 passed**, 0 failed, 16:45 | `_midpoint_pytest.txt` |
| Release-final, as-is (the `6c0b60a` content) | **1931 passed**, 0 failed, 17:57 | `_final_pytest.txt` |
| Release-final, keys cleared (the `6c0b60a` content) | **1931 passed**, 0 failed, 18:09 | `_final_pytest_keys_cleared.txt` |

Midpoint → final: +35 tests from the adversarial-fix round (the supersession
crash / tamper / provenance tests, the D6 / namespace / MODEL_FEATURE /
decided-at tests, the report-gate tests, the event-regime summary tests, the
streaming-writer / producer tests, the deep-immutability test, the
distinct-view pairing test, the executor-ref tests, the S05 chart-binding
test, the S08 schedule test, the S02 reproduction test, the parametrized
zero-fitting runs, the supervised panel E2E, the S07 refusal test, the UI
readiness AppTest, the missing-dependency test).

## Targeted suites (release-final states, mixed worktree before the commit)

| Suite | Count |
|---|---|
| `tests/agents/ifvg_search/test_pipeline_regime.py` (16 terminal states for the descriptive candidate / descriptive panel / supervised candidate / supervised panel studies; second-attempt reuse with identical stage-result ids; S05 chart binding + the foreign-chart refusal; S08 schedule from the candidate view; S02 verified reuse by reproduction; S07 / label-policy refusal; zero fitting on all three shapes) | **19 passed** (4:36) |
| `tests/agents/data_infra/ifvg/test_regime_stratification.py` (five classes; the event-regime summary schema / aggregates / determinism / budget / tamper; `delivered_by`; D15 precedence + own trading day) | 15 passed |
| `tests/agents/data_infra/ifvg/test_regime_stratification_gate.py` (owner authorization under the run scope; structural gate re-derived from the loaded assessment; superseded owner refused; synthetic scope confined) | 6 passed |
| `tests/agents/ifvg_search/test_owner_decisions.py` (decisions 25/28/29/30 value verification; hash-chained supersession: crash, deleted / edited / headless log, weaker provenance, forged line, idempotent re-persist; stale lock; namespace confinement) | governance total **53 passed** together with |
| `tests/agents/data_infra/ifvg/test_regime_store.py` + `tests/agents/test_ifvg_regime_promotion_cli.py` (D6 at persistence; MODEL_FEATURE unpersistable; derived `decided_at` reproducing S10; monotone chain; sanitized CLI exits) | (in the 53 above) |
| Governance + stratification + `tests/propsim` + supervised studies + service + contracts, run together | **255 passed** (1:29) |
| `tests/propsim/test_account_event_detail.py` (streaming writer determinism; one block at a time; producer bookkeeping refusals; reuse with a producer; `none_v0` never widened) | 13 passed |
| `tests/agents/data_infra/ifvg/test_regime_supervised_studies.py` (three-rung bundle ladder on both arms; distinct-view pairing; deep-immutable summaries with unchanged ids) | 10 passed (1:18) |
| `tests/agents/test_ifvg_pipeline_tab.py` (regime surfaces incl. the S14 `delivered_by` rows and the pre-persist readiness refusal) | **60 passed** |
| `tests/agents/ifvg_search/test_context_bar_panel.py` + `test_regime_observation_source.py` + UI regime/launch/preview | 44 passed |
| `test_pipeline_run.py` + `test_identities.py` + `test_regime_supervised_studies.py` + `test_regime_fold_features.py` + `test_regime_oos_assignment.py` | 39 passed |

## Lint and hygiene (the committed content)

```text
ruff check src tests scripts   → All checks passed!            (exit 0)
git diff --check               → clean (tracked files)          (exit 0)
git add -N <46 new files>; git diff --check → one "new blank line at EOF"
  (tests/agents/data_infra/ifvg/ml_fixtures/synthetic_context_panel.py) FIXED,
  re-check clean                                                (exit 0)
```

(raw: `_ruff_and_diffcheck.txt`)

## Post-commit shared-doc verification

`stage_shared_docs.py --apply-worktree` replayed the three R6.1 lane
transforms into the worktree; the surviving worktree diff on the four
user-owned files (`ARCHITECTURE.md`, `docs/README.md`,
`docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md`) is
content-identical to `../R1/PRE_EXISTING_DIFF.patch` — the diff-of-diffs
with the git `index` header lines removed is EMPTY
(`_surviving_shared_doc_diff.patch`, 4 files, 12,348 bytes).

## Source-review evidence

`R6.1.patch` = `git format-patch --stdout f3f9ac2..6c0b60a` (1,310,228
bytes); sha256 in `R6.1.patch.sha256`
(`6a166afe752350d6a432b9f1c635cf6756de184330618898047c6834d06bfe7a`).

## Browser smoke (plan §6.K; bound to the commit)

`r61_smoke_app.py` served three completed synthetic 16-stage runs under
`%TEMP%\ifvg_r61_smoke\d8dae9697cf5b3bb` (descriptive candidate, descriptive
5 m panel, model-bearing candidate frozen to a synthetic owner decision +
FEATURE_ELIGIBLE promotion — the two-pass workflow) from
`python -m streamlit run … --server.port 8611 --server.headless true
--logger.level=info` with stdout+stderr captured
(`browser-smoke/_smoke_server.log`: 26 `[r61-smoke]` marker lines —
authority / prepare / three `run_pipeline` phases / manifest written / the
served views — **0 Traceback, 0 DeprecationWarning**). 25 screenshots were
captured through Chrome (viewport 1440×1100) across the Configure / Preview
/ Monitor phases and the Regime Lane of all three runs
(`browser-smoke/*.jpg`, described in `MANIFEST.json`), including the S14
table that renders the modeled classes as "delivered by S09c" (F6), the
owner-decision artifact view with its synthetic-scope note, the panel-grain
model card with typed nulls and consecutive-bar transitions, and the MBP-1
coverage-evidence expander. `MANIFEST.json` (schema v2, built by
`build_browser_manifest.py`) binds every file's sha256 / bytes, the commit
`6c0b60a…`, `commit_is_head_at_capture: true`, `worktree_dirty_in_scope:
false`, the committed-tree digest over the harness digest globs, the server
log counts and the bound evidence ids;
`python verify_browser_manifest.py browser-smoke/MANIFEST.json` →
**OK: manifest v2 bound to commit 6c0b60a1ddde, 26 file(s), 25
screenshot(s), committed-tree digest verified** (exit 0). No launch,
promote, rank, or retrain control exists on the served surface; the repo's
`data/` tree gained zero files.
