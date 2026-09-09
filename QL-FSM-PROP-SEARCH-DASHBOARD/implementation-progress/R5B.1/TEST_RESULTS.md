# R5B.1 — Test Results

All commands run from the repo root on Python 3.13.1 / pytest 9.0.2
(scikit-learn 1.7.0, catboost 1.2.10, pyarrow 20.0.0, pandas 2.3.1,
numpy 2.3.1, pydantic 2.12.5, streamlit 1.54.0 — unchanged).

## Release-final full suite (the committed tree `f3f9ac2`, git tree `b9fa5b84…`)

The worktree concurrently carried the R6.1 regime-lane files, so the
release-final runs were executed in the MAIN tree with those files
temporarily removed by a pathspec-limited `git stash push -u` (`git
write-tree` = `b9fa5b84d2745a52bd7fced2de3428b79c1fcea7` = the tree of
the R5B.1 commit, asserted before each run and before the commit) and
restored by `git stash pop` afterwards.

```text
python -m pytest -q -p no:cacheprovider                                   (environment AS-IS)
→ 2 failed, 1756 passed, 84 warnings in 580.27s (0:09:40)   exit=1

env -u POLYGON_API_KEY -u DATABENTO_API_KEY python -m pytest -q -p no:cacheprovider
→ 1758 passed, 84 warnings in 594.50s (0:09:54)             exit=0
```

(raw: `_final_pytest.txt` / `_final_pytest_keys_cleared.txt`; each file
records the tree id, the names-only environment snapshot, the start /
finish instants, and the pytest exit code)

**The two as-is failures are the PRE-EXISTING environment-dependent pair
from the baseline** (`test_connect_without_api_key_raises` /
`test_connect_without_key_raises`): this session's environment carries
`POLYGON_API_KEY` / `DATABENTO_API_KEY`, so the "no key → raise" tests
legitimately do not raise. With both variables removed the SAME tree
reports **0 failed, 1758 passed** (a full-suite run, not a two-test
re-check). The hermetic `monkeypatch.delenv` fix for those two tests is
R6.1 workstream K by plan.

**Baseline → final: 1733 → 1756 passed as-is (+23 net new tests), 0 new
failures; 1758 passed with the keys cleared.**

## Sequence

| Run | Result | Raw |
|---|---|---|
| Pre-R5B.1 baseline (= R6 release-final, tree of HEAD `179a2c9`) | 2 failed (env pair), 1733 passed, 10:02 | `../R6/_final_pytest.txt` |
| Verification tree in two DETACHED worktrees (`c9349f6`, same git tree `b9fa5b84…`) | 4 failed, 1707 passed, 47 skipped (as-is) / 2 failed, 1709 passed, 47 skipped (keys cleared), 3:12 each | `_detached_worktree_pytest_as_is.txt` / `_detached_worktree_pytest_keys_cleared.txt` |
| Release-final in the MAIN tree (R6.1 files stashed), as-is | 2 failed (env pair), 1756 passed, 9:40 | `_final_pytest.txt` |
| Release-final in the MAIN tree (R6.1 files stashed), keys cleared | **0 failed, 1758 passed**, 9:54 | `_final_pytest_keys_cleared.txt` |

The detached-worktree runs are retained as evidence of the same tree but
are NOT the headline: a worktree under `%TEMP%` lacks the untracked local
real-data fixtures (47 real-data tests skip by their own `skipif` guards)
and two launch-handler tests (`test_launch_button_freezes_and_spawns_exactly_once`,
`test_freeze_and_launch_only_in_the_button_handler`) fail there because
the launch path is location-dependent — the identical test files pass in
the main tree on the same tree content (re-run individually: 2 passed), and
neither test touches an R5B.1 file. The main-tree runs above are the
comparable, complete numbers.

## Targeted suites (release-final states, run in the mixed worktree before the commit)

| Suite | Count |
|---|---|
| `tests/agents/ifvg_search/test_mbp1_coverage_evidence.py` (the owner's six proofs, the §9.1 MBP-1 rows, multi-UTC physical denominators + channel/publisher scope, the positive-completeness compiler, scope mismatch / empty span / head-tail, evidence store round trips, protected/sealed unrepresentability, real-read clipping, synthetic-provenance refusal, the diagnostic gate matrix + fail-before-path, the evidence-manifest seam, the CLI) | **18 passed** |
| `tests/agents/ifvg_search/test_mbp1_materializer.py` (giant sequence jump never lowers coverage; declared gap → `declared_source_gap` on exactly the intersecting windows; no partition evidence → `coverage_evidence_unavailable`; superseded resolution refused; the R5B formula/window proofs) | **26 passed** |
| `tests/agents/ifvg_search/test_mbp1_schemas_and_source.py` (sequence jumps and resets are diagnostics, never gaps; schema hashes incl. `publisher_id` + `flags`) | **22 passed** |
| `tests/agents/ifvg_search/test_feature_blocks.py` (the v2 re-resolution as a second versioned event; MBP-10 guards) | 11 passed |
| `tests/agents/ifvg_search/test_identities.py` (projection audit incl. the three new pairs) | 19 passed |
| `tests/agents/ifvg_search/test_bundle_feature_view.py` | 9 passed |
| `tests/agents/ifvg_search/test_pipeline_run.py` (S05 synthetic-provenance check; MBP-1 plans under coverage v2) | 19 passed |
| `tests/agents/data_infra/ifvg/test_controlled_feature_study.py` (study ids move with the v2 block) | 7 passed |
| `tests/agents/test_ifvg_pipeline_tab.py` (the stamped coverage-policy defaults; evidence-based coverage view) | 32 passed |
| `tests/agents/test_ifvg_study_scans.py` (FUX source scans over the panel script) | 14 passed |
| Combined lane re-run (the ten suites above) | **177 passed** (1:44) |

## Lint and hygiene (on the committed tree)

```text
ruff check src tests scripts   → All checks passed!            (exit 0)
git diff --check               → clean                          (exit 0)
```

(raw: `_ruff_and_diffcheck.txt`, run in the detached verification worktree
of the same tree)

## Post-commit shared-doc verification

`stage_shared_docs.py --apply-worktree` replayed the three R5B.1 lane
transforms into the worktree; the surviving worktree diff on the four
user-owned files (`ARCHITECTURE.md`, `docs/README.md`,
`docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md`) is
content-identical to `../R1/PRE_EXISTING_DIFF.patch` — the diff-of-diffs
with the git `index` header lines removed is EMPTY
(`_surviving_shared_doc_diff.patch`).

## Source-review evidence

`R5B.1.patch` = `git format-patch --stdout 179a2c9..f3f9ac2` (260,988
bytes); sha256 in `R5B.1.patch.sha256`
(`517b548c508b4daa837e10f69668b28acfbc00fd73619530bececd96a10e0a32`).

## Browser smoke

Not re-run for R5B.1 by plan: the R6.1 browser smoke (§6.K) runs AFTER the
R6.1 commit, binds to that commit, and includes the MBP-1
coverage-evidence view (diagnostics vs evidence, open-interval facts, the
stamp table). The R5B.1 UI change is covered by the two AppTests
(`test_mbp1_panel_renders_the_stamped_coverage_policy_defaults`,
`test_mbp1_panel_autofills_and_renders_coverage_and_comparison`) and the
FUX source scans.
