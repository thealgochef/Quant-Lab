# Pre-R6.1 Baseline

Recorded 2026-08-29, immediately after the R5B.1 commit, before the R6.1
release-scoped commit. R6.1 is the SECOND of the two correction-release
commits planned by `../R6.1-CORRECTION-PLAN-DOCS-FINAL/R6.1_IMPLEMENTATION_PLAN.md`
(revision 3, owner-approved): `R5B.1` (MBP-1 coverage policy v2) landed
first as `f3f9ac2`; `R6.1` (the regime-lane correction) lands second with its
own evidence folder (this) and its own adversarial round.

- **Branch**: `feature/ifvg-prop-robust-config-search-v1`
- **HEAD**: `f3f9ac2` (R5B.1; parent `179a2c9` = R6)
- **Worktree at the R5B.1 commit**: the same four user-owned dirty files
  (`ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` — pre-existing hunks, content-verified
  byte-equal to `../R1/PRE_EXISTING_DIFF.patch` after the R5B.1
  `--apply-worktree` replay: `../R5B.1/_surviving_shared_doc_diff.patch`) +
  the long-standing untracked entries (incl. the plan-document tree) + the
  R6.1 working set that was authored concurrently with R5B.1 (stashed for
  the R5B.1 release-final runs and restored afterwards).
- **Python**: 3.13.1 · pytest 9.0.2 · scikit-learn 1.7.0 · catboost 1.2.10
  · pyarrow 20.0.0 · pandas 2.3.1 · numpy 2.3.1 · pydantic 2.12.5 ·
  streamlit 1.54.0 · threadpoolctl 3.6.0 (unchanged; no new package).
- **Environment**: `POLYGON_API_KEY` and `DATABENTO_API_KEY` are SET in this
  session (names only recorded; values never). R6.1 makes the two
  environment-dependent provider tests hermetic (`monkeypatch.delenv`).

## Baseline test run

The R5B.1 release-final runs ARE the R6.1 baseline (the tree of HEAD
`f3f9ac2`, run in the main tree with the R6.1 working set stashed):

```text
python -m pytest -q -p no:cacheprovider                                   (as-is)
→ 2 failed, 1756 passed, 84 warnings in 580.27s (0:09:40)   exit=1
env -u POLYGON_API_KEY -u DATABENTO_API_KEY python -m pytest -q -p no:cacheprovider
→ 1758 passed, 84 warnings in 594.50s (0:09:54)             exit=0
```

(raw: `../R5B.1/_final_pytest.txt` / `../R5B.1/_final_pytest_keys_cleared.txt`;
the 2 as-is failures are the pre-existing environment-dependent provider-key
pair, hermetic after R6.1.)

`ruff check src tests scripts` — `All checks passed!` at HEAD
(`../R5B.1/_ruff_and_diffcheck.txt`).

## R6.1 scope authority

`R6.1_IMPLEMENTATION_PLAN.md` §4 (closes the audit's seven items + the
additional gaps + owner decisions 1–5 + the plan-review corrections + the
final contract-closure corrections), §5 (D1–D15), §6.A–H/J/K/L (workstreams;
§6.I landed as R5B.1), §7 (identity consequences), §8 (stamped defaults),
§9 (tests), §11 (verification), §12 (open blockers) +
`OWNER_PLANNING_DECISIONS_2026-08-28.md` Q2–Q4 + `OWNER_PLAN_REVIEW_CORRECTIONS_2026-08-28.md`
1–7 + `FINAL_PLAN_CORRECTIONS_2026-08-28.md` 1, 3–6, 8.

## Dependencies

No new package. `threadpoolctl` (a scikit-learn dependency already
installed) is used to pin the regime kernel to one BLAS/OpenMP thread.
