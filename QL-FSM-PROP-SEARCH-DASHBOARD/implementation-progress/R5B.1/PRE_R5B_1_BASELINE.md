# Pre-R5B.1 Baseline

Recorded 2026-08-28, immediately after the R6 commit, before any R5B.1 /
R6.1 change. R5B.1 is the FIRST of the two correction-release commits
planned by `..\R6.1-CORRECTION-PLAN-DOCS-FINAL\R6.1_IMPLEMENTATION_PLAN.md`
(revision 3, owner-approved): `R5B.1` (MBP-1 coverage policy v2) lands
first, `R6.1` (regime-lane correction) second; each carries its own
evidence folder and adversarial round.

- **Branch**: `feature/ifvg-prop-robust-config-search-v1`
- **HEAD**: `179a2c9` (R6)
- **Worktree**: the same four user-owned dirty files (`ARCHITECTURE.md`,
  `docs/ML_TRAINING_WORKBENCH.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` — pre-existing hunks, content-verified
  byte-equal to `../R1/PRE_EXISTING_DIFF.patch` at the R6 close,
  `../R6/_surviving_shared_doc_diff.patch`) + the long-standing untracked
  entries (incl. the plan-document tree `QL-FSM-PROP-SEARCH-DASHBOARD/`).
- **Python**: 3.13.1 · pytest 9.0.2 · scikit-learn 1.7.0 · catboost 1.2.10
  · pyarrow 20.0.0 · pandas 2.3.1 · numpy 2.3.1 · pydantic 2.12.5 ·
  streamlit 1.54.0 (unchanged).
- **Environment**: `POLYGON_API_KEY` and `DATABENTO_API_KEY` are SET in this
  session (names only recorded; values never) — the two environment-
  dependent provider tests therefore fail as-is and pass with the keys
  cleared (`../R6/_env_dependent_check.txt`).

## Baseline test run

The R6 release-final run IS the R5B.1 baseline (same tree as HEAD `179a2c9`):

```text
python -m pytest -q -p no:cacheprovider → 2 failed, 1733 passed, 84 warnings (0:10:02)   exit=1
```

(raw: `../R6/_final_pytest.txt`; the 2 failures are the pre-existing
environment-dependent provider-key pair `test_connect_without_api_key_raises`
/ `test_connect_without_key_raises` — both pass with the API keys cleared.)

`ruff check src tests scripts` — `All checks passed!` at HEAD (re-run
2026-08-28 before the first R5B.1 edit).

## R5B.1 scope authority

`R6.1_IMPLEMENTATION_PLAN.md` §6.I (workstream I) + §9.1 rows 4–6/10 +
§9.2 `test_mbp1_coverage_evidence.py` + D12 + `OWNER_PLANNING_DECISIONS
_2026-08-28.md` Q1 (binding: withdraw "raw venue sequence jump > 1 = source
gap"; evidence-based versioned coverage policy; diagnostics only; re-mint
every affected identity; synthetic proofs; bounded real-data diagnostic under
the canonical ≤5-day fixture once `VerificationAuthorizationRef` exists; real
MBP-1 research and R5B acceptance stay blocked) + `OWNER_PLAN_REVIEW_
CORRECTIONS_2026-08-28.md` §4 (no "next unflagged row closes the interval";
documented recovery boundary or fail closed to the partition end;
dataset-condition vocabulary; coverage-calculation definition) +
`FINAL_PLAN_CORRECTIONS_2026-08-28.md` #2 / #7 (physical-partition
denominator, channel/publisher scope, trusted start, non-self-referential
`source_document_sha256`).

## Dependencies

No new package. pyarrow/pandas/numpy/pydantic all pre-existing.
