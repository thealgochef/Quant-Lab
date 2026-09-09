# Pre-R6 Baseline

Recorded 2026-08-26, immediately after the R5B commit, before any R6 change.

- **Branch**: `feature/ifvg-prop-robust-config-search-v1`
- **HEAD**: `7f018f5` (R5B)
- **Worktree**: the same four user-owned dirty files (pre-existing hunks,
  content-verified byte-equal to `../R1/PRE_EXISTING_DIFF.patch` at the R5B
  close) + the long-standing untracked entries.
- **Python**: 3.13.1 · pytest 9.0.2 (unchanged).

## Baseline test run

The R5B release-final run IS the R6 baseline (same tree as HEAD):

```text
python -m pytest -q → 2 failed, 1682 passed, 84 warnings (0:11:08)
```

(raw: `../R5B/_final_pytest.txt`; the 2 failures are the pre-existing
environment-dependent provider-key pair — both pass with the API keys
cleared, `../R5B/_env_dependent_baseline_check.txt`.)

`ruff check src tests scripts` — clean at HEAD.

## R6 scope authority

`PHASED_DELIVERY.md` "Release 6 — V1 KMeans Regime Lane" + the V1 boundary
(V3 P1-6: the GMM/minibatch/spectral/Nyström expansion is post-V1);
`ML_REGIME_CONTRACT_PLAN.md` §3–§5, §9 (fixtures 2 + 4-KMeans), §10–§12;
`TEST_MATRIX.md` §3.6/§3.8 (panel row)/§3.9 (protocol-fit-promotion
separation, V1 boundary)/§3.10 (regime fit provenance);
`FRONTEND_UX_CONTRACT.md` §35 (R6); `OWNER_DECISIONS.md` 25–30 (all
`proposed_protocol_default` — nothing ratified).

## Dependencies

scikit-learn 1.7.x, scipy, joblib, numpy, pandas, pyarrow — all
pre-existing (the plan's dependency check: no new packages for V1).
