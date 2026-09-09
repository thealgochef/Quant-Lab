# R5 — Pre-implementation baseline

Recorded 2026-08-21 (UTC) before any R5 code was authored.

## Repository state

- **Branch:** `feature/ifvg-prop-robust-config-search-v1`
- **HEAD:** `6b820adff8b79643b9772d2bc314f76267e2a6e9` (= the R4 release
  commit, parent `a23893c` = R3)
- `git status --short`: the four user-owned pre-existing modified files
  (`ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` — byte-preserved through R1–R4 per
  `../R1/PRE_EXISTING_DIFF.patch`) plus the untracked user research
  artifacts and the `QL-FSM-PROP-SEARCH-DASHBOARD/` plan/progress tree.
  No file R5 plans to create exists yet.

## Environment

- Python 3.13.1 (Windows 11)
- streamlit 1.54.0 · scikit-learn 1.7.0 · catboost 1.2.10 · numpy 2.3.1 ·
  pandas 2.3.1 · scipy 1.16.0
- strategy_core installed at
  `%LOCALAPPDATA%\Programs\Python\Python313\Lib\site-packages\strategy_core`
  (pip-installed package; Strategy-Core repo treated read-only; not
  modified by any release of this program)

## Baselines (exact commands, this session)

| Command | Result | Exit |
|---|---|---|
| `python -m pytest -q` (full repo at `6b820ad`) | **1492 passed, 7 warnings in 404.82s** | 0 |
| `python -m ruff check src tests scripts` | All checks passed | 0 |
| `python -m pip check` | pre-existing known conflict only: `async-rithmic 1.5.9 has requirement protobuf<5,>=4.25.4, but you have protobuf 6.33.6` (unchanged since the R1 baseline) | 1 |

## Authoritative plan package

`QL-FSM-PROP-SEARCH-DASHBOARD\FINAL-IMPLEMENTATION-PLAN-DOCS\` (read-only):
README, IMPLEMENTATION_PLAN, FRONTEND_UX_CONTRACT,
FRONTEND_RETENTION_VERIFICATION, PHASED_DELIVERY, CONTRACTS_AND_SCHEMAS,
TEST_MATRIX, OWNER_DECISIONS, DELTA_TAXONOMY, ML_REGIME_CONTRACT_PLAN,
ARCHITECTURE_MAP, FINAL_CONSISTENCY_AUDIT, REVISION_CHANGELOG.

## R5 scope entering this release (PHASED_DELIVERY R5 + carried obligations)

New: `search/pipeline.py` (CS §7 contracts + 16-stage executors,
semantic/attempt split), `scripts/ifvg_pipeline_tab.py` +
`ifvg_pipeline_job.py` (FUX §30, FUX-PIPE-001..006),
`features/bundle_feature_view.py` (available blocks only), MBP-1
source/stage-cutoff **refusal tests only** (contracts landed R1),
`ml/` (model_protocols, logistic_model, supervised_ladder,
calibration_policies, decision_policies, drift_monitoring), ML fixture 1 +
portable-artifact relocation test.

Carried obligations closed in R5: real runner-registry executors
(DEV-R4-5), live population/funnel delta builds + persisted
comparison-contract consumption (DEV-R4-16), real
`AccountPolicySetEnvelope` construction/persistence (DEV-R4-17),
insights-store persistence (DEV-R4-7 / S14), scenario-seam bridging at the
production prop seam (DEV-R3-11).

Gate: pipeline E2E under `verification_5d` semantics on synthetic
fixtures (16 stages terminal; S11 BLOCKED with the exact reason text;
attempt-identity test), FUX §30 UX + FUX-PIPE rows, no process launch on
render/import/AppTest, ladder parity on fixture 1 (the real R1 slice's
candidate view remains blocked on the owner's fixture authorization),
`IFVG_ORDER_FLOW_MBP1_V1` stays `planned` with refusals, portable-artifact
relocation, decision-policy/schedule envelope closure (TEST_MATRIX §3.10),
capability-gated operator readiness (TEST_MATRIX §3.9).

## Program-wide first blocker (unchanged)

Owner fixture authorization (decisions 21 + R-5): no
`VerificationAuthorizationRef` exists. R1 acceptance stays
`blocked_verification_authorization`; R5 acceptance is transitively
blocked regardless of implementation status.
