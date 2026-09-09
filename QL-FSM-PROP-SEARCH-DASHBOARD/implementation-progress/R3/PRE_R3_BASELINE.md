# R3 — Pre-implementation baseline (reconstructed)

**Honesty note.** The R3 propsim modules and their five test suites were
authored in a prior working session on 2026-08-18 that was interrupted before
any R3 progress document, verification pass, or commit was produced (the
`implementation-progress/R3/` folder existed but was empty). This baseline is
therefore RECONSTRUCTED at resumption on 2026-08-19 from R2's closing
evidence plus direct re-verification, rather than captured before the first
R3 edit. Everything below was re-verified this session.

## Repository state at R3 start (= R2 close)

| Item | Value |
|---|---|
| Branch | `feature/ifvg-prop-robust-config-search-v1` |
| HEAD (R2 commit) | `e050a25` — `R2: add child orchestration, lineage, and deltas` (parent `3546254` = R1) |
| Baseline suite (R2 final, from `R2/GATE_SUMMARY.md`) | full repo `python -m pytest -q`: **1299 passed**, 7 pre-existing-pattern warnings; lane + verifier-tab suites 216 passed; ruff all-clean |
| Pre-existing user-owned hunks | `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md` — byte-verified this session against `../R1/PRE_EXISTING_DIFF.patch` (diff-line comparison identical) BEFORE any R3 shared-doc edit |
| Strategy-Core | untouched (v1 hard constraint); no file outside Quant-Lab modified by R3 |
| Trade-Lab | untouched |

## Verification at resumption (2026-08-19, before continuing R3)

| Command | Result |
|---|---|
| `python -m pytest tests/propsim tests/agents/ifvg_search -q` | 296 passed, 4 pre-existing-pattern warnings, 7.3s |
| shared-doc worktree diff vs `R1/PRE_EXISTING_DIFF.patch` | identical (user hunks only — the prior session had written NO R3 lane appends) |
| `implementation-progress/R3/` | empty (no progress docs existed) |
| `docs/DECISIONS.md` | D-041 still reserved, not yet written |
| `docs/pipeline_state.yaml` | `R3_prop_lifecycle: implementation_status: "not_started"` (stale) |
| Orchestrator prop seam (`prop_simulator`, `evaluate_prop_gates`) | present in the worktree but UNTESTED — no test referenced either symbol |

## Inherited-at-resumption change set (authored 2026-08-18, uncommitted)

New: `src/alpha_lab/propsim/{trade_path,calendar,firm_contracts,contract_evidence,account,risk,withdrawal,adapters,portfolio,stress,simulation,prop_metrics}.py`;
`tests/propsim/{test_trade_path_fidelity,test_calendar_and_evidence,test_account_walk_rules,test_risk_and_withdrawal,test_simulation_identity}.py`.
Modified: `search/gates.py` (+`evaluate_prop_gates`), `search/orchestrator.py`
(+`prop_simulator` seam), `tests/agents/ifvg_search/test_orchestrator.py`
(skip-note wording), `tests/agents/ifvg_search/test_identities.py`
(`extra_envelope_fields` placeholder facts in the projection audit).

The July evaluation-only walker (`engine.py`, `models.py`, `presets.py`,
`bootstrap.py`, `loaders.py`, `report.py`, `cli.py`, `__init__.py`,
`__main__.py`) carries zero modifications (`git diff` empty for those paths).
