# R3 — Test results (final, post-adversarial-fix)

All commands run from the repo root, 2026-08-19.

| Command | Result | Exit | Time |
|---|---|---|---|
| `python -m pytest tests/propsim tests/agents/ifvg_search -q` (resumption baseline, before the fix pass) | 296 passed, 4 warnings | 0 | 7.3s |
| `python -m pytest -q` (full repo, pre-review checkpoint: seam fix + seam tests in, review fixes not yet) | 1341 passed, 7 warnings | 0 | 9:07 |
| `python -m pytest tests/propsim tests/agents/ifvg_search -q` (post-review-fix lanes) | **334 passed**, 4 pre-existing-pattern warnings | 0 | 8.0s |
| `python -m pytest -q` (full repo, FINAL) | **1372 passed** (1299 R2-final + 73 net new R3), 7 pre-existing-pattern warnings | 0 | 8:24 |
| `python -m ruff check src tests scripts/ifvg_search_job.py scripts/ifvg_verifier_tab.py QL-.../R3/stage_shared_docs.py` (final) | All checks passed | 0 | <10s |
| `git diff --check` | clean | 0 | <1s |
| `python -c "yaml.safe_load(docs/pipeline_state.yaml)"` | parses | 0 | <1s |

## Suite composition (new/renewed in R3)

- `tests/propsim/test_trade_path_fidelity.py` — 8 (fidelity truthfulness, scenario identities, per-trade capability refusal incl. the mixed-bundle fixture, bundle identity, positive path-event linkage, package-wide wording scan)
- `tests/propsim/test_calendar_and_evidence.py` — 8 (day-count bases, bootstrap fail-closed, forward-only clock, compilation + ladder + document-set binding, supersession retirement, firm/scenario separation)
- `tests/propsim/test_account_walk_rules.py` — 23 (parity 3×2, DLL, trail styles, payouts/fees incl. RECURRING on trading-day and calendar-month bases, post-payout rules, min/winning/consistency, replacement, expiration, funded consistency, payout-processing refusal, unrealized-required refusal, intraday ratchets, open-ts, total order, path-event citation)
- `tests/propsim/test_risk_and_withdrawal.py` — 7 (all sizing families + all 11 skip reasons, withdrawal behaviors incl. max_allowed_each_period, P0-15 two-run payout-stream diff)
- `tests/propsim/test_simulation_identity.py` — 11 (identity sensitivity, constructor-surface audit with zero guard exemptions, runner revalidation, supersession refusal, bootstrap protocol family, mode refusals, SCENARIO RESULT DIVERGENCE, scenario-policy consistency, stress, portfolio common path + leg binding + bootstrap shared sequence)
- `tests/propsim/test_synthetic_contract_e2e.py` — 2 (THE PHASED R3 gate E2E; production simulator through `run_search` to a prop-metric frontier champion)
- `tests/propsim/test_prop_metrics.py` — 3 (event-based pass/breach/expiry scoping, horizon windowing)
- `tests/propsim/test_adapters.py` — 2 (tick→point/cost mapping, sequence-pinning hash + refusals)
- `tests/agents/ifvg_search/test_prop_seam.py` — 9 (prop-gate rows, worst-firm flip, ALL-legs, scoping, exclusions, exception containment, no-simulator exclusion)
- July propsim suites (evaluation-only walker) — 27, untouched and passing
- `tests/agents/ifvg_search` — 218 (R2's 216 + charter kwargs ripple + extended projection audit covering the nine propsim pairs)

No test reads real data; every write is under pytest tmp; all randomness is
seeded (`np.random.default_rng`); zero wall-clock dependence.
