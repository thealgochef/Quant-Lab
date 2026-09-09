# R3 — Repository files touched

(Amended after the adversarial-review fix pass — the review-driven changes
are folded into the per-file notes; see `ADVERSARIAL_REVIEW_RESOLUTION.md`.)

## New source modules (13) — `src/alpha_lab/propsim/`

| Module | Content (contract) |
|---|---|
| `trade_path.py` | TradePathEvent/Artifact/Bundle payload+envelope identities, `TradePathFidelity`, scenario policies (`bar_adverse_extreme_first_v1`/`bar_favorable_extreme_first_v1`), `PropRulePathRequirement`, `PathCapability`, fail-closed `PathCapabilityReport` (CS §5.1; P0-12/P0-H) |
| `calendar.py` | `DayCountBasis`, `DurationRule`, `FirmCalendarPolicy`, `SimulatedClockPolicy`, fail-closed `UnsupportedCalendarRuleError` (CS §5.2; P0-14) |
| `firm_contracts.py` | `PhaseRules`/payout/fee rules + rule→capability matrix, `PropFirmContractPayload/Envelope`, synthetic fixture firm (CS §5.2; P0-17) |
| `contract_evidence.py` | source-document → per-field evidence → compilation → review → supersession; one-way status ladder; synthetic ≠ `first_party_verified` (CS §5.5; P0-17 + P1-A) |
| `account.py` | full-lifecycle `AccountWalk` + `PropAccountState` + totally ordered `PropAccountEventEnvelope` stream (`prop_account_event_order_v1`); `AccountWalk`≡`EvaluationWalk` one-contract parity target (CS §5.3/§5.4; P0-13) |
| `risk.py` | every risk-family sizing policy with typed skip reasons (CS §5.4) |
| `withdrawal.py` | trader withdrawal policy — separate identity from the firm contract (CS §5.2; P0-15) |
| `adapters.py` | v2 executed-trade → account-trade tick→point/cost mapping + stable order-sensitive gross stream hash (TEST_MATRIX §3.3) |
| `portfolio.py` | copied accounts on ONE common correlated path; no per-account resampling path (CS §5.4 + §9 guardrail) |
| `stress.py` | nine seeded deterministic day-block scenarios; identity rides the simulation id (CS §5.4) |
| `simulation.py` | `AccountSimulation`/`PortfolioSimulation` identities + deterministic runners; constructor-surface audit target; `path_instance_id` + `sampled_index_sequence_hash` (CS §5.4; P0-16/P0-21) |
| `prop_metrics.py` | `EvaluationFitness`/`FundedFitness`/`CashExtraction`/`PortfolioFitness`/`PayoutReliabilityVector` builders, lower-tail first, EVENT-based pass/breach detection with day ordinals (CS §5.6) |
| `search_bridge.py` | the PRODUCTION `prop_simulator` for the orchestrator seam: tables → adapters → artifacts → bundle → report → walk → vector (PHASED R3 gate closure) |

## New test suites (9)

- `tests/propsim/test_trade_path_fidelity.py` — fidelity truthfulness, scenario identities/results, capability refusal, bundle identity/completeness
- `tests/propsim/test_calendar_and_evidence.py` — day-count bases through weekend/halt fixtures, unsupported-basis fail-closed, contract-evidence compilation + status ladder + supersession
- `tests/propsim/test_account_walk_rules.py` — the §16.4 rule-by-rule deterministic fixtures + `AccountWalk`≡`EvaluationWalk` parity + envelope total order
- `tests/propsim/test_risk_and_withdrawal.py` — sizing families + skip reasons; withdrawal/replacement policy identity propagation
- `tests/propsim/test_simulation_identity.py` — simulation identity completeness, constructor-surface audit, bootstrap path semantics, common correlated path, stress determinism
- `tests/agents/ifvg_search/test_prop_seam.py` — `evaluate_prop_gates` rows + orchestrator seam (ALL-legs rule, worst-firm merge representative-flip, simulator scoping, exclusion paths, exception containment, no-simulator prop-objective exclusion) [authored at resumption, 2026-08-19]
- `tests/propsim/test_synthetic_contract_e2e.py` — THE gate E2E: compile → `synthetic_fixture_verified` (capped) → production simulator on the R1 baseline stream → vector → gates → `run_search` frontier champion
- `tests/propsim/test_prop_metrics.py` — event-based builder fixtures (pass-then-funded-breach, phase-scoped breach/expiry, horizon windowing)
- `tests/propsim/test_adapters.py` — tick→point/cost mapping, canonical ordering, sequence-pinning hash rows (refusal on missing columns)

## Modified source (3)

- `src/alpha_lab/agents/data_infra/ifvg/search/gates.py` — `evaluate_prop_gates` over a `PayoutReliabilityVector`-shaped vector (duck-typed; no propsim import in the search lane)
- `src/alpha_lab/agents/data_infra/ifvg/search/orchestrator.py` — `prop_simulator` seam (ALL-legs feasibility, worst-firm merge) + the deferred prop-owned-objective partition at the strategy stage (defect found by the seam tests at resumption) + per-child exception containment with sanitized reasons (review R2-F5)
- `src/alpha_lab/agents/data_infra/ifvg/search/identities.py` — `registered_identity_pairs()` imports the seven propsim contract modules so the CANONICAL audit enumerates the R3 pairs (review R2-F2)

## Modified tests (2)

- `tests/agents/ifvg_search/test_orchestrator.py` — `_charter()` gains `pareto_objectives`/`lexicographic_tie_breaks`/`prop_feasibility_gates` kwargs; skip-note assertion follows the new no-simulator wording
- `tests/agents/ifvg_search/test_identities.py` — projection audit supplies placeholder post-materialization facts for pairs declaring `extra_envelope_fields`; force-imports the propsim identity modules for standalone runs; required-name superset extended with the nine propsim pairs

## Shared docs (lane-staged: commit = HEAD + R3 transforms; user hunks preserved in worktree)

- `ARCHITECTURE.md` — R3 additions block (appended)
- `docs/README.md` — lane paragraph extended to R3; decision list moves D-041 out of reserved
- `docs/pipeline_state.yaml` — `R3_prop_lifecycle.implementation_status: "complete"`
- `docs/DECISIONS.md` — D-041 authored; reservation note updated (no user hunks in this file; staged normally)

## Implementation-progress files (R3/)

`PRE_R3_BASELINE.md`, `FILES_TOUCHED.md` (this file), `DEVIATIONS.md`,
`TEST_RESULTS.md`, `ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md`,
`ADVERSARIAL_REVIEW_RESOLUTION.md`, `GATE_SUMMARY.md`, `stage_shared_docs.py`
— plus `../DECISIONS_TAKEN.md` R3 entries 17–21.
