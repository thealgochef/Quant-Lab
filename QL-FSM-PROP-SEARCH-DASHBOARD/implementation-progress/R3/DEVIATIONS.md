# R3 — Deviations and scoping notes

- **DEV-R3-1 — Interrupted authoring session; reconstructed baseline.** The
  R3 modules/tests were authored 2026-08-18 in a session that ended before
  any progress doc, verification pass, or commit. Work resumed 2026-08-19:
  baseline reconstructed (`PRE_R3_BASELINE.md`), user-hunk byte-identity
  re-verified BEFORE any R3 shared-doc edit, then the release was completed
  under the normal gate discipline. No plan content changed.

- **DEV-R3-2 — Orchestrator deferred-objective defect, found and fixed at
  resumption.** The inherited `prop_simulator` seam excluded every child at
  the strategy-gate stage whenever the charter's pareto objectives included a
  prop metric (`getattr(StrategyMetrics, prop_metric)` → None → "objective
  unavailable"), so the simulator could never run and prop metrics could
  never reach the frontier. Fixed by partitioning pareto objectives at the
  strategy stage: strategy-owned objectives (attributes of
  `StrategyMetrics`) must resolve there; prop-owned objectives defer to the
  prop phase ONLY when a simulator is wired, else the child is explicitly
  excluded. Caught by the new `test_prop_seam.py` before any commit
  (`DECISIONS_TAKEN.md` #17).

- **DEV-R3-3 — `evaluate_prop_gates` is duck-typed in the search lane.** The
  search lane does not import `alpha_lab.propsim`; the gate evaluator reads
  the vector's fields structurally. The cross-package field contract is
  proven by `test_prop_seam.py`, which feeds the REAL
  `PayoutReliabilityVector` through the evaluator and the orchestrator seam.
  Chosen to keep the search lane importable without the prop stack.

- **DEV-R3-4 — Projection-audit self-containment.** Identity pairs register
  at module import; `test_identities.py` standalone previously audited only
  the search-lane pairs (full-session collection imported the propsim tests
  and hence the pairs, so full runs were covered — but standalone runs were
  silently narrower). Fixed by force-importing the seven propsim identity
  modules in `test_identities.py`. The audit's `extra_envelope_fields`
  handling supplies placeholder post-materialization facts and still enforces
  payload → id → envelope → reload determinism (`DECISIONS_TAKEN.md` #21).

- **DEV-R3-5 — Mechanical ruff import-sort fixes** applied to
  `tests/propsim/test_account_walk_rules.py` and 4 sibling authored files
  (`ruff --fix`, I001 only; no semantic change).

- **DEV-R3-6 — Bootstrap protocol ids are a parametric FAMILY.** The
  sampled-path horizon is encoded in the registered id
  (`day_block_bootstrap_h<horizon>_v1`, validated by pattern) rather than a
  fixed enumeration, so the horizon rides the simulation identity with no
  registry file to grow. Any out-of-family id is refused.

- **DEV-R3-7 — Breach-observation compatibility is one-directional.** A firm
  requiring unrealized equity observation REFUSES a realized-only walk
  (under-observation = untruthful breach risk). The reverse — walking a MORE
  conservative unrealized scenario than the firm mandates — is permitted and
  labeled by the breach mode itself (it is also what keeps the
  `AccountWalk`≡`EvaluationWalk` parity surface runnable). Recorded as
  `DECISIONS_TAKEN.md` #23.

- **DEV-R3-8 — `FirmCalendarPolicy.holidays` is an additive extension** to
  the CS §5.2 field set (needed by the business-day arithmetic fixtures);
  documented here per the schema-deviation rule.

- **DEV-R3-9 — `manifest_payload_sha256` is a derived placeholder in R3.**
  No artifact store exists yet, so the envelope field is a pure function of
  the payload id and attests no materialized bytes (documented at the
  constructor). Real manifests land with the store wiring (R4/R5); until
  then the two-factor pin adds no provenance beyond the id (reviewer 2 F10).

- **DEV-R3-10 — `payout_processing` is implement-or-refuse.** The walk
  refuses any nonzero payout-processing delay (`UnsupportedFirmRuleError`)
  instead of silently settling payouts instantly; implementing settlement
  delay is deferred until a firm contract that needs it is compiled.

- **DEV-R3-11 — The production bridge runs `historical_closed_trade`.** The
  seam's default chain builds closed-trade artifacts (honest base fidelity);
  firms whose rules demand market-price chronology are refused by the
  capability report and excluded per child. Scenario-mode bridging (assumed
  intrabar artifacts from bar evidence at the seam) is future wiring; the
  scenario machinery itself is fully implemented and walk-proven.
