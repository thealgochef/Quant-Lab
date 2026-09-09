# R3 — Adversarial review resolution

Dedup map: R1-F1+R2-F3 (scenario inertness) · R1-F4+R2-F1 (bootstrap-length
identity) · R1-F3+R2-F14 (calendar bases) · R1-F6+R2-F4 (runner trust) ·
R1-F10+R2-F9 (Literals) · R1-F13+R2-F12 (stream hash). Net: **4 blockers,
7 majors, 11 minors — ALL RESOLVED** (fixed in code/tests; none dismissed).
Post-fix verification: `tests/propsim` + `tests/agents/ifvg_search` = 334
passed; full repo green (`TEST_RESULTS.md`); ruff all-clean.

## Blockers

1. **Scenario policies inert (R1-F1 / R2-F3)** — FIXED. The walk gains
   `unrealized_favorable_first` (`account.py::BREACH_MODES`; favorable-first
   raises the intraday-trail peak from MFE BEFORE the MAE test); the runner
   maps the mode from the payload's policy id
   (`simulation.py::_SCENARIO_BREACH_MODES` + `_breach_mode_for`), validates
   the id against `INTRABAR_SCENARIO_POLICIES`, and refuses scenario
   artifacts generated under a different policy than the payload claims
   (`_revalidate_identities`). The artifact builder's "adverse" is now
   POSITION-adverse (`trade_path.py::build_assumed_intrabar_artifact` gains
   `trade_direction`; long→low/short→high), matching the walk's MAE-first
   semantics under the same policy id. Proof:
   `test_simulation_identity.py::test_two_scenario_policies_produce_two_identities_and_two_results`
   (favorable-first → `breached_out`; adverse-first survives; distinct ids)
   and `test_scenario_artifacts_must_match_the_payload_policy`.
2. **Gate E2E missing (R1-F2)** — FIXED. New production bridge
   `propsim/search_bridge.py` (`FirmSimulationSpec` + `make_prop_simulator`:
   executed-trade tables → adapters → closed-trade artifacts → bundle →
   capability report → `run_account_simulation` → reliability vector) and the
   new `tests/propsim/test_synthetic_contract_e2e.py`: compiles the synthetic
   contract from synthetic documents (PASSED compilation; capped at
   `synthetic_fixture_verified` inside the same chain), simulates the R1
   baseline synthetic stream through the production chain, feeds
   `evaluate_prop_gates` (6 checks), and wires the whole thing through
   `run_search` to a frontier champion on `expected_net_payout_90d`.
   Adapters and both metric builders now have direct tests
   (`test_adapters.py`, `test_prop_metrics.py`) and production callers.
3. **Calendar bases silently dropped (R1-F3 / R2-F14)** — FIXED.
   `AccountWalk._validate_firm_rules` refuses ANY declared duration rule
   whose basis the active clock cannot represent (walk-level P0-14, at
   construction); recurring fees route through
   `SimulatedClock.elapsed_since` (trading-day semantics preserved exactly;
   calendar-month works under the historical clock); `max_eval_days` /
   `min_days` non-trading bases route through the clock; the withdrawal
   cadence elapses in ITS OWN basis (`plan_withdrawal_request` param renamed
   `elapsed_since_last_payout`; account.py computes it via the clock). Dead
   assertion deleted. Proofs:
   `test_account_walk_rules.py::test_recurring_fee_charges_on_its_trading_day_period`,
   `::test_calendar_month_recurring_fee_works_under_the_historical_clock`,
   `::test_calendar_basis_rules_fail_closed_under_the_bootstrap_clock`.
4. **Bootstrap length outside the identity (R1-F4 / R2-F1)** — FIXED.
   `max_days_per_path` DELETED from the runner; the horizon is encoded in the
   registered protocol family `day_block_bootstrap_h<horizon>_v1`
   (`simulation.py::bootstrap_horizon_for`), REQUIRED for bootstrap mode in
   `_require_mode_support`, and therefore rides the payload identity. The
   constructor-surface audit no longer carries any guard exemption and
   asserts the old argument is gone. Proof:
   `test_bootstrap_paths_are_unique_instances_with_stored_sequence_hashes`
   (h3 vs h40 → different ids AND different draw sequences; unregistered
   family refused; bootstrap without a protocol id refused).

## Majors

5. **Canonical audit enumeration (R2-F2)** — FIXED. `identities.py::
   registered_identity_pairs()` imports the seven propsim contract modules;
   the audit's required-name superset now names all nine propsim pairs (plus
   the standalone force-imports in `test_identities.py` for standalone runs).
6. **Mixed-fidelity bundle hole (R1-F5)** — FIXED. `evaluate_path_capabilities`
   evaluates PER TRADE from artifact-derived fidelities (the caller-asserted
   `bundle_fidelities` side channel is gone): any trade without an accepted
   class → `unsupported` with the trade ids in the reasons; any accepted
   scenario-only trade degrades the whole rule to at most `scenario_only`.
   Proof: `test_trade_path_fidelity.py::test_mixed_fidelity_bundle_degrades_per_trade_not_per_bundle`.
7. **Runner trust (R1-F6 / R2-F4)** — FIXED. `run_account_simulation` accepts
   the bundle ENVELOPE + artifacts and `_revalidate_identities` fails closed
   on: policy-set/firm/risk/withdrawal hash mismatches, capability-report
   hash + bundle binding, report rule coverage ⊇ firm rules (an empty report
   is refused), artifact-set ↔ bundle equality, bundle core/stream-hash vs
   payload, scenario-policy consistency, and day-block trades outside the
   bundle. Fidelities + path-event ids are DERIVED from artifacts. The old
   exploiting fixtures were rewritten to real evidence. Proof:
   `test_runner_revalidates_every_pinned_artifact`.
8. **Supersession retirement (R1-F7)** — FIXED. `PropContractSupersession`
   gains `effective_at`; `assert_contract_not_superseded` +
   `SupersededContractError` added and wired into the runner (`supersessions`
   refusal registry). Proofs: `test_calendar_and_evidence.py::
   test_supersession_retires_the_prior_contract_id`,
   `test_simulation_identity.py::test_superseded_contract_is_refused_for_new_simulations`.
9. **Inert firm fields (R1-F8)** — FIXED. `account_expiration` implemented
   (per-account, trading-day + clock bases; reason `account_expiration`);
   funded-phase `consistency_pct` gates payout eligibility (phase-entry
   balance denominator); `payout_processing` delays REFUSE at construction
   (`UnsupportedFirmRuleError` — implement-or-refuse, never silent); a firm
   requiring unrealized breach observation refuses a realized-only walk
   (under-observation direction; the more-conservative unrealized scenario of
   a realized-only firm stays permitted and is labeled by its mode —
   engineering default recorded in `DECISIONS_TAKEN.md` #23). Proofs: four
   new walk-rule tests.
10. **Metric builders wrong/untested (R1-F9)** — FIXED. `prop_metrics.py`
    rewritten event-based: passes detected from `phase_transition` reasons
    (`evaluation_passed*`), days-to-pass = the event's played-day ordinal,
    breach probability phase-scoped via the envelope's `account_phase`,
    expiry from `to_phase=EXPIRED`, vector breach windowed by event day
    ordinal ≤ horizon. Proofs: new `test_prop_metrics.py` (includes the exact
    pass-then-funded-breach repro).
11. **Seam exception containment (R2-F5)** — FIXED. The orchestrator wraps
    the simulator per child: `FailureReason.REPLAY` + sanitized explanation,
    child popped from the feasible set, search completes; the reused-child
    `result is None` contract is documented at the seam and the production
    bridge refuses it explicitly. Proof: `test_prop_seam.py::
    test_prop_simulator_exception_is_contained_per_child` (sanitization
    asserted) and the bridge's REUSED refusal in the E2E.

## Minors

12. **Literals (R1-F10/R2-F9)** — FIXED (`risk.instrument`,
    `copied_market_path_policy` restored to spec Literals). `holidays` on
    `FirmCalendarPolicy` recorded as a documented additive extension
    (DEV-R3-8). `effective_at` covered under #8.
13. **Intraday ratchet events (R1-F11)** — FIXED. `_raise_peak` emits
    `threshold_ratchet` with the source trade + mid-day ts; the EOD block no
    longer double-emits. Proof: `test_intraday_trail_raises_emit_ratchet_events_with_trade_links`.
14. **Coverage gaps (R1-F12)** — FIXED: recurring-fee fixtures (#3);
    `daily_stop_r_reached`/`zero_contracts_configured`/`capped_to_zero` +
    `max_allowed_each_period` (`test_risk_and_withdrawal.py`); P0-15 two-run
    payout-stream diff (same file); portfolio bootstrap runner
    (`portfolio.py::run_portfolio_bootstrap`, ONE draw per path shared by
    every copy) + leg-pin binding + shared-sequence determinism tests;
    positive path-event linkage test; wording scan widened to all 13
    lifecycle modules; the no-simulator deferred-objective branch tested
    (`test_prop_objective_without_simulator_excludes_explicitly`).
15. **Stream hash (R1-F13/R2-F12)** — FIXED. Required-column refusal, named
    column→value records in the digest, honest docstring (sequence-pinning,
    canonical order). Proof: `test_adapters.py`.
16. **Open-ts sentinel (R1-F14)** — FIXED. The initial evaluation fee defers
    to the first played day (`T00:00:00+00:00`); replacements keep their real
    ts; every `event_ts_utc` parses. Proof:
    `test_initial_account_open_timestamp_is_never_a_sentinel`.
17. **Ladder document binding (R2-F6)** — FIXED. `advance_verification_status`
    refuses a compilation citing documents not presented. Proof:
    `test_status_advancement_binds_the_compilations_document_set`.
18. **Live state exposure (R2-F7)** — FIXED. `result()` returns a defensive
    copy of the state.
19. **Portfolio dead check / id collisions (R2-F8)** — FIXED. Real
    hash-binding per leg (`canonical_contract_sha256(policy_set) ==
    leg.account_policy_set_id`); `AccountWalk` gains `account_namespace`
    (leg#copy), included in account ids AND event-id fields. Proofs:
    portfolio tests (distinct account/event ids across copies; bad pin
    refused).
20. **Pseudo-manifest (R2-F10)** — RESOLVED AS DOCUMENTED LIMITATION. No
    artifact store exists in R3; the field is now explicitly documented as a
    derived placeholder attesting no bytes (comment at `_artifact_envelope`
    + DEV-R3-9). Real manifests land with the store wiring (R4/R5).
21. **Empty-bundle fabricated core id (R2-F11)** — FIXED. Empty bundles are
    refused; the `"0"*64` fallback is gone. Proof: mixed-bundle test.
22. **Dead assertion (R2-F13)** — FIXED (deleted).

## Residuals (explicit, none blocking)

- `historical_ordered_event_replay` remains structurally unreachable until
  real ordered sources exist (R5B+) — by design, tested as a refusal.
- The bridge's closed-trade artifact entry/exit prices are evidence derived
  from tick columns (entry + realized); scenario-mode bridges (OHLC/assumed
  artifacts from bar data) land when bar evidence is plumbed to the seam.
- `PayoutReliabilityVector` fields other than the three `*_90d` fields are
  whole-walk measurements (documented; unchanged from the authored design).
- The R2-recorded acceptance-time slice requirements and R4 obligations
  remain tracked in `R2/GATE_SUMMARY.md` / `R2/DEVIATIONS.md`.
