# R5 — Deviations and scoping notes

Each entry records a deliberate implementation deviation from the plan
text, its trigger, and why it does not invent a third design. Every
fail-closed rule is preserved or strengthened. (Engineering defaults are in
`../DECISIONS_TAKEN.md` §R5.)

- **DEV-R5-1 — Additive modules beyond PHASED's R5 file list.**
  `ifvg/search/executors.py` (the real runner-entry factories the registry
  names — PHASED assigns the registration but no home module) and
  `scripts`-side `tests/agents/ifvg_search/pipeline_fixture.py` /
  `test_pipeline_job_script.py` (the synthetic pipeline wiring). Same
  precedent as DEV-R1-3 and DEV-R4-1. *(R5-FIX finding 3 supersedes the
  original "sanctioned tests-package pattern" claim: the PRODUCTION
  registry no longer names any `tests.*` path — the synthetic entries are
  registered by the development checkout itself through the guarded
  `register_development_runner_entries` extension point, DECISIONS_TAKEN
  #43.)*

- **DEV-R5-2 — Two additive store names.** `pipeline_stage_results` (per
  attempt-stable stage-result envelopes + their evidence sidecars: lineage
  maps, day funnels, coverage/ladder diagnostics, prop vectors) and
  `pipeline_specs` (the frozen `PipelineSemanticIdentity` the detached
  worker loads by id). CS §8's list predates the semantic/attempt split's
  storage needs; the store implementation had already extended the list
  seven names beyond CS §8 (R1), and both stores follow the same manifest
  protocol. Additive; no existing store's semantics change.

- **DEV-R5-3 — `merge_prop_vectors` extracted from `run_search`.** The
  ALL-legs feasibility + worst-firm merge loop moved into a public pure
  function that `run_search` now calls — a behavior-identical refactor of
  an R2/R3 module (the orchestrator/prop-seam suites prove byte-identical
  explanations), permitted as a minimal compatibility edit so the pipeline
  and study lanes share ONE implementation instead of two divergable
  copies. Same precedent as DEV-R4-2's minimal orchestrator edit.

- **DEV-R5-4 — `search_bridge` extended additively (DEV-R3-11 closure).**
  `make_prop_simulator` gained keyword-only mode/persistence parameters
  whose defaults reproduce the R3 behavior byte-for-byte (existing suites
  unchanged); `FirmSimulationSpec.policy_set_envelope()` and
  `persist_account_simulation` are new additive surfaces. The
  evaluation-only propsim API is untouched.

- **DEV-R5-5 — S15's `zero_forbidden_counters` gate is DERIVED from the
  run** *(brought forward per R5-FIX finding 9; the original entry's
  "true-by-construction" framing predated the adversarial F1 fix)*. The
  per-child containment records any tripped zero-counter access assertion
  on `_RunContext.forbidden_access_detected`, and S15's gate READS that
  record — test-witnessed
  (`test_tripped_access_assertion_fails_the_derived_counter_gate`: a
  tripping drive fails the child, the gate, the report, and publication).
  The structural facts remain (no real source path exists under a
  synthetic-marker charter; on the REAL branch every drive of
  `run_child_replay` asserts its own zero counters before returning), but
  the persisted gate is measured from the attempt, never asserted, so the
  report can never contradict the audit it names; the runtime audit
  evidence stays attempt-scoped per §1.1.

- **DEV-R5-6 — The ladder's identity-parity claim vs the "real R1 slice's
  candidate view" gate item.** PHASED R5's gate names "ladder parity on
  fixture 1 + the real R1 slice's candidate view (control-flow only)". The
  fixture-1 half is proven (47 ML-lane tests). The real-slice half CANNOT run: the
  real slice itself is blocked on the owner's fixture authorization
  (program-wide first blocker). The pipeline E2E proves the equivalent
  control-flow shape on the synthetic ≤5-day window (0 valid folds under
  the frozen 40/5/5/2 protocol; empty-prediction reports; every stage
  terminal) — the same safe-failure states the ML plan §9 names for the
  real mini-run. The real-slice ladder run remains part of R1's
  acceptance-time slice execution.

- **DEV-R5-7 — `DecisionPolicyPayload` carries
  `resolved_regime_protocol_id: str | None`.** `ML_REGIME_CONTRACT_PLAN.md`
  §7 (the module-owning authority for `decision_policies.py`) includes the
  field; CS §9's summary omits it. The regime-conditioned policy is
  unbuildable without it, so the module follows its owning plan; recorded
  here because the two authoritative documents differ on the optional
  field.

- **DEV-R5-8 — S14 persists search-lane comparison results without
  study-cell envelopes.** DT §2's `ComparisonEnvelope` requires baseline/
  challenger STUDY CELLS, whose `DataLineagePayload` requires published v2
  dataset references — synthetic control-flow children never publish v2
  datasets, and fabricating dataset ids into a semantic identity would be
  identity abuse. The persisted `ComparisonResult` therefore carries the
  derivation id of DECISIONS_TAKEN #42 — since R5-FIX finding 6 as the
  TYPED `SearchDerivationComparisonSubject` (discriminated union,
  DECISIONS_TAKEN #45), never a bare id string — with the
  lineage-uniqueness envelope ids as its evidence links; cell-based
  comparisons remain the study-lane contract and gain nothing false here.

- **DEV-R5-9 — The UI's "verified reuse hits" stays launch-resolved
  (DEV-R4-6 carried).** The pipeline Preview repeats the R4 rule: counting
  reuse at preview would construct `ReplayInputBundle` source hashes before
  authorization; the disclosure names launch-time resolution instead.

- **DEV-R5-10 — Search-shim real-entry store root.** The search job shim
  invokes runner-entry factories as `factory(charter)` (the R2 contract);
  the real search entry therefore defaults its store root to the canonical
  verification namespace (`search_test/v1`) rather than receiving the
  worker's `--store-root`. The pipeline shim's newer contract passes the
  store root explicitly (`factory(charter, semantic, store_root=...)`).
  Coherent because the canonical program allowlist confines the real slice
  to that one namespace (V3 P0-7); flagged for the R5B shim alignment.

## Scoping notes (not deviations)

- `pipeline_runner_planned` → `pipeline_no_runs`: presentation-registry
  change only (see DECISIONS_TAKEN #39); the §31 state-set test updated in
  the same change.
- The wizard's mode-5 step 8 persists the visible step fields BEFORE
  rendering the pipeline surface, so the Launch handler assembles exactly
  the state the user sees.
- The pipeline state file's `stages` map covers ALL 16 stages; stages
  outside the plan carry `in_plan: false` and present as "Not required"
  (`StageStatus` deliberately has no such member — CS §7 is unchanged).
- The E2E fixture's charter relaxes two strategy-gate thresholds exactly
  as the R2 orchestration fixture does (min 10 trades / 3 days) — research
  gates are never applied to verification fixtures (TEST_MATRIX §1).

## Post-adversarial additions (recorded with the resolutions)

- **DEV-R5-11 — §30.4 monitor field subset.** The Monitor renders elapsed
  time, a remaining-stages note, the configured worker count, the running
  child (ordinal + short id), and the warnings list (adversarial M-3
  closure) — but NOT a live "current fold": fold execution happens inside
  one in-memory ladder call with no per-fold checkpoint, so a live fold
  indicator would be fabricated rather than observed. "Worker utilization"
  renders as the configured count with the truthful note that R5 children
  execute sequentially. The state file's `warnings` list is rendered (and
  stays empty until a runner condition populates it).

- **DEV-R5-12 — FUX §30.1 Configure is split across the wizard and the
  pipeline phases** (adversarial m-15). Steps 1–7 of the mode-5 wizard
  carry the charter half of §30.1's select-list (scope, baseline, dates,
  prop contracts, policy selections, simulation protocol); the Configure
  phase adds the pipeline-only selections (stage plan, bundle, model,
  labels, worker limit) and captions the split. Mode-5 step 8 persists the
  visible fields before rendering the surface, so Launch assembles exactly
  what both halves show.

- **Scoping notes:** prop-simulation runtime failures keep the study
  lane's `FailureReason.REPLAY` labeling (identical to `run_search`'s R3
  seam — cross-lane consistency wins; the explanation prefix "prop
  simulation failed:" carries the truth) — adversarial m-8.
  `assert_single_frozen_selection`'s ladder call site guards a
  single-valued surface by construction; the enumeration defense is
  structural (ids-only entry, no multi-value parameter exists) plus the
  direct-call tests — adversarial m-13. Preview estimates render per-phase
  NUMBERS as operational annotations (m-17).
