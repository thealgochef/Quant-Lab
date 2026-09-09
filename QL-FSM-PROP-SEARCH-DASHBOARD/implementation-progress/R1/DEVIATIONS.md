# R1 — Deviations from the Final Plan Package

Each entry records a deliberate implementation deviation, its trigger, and why
it does not invent a third design. None alters plan semantics; every
fail-closed rule is preserved or strengthened.

## DEV-R1-1 — `build_ifvg_v2_capture` gained three parameters (dataset.py beyond "hash promotion")

- **Plan text**: IMPLEMENTATION_PLAN §3/§18 list dataset.py as "hash
  promotion" only; but IMPLEMENTATION_PLAN §17 and PHASED_DELIVERY R0
  explicitly sanction the fallback "`start_after_artifact` driver parameter
  (QL-only)" for the R0→R1 mid-chain-start item.
- **Deviation**: `build_ifvg_v2_capture` gained `start_after_artifact:
  ChainStart | None = None` (the sanctioned fallback), `audit_capture_mode:
  str = "disabled"` (dual-drive neutrality needs the audit channel through
  the same proven chain path), and `final_day_exhausts_dataset: bool = True`.
  All default to existing behavior; every pre-existing call site is
  byte-compatible.
- **Trigger for the third parameter (discovered fact)**: the chain marks its
  final day `dataset_exhausted=True`, which alters the end seed. A
  snapshot-producing prefix run must therefore NOT exhaust its final day, or
  the snapshot seed diverges from the continuous chain
  (`test_build_ifvg_v2_capture_supports_mid_chain_start` proves both sides).
  Without this, the plan's seed-snapshot design would be silently wrong.

## DEV-R1-2 — `ArtifactProvenanceReadAdapter` (new, additive, in child_replay.py)

- **Discovered codebase fact**: `day_artifacts.load_day_artifacts` verifies
  the cache stamp `source_allowlist_sha256` against the LOADING policy's
  allowlist hash; dev-chain caches were stamped by the development policy, so
  a 5-day `VerificationReplayPolicy` can never load them directly. The plan
  did not anticipate this seam (it assumed cached artifacts are loadable
  under the verification policy).
- **Resolution chosen** (instead of stopping): an additive read adapter that
  exposes the recorded artifact-provenance allowlist for the stamp
  comparison ONLY while delegating every date authorization, path
  construction, audit event, and forbidden-access assertion to the strict
  inner 5-day policy. No existing loader semantics changed; a 6th or
  off-allowlist date is still refused before any path exists
  (`test_provenance_adapter_reads_dev_caches_under_strict_authorization`).
  Documented in D-040's trade-off paragraph. This is a seam adaptation, not
  a third design: the plan's fail-before-path and ≤5-day rules are enforced
  by the same policy object they were designed around.

## DEV-R1-3 — R2-scheduled `study/cohort.py` authored in R1

- `StudyCellSemanticPayload.observation_cohort` requires the CohortSpec hash;
  authoring the CS §9.1 contract early (schedule-forward, content exactly per
  the plan) avoided a placeholder id that would churn every cell identity at
  R2. Scope unchanged; R2's cohort interpretation tests still land in R2.

## DEV-R1-4 — Axis value-id naming uses the full technical key

- CS §2 shows illustrative ids like `parent_retest_timeout.480`; the
  implementation uses the deterministic rule `<technical_key>.<token>`
  (`parent_retest_timeout_1m_bars.480`) for collision-free ids across all 44
  axes. The doc's ids are marked "e.g." (illustrative); the registry is the
  binding surface.

## DEV-R1-5 — Decision/model resolved-id "none" sentinels in cell identities

- Strategy-only cells need typed null identities before the R5 decision/model
  payload modules exist. `resolved_decision_policy_id` /
  `resolved_model_protocol_id` accept `none_*_v1` sentinels (P1-G's typed
  nulls) in addition to 64-hex resolved ids; R5 mints real resolved ids for
  real policies. Pattern-validated; no free-form strings.

## DEV-R1-6 — R1 gate's "verifier link resolves" executes at acceptance time

- The exact `setup_id` verifier jump and the neutrality-aware audit/chart
  companion builders are R2-assigned seams (PHASED R2 modified-files list).
  `run_baseline_verification_slice` therefore takes injectable
  `companion_builders`; the corresponding control-flow gates evaluate False
  until those builders run. R1 acceptance is blocked on the owner fixture
  regardless (authoring-vs-acceptance model, V3 P0-8), so no gate is claimed
  passed that has not run. Recorded in GATE_SUMMARY as an open sub-gate.

## DEV-R1-8 — Baseline cell bundle key: `B0_CORE` (plan shorthand conflict)

- DELTA_TAXONOMY §2 lists the baseline cell's bundle as `IFVG_CORE_BASELINE_V1`
  — a feature-BLOCK key. The typed bundle registry (DT §7) is the binding
  surface for bundle keys, and the cell validator (adversarial finding C2)
  fails closed on unregistered bundle keys, so the baseline cell carries the
  lawful bundle key `B0_CORE` (= the CORE block + SESSION). The resolved
  bundle id remains the strategy-only `none_feature_bundle_v1` either way.
  Conflict resolved toward the typed registry; noted inline in
  `study_cell.py`.

## DEV-R1-7 — Coverage evidence surfaced a better-scoring window than the plan's candidate

- Not a code deviation: the coverage matrix (built per P0-7 from the accepted
  artifacts only) shows the candidate `2026-06-04…06-10` covers source/replay
  integration but zero candidate/decision/trade paths (May–June is
  funnel-quiet), while `2026-02-06…02-11` covers all four funnel paths. Both
  lawful options are presented neutrally in
  `DRAFT_VERIFICATION_AUTHORIZATION.md`; the date choice remains the owner's
  BLOCKING-VERIFICATION decision (21/R-5). No allowlist was registered.
