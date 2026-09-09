# R2 — Deviations from the Final Plan Package

Each entry records a deliberate implementation deviation, its trigger, and why
it does not invent a third design. None alters plan semantics; every
fail-closed rule is preserved or strengthened.

## DEV-R2-1 — audit-mode capture retains `trace_audit_rows` + stamps audit frames (dataset.py)

- **Plan text**: PHASED R2's modified list names `fsm_audit_preparation.py`
  for the per-child neutrality-aware audit build; dataset.py is not on R2's
  list.
- **Discovered codebase fact**: the six trace-cut audit tables (htf_tap …
  setup_resolution) derive from core-trace rows that
  `build_ifvg_v2_capture` discards at partition time, and the audit-channel
  frames it returned lacked the warmup/eval-hash/global-ordinal stamps the
  audit contracts REQUIRE (`validate_audit_table` refuses null `is_warmup`).
  Without both, the plan's own per-child audit build is impossible short of a
  second heavyweight replay through the doc-default-parity builder — which
  D-039/D-040's child lane exactly must not use.
- **Deviation**: under `audit_capture_mode != "disabled"` ONLY,
  `V2CaptureResult` retains the trace-cut audit rows (with their global
  `trace_ordinal`) and the per-day audit frames gain the identical stamps the
  fsm-audit builder path applies. The plain (audit-disabled) capture path is
  byte-identical to before; this is the same seam R1 opened under DEV-R1-1
  (`audit_capture_mode`/`audit_frames`/`end_seed`), completed. Proven by
  `test_dual_drive_retains_audit_channels_and_neutrality` and the untouched
  full-suite/native-determinism results.

## DEV-R2-2 — companion store names + storage-envelope classes

- CS §8's store list (as already extended by R1 for
  `replay_input_bundles`/`seed_snapshots`/`coverage_matrices`/
  `verification_runs`/`neutrality_reports`) gains `fsm_audit_companions`.
  CS §1.5 defines `FsmAuditArtifactIdentity` but assigns it no store; the
  per-child companion must persist somewhere immutable inside the search
  namespace. Two storage envelopes were added for existing §1.5/§3.3
  contracts (`ChildFsmAuditEnvelope`, `ChildAuditNeutralityEnvelope`) —
  payloads are exactly the plan's contracts (no content hashes; table bytes
  are manifest sidecar facts) and both are registered in the
  identity-projection audit.

## DEV-R2-3 — job-shim execution requires an explicit runner entry

- IMPLEMENTATION_PLAN §6 requires the detached CLI shim; PHASED defers the
  real pipeline executors to R5 (`search/pipeline.py`). The R2 shim therefore
  implements start/status/cancel process control and a worker that REFUSES to
  execute without an explicit `--runner-entry module:function` supplying the
  replay wiring. No built-in real-data execution path exists in R2; nothing
  launches at import; the R5 pipeline registers the real executors.

## DEV-R2-4 — `canonicalize_section` content-equality guard (defect fix, recorded for transparency)

- The R1-committed helper kept ANY section whose `profile_name` was in the
  fixed registry — including derived children that still carried their
  baseline's name (the normal `resolve_profile_config` + overrides output),
  so generated children were never renamed and `GeneratedProfileCapability`
  blocked them as non-canonical. The guard now keeps a registered name only
  when the name-free content is byte-identical to the registered baseline's;
  otherwise the canonical `ifvg_search_profile_<hash16>` name applies. This
  is CS §1.3's own rule ("two different names for identical semantics are
  impossible" — and one baseline name for different semantics equally so).

## DEV-R2-5 — contrast evaluation scope: declared interactions evaluate at R4

- DT §5's payload declares main effects (1 axis) AND two-way interactions
  (2 axes); the R2 evaluator computes fully-crossed paired MAIN effects (with
  the seed-7 pair bootstrap) and refuses interaction evaluation with a typed
  `UnbalancedDesignError` naming the missing registered adjustment method.
  Interactions remain declarable (schema + identity complete, no redesign);
  their evaluation lands with the results/comparison surfaces (R4), where
  §7A.19.11's acceptance row is gated. Nothing silently computes.

## DEV-R2-6 — contrast pair orientation

- DT §5 does not fix a direction for paired deltas. The evaluator orients
  every pair by the deterministic lexicographic sort of the two registered
  value tokens (delta = second − first) and RECORDS the exact
  `matched_pairs`, so the direction is always reconstructible; no hidden
  convention.

## DEV-R2-7 — `EvidenceRef` kinds: +candidate/+decision (plan-vocabulary patch)

- CS §12 registers setup/trade/account-event/child/simulation/axis. The delta
  layer legitimately emits population evidence for the candidate and decision
  entity kinds, so both are registered kinds; the never-constructed
  `artifact`/`report` members were deleted (no dead vocabulary). Adversarial
  finding A-F8.

## DEV-R2-8 — locked-invariant check exempts the resolver's provenance relabel

- `resolve_profile_config` stamps every override-bearing derived profile
  `qualification_mode="custom_profile"` (profiles.py) — a provenance label
  the Strategy-Core reducer stores into records but never branches on
  (verified: reducer.py/replay.py stamp-only usage). The
  `blocked_invariant_failure` check therefore exempts exactly the
  `→ "custom_profile"` transition; every other qualification transition
  (broad_capture/ict_clean, or a child spoofing a baseline label) and every
  other LOCKED_INVARIANT field remains a violation. Adversarial finding A-F6.

## DEV-R2-9 — charter schema additions closing inherited plan gaps

- `SearchCharterPayload.declared_contrast_ids` (DT §5 requires charter-frozen
  declarations, but the CS §3.1 schema carried no field — the evaluator is
  now anchored on the frozen charter identity; adversarial finding A-F3), and
- `OBJECTIVE_DIRECTIONS` + the `ObjectivePolicy` validator (CS §3.1's
  `pareto_objectives` are bare names with no direction authority anywhere —
  unregistered objective/tie-break names now refuse at charter validation;
  adversarial finding A-F16).
- No real charter artifact predates these additions (no research store
  exists), so no persisted identity changed meaning; `schema_version` stays 1.

## Scoping notes (not deviations)

- The slice companion builder requires explicit `repository_states` (bound by
  the caller via functools.partial); it never fabricates repository evidence,
  and the acceptance run binds the real states.
- `verify_exact_drill_targets` marks a zero-entity artifact as
  `vacuous_zero_targets=true`; the slice summary surfaces the flag beside the
  gate policy, and the ACCEPTANCE runner must require a non-vacuous link on
  the real slice (GATE_SUMMARY blocker list).
- Reused children RELOAD their published costed evaluations (metrics keyed on
  core_replay_id × cost policy) and gates re-evaluate per charter; a reused
  child whose evaluation was never published (pre-R2 stores) is explicitly
  annotated, never silently gate-free. The `frontiers` store holds the
  persisted `FrontierResult` per search.
- R4's comparison surfaces must call `persist_lineage_uniqueness` before
  surfacing any cross-profile delta, and must registry-gate the job shim's
  `--runner-entry` (the UI never passes user-shaped strings).
