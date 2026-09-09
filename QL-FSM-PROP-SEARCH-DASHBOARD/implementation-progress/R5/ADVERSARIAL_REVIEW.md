# R5 — Adversarial review (two independent read-only reviewers)

Reviewer reports (verbatim summaries of the delivered reports).
Dispositions in `ADVERSARIAL_REVIEW_RESOLUTION.md`.

---

# Reviewer 1 — contract-fidelity lens

**Overall verdict:** the core P0-3 semantic/attempt split, the S11
blockade, the MBP-1 planned-state refusals, the capability-scoped
readiness, the ladder parity gate, and the publication boundary are
genuinely implemented and honestly tested — I could not break any of
them. **0 blockers, 4 majors, 13 minors.** Full-repo suite independently
reproduced (1595 passed, 44 warnings).

## MAJOR findings

### M-1 — S07 label artifact identity can collide across different labelings and forks on row order
`_stage_s07_labels` hashes `candidate_ids` sorted but `binary_targets`
in frame row order, so the identity does not bind candidate→target. Two
genuinely different labelings can share one `label_artifact_id`, and one
identical labeling in a different row order mints a different id.
Reproduction executed: labeling {A→0, B→1} in row order (A,B) and
labeling {A→1, B→0} in row order (B,A) hash EQUAL; the same labeling
reordered hashes DIFFERENT. Contract: CS §0.1 + TEST_MATRIX §3.1 P0-3.
Proposed: hash ordered (candidate_id, target) pairs.

### M-2 — DEV-R4-16's "persisted comparison-contract consumption" claimed closed but the consumption half is dead code
R5 builds and persists `ComparisonResultEnvelope`s (S14, genuinely
tested) and adds `load_comparison_results_for_search` — but nothing
consumes it: zero call sites, zero tests. The provider docstring claims
"the UI consumes persisted contracts" — no UI does. Proposed: wire the
provider into a surface with an AppTest, or downgrade the claim to an
explicit deviation.

### M-3 — FUX §30.4 Monitor omits contract-required fields; no deviation recorded
§30.4 requires current date/child/fold, elapsed time, estimated
remaining time, worker utilization, and warnings; the R5 Monitor renders
none of these; the state file's `warnings` list is dead. Some fields are
structurally hard on a completed run — exactly what a DEVIATIONS entry
should have said. Proposed: render elapsed / children progress / worker
line / warnings, or record the subset as a deviation.

### M-4 — ruff not green on the delivered tree; TEST_RESULTS records "All checks passed"
`python -m ruff check src tests scripts` failed with 2 F401 errors in
the new `tests/agents/test_ifvg_pipeline_tab.py` (unused `json`,
`WorkerPolicy`) at review time. Trivial to fix; major for the gate
record. (The full-suite pytest claim was independently reproduced.)

## MINOR findings

- **m-5** — S12 asserts persistence that did not happen in the
  zero-simulation edge (policy-set envelopes persist only inside the
  bridge's per-child call). Fix: persist unconditionally in the stage.
- **m-6** — Publish phase caches gate results in a session key not
  scoped to the selected run; the Activate button could enable from
  another run's gates (server-side refusal still holds). Fix: key by
  pipeline id.
- **m-7** — `merge_prop_vectors` extraction micro-divergence: failure
  reason assigned only when not None (old code unconditional).
  Observable only for a non-constructible failed-report-with-None-reason.
- **m-8** — Prop-simulation runtime failure labeled
  `FailureReason.REPLAY`; vocabulary-exact but semantically loose (the
  explanation prefix mitigates).
- **m-9** — `ladder_id` under-specifies its inputs (no labels/folds in
  the hash); ambiguous as an output artifact id.
- **m-10** — S00's wiring-closure check does not require
  `candidate_view_source` for a plan with S07 but not S05.
- **m-11** — Stage-result publication errors escape the per-stage
  containment (a store refusal crashes the runner, stage left RUNNING).
- **m-12** — A strategy-only plan may carry unresolvable (e.g. MBP-1)
  bundle ids in its semantic identity (inert but carried).
- **m-13** — `ProhibitedSelectionError`'s only production call site is
  tautological; the 7B.22-15 defense is structural + direct tests.
- **m-14** — The 7B.22-2 "leak test" is structural, not a direct probe;
  a perturb-test-row probe would be stronger. (Permutation importance
  examined for leakage and REFUTED.)
- **m-15** — Configure phase presents only the pipeline-half fields
  (charter half rides wizard steps 1–7); reasonable but unrecorded.
- **m-16** — S00's verification binding partially self-referential
  (expected seed sourced from the same envelope; spec allowlist never
  compared to the run payload). Transitively closed on the registered
  path via the executor factory's charter cross-checks.
- **m-17** — Preview's "runtime/storage estimate by phase" rows are
  fixed sentences, not per-phase estimates.

Also recorded: the review-round evidence files did not yet exist
(expected — this review is that round's input); DEV-R5-6 said "45 tests"
where the ML lane holds 47 (cosmetic); prevalence rung lacked an in-lane
duplicate guard (unreachable today — footnote to refuted suspicion 7).

## Suspicions raised and REFUTED by verification

1. P0-3 leaks/nondeterminism — refuted (no operational field in any
   hashed payload; attempt contract unregistered by design; sidecars
   sort_keys-deterministic; the E2E attempt-identity test real).
2. REUSED marking assumed — refuted (identity-proven; the store fails
   closed on same-id divergence, sidecars included).
3. Cross-attempt S14 drift via the lineage sidecar round-trip — refuted.
4. `merge_prop_vectors` behavior change — refuted by line-level diff
   (identical strings/ordering/semantics; m-7 micro-edge only).
5. MBP-1 reachable — refuted at block, bundle, readiness, and UI layers
   (no resolution envelope exists; `with_activated_block` pure).
6. S11 escapable — refuted on every surface; no executor entry exists;
   `ModelGatedReplayRequest` has zero consumers; the reason string is
   verbatim-locked across three modules by test.
7. Ladder parity passable with different row semantics — substantially
   refuted (oos_row_id binds view/fold/candidate; one_to_one merges).
8. Prop targets reachable by models — refuted structurally + scans.
9. Publication state in hashed identity / S15 self-reference — refuted.
10. Scenario fabrication / ordered-replay leakage at the bridge —
    refuted.
11. UI badge vs frozen scope divergence — refuted (same source field).

## Gate-item coverage

Every PHASED R5 gate bullet mapped to a passing test or an honest
deviation; "complete §30 UX" Partial pending M-3/m-15; DEV-R4-16
half-closed pending M-2; the real-slice ladder half honestly deviated
(DEV-R5-6). Standing constraints: full pytest reproduced green; ruff not
green at review time (M-4); Strategy-Core untouched; no real source path
constructible in anything R5 executes; the acceptance chain remains
truthfully `transitively_blocked_by_R1`.

---

# Reviewer 2 — safety / access / identity-abuse lens

**Verdict: no blocking access-safety defect.** Protected/sealed
zero-counter verdict for the R5 layer: **AFFIRMED — zero.**

## Findings

- **F1 (MEDIUM)** — S15's `zero_forbidden_counters: True` is asserted,
  not derived: a tripped drive assertion fails the child but the
  persisted gate line-item would still read True (overall passed=False
  and publication refusal still hold). Proposed: derive the line-item
  from child outcomes.
- **F2 (LOW-MEDIUM)** — Synthetic-branch verification stamps label
  synthetic fixture days as real (`real_date_count: 3` for the synthetic
  days). Proposed: `real_date_count=0` + a synthetic count.
- **F3 (LOW)** — The real executors bind S00's authorization validation
  tautologically (authorization + expected seed sourced from the same
  envelope; the independent charter cross-checks retain the teeth).
- **F4 (LOW)** — Search-entry authorization root diverges from the
  worker's write root (documented DEV-R5-10; R5B shim alignment).
- **F5 (LOW)** — Sidecar filename validation weaker than envelope-id
  validation (drive-relative/ADS names unrefused; no user-shaped name
  reaches `extra_files` today). Proposed: allowlist regex.
- **F6 (LOW)** — "real AccountPolicySetEnvelopes persisted" wording in a
  synthetic run's monitor ("real" ambiguous vs real-firm).
- **F7 (LOW)** — `full_pipeline_not_run: True` unconditional in the
  state file (untruthful for a future dev-scope run; unreachable today).
- **F8 (INFO)** — Launch-phase scope badge derives from the draft, not
  the assembled charter (same source field in practice — refuted as a
  divergence in the contract reviewer's #11).
- **F9 (INFO)** — `sanitize_failure_message` token list misses non-C:
  drive paths (the UI re-sanitizes; state files could carry them).
- **F10 (INFO)** — Observations: attempt-invariant empty-output stage
  ids fail closed on sidecar divergence (honest); the
  `ComparisonResultEnvelope` docstring overstates for the search lane;
  the "parity held over 0 OOS rows" caption is a vacuous truth with the
  count displayed; the launch docstring said "ONLY code path" while the
  resume handler also spawns (both handler-confined, scan-pinned).

## Refuted suspicions (each verified)

Fail-before-path holds (no config/policy/path before the refusal; the
provider never lists store roots; production code never creates a
`VerificationRunEnvelope`); zero protected/sealed constructions in the
complete R5 diff; the registry gate is airtight (regex + exact-membership
before import; both shims AST-clean; no eval/exec/os.system); no AppTest
spawns a real process; all tests tmp-only and the repo's real data
namespaces do not exist on disk; all eight new envelopes pass the
projection audit and `ExecutionAttemptIdentity` is unregistered by
design; store overwrite/sidecar-divergence refusals hold; M0–M3 /
pre-existing propsim / Strategy-Core / Trade-Lab untouched
(`search_bridge` additive with R3-identical defaults;
`historical_ordered_event_replay` refused; `first_party_verified`
unconstructible); MBP-1 depth guarded at every layer; no
full-development run possible (no registered entry; synthetic marker
refused under dev scope; typed ack then the no-executor state;
verification scope can never activate).

## PROVABLE claims

Ten enumerated provable claims for ACCESS_SAFETY (zero protected/sealed
access; fail-before-path executors; no dev-scope entry; two scan-pinned
UI Popen seams + one per shim; projection-audited envelopes;
store-integrity refusals; synthetic-firm pinning; untouched boundary
repos/lanes; verification-activation refusal; tmp-only writes), with two
claims to qualify pending the F1/F2 fixes.
