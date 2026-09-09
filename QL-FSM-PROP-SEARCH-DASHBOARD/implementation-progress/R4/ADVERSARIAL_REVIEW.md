# R4 — Adversarial review (two independent read-only reviewers)

Both reviewers examined the complete uncommitted R4 tree (all 9 scripts,
5 new + 2 modified src modules, 14 test files, config, deviations doc)
against `FRONTEND_UX_CONTRACT.md`, `TEST_MATRIX.md` §3.11,
`PHASED_DELIVERY.md` R4, `CONTRACTS_AND_SCHEMAS.md`, the protected/sealed
rules, and the M0–M3/immutability boundaries. Findings only; no reviewer
edited implementation files. Verbatim findings below (resolutions in
`ADVERSARIAL_REVIEW_RESOLUTION.md`).

## Reviewer 1 — contract-fidelity lens

**Verdict: no blockers; 10 majors, 10 minors.** "R4's skeleton is genuinely
contract-driven — the quoted strings, routes, status/glyph vocabulary,
wizard shape, registry-gated launch, namespace confinement, draft
immutability, and the interaction-contrast mathematics are all exact and
well-tested … the majors are concentrated exactly where the contract
demands truthfulness rather than machinery, and each is fixable locally
without re-architecting R4." (DiD interaction math independently
hand-verified correct; no counterexample found.)

### Majors

- **F1** | FUX §24 + FUX-LABEL-001 — the explorer's "90-day survival"
  column displayed **breach probability** (inverted semantics); no
  value-level Prop-preset test.
- **F2** | FUX §10.5 — the interpretation selector rendered but was never
  consumed; "Descriptive Slice" still enumerated executable children.
- **F3** | FUX §14.2 + CS §12 — the verification-authorization state was
  hardcoded "missing" (and the checklist suffix asserted a world-state);
  no provider existed.
- **F4** | FUX §25/§3.2 + DT §3.1 — (a) membership/funnel deltas and exact
  affected-ID actions unreachable live; (b) ribbon compatibility computed
  ad hoc rather than from a stored comparison contract; (c) config-diff-
  only pairs still rendered populated Strategy/Prop delta tables; the
  deferral was under-classified as a scoping note.
- **F5** | FUX §33/§20 — the heatmap silently kept the LAST child per
  coordinate (single-axis view collapsed children with no aggregation or
  omission entry).
- **F6** | FUX §11 — `launchable = not superseded` let unverified
  intermediates (`first_party_evidence_compiled`, `owner_reviewed`) render
  a selectable checkbox; staleness had no implementation.
- **F7** | FUX §12/§15 — the frozen charter dropped `risk_value`,
  `n_accounts`, `replacement_policy` (reduced to template tokens);
  result-changing parameters survived only in the mutable draft.
- **F8** | FUX §5.2/§17 — scope captions hardcoded "Bootstrap Simulation"
  over artifacts declaring `historical_closed_trade`; one caption rendered
  the snake_case enum value.
- **F9** | FUX §31/§16.4 — wrong-state reuse: the §16.4 gate-skip sentence
  for missing simulations; "blocked search axis" copy for a blocked
  BASELINE; MBP-1 feature-block copy for the missing runner executor and
  the pipeline mode.
- **F10** | FUX §7/FUX-WIZ-002 — fixed widget keys leaked mounted values
  across a draft switch (cross-draft contamination on Save/Next).

### Minors

- **F11** §14.1 fields missing (real-date count, warmup, runtime/storage
  on Validation) · **F12** §15 review omissions (authorization checklist,
  frozen dimensions, baseline identity) · **F13** §16.2 quoted count
  labels absent · **F14** §16.5 membership/costed identities absent from
  child detail · **F15** heatmap table twin lacked the evidence column;
  timeline daily-loss drawn as markers, phase transitions table-only ·
  **F16** primaries unimplemented (no `on_select`, no pinned columns;
  details below the table) · **F17** vacuous `or True` override-scan +
  missing Best/Winner/Validated title scan · **F18** coverage gaps
  (FUX-WIZ-009 untested; 4 validators untested; no Back test; largest
  table 4 children; stale/first-party card states untested) · **F19** the
  no-pass sentence rendered for in-progress runs · **F20** sanitizer
  missed UNC/relative paths.

## Reviewer 2 — safety / identity / frozen-lane lens

**Verdict: no blockers, no majors; 6 minors.** "Protected/sealed:
zero-counter PROVABLE at the R4 layer … launch paths are registry-gated
and button-confined (execution-proven), immutable artifacts and catalogs
are never overwritten or deleted from any UI path, the R2→R4 obligations
are verifiably closed, identity surfaces are untouched by the orchestrator
and contrasts edits, and the M0–M3 lane change is exactly the
plan-authorized one-hunk delegation with its untouched suite still green."
Disproved suspicions (protected access, registry bypass, import-time
launches, store listing, History delete, orchestrator/contrasts identity
drift, fixture confinement, frozen-lane byte-behavior) are enumerated in
the review output and were re-confirmed during resolution.

### Minors

- **S1** — `_commit_of` is a second (read-only) subprocess seam with a
  false "only seam" docstring, unrecorded; silent `"unknown"` fallback
  into the hashed charter; case-guessed sibling SC root.
- **S2** — draft-layer TOCTOU family (save-vs-freeze race, discard race,
  double freeze).
- **S3** — draft-id validation gaps (Windows drive-relative components,
  device names; file-vs-directory identity trust).
- **S4** — the account-timeline read path trusted sidecars behind
  `has_envelope` without load-verifying the envelope.
- **S5** — scan-test witnesses weaker than claimed (the `or True`
  tautology; literal-only session-key regex missing f-strings).
- **S6** — sanitizer regex blind spots (UNC; quoted paths with spaces).
