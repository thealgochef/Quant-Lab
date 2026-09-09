# R4 — Adversarial review resolution

Every finding from both reviewers was resolved in code/tests before the
release commit (or, for two structurally-deferred items, upgraded to
first-class recorded deviations with the deferral rationale and partial
delivery). Post-fix verification: the complete R4-adjacent suite
(454 tests incl. every new witness) green, full-repo suite green, ruff
clean, `git diff --check` clean.

| Finding | Resolution | Post-fix witness |
|---|---|---|
| R1-F1 (survival column) | The Prop preset's "90-day survival" now displays **1 − breach_probability_90d** as a percentage | `test_prop_preset_shows_true_survival_values` (80.0% asserted; breach value asserted absent) |
| R1-F2 (interpretation inert) | Enforced at validation: changed session/direction axes + a non-sequential interpretation REFUSE Next with the exact explanation (DEV-R4-15); nothing descriptive can enumerate replay children | `test_interpretation_selector_gates_enumeration` (unit) + `test_descriptive_interpretation_blocks_enumeration` (AppTest, Next disabled) |
| R1-F3 (hardcoded auth state) | New provider `verification_authorization_state` scans load-verified `VerificationRunEnvelope`s for a bound `VerificationAuthorizationRef`; the wizard renders EXISTS (with run ids) or the missing state from the provider; the checklist suffix now states the registry fact ("no registered evidence artifact"), not a world-claim | `test_verification_authorization_state_is_derived` (missing AND present directions) |
| R1-F4 (comparison surface) | (c) FIXED: config-diff-only pairs suppress the Strategy/Prop delta panels entirely (parameter diff only); (a)+(b) upgraded to **DEV-R4-16** — live delta builds and stored-comparison-contract consumption land with the R5 stages that persist their inputs; the lineage-gated constructor + suppression + affected-ID rendering are delivered and proven | `test_incompatible_pair_suppresses_membership_claims` (2 suppression notices; no delta table) + provider delta tests |
| R1-F5 (heatmap keep-last) | The results layer AGGREGATES per coordinate (mean, summed samples, per-cell evidence reason, visible "N configurations aggregated (mean)" notes); the builder reports any residual duplicate coordinate as an omission | `test_heatmap_single_axis_view_aggregates_honestly` |
| R1-F6 (launchable semantics) | `launchable ⇔ status ∈ {synthetic_fixture_verified, first_party_verified}`; intermediates render their reason and NO checkbox; superseded unchanged. Staleness: no staleness policy is registered in the v1 contracts (nothing to evaluate against) — the card shows `effective_date` + the ladder status, and unverified statuses cannot launch; recorded in the resolution rather than inventing a staleness rule | `test_unverified_intermediate_contracts_are_not_launchable` (4 statuses) |
| R1-F7 (dropped risk params) | Charter risk ids are now CONTENT hashes of the complete per-firm policy dicts (template + value + n_accounts + replacement) — no result-changing parameter can be dropped from the charter identity; real policy-set envelope construction deferred to R5 S12 (**DEV-R4-17**) | freeze AppTests exercise the assembly; identity sensitivity follows from `canonical_contract_sha256` |
| R1-F8 (hardcoded scopes) | Every prop scope caption derives from the persisted `simulation_mode` via `_MODE_TO_SCOPE` → `RESULT_SCOPE_LABELS` (overview, frontier, firm matrix, explorer, compare Prop panel); the snake_case render replaced by the quoted label | `test_common_frame_picker_disclosure_badge_scopes` now asserts "Prop Historical Closed-Trade Replay" present AND "Bootstrap Simulation" absent for the closed-trade fixture |
| R1-F9 (wrong-state copy) | (a) missing-simulation views render `artifact_unavailable` (never the §16.4 gate-skip sentence); (b) blocked baselines render a dedicated FUX §9 error card; (c) two new dedicated presentations `runner_executor_planned` / `pipeline_runner_planned` with correct copy (registry updated to 16 entries) | `test_missing_simulations_render_artifact_unavailable_not_gate_skip` + `test_every_section31_state_is_registered` + wizard planned-state AppTest |
| R1-F10 (cross-draft leak) | On a draft switch the shell purges every not-yet-instantiated step-widget key (header widgets excluded) before rendering — draft B can never inherit draft A's mounted values | `test_no_cross_draft_widget_leak` (B shows empty; Save writes B's state; A untouched) |
| R1-F11 (§14.1 fields) | Validation step adds: real-date count caption, warmup display (verification: "0 real days — seed snapshot"; full scope: a warmup_dates input feeding the ≤5 validator), runtime/storage estimate rows | wizard validation AppTest re-run |
| R1-F12 (§15 omissions) | Review step adds: frozen-dimensions row, the baseline resolved-section identity block, and the resolved authorization-checklist summary | freeze AppTests re-run |
| R1-F13 (§16.2 labels) | The monitor renders the five exact quoted labels with counts (Profiles Generated / Replays Completed / Strategy-Gate Passes / Prop-Feasible Configs / Robust Finalists) above the funnel | monitor test asserts all five |
| R1-F14 (§16.5 identities) | Child detail derives + renders `membership_id` (from the state row) and `costed_evaluation_id` (core replay × charter cost policy) | monitor detail test asserts both labels |
| R1-F15 (twin/timeline detail) | Heatmap table twin gains the "evidence reason" column; the timeline draws the daily-loss threshold as a step LINE (markers only for a single observation) and plots phase transitions with from→to labels | `test_timeline_builder_marker_shapes_and_display_timezone` (dll line mode + "evaluation→funded") + heatmap twin test |
| R1-F16 (primaries) | Frontier chart gains `on_select="rerun"` point selection syncing the selected configuration (selectbox twin unchanged as the accessible/AppTest path; `TypeError` fallback for older pins); the explorer's repeated selected-config details moved ABOVE the table per §4.1 | AppTests re-run (selection sync exercised interactively at hardening) |
| R1-F17 + R2-S5 (weak scans) | The `or True` tautology removed — the override scan now asserts: `section_overrides` absent from the wizard, exactly two text_areas (both date lists), no user-JSON parsing; the session-namespace scan covers literal, f-string, AND variable-keyed writes against a known-prefix allowlist; a Best/Winner/Validated/Production Ready/Live Ready display-title scan added | `test_no_raw_override_editor_exists`, `test_session_namespace_is_the_contracted_prefix`, `test_no_standalone_best_winner_validated_titles` |
| R1-F18 (coverage gaps) | Added: FUX-WIZ-009 risk-step AppTest (tree + policy widgets + cap-2 + shared-path note); unit tests for the four uncovered validators; a Back-preserves-fields AppTest; a 64-child state-only pagination AppTest ("Rows 1–25 of 64"); intermediate/first-party contract-card states (F6 witness) | the named tests, all green |
| R1-F19 (premature no-pass) | The no-pass sentence renders ONLY for `search_complete`; non-terminal runs render "Run in progress (phase: …)" | `test_in_progress_run_never_shows_the_no_pass_verdict` |
| R1-F20 + R2-S6 (sanitizer) | Quoted-path redaction (spaces inside quotes) + UNC (`\\server\…`) patterns added ahead of the bare-token pass | `test_sanitizer_redacts_unc_and_quoted_space_paths` |
| R2-S1 (second subprocess seam) | Docstring corrected (names both seams); **DEV-R4-14** recorded; `"unknown"` provenance now REFUSES the freeze; the SC root resolves from the installed `strategy_core` package (walk to `.git`), sibling guess only as last resort | `test_unresolved_commit_provenance_refuses_freeze` |
| R2-S2 (draft TOCTOU) | Per-draft `O_EXCL` lock (catalog-lock idiom, stale-break at 30 s) held across every check-then-write transition: save re-checks the STORED status under the lock; mark_frozen re-checks the stored record (second freezer refuses); discard loads-checks-unlinks under the lock | extended `test_frozen_drafts_refuse_mutation_and_discard` (stale-copy save refusal + racing-freezer refusal) |
| R2-S3 (draft-id validation) | Allowlist `^[0-9a-f]{32}$` (drive-relative components and device names cannot match); `load_draft` refuses a record whose embedded id disagrees with its directory | extended `test_discard_removes_only_mutable_drafts` (6 hostile ids + the tampered-record refusal) |
| R2-S4 (sidecar trust) | `load_account_simulation_events` load-VERIFIES the envelope (manifest hash + identity) before reading either sidecar, matching `load_prop_vectors` | provider suite re-run |

## Residuals (recorded, none blocking)

- Live population/funnel delta builds + stored-comparison-contract
  consumption: **DEV-R4-16** (R5 persists the inputs). FUX-RES-007 carried
  as R4-partial on the gate summary.
- Real `AccountPolicySetEnvelope` construction: **DEV-R4-17** (R5 S12).
- Contract staleness: no registered staleness policy exists in the v1
  contract schema to evaluate; unverified statuses cannot launch, which is
  the §11 fail-closed core. Revisit when a staleness policy is registered.
- Pinned/sticky dataframe columns: the §4.1 FALLBACK (identity columns
  first + selected-config details repeated above the table) is the shipped
  behavior; a native pinned-column upgrade is a hardening-window item.
- Interactive point-selection evidence (`on_select`) is code-complete;
  interactive verification belongs to the hardening browser pass.
