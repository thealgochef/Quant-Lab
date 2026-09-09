# R4 — Deviations and scoping notes

Each entry records a deliberate implementation deviation, its trigger, and
why it does not invent a third design. Every fail-closed rule is preserved
or strengthened.

- **DEV-R4-1 — Two additive src modules beyond PHASED's R4 file list**
  (`ifvg/study_drafts.py`, `ifvg/study_providers.py`). FUX-WIZ-002 requires
  a disk draft layer and no mutable-draft persistence existed anywhere in
  the repo; CS §12 names the "UI providers" surface but PHASED's list gave
  it no home. Scripts must stay thin (FUX-MOD-001), so both live under
  `src` beside the contracted `study_status.py`/`study_presentation.py`.
  Same precedent as R1's early `study/cohort.py` (DEV-R1-3).

- **DEV-R4-2 — The orchestrator records the frontier envelope id as a
  state-file phase note.** The immutable stores are exact-ID-only and never
  listed (R1 store contract), and the frontier envelope id is a content
  hash — WITHOUT a recorded pointer no consumer can locate a search's
  frontier artifact. `run_search` now stamps
  `phase_notes["frontier_id"] = <envelope id>` after the immutable save.
  The note is an operational locator in the MUTABLE state file; no
  identity, payload, or store changes. (Minimal additive edit to an R2
  module, tested by `test_study_providers.py`.)

- **DEV-R4-3 — `EmptyStateKey.BROWSER_QA_UNAVAILABLE` is an additive enum
  member.** FUX §31 requires the UI to render the "browser QA unavailable"
  state; CS §13's enumeration predates that row. Additive, presentation-
  only; recorded per the schema-deviation rule (DEV-R3-8 precedent).

- **DEV-R4-4 — `InteractionEvaluationUnavailableError` deleted.** DEV-R2-5
  landed interactions as declarable-but-unevaluable with that typed
  refusal naming the R4 landing; R4 lands the evaluation (balanced
  two-way difference-of-differences, lexicographic orientation per
  DEV-R2-6, seed-7 bootstrap, `n_quartets` recorded, matched pairs = the
  two oriented A-pairs per stratum). The now-dead exception class is
  removed under the no-dead-vocabulary rule (DEV-R2-7 precedent);
  unbalanced/hole/duplicate/higher-order designs still refuse with
  `UnbalancedDesignError`.

- **DEV-R4-5 — The runner registry's only pre-R5 entry resolves into the
  `tests` package.** `REGISTERED_RUNNER_ENTRIES` maps
  `synthetic_search_job_fixture_v1` to the synthetic wiring factory that
  already lived in `tests/agents/ifvg_search/test_search_job_script.py`
  (DEV-R2-3's sanctioned synthetic path). In a development checkout it
  imports; anywhere it cannot, execution fails closed exactly like an
  unregistered entry. Raw `--runner-entry` strings are now refused BEFORE
  any import unless they byte-match a registered value; the UI only ever
  passes keys. The R5 pipeline registers the real executors.

- **DEV-R4-6 — Review-step "verified reuse hits" reads *resolved at
  launch*.** Counting reuse at preview would resolve every child's
  `CoreStrategyReplayIdentity`, which constructs `ReplayInputBundle`
  source hashes — path construction before authorization. Fail-before-path
  wins over a numeric preview; launch resolves reuse identity-deduped as
  R2 designed.

- **DEV-R4-7 — Insights render deterministically from persisted artifacts;
  the insights STORE stays empty until R5.** S14
  (`BUILD_FRONTIER_AND_INSIGHTS`) is a pipeline stage (R5). R4's panel is
  a pure function of the persisted frontier/metrics (identical text every
  render) and labels its derivation source in the panel; nothing writes
  research artifacts from a render path.

- **DEV-R4-8 — History's Clone as New Search clones the originating frozen
  DRAFT.** Every charter frozen through this workspace has one
  (`mark_frozen` stamps the linkage). A charter created outside the
  workspace (CLI/tests) has no draft; its row states "clone unavailable —
  originating draft not found" instead of reverse-engineering wizard
  steps from a charter payload.

- **DEV-R4-9 — A namespace selector (research `search/v1` vs verification
  `search_test/v1`) heads the study routes.** The verification contract
  publishes into `search_test/v1` (CS §6 `output_namespace`), so the
  monitor/results MUST be able to read it; FUX names no switch, so the
  selector is explicit, truthfully labeled ("Verification artifacts are
  never research evidence"), and verification-namespace Results render the
  non-dismissible verification badge. Synthetic-marker charters freeze
  only into the verification namespace (`save_charter` P0-4 confinement —
  witnessed by the wizard AppTest for both directions).

- **DEV-R4-10 — No preparatory lint-fix commit.** PHASED R4 anticipated
  one before widening Ruff to `scripts/`; measured reality:
  `ruff check scripts` was ALREADY clean at R3 HEAD, so the CI widening
  (`ruff check src tests scripts`) ships alone.

- **DEV-R4-11 — Account-simulation store wiring lands with UI sidecars**
  (`walk_summary.json`, `account_events.json` via
  `save_envelope_immutable`), closing DEV-R3-9's store half: manifests
  over these artifacts are now real (hashed + reverified on load).
  Production WRITERS for them are the R5 pipeline's S12/S13/S15;
  fixtures/tests are the only writers in R4. Scenario-mode bridging at
  the production seam (DEV-R3-11) remains R5 wiring.

- **DEV-R4-12 — Mode 5 (Full Pipeline Run) freezes nothing in R4.** Its
  wizard steps render and validate; step 8 renders the DEDICATED
  `pipeline_runner_planned` capability state (heading/explanation name the
  R5 operator workflow — the adversarial round found the first cut reusing
  the MBP-1 feature-block copy; fixed). The question↔mode compatibility
  matrix (each search mode answers its natural question; the pipeline mode
  accepts all four) is an engineering default recorded in DECISIONS_TAKEN.

- **DEV-R4-13 — Funnel counts and child stage cells derive from
  (state, failure_reason, exact orchestrator explanation sentinels).**
  The atomic state file carries no per-stage booleans; the derivation
  table (replay/strategy/prop stage reason sets, the reused-unevaluated
  and frontier-exclusion sentinels, the prop-crash `REPLAY`-on-completed
  distinction) is pinned by a contract test that scans the orchestrator
  source for the exact strings. The persisted frontier envelope is
  authoritative for Prop Feasible / Robust when present; before it
  exists, Robust renders `–` (unknown), never a guess.

- **DEV-R4-14 — A second, read-only subprocess seam exists in the wizard**
  (`_commit_of`: `git rev-parse HEAD`), found undocumented by the safety
  review (F1). Now: the `_spawn_search_job` docstring names it; the
  Strategy-Core root resolves from the INSTALLED `strategy_core` package
  (walking up to `.git`) instead of a case-guessed sibling folder; and a
  provenance resolution of `"unknown"` REFUSES the freeze (two source
  states must never share a charter identity through a silent fallback) —
  AppTest-witnessed.

- **DEV-R4-15 — Descriptive/specialized interpretations BLOCK enumeration
  in v1** (adversarial F2 closure). The FUX §10.5 selector is now
  enforced at validation: changed session/direction axes with a
  non-sequential interpretation refuse Next with the exact explanation
  (the CohortSpec descriptive flow is a study-lane surface, not an R4
  charter path). Nothing descriptive can silently become a sequential
  replay child.

- **DEV-R4-16 — Cross-profile population/funnel delta BUILDS and stored
  comparison-contract consumption are deferred to the release that
  persists their inputs** (upgraded from a scoping note per adversarial
  F4). What R4 delivers: `prepare_cross_profile_deltas` is the ONLY
  constructor (lineage-uniqueness persisted first, provider-proven); the
  ribbon derives compatibility/match_basis from the REGISTERED
  `derive_lineage_validity` (not a nearest match); incompatible pairs
  suppress the Strategy/Prop delta panels entirely (parameter diff only —
  AppTest-witnessed); injected reports render exact affected-ID actions.
  What is deferred: building lineage maps from live runs (their record
  tables are not persisted by R1–R3 stores) and consuming persisted
  `ComparisonResult` contracts — both land with the R5 pipeline stages
  that persist those inputs. FUX-RES-007's affected-ID rows therefore
  surface live only after R5; the acceptance row is carried on the gate
  summary as R4-partial/R5-completing.

- **DEV-R4-17 — Charter risk ids are CONTENT hashes of the complete
  per-firm policy dicts** (adversarial F7 closure): template + value +
  n_accounts + replacement hash into `authorized_risk_policy_ids`, so no
  result-changing parameter can be dropped from the charter identity; the
  frozen draft retains the resolved parameters as provenance. Real
  `AccountPolicySetEnvelope` construction (64-hex policy-set ids consumed
  by `AccountSimulationPayload`) is the R5 pipeline's S12 job.

## Scoping notes (not deviations)

- `enumerate_child_count` counts the LAUNCH enumeration (per-axis baseline
  value merged with the selected challengers) so the preview can never
  undercount what `enumerate_children` will produce.
- The wizard displays the protected boundary ("2026-06-11") as read-only
  COPY (FUX §14.1 requires showing it); no code constructs, lists, stats,
  or reads any path for it — see `ACCESS_SAFETY_EVIDENCE.md`.
- `catalog_annotations` display-name payloads are normalized (dict or bare
  string) because the R1 catalog stores the event payload verbatim.
- Post-review hardening (adversarial round, all witnessed by tests):
  draft-id allowlist (`^[0-9a-f]{32}$`) + file-vs-directory identity check
  + per-draft O_EXCL lock around every check-then-write transition; the
  account-timeline read path load-verifies its envelope before trusting
  sidecars; scope captions derive from the persisted `simulation_mode`;
  the explorer's "90-day survival" column displays 1 − breach; the
  heatmap aggregates coordinate collisions (mean + n, reported); the
  no-pass verdict renders only for terminal runs; wrong-state empty-state
  reuse replaced by dedicated presentations; UNC/quoted-space paths
  redact; the session-namespace and override-editor scans now cover
  f-string keys and assert real properties; cross-draft widget state is
  purged on draft switch.
