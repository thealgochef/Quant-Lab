# R6.1-FIX — Access-Safety Evidence

Implementer's audit, written before the adversarial round (2026-09-01) and
corrected after it: both reviewers' access-safety verdicts AFFIRMED the
first draft (Reviewer A: "the diff introduces no `data/`, store-root,
dataset or date constants … Strategy-Core worktree is clean; Trade-Lab's
dirty files date 2026-07-31 (pre-existing, not touched by this release)";
Reviewer B: no production `assert`, no broad recovery around an evidence
load, no store listing on the S02/S14 paths); the paragraphs below state
what the code does AFTER the fix round (`ADVERSARIAL_REVIEW_RESOLUTION.md`).

## Protected/sealed counters: ZERO

- **No real source path was ever constructed** during the R6.1-FIX
  implementation, its tests, the reviewers' probes, the fix round, the
  golden-identity run, or the full-suite runs. The release touches no
  data-access module: the R6.1-FIX file set (`FILES_TOUCHED.md`) contains no
  `data/databento`, `DEFAULT_DATA_DIR` or `IfvgCaptureConfig(` reference
  (`grep` over every touched `src` / `scripts` file: no hit), and the only
  real-path reader the regime lane can reach — the verified replay-chart
  pair loader behind the `context_bar_source` seam (R6.1) — is untouched.
- **Date literals in the R6.1-FIX set** (`grep` for `2026-06-1x` /
  `2026-05..09-xx` over every new / modified code and test file): the only
  hits are August-2026 ISO instants used as owner-decision `approved_at` /
  `effective_from` / `decided_at` / `declared_at` values in fixtures and the
  contract examples — instants from which no path is derived. The R1 guard
  constants (`charter.py` `_DEVELOPMENT_CUTOFF_DAY = "2026-06-10"`,
  `_PROTECTED_BUFFER_DAY = "2026-06-11"`) are unchanged at HEAD; `charter.py`
  is not in this release.
- **On disk**: `find data -type f -newermt "2026-09-01 19:00:00"` → **0
  files** (the R6.1-FIX baseline was recorded at 19:30 local; re-checked
  after the fix round). The 13 files under `data/` newer than the R6.1 audit
  are `data/ifvg_study_drafts/*/draft.json` written by the user's Streamlit
  session between 15:48 and 17:24 local on 2026-09-01 — untitled
  `fsm_config_search` drafts (the R4 mutable authoring surface;
  `frozen_search_id: null`), created BEFORE the R6.1-FIX baseline and not by
  any R6.1-FIX test (every test store is `tmp_path`-rooted). No `search`,
  `search_test`, `regime_*`, `executed_trade_tables`, `owner_decisions`,
  `fold_*` or `context_bar_panels` directory exists under `data/`
  (`find`-verified).
- **The new store is synthetic-only so far**: `executed_trade_tables` entries
  were written exclusively under `tmp_path` by the pipeline fixtures and the
  reviewers' probes; the store name is registered in `SEARCH_STORE_NAMES` and
  follows the manifest / immutability protocol (`save_or_reuse_envelope`;
  identical bytes reuse, different bytes under one id refuse; a tampered or
  manifest-less entry fails closed on probe AND load — corrupt is never
  absent, review B-02).
- **Strategy-Core clean at the pinned commit
  `a4e3303179ac6a1088aecaaa3482934cf1aec4d7`** (`git status --short` empty);
  **Trade-Lab untouched** by this release — its worktree carries the user's
  pre-existing dirty files dated 2026-07-31 (no file newer than 2026-08-30
  outside `.git` / `node_modules`; nothing from 2026-09-01/02). The plan
  package, `../R5B.1/`, `../R6/` and `../R6.1/` are unmodified.

## No automated selection or promotion

- Nothing in R6.1-FIX adds a selection, ranking, promotion, activation or
  launch path: `S11` stays blocked with its registered reason; S14 performs
  zero fitting (the parametrized `test_s14_performs_zero_fitting` runs
  unchanged); the stratified frontier remains descriptive; owner-decision
  authority (R6.1) is untouched.
- **Activation cannot ride a prior attempt's gates** (review B-04): a halted
  or cancelled attempt resets the state file's publication block and
  `activate_pipeline_result` re-derives the publication gates from the
  LATEST attempt before it activates anything.
- The typed `children_skipped` record (`child_not_completed_or_reused`,
  `strategy_gates_not_passed`, `executed_trade_table_unavailable`) makes S14's
  child set auditable against the charter — a child can no longer vanish from
  the stratified reports silently, a corrupt table is a typed failure, never
  a skip, and no prior attempt's report record is ever reused in place of
  this attempt's evidence (review B-03).
- `RegimeNetRAccounting` is arithmetic over per-trade net R, recomputed and
  checked by its own validator; it infers no statistical confidence for a
  thin regime (the reportability floor still types a thin stratum
  `insufficient_regime_partition`), so no regime becomes "reportable" or
  "selected" by this release.

## Verified evidence, never in-memory objects

- Every persisted artifact derived from a fit's assignments consumes the
  manifest-verified sidecar bytes: `persist_regime_fit` reuses a fit only when
  the candidate assignment bytes equal the stored sidecar byte-for-byte
  (different values under the same identity fail closed; joblib bytes are
  never rewritten); the executor exact-loads every fit it persisted
  (`verified_fit_assignments_for_run`, id + fold index re-checked) and the
  descriptive OOS artifact, the panel PIT assignment, S10's evidence as-of
  and the fold-local features consume ONLY those `VerifiedFitAssignments`
  (an in-memory frame is refused by type — `TypeError` "not evidence").
- Identity binds bytes: `FitAssignmentRef` / `FoldFitRef` carry the sidecar
  SHA-256 and the enforced schema hash; the OOS payload's
  `consulted_assignments_hash` covers every consulted value and the payload
  records `candidate_as_of_stage`; the stratified reports pin
  `assignment_table_sha256` + `assignment_schema_hash`; the fold-feature
  loader re-checks every ref against the store by exact id (a tampered or
  re-typed sidecar fails the load closed); every assignment row is linked
  and arithmetically self-consistent (review RA-06).
- Executed trades: S02 persists the exact 42-column projection after a fresh
  completion (the neutrality report's core-table hash must agree) and after a
  verified reproduction (projection bytes AND raw core-table hash), exact-loads
  it back and computes EVERY costed evaluation from the loaded projection
  (review B-01); S14 verified-loads every gated child's table by its DERIVED
  id (no listing) and re-checks the hash S02 recorded; the stratification
  service re-verifies the caller's frame against the artifact and binds the
  artifact's projection hash into the `cohort_descriptive` body (review
  RA-01); the body hashes, joins and stratifies the NORMALIZED projection
  only.
- Prior-stage sidecars (prop vectors, account simulations, lineage maps, the
  S09 run record) load through the typed probe: corruption, manifest /
  identity mismatch, a manifest-less entry and I/O errors propagate as
  sanitized typed stage failures (`load_verified_envelope` and
  `load_sidecar_bytes` raise the reason at the detection point — review
  B-08); only a manifest-proven "not produced" sidecar is optional; S15
  records every reload failure by store / id in the state file AND
  immutably on the pipeline result (`reload_failure_reasons`, review B-09),
  re-loading every executed-trade table and stratified report the S14 record
  names; a halted attempt marks every later planned stage PENDING and resets
  the publication block.

## Determinism and store discipline

- The enforced `FIT_ASSIGNMENT_SCHEMA` makes the persisted sidecar bytes
  depend on the values and the declared schema only; the executed-trade
  projection is declared, typed and sorted by `trade_id` (mergesort) so
  identical replays produce identical bytes and identical costed-evaluation
  bytes regardless of run order (`test_costed_evaluations_are_provenance_independent_across_run_orders`);
  the double-run tests prove REUSED stages with identical stage-result ids —
  zero replay for non-stratified reuse and exactly one verified reproduction
  per reused child under stratified reporting (`replay_invocations` asserted;
  DEV-R6.1-FIX-20).
- One new store (`executed_trade_tables`) — manifest protocol, exact-id loads,
  never listed, relocation-safe, tamper fails closed
  (`test_a_tampered_trade_table_is_refused_not_skipped`,
  `test_every_corruption_state_is_a_typed_failure`,
  `test_a_manifest_less_entry_is_corrupt_never_absent`). `has_envelope` now
  raises typed on a manifest-less existing entry (its UI callers in
  `study_providers` already sanitize errors). The UI
  (`scripts/ifvg_regime_panels.py`) still loads exact ids only and never
  unpickles.

## Frozen lanes

`ifvg/context_model.py` and the M0–M3 lane are byte-unchanged (`git status` /
`git diff HEAD --stat` list no M0–M3 lane file) and the frozen M0 CatBoost
`resolved_hash` is golden-tested unchanged (`_goldens_pytest.txt`); the R6
golden `regime_fit_id`, `resolved_regime_protocol_id`, `core_replay_id`,
`account_simulation_id` and `feature_block_registry_hash` are unchanged; the
frozen-tier ladder ids do not move (the ladder identity's `label_content_hash`
key is unchanged; only helper runs without an exact label id re-mint their
bundle-path ladder ids — review RA-05); Strategy-Core / Trade-Lab untouched;
no new package dependency.
