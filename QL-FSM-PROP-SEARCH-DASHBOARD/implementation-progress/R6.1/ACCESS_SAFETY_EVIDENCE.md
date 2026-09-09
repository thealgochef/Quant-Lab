# R6.1 — Access-Safety Evidence

Implementer's audit, written before the adversarial round and corrected
after it: the safety-lens reviewer (`ADVERSARIAL_REVIEW.md`, Reviewer 2)
challenged the supersession, S02-reproduction, "only spawn" and
"golden-tested" statements of the first draft (S2 / S4 / S8); the
paragraphs below state what the code does AFTER the fix round
(`ADVERSARIAL_REVIEW_RESOLUTION.md`).

## Protected/sealed counters: ZERO

- **No real source path was ever constructed** during R6.1 implementation,
  its tests, the release-final full-suite runs, or the browser smoke. The
  ONLY real-path reader the regime lane can reach is the replay-chart pair
  loader (`load_verified_replay_chart_artifact(_v2)`) behind the pipeline's
  `context_bar_source` seam, which the synthetic fixtures satisfy with a
  synthetic VERIFIED artifact written under `tmp_path` / the `%TEMP%` smoke
  scratch through the production resampler and identity functions. The
  panel materializer accepts ONLY a `VerifiedReplayChartArtifact` (a frame is
  refused by type), re-reads `bars_tf.parquet` from the artifact directory,
  and rehashes it against the manifest before any bar is interpreted.
- **Date literals in the R6.1 set** (`grep` over every new/modified code,
  test, and harness file): none protected/sealed. The one hit,
  `search/charter.py:65 _PROTECTED_BUFFER_DAY = "2026-06-11"`, is the
  pre-existing R1 guard constant (unchanged at HEAD `f3f9ac2`; charter.py
  changed only for the D15 `SimulationProtocol` policy field). Every fixture
  day is a synthetic business day from `2026-01-05`; decision timestamps are
  August 2026 instants from which no path is derived.
- **On disk**: `find data -type f -newermt 2026-08-25` → **0 files**; no
  `regime_*`, `context_bar_panels`, `owner_decisions`, `fold_*`,
  `bundle_feature_views`, `search` or `search_test` directory exists under
  `data/`; every store write in tests is `tmp_path`-rooted and the smoke
  writes only under the content-addressed `%TEMP%\ifvg_r61_smoke\<key>`.
- **No real-path construction in the new source set** (`grep` for
  `data/databento`, `DEFAULT_DATA_DIR`, `IfvgCaptureConfig(` over every
  new/modified `src/` file: no hit). The bounded MBP-1 diagnostic and the
  five-day regime mini-run remain the owner-blocked real slice (plan §12
  blocker 1): nothing in R6.1 runs without the `VerificationAuthorizationRef`.
- **Strategy-Core clean; Trade-Lab untouched** by this release (its 27
  pre-existing user worktree entries; the only files newer than 2026-08-01
  are the user's frontend build outputs of 2026-08-04 and a pytest cache —
  nothing from 2026-08-28/29). The plan package is unmodified; the R5B.1 and
  R6 evidence folders are never edited.

## No automated selection or promotion (kickoff §9 ML/regime)

- Cluster count stays `fixed_k`; every planned algorithm/policy still
  refuses before any preprocessing or fit (R6 invariants re-run green).
- **Promotion authority is verified evidence, never a string** (D5):
  `persist_regime_promotion` loads the owner-decision artifact through the
  manifest-checked `owner_decisions` store, re-verifies decisions 25/28/29/30
  (incl. the exact pinned KMeans parameter snapshot/hash) against the loaded
  registry / protocol / assessment, the transition, effectivity, and the
  store-owned supersession chain — hash-chained lines + `SUPERSESSIONS.head`,
  appended BEFORE the replacement is published; any edited / deleted /
  reordered line, a missing head, a dangling line or a forged line fails the
  whole chain closed (the only undefended case, deleting log AND head
  together, is the DEV-R6-8 trust boundary — DEV-R6.1-15); a bare 64-hex
  reference is refused; `MODEL_FEATURE` is unpersistable in V1; STRATIFICATION_READY
  structurally requires passing coverage gates + an OOS assignment at
  persistence and is re-derived from the loaded assessment at the report
  gate; synthetic provenance is lawful only in the `synthetic_fixture` run
  scope, which `assert_run_scope_lawful_for_root` confines to test
  namespaces (a `search` segment other than `search_test` refuses the scope
  and refuses to persist OR load a synthetic-provenance owner artifact); the
  pipeline's readiness / S05 / S09a / S10 / S14 gate / block activation and
  the CLI all pass the run scope through; `decided_at` derives from verified
  artifacts (never a flag or the wall clock).
- **No "latest" lookup exists**: model-bearing requests freeze the exact
  promotion / owner / assessment ids into the semantic identity (the tab's
  Preview and Launch handler verify that authority with the store root and
  run scope BEFORE any charter / spec envelope is saved or a job is spawned —
  review S7); descriptive runs derive their status from their OWN assessment
  at the evidence as-of instant (never the wall clock); `IFVG_REGIME_CONTEXT_V1`
  activates only as a pure event bound to the frozen ids; `model_feature` is
  refused by the store AND the CLI with the exact shared reason; S11 stays
  blocked.
- **S14 performs zero fitting** (test-enforced by monkeypatched estimators on
  the descriptive, supervised-candidate and supervised-panel runs — review F4);
  S10's STRATIFICATION_READY is structurally reachable only when the coverage
  gates passed AND an OOS assignment exists AND descriptive classes were
  requested; the stratified frontier view is never a selection input; the
  Regime Lane and the Monitor's regime expander carry no button / form /
  toggle / select (source-scanned); no new spawn site — the two
  `_spawn_pipeline_job(` call sites (Launch, Resume/Retry) pre-exist at HEAD
  and are unchanged.

## Verified reuse, never rewriting

- Fits: an existing fit identity is reloaded and must reproduce this fit's
  transform and labels from its verified bytes — equal → reused, different →
  refused; joblib bytes are never rewritten.
- Reused children needed by stratified reports are re-derived through the
  wired runner; the re-derived tables become this run's evidence ONLY when
  they reproduce the persisted costed evaluation of THIS cost policy
  (metrics byte-equal) — with no such evaluation, or a mismatch, the tables
  are not adopted (the child stays reused, no gates evaluated, no evaluation
  published, out of the stratified reports; review S4 / DEV-R6.1-8); the
  immutable core replay is never rewritten. S05 binds the materialized
  panel to the chart it asked for (id + pair; review S1) and S14's panel
  assigner loads every artifact by exact id (review F14).
- Prior-attempt S14 reports are recovered only by verified reload of the
  exact ids in the prior stage sidecar; S12/S13 record the exact persisted
  account-simulation ids (never a store listing).
- D15: budgets fail BEFORE atomic publication with no partial artifact
  (the writer streams one path block at a time into the store's temporary
  publication directory through the sidecar-producer protocol; the store
  re-hashes every produced file and refuses any bookkeeping lie, stray or
  ghost file with nothing published — DEV-R6.1-14); `none_v0` artifacts are
  never widened (the store refuses different sidecars under one id); readers
  verify bytes / rows / schema against the detail manifest behind the store
  manifest; the stratified-prop report's `account_event_regime_summary.parquet`
  is bounded by its own registered budget and verified on load (DEV-R6.1-16).

## Determinism as a safety property

`run_regime_protocol` executes under `threadpool_limits(1)` so the persisted
assignment tables (and every artifact hashing them) are byte-reproducible
across attempts and machines — the double-run pipeline test proves every
stage REUSED with identical stage-result ids; nothing about fit identity or
science changes (DEV-R6.1-4).

## Store discipline and UI safety

- Ten new stores (`bundle_feature_views`, `context_bar_panels`,
  `fold_schedules`, `fold_sets`, `regime_oos_assignments`,
  `regime_fold_features`, `owner_decisions`, `regime_stratified_reports`,
  `regime_controlled_studies`, `regime_cohort_model_studies`) follow the
  manifest protocol via `save_or_reuse_envelope`; every sidecar is rehashed
  on load; relocation-tested; tamper fails closed; every identity pair is
  enumerated by the projection audit (deep-immutable payloads — the nested
  owner-decision values are `ImmutableMap`).
- The UI loads exact ids only (64-hex validated at the store layer; stores
  never listed), never unpickles (JSON + Arrow readers), sanitizes errors,
  and surfaces a store-integrity note when an auto-fill sidecar fails
  verification.
- Trust boundary (DEV-R6-8) stated in-tree: integrity is not authenticity;
  signing / non-pickle fit serialization remains a hardening candidate.

## Frozen lanes

`ifvg/context_model.py` is byte-unchanged (`git status` / `git diff HEAD
--stat` list no M0–M3 lane file; there is no source-hash test) and the
M0–M3 CatBoost protocol hash is unchanged (golden); the tier-path ladder ids
are unchanged; propsim's public API is compatible (new keyword-only fields
with `none_v0` defaults); Strategy-Core / Trade-Lab untouched; no new
package dependency (`threadpoolctl` ships with scikit-learn).
