# R5B — Access-Safety Evidence

Implementer's audit; independently corroborated (or challenged) by the
safety-lens adversarial reviewer in `ADVERSARIAL_REVIEW.md`.

## Protected/sealed counters: ZERO

- **No real source path was ever constructed** during R5B implementation
  or its tests. Every MBP-1 suite runs on synthetic in-memory frames
  (`tests/agents/ifvg_search/mbp1_fixture.py` — crafted events on the
  synthetic days `2026-01-13…15`, the same fixture window every release
  has used) with store writes to pytest `tmp_path` roots only.
- **2026-06-11 and the sealed range**: never constructed, listed, stat'ed,
  opened, or read. The only date literals in the new code/tests are the
  synthetic fixture days and the D-046 doc text naming the candidate
  allowlist as a proposal.
- **Authorize-before-path is structural** in the one real-source seam:
  `read_mbp1_partition_frame` refuses `access_policy=None` BEFORE invoking
  the path factory (test-witnessed: the factory records zero calls), and a
  denying policy refuses before any filesystem call
  (`test_denied_day_refuses_before_any_path_construction`). The real
  builder (`build_mbp1_source_artifact_from_paths`) refuses a missing
  policy and an empty day set at the top of the function (review S3) and
  reaches paths only through `access_policy.resolve_source_path` — the
  verification policy family's gate — validating the pinned source schema
  on the parquet header before any rows decode.
- **Ingestion-side depth boundary (review S1)**: a file whose ACTUAL
  columns carry any book level beyond 00 (a legacy-era mbp10-shaped
  parquet) is refused before any row decodes, and even a lawful file is
  read through the pinned mbp-1 column projection — deeper-book data is
  never materialized in memory, complementing the exposure-side schema
  guards. The deep-book identifier regex was strengthened to cover every
  depth beyond one (mbp2…mbp9 included), keeping the plan's mandated
  pattern as a strict subset (review S2).
- **The real pipeline entry wires NO MBP-1 evidence source**
  (`pipeline_baseline_verification_entry` — DEV-R5B-1): a real MBP-1 plan
  refuses at S00 with the exact missing-seam reason. The real five-day
  MBP-1 control-flow verification stays blocked on the owner's
  `VerificationAuthorizationRef`, exactly like every real half since R1.

## No full-development runs

No full-development replay, feature materialization, model fit,
configuration search, prop search, or operator pipeline ran. The largest
computations this release: the synthetic three-day pipeline E2E (24
candidates) and the 60-synthetic-day controlled-study unit fixture
(240 candidates, logistic fits only) — both entirely synthetic,
tmp-rooted, and seconds-to-minutes scale.

## Order-flow boundary (kickoff §9)

- MBP-1 remains the maximum representable depth: the deep-book guard runs
  at import over every new schema/registry/feature namespace
  (`assert_no_deep_book_identifiers` in `mbp1_arrow_schemas`,
  `feature_blocks`, `mbp1_feature_join`), and the four new Arrow schemas
  expose top-of-book fields only (`*_00`).
- `legacy_verified_replay_source` is refused outright at the
  materialization layer (`_require_no_legacy_provenance`), is
  unrepresentable in `Mbp1SourceContract` (Literal `"mbp-1"`), and remains
  a provenance-only literal elsewhere (R1 guards re-run green at R5B).
- `research_only_offline` is structural: the activated definition carries
  `expected_computation_path="offline_research_feature_materialization_v1"`
  and `can_affect_execution=False` (import-time asserted); S11 stays
  BLOCKED with the exact registered reason; no live, serving, order,
  promotion, or Trade-Lab control exists on any new surface (the MBP-1
  panels are read-only exact-ID views + the persistent boundary badge).

## Store discipline

- The four new stores (`mbp1_source_artifacts`, `mbp1_feature_artifacts`,
  `mbp1_coverage_reports`, `controlled_feature_studies`) follow the
  manifest protocol (refuse-if-exists → tmp write → manifest →
  `os.replace` → reload-verify) via `save_or_reuse_envelope`; sidecar
  bytes are manifest-hashed AND rehashed against envelope facts on load
  (`load_partition_events`, `load_feature_frame`,
  `load_stage_evidence_frame`, `load_controlled_study_detail`).
- Tests write only under `tmp_path`/`tmp_path_factory` roots and the
  keyed `%TEMP%` smoke scratch; the repo's `data/` namespaces receive no
  writes (verified by the safety reviewer's on-disk audit).
- The UI performs exact-ID reads only; store roots are never listed; text
  inputs cannot traverse (`envelope_destination` accepts full 64-hex ids
  only).

## Frozen lanes

- Strategy-Core and Trade-Lab: untouched by this release (no path outside
  the repo was modified; no Trade-Lab file exists in the change set;
  `git -C ..\Strategy-Core status` clean). The safety reviewer's on-disk
  audit additionally observed that the Trade-Lab WORKING TREE carries
  PRE-EXISTING user modifications (~27 files, every mtime ≤ 2026-07-31 —
  a month before this session) — user-owned prior work, acknowledged here
  per the reviewer's S5 finding, not touched or absorbed by R5B.
- M0–M3 lane modules (`context_model.py`, `context_feature_view.py`,
  `context_folds.py`, `context_experiment_contracts.py`, and the rest):
  byte-unchanged this release — the CatBoost fold runner's tier lock is
  honored by REFUSING bundle parametrization rather than editing the lane.
- The evaluation-only propsim API: untouched (no propsim file in the R5B
  change set).
- The plan package: unmodified.

## Shim alignment cannot widen access (DEV-R5-10 closure)

`ifvg_search_job.py` now passes the worker's `--store-root` to
runner-entry factories. The real search entry still fails BEFORE path
construction without a persisted `VerificationRunEnvelope` discovered
through the provider at whatever root it is given — an arbitrary store
root yields the same fail-before-path refusal, never data access; the
synthetic entry ignores the argument. Registry validation (production
keys unshadowable, exact `module:function`) is unchanged.
