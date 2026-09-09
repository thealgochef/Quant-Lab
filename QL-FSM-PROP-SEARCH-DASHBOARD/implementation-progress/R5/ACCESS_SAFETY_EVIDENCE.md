# R5 — Access-safety evidence

**Protected/sealed counters: ZERO at the R5 layer.**

## What R5 can and cannot touch

- **No real source path is constructible in anything R5 executed.** Every
  R5 test runs on synthetic fixtures under pytest `tmp_path` roots; the
  pipeline E2E's store/state roots are tmp-only; the ml fixtures are
  `numpy.random.default_rng(7)` synthetic frames. Nothing reads
  `data/databento/`, the real `data/ifvg_datasets/` stores, or any source
  parquet.
- **2026-06-11 / sealed range:** no R5 code or test constructs, lists,
  stats, opens, or reads either. The only occurrences of the protected
  date in the R5 diff are the pre-existing plan-required read-only display
  copy (unchanged from R4) and charter validation REFUSALS
  (`DatePolicy._dates_are_lawful` — unchanged).
- **The real executors registered this release fail BEFORE path
  construction** (`search/executors.py`): without a persisted
  `VerificationRunEnvelope` the factory raises `PermissionError` before
  any `IfvgCaptureConfig`, `VerificationReplayPolicy`, or path exists —
  registration unblocked the LAUNCH SURFACE only, never the data
  (test-witnessed: `test_real_executor_factories_fail_before_any_source_path`).
  Discovery goes through `verification_authorization_state` (persisted
  envelopes/catalog; immutable store roots are never listed).
- **No full-development execution:** `full_authorized_development` has NO
  registered runner entry (search or pipeline); the S00 executor refuses
  the synthetic marker under development scope; the UI's full-scope launch
  requires the exact §15 typed confirmation AND then renders the
  no-registered-executor state. `full_pipeline_not_run=true` is stamped on
  every verification pipeline result and state file.
- **Verification honesty:** the synthetic E2E branch stamps
  `verification_only=true`, `not_for_research_interpretation=true`,
  `synthetic_fixture_ids=[charter id]`, and — after the safety review's
  F2 fix — `real_date_count=0` with a separate `synthetic_date_count`
  (synthetic fixture days are never counted as real); publication stays
  `prepared_not_published`; activation for verification scope raises
  `PublicationError` with no override (test- and CLI-witnessed). The S15
  `zero_forbidden_counters` gate is DERIVED (safety review F1): every
  drive asserts its own zero counters before returning, a tripped
  assertion is recorded at the per-child containment and fails the gate —
  test-witnessed
  (`test_tripped_access_assertion_fails_the_derived_counter_gate`) — so
  the persisted report can never contradict the audit it names.
- **Synthetic contracts stay synthetic:** the pipeline's prop stages use
  `SYNTHETIC_FIXTURE_FIRM` via the canonical `synthetic_firm_specs()`;
  nothing constructs or advances a `first_party_verified` status (the R3
  ladder cap is untouched and re-exercised by the unchanged propsim
  suites).
- **MBP-1 depth:** `IFVG_ORDER_FLOW_MBP1_V1` remains `planned`; every
  bundle carrying it refuses at resolution, in the stage-plan readiness,
  and in the UI (visible-disabled); the deep-book identifier guard runs on
  every bundle-view name set (defense in depth).
- **Launch surfaces:** exactly TWO detached `subprocess.Popen` seams exist
  in the UI lane (`_spawn_search_job`, R4; `_spawn_pipeline_job`, R5) and
  one per job shim — all pinned by the widened source scan
  (`test_process_launch_exists_only_in_the_designated_seams`). The worker
  executes registry-resolved entries only; raw `module:function` strings
  are refused before any import. Importing either shim or any tab launches
  nothing (AST-scanned).
- **Repository boundaries:** Strategy-Core and Trade-Lab untouched (R5
  reads only the installed `strategy_core` package for identity, as
  before); every M0–M3 lane module byte-unchanged; the evaluation-only
  propsim API untouched (`search_bridge` changes are additive keyword
  parameters whose defaults reproduce R3 byte-for-byte — existing suites
  green unmodified).
- **Worktree writes:** every artifact the R5 tests produced lives under
  pytest tmp directories; the repo's `data/` tree is byte-unchanged from
  the R4 baseline (`git status` untracked set identical). No push, no
  merge.

## Corroboration

The independent safety-lens adversarial reviewer's verdict and grep audit
are recorded in `ADVERSARIAL_REVIEW.md`; resolutions in
`ADVERSARIAL_REVIEW_RESOLUTION.md`.
