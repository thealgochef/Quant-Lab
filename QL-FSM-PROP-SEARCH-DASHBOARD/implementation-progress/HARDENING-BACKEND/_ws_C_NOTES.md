# WS-C notes — §4.5 warning policy (F-18) + §4.6 sequential-execution truth (F-20)

Written 2026-09-02 by the WS-C workstream of HARDENING-BACKEND. Everything
below states what the CODE does; the main agent folds it into
`FILES_TOUCHED.md`, `DEVIATIONS.md`, `TEST_RESULTS.md`.

## Files touched

| File | Change |
|---|---|
| `pyproject.toml` | `[tool.pytest.ini_options].filterwarnings = ["error", <one exact rule>]` — `hardening_warning_policy_v1`: warnings are errors; the ONLY ignore rule is the sklearn 1.7.0 / SciPy 1.16 L-BFGS-B `disp`/`iprint` `DeprecationWarning` (exact message regex with the colon as `\x3a`, category `DeprecationWarning`, module `sklearn\.linear_model\._logistic`); the reason, owner and expiry condition are in the TOML comment and `WARNING_BASELINE.json` |
| `src/alpha_lab/agents/data_infra/ifvg/dataset.py` | `concat_schema_aligned(frames)` (exported) + `_na_capable` / `_common_with_all_na_entry`: the explicit schema-aligned concat of the per-day audit-channel frames in `assemble_fsm_audit_tables` (the `:813` FutureWarning); reproduces the pre-deprecation result exactly, never drops an all-null column; `import numpy as np` added |
| `src/alpha_lab/agents/data_infra/ifvg/search/pipeline.py` | `SUPPORTED_CHILD_WORKERS = 1`, `EXECUTION_MODE_V1 = "sequential_children_v1"`, `UNSUPPORTED_WORKER_PARALLELISM_REASON`, typed `UnsupportedWorkerParallelismError(ValueError)` (`.reason`, `.requested_workers`), `assert_supported_worker_parallelism`, `worker_parallelism_refusal` (extracts the typed error from a pydantic `ValidationError`); `WorkerPolicy.max_workers` keeps its field (`ge=1`; the `le=4` ceiling is gone) and a model validator refuses any value other than 1 with the typed error (never coerced); `ExecutionAttemptIdentity.effective_workers: Literal[1]` + `execution_mode: Literal["sequential_children_v1"]` (operational metadata; the contract stays unregistered); `run_pipeline` sets both on every attempt receipt; `__all__` extended |
| `scripts/ifvg_pipeline_job.py` | `_refuse_unsupported_parallelism`: `start` / `resume` refuse `--max-workers != 1` BEFORE the job directory or the detached worker exist; `worker` refuses BEFORE any store load — JSON `{"status": "refused", "reason": "unsupported_worker_parallelism_v1", "requested_workers", "supported_child_workers": 1, "execution_mode", "detail"}`, exit code 2; `--max-workers` help text states the V1 capability |
| `tests/agents/test_hardening_warning_policy.py` (new, 11 tests) | the concat helper against the old result on every warning shape (all-None object vs float / bool / tz datetime; all-null column everywhere; absent column; all-NaN float vs strings; int / bool vs all-None and all-NaN entries; empty inputs) and on the REAL synthetic FSM-chain audit frames (`assert_frame_equal`, dtype-checked, under `simplefilter("error")`); the typed-row append of the test-side fix; the registered pytest policy (error + one exact rule; agrees with `WARNING_BASELINE.json`); the typed worker refusal (2 / 4 / 8 refused, 1 accepted) |
| `tests/agents/ifvg_search/test_pipeline_contracts.py` | `test_worker_policy_rejects_more_than_four_workers` → `test_worker_policy_refuses_any_parallelism_with_the_typed_reason` (2 / 4 / 8 refused with `reason` + `requested_workers`; 1 accepted); new `test_attempt_receipt_states_sequential_execution_truthfully` (receipt fields; `effective_workers=2` unconstructible) |
| `tests/agents/ifvg_search/test_pipeline_run.py` | the attempt-identity test clones resources WITHOUT parallelism (`max_workers=1, max_tasks_per_child=2, memory_budget_bytes=1 << 31`); its semantic-identity and `replay_invocations == 0` assertions stand; every attempt receipt in the state file is asserted to carry `effective_workers == 1`, `execution_mode == "sequential_children_v1"`, `worker_policy.max_workers == 1` |
| `tests/agents/ifvg_search/test_pipeline_job_script.py` | the worker end-to-end test asserts the receipt fields in `pipeline_state.json`; new `test_start_and_resume_refuse_parallelism_before_spawning` (parametrized start / resume; `subprocess.Popen` spied; nothing spawned; no state dir) and `test_worker_refuses_parallelism_before_any_store_access` (a nonexistent store root is never touched) |
| `tests/agents/test_ifvg_context_experiment_engine.py` | `append_censored_row(frame, row)`: the typed one-row append replacing `frame.loc[len(frame)] = censored` (the `:260` FutureWarning); the old result (object `binary_target`, tz-aware NaT) is reproduced exactly |
| `WARNING_BASELINE.json`, `WARNING_POLICY_REPORT.md`, `_red_ws_C.txt`, `_ws_C_pytest_core.txt`, `_ws_C_pytest_ifvg_search.txt`, `_ws_C_pytest_data_infra_ifvg.txt`, `_ws_C_pytest_propsim_ifvg_misc.txt` | evidence |

Not touched (by directive): `scripts/ifvg_pipeline_tab.py`, `search/store.py`,
`search/identities.py`, docs, git state.

## Deviations / decisions

1. **The typed error rides inside pydantic's `ValidationError`.** pydantic v2
   converts a `ValueError` raised in a validator into `ValidationError`, so
   `WorkerPolicy(max_workers=2)` raises `ValidationError`; the original
   `UnsupportedWorkerParallelismError` (with `.reason`) is carried in the
   error's `ctx.error` and `worker_parallelism_refusal(error)` extracts it.
   The plain function `assert_supported_worker_parallelism` raises the typed
   error directly — that is what the job shim calls BEFORE any object exists.
2. **`WorkerPolicy.max_workers` lost its `le=4` ceiling**: the ceiling was a
   claim of parallelism the executor never had; the validator's `== 1` rule
   replaces it (the R5 UI slider's 1–4 range is the UI plan's concern — see
   the UI note below).
3. **The message regex encodes the colon as `\x3a`** because pytest splits an
   ini `filterwarnings` rule on `:` (a literal colon in the message would be
   parsed as the category field). The rule still matches the exact message
   at its start (case-insensitively, per the `warnings` module) and nothing
   else.
4. **`concat_schema_aligned` reproduces pandas' PRE-deprecation semantics
   exactly, including the odd corner**: an all-NA present entry joining a
   numpy int / bool column yields the common type of the RAW value concat
   (int + all-None object → object with the ints and `None`s kept; int +
   all-NaN float → float64), because the old path only excluded the entry
   from dtype determination and then concatenated its raw values. This is
   proven against the old call on the real synthetic audit frames and on
   every synthetic shape; no frozen table changed (the audit tables the
   goldens hash are cut from this frame).
5. **Only the two project-owned warning sites were touched** (`dataset.py`
   `:813` and the test file `:260`); the six baseline project-owned warnings
   all originate there (5 + 1).

## UI note (not changed; belongs to the UI/UX redesign plan)

`scripts/ifvg_pipeline_tab.py` still renders the "Worker limit" slider with
`min_value=1, max_value=4` (lines ~693 and ~1660) and forwards the chosen
value as `--max-workers` to `ifvg_pipeline_job.py start` / `resume`. With
HARDENING-BACKEND the backend refuses any value other than 1 BEFORE job
creation (JSON refusal, exit 2, reason `unsupported_worker_parallelism_v1`),
so a slider value of 2–4 now yields a typed refusal instead of a launch. The
slider itself is governed by the separate UI plan (this plan makes no
Streamlit change). `max_workers=` values > 1 elsewhere: none in `src/`;
`scripts/ingest_databento_batch.py` and `scripts/w3_cache_warmer.py` pass
`ProcessPoolExecutor(max_workers=args.workers)` for DATA INGEST / cache
warming — unrelated to the search-lane child executor and out of scope.

## Test evidence

Red first: `_red_ws_C.txt` — the new/changed modules failed collection
(`ImportError: concat_schema_aligned`, `ImportError: EXECUTION_MODE_V1`)
before the code landed (pytest interrupted at collection, so the job-script
refusal tests were not reached in that run; they were red for the same
reason — `_refuse_unsupported_parallelism` did not exist).

Green (under the registered policy, warnings as errors):

- `_ws_C_pytest_core.txt`: the hardening module + pipeline contracts / job
  script / run + context-experiment engine + FSM audit contracts + child
  audit companion + dataset + v2 replay reconciliation + goldens +
  `test_ml_pipeline` (the `pytest.warns` module) + logistic model —
  **117 passed, 2 failed**; the 2 failures are
  `test_child_audit_companion.py::test_slice_companions_close_the_dev_r1_6_gates_synthetically`
  and `::test_slice_companions_require_repository_states`, which construct
  `VerificationAuthorizationRef` without the fields WS-A added concurrently
  (`store_namespace_id`, `supersession_head_witness`) — WS-A's fixture
  update, not a WS-C regression (they do not touch the concat or the worker
  policy).
- `_ws_C_pytest_ifvg_search.txt` — `tests/agents/ifvg_search` (the four
  modules other workstreams were rewriting at the time ignored:
  `test_owner_decisions` [WS-A], `test_seed_production` /
  `test_trading_calendar` / `test_verification_window` [WS-D]):
  **484 passed, 9 failed, 5 errors in 6:56**. Every failure / error is the
  concurrent WS-A contract change, not a warning and not WS-C code:
  `VerificationAuthorizationRef` / `OwnerAuthorizationBundle` now REQUIRE
  `store_namespace_id` + `supersession_head_witness` (test_authorization,
  test_charter, test_child_audit_companion ×2, test_mbp1_coverage_evidence
  ×2, test_study_providers, the five `test_verification_slice` setup
  errors) and `validate_verification_run(..., store_root=)` gained a
  required argument (test_verification ×2). Those fixtures are WS-A's to
  update.
- `_ws_C_pytest_data_infra_ifvg.txt` — `tests/agents/data_infra/ifvg`:
  **212 passed, 5 failed in 11:21**: `test_regime_store.py::test_promotion_persists_only_against_verified_evidence`
  (WS-A: `store_namespace_missing` — the owner-decision store now requires
  the marked namespace) and the four
  `test_stratified_prop_external_aggregation.py` tests (WS-B's red-first
  module for the external DuckDB aggregation, still in progress at run time).
- `_ws_C_pytest_propsim_ifvg_misc.txt` — `tests/propsim` + every
  `tests/agents/test_ifvg_*.py` module + `test_ml_pipeline`:
  **539 passed, 1 failed in 4:19**:
  `test_ifvg_regime_promotion_cli.py::test_feature_eligible_requires_verified_owner_evidence_in_a_lawful_scope`
  (WS-A: the CLI's refusal text is now the typed
  `store_namespace_missing` message).
- **Zero warnings surfaced in any run** (no pytest "warnings summary"
  section anywhere): under `filterwarnings = error` every warning would
  have been a failure, and none of the 1,352 collected tests across the
  four runs failed for a warning — the six project-owned sites are closed
  and the single third-party rule covers exactly the sklearn / SciPy
  deprecation.

Lint: `ruff check` over every touched file → `All checks passed!`.

Test modules WS-C did NOT run (the main agent's full-suite gate covers them):
everything outside `tests/agents/ifvg_search`, `tests/agents/data_infra/ifvg`,
`tests/propsim`, `tests/agents/test_ifvg_*.py`, `tests/agents/test_ml_pipeline.py`
— i.e. `tests/agents/test_data_infra.py`, `test_databento_*`,
`test_decision_repoint_parity`, `test_execution`, `test_ingest_databento_batch`,
`test_ml_extrema`, `test_ml_features`, `test_ml_training_reporting`,
`test_monitoring`, `test_orchestrator`, `test_signal_eng`,
`test_strategy_contract_*`, `test_stream_labeling_e2e`, `test_tick_store`,
`test_validation`, `test_cache_seed_guard`, and every non-`agents` test
package. Any warning those emit will surface as a failure in the gate run
and must be fixed at the source (or, if third-party, added as an exact rule
with owner / expiry) — not suppressed broadly.

## For the main agent

- Fold the UI-slider note into `DEVIATIONS.md` and the `WorkerPolicy`
  ceiling change into `FILES_TOUCHED.md` / `docs` (pipeline_state:
  `execution_mode: sequential_children_v1`, `supported_child_workers: 1`).
- Update `test_child_audit_companion.py`'s `VerificationAuthorizationRef`
  fixture for the WS-A namespace / witness fields (WS-A's lane).
- The two `_ws_C_pytest_*` directory runs were executed while WS-A / WS-B /
  WS-D were still editing their own modules in the same working tree; any
  failure in a module owned by another workstream is theirs to read.
