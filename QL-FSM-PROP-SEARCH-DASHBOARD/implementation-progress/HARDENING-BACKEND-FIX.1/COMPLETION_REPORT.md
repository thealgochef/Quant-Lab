# HARDENING-BACKEND-FIX.1 — Completion Report

**Feature:** `ifvg_prop_robust_config_search_v1`
**Release type:** backend micro-release (the four remaining code issues of the HARDENING-BACKEND-FIX independent review + closure of the open capacity gate; nothing else) — plus the one blocker the owner's independent complete lock-safety review found in the committed result (R3, §13: a failed Win32 exit-code query was classified dead), corrected in a follow-up commit with the proportionate gates rerun (§11, §13).
**Authority:** the task instructions for HARDENING-BACKEND-FIX.1 (verbatim requirements in §0 below), read with `HARDENING_BACKEND_FIX_IMPLEMENTATION_PLAN.md` and the completed HARDENING-BACKEND-FIX evidence (`../HARDENING-BACKEND-FIX/`). `HARDENING_BACKEND_FIX_INDEPENDENT_REVIEW.md` was **not present** anywhere on the host (repo, dashboard tree, Downloads, Documents, job directories were searched); the four findings as stated in the task were treated as the hypotheses and each was confirmed against the current source before editing (§2).
**Baseline:** parent commit `a5eee1aa1e04127809eb57fec1e61331a1d692ef` (HARDENING-BACKEND-FIX) on `feature/ifvg-prop-robust-config-search-v1`; HEAD had not advanced.
**Commits (two, both focused, neither pushed nor merged):** `7491769` (`74917692a96b725d7e1f6f2c9ec8160e3a18afec`; tree `6608a9414080…`; parent `a5eee1a`; the 13 release files of the four fixes) and **R3** `ffcf39b` (`ffcf39bdcfb7b4d7ca8995886bb23e682f11e30b`; tree `597c10feb59a…`; parent `7491769`) (the owner-reported Win32 liveness blocker, §13; `owner_decision_lock.py`, the test module, `docs/DECISIONS.md`). Release head = the R3 commit.

```text
implementation_status: complete
backend_dev_complete_for_ui: true
ui_implementation_may_begin: true
capacity_gate: closed (formal artifact PASS over the final tree, 2026-09-04T04:34Z)
formal_acceptance_status: transitively_blocked_by_R1
real_verification_run_completed: false
full_authorized_development_run_completed: false
```

Not performed (by design): no real replay, no seed production, no bounded verification, no model fitting, no prop simulation, no search, no Full Authorized Development pipeline, no owner action, no push, no merge. `data/` gained zero files. Owner verification and R1 acceptance remain separate later actions.

---

## 0. Required scope (the task, condensed)

1. Strict lock-body validation in `search/owner_decision_lock.py` (exact schema version; native positive in-range pid, never bool; 32-lowercase-hex token; string-or-null host / start token; timezone-aware ISO-8601 timestamps; no coercion; a malformed body stays non-reclaimable) and, after `O_EXCL` creation, acquisition only on an exact persisted readback proving this writer's token — missing / malformed / unreadable / foreign-token readback fail typed; cleanup only when ownership of the file is proven.
2. Recoverable namespace initialization in `search/store_namespace.py` and its CLI / callers (explicit `store_instance_id` required for the first initialization of every real persistent store; no unrecoverable random id once publication has begun; a clearly separate disposable test helper generating an id BEFORE the real initializer; exact idempotent reuse and explicit-id crash recovery preserved; concurrent initialization serialized and fail-closed for divergent class / instance).
3. Native candidate-id validation in the candidate-as-of construction path (before `astype(str)` or any conversion; native non-empty strings; numeric / bool / null / NaN / blank / object-shaped ids rejected; canonical output for valid inputs preserved).
4. Capacity acceptance closure: the unchanged final implementation benchmarked on an idle machine with ≥ 8 GiB available, regenerating `_capacity_benchmark.txt`, `CAPACITY_BENCHMARKS.json` (`passed=true`), `CAPACITY_BENCHMARKS.md` (B1 / B2 Overall PASS) — no limit weakened, bypassed or lowered.

Effort limits honoured: one main agent; no dynamic / nested subagent workflows; exactly one read-only reviewer, one pass, over the four fixes only (§6); no review-until-consensus loop, no second adversarial round. The R3 correction (§13) followed the owner's own independent review and used no further reviewer.

---

## 1. Baseline and preflight

| Item | Evidence |
|---|---|
| Branch / HEAD | `feature/ifvg-prop-robust-config-search-v1` @ `a5eee1a` (the HARDENING-BACKEND-FIX commit; expected parent; no later commit) |
| Pre-existing user-owned hunks | `ARCHITECTURE.md` (+74/−1), `docs/ML_TRAINING_WORKBENCH.md` (+28/−1), `docs/README.md` (+5/−1), `docs/pipeline_state.yaml` (+44/−1) and the two deleted `data/experiment/honest_edge/*.json` — present before this release, **not touched, not staged, not committed** (the same `git diff --stat` before and after) |
| Prior evidence read | `../HARDENING-BACKEND-FIX/_PROGRESS_CHECKPOINT.md`, `COMPLETION_REPORT.md`, `FOCUSED_REVIEW.md`, `_capacity_benchmark.txt`, `CAPACITY_BENCHMARKS.md` (every numerical gate PASS twice; `overall: FAIL` on the ≥ 8 GiB available-RAM precondition only: 7.647 / 7.905 GiB) |
| Host at start | Windows 11 (10.0.26200), 16 logical CPUs, 31.1 GiB RAM; ≈ 14.9 GiB free at session start, no Python process running |
| Evidence folder | `implementation-progress/HARDENING-BACKEND-FIX.1/` (created; only the items the task lists; no prior-release evidence folder edited) |

---

## 2. The four findings — confirmed against the current source before editing

| # | Hypothesis (from the task) | Confirmation at `a5eee1a` |
|---|---|---|
| 1 | Lock body coerced / unvalidated; post-create readback not proven | `LockBody.parse` built the body with `int(document["lock_schema_version"])`, `int(document["pid"])`, `str(document["lock_token"])`, `str(created_at)`, `str(heartbeat_at)` and `document.get(...)` for host / start token — a string / float / boolean pid, a `"1"` or `1.0` schema version, any token shape and any timestamp text were coerced; the schema version was never compared to `LOCK_SCHEMA_VERSION`; `_parse_iso` did `str(value).replace("Z", …)`. `acquire()` returned `body if body is not None and body.lock_token == self.token else self._body()` — a missing, malformed or foreign-token readback still set `_held = True` with a synthesized body; the review-round RA-04 branch unlinked the freshly created file on a persistent read failure without any proof of ownership. |
| 2 | Random instance id minted by the real initializer | `_initialize_under_mutex`, both-absent branch: `instance = store_instance_id if store_instance_id is not None else uuid.uuid4().hex` — a crash between the head publication and the namespace publication lost the only key that the one-present recovery branches require; the CLI's `--store-instance-id` help read "generated once when omitted"; `initialize_test_namespace` passed no id. |
| 3 | Candidate ids coerced before validation | `candidate_as_of_frame`: `ids = frame["candidate_id"].astype(str)` with no prior validation — `1` / `True` / `None` / `NaN` became `"1"` / `"True"` / `"None"` / `"nan"` and could collide with real ids or pass the duplicate check as distinct strings. |
| 4 | Capacity gate open | `../HARDENING-BACKEND-FIX/CAPACITY_BENCHMARKS.json` `passed = false`; the only failing gate in B1 and B2 was "minimum available RAM at start ≥ 8 GiB" (7.647 / 7.905 GiB); `scripts/hardening_capacity_benchmark.py` unchanged since. |

---

## 3. Corrections delivered — exact files changed

### Source (4 files modified; no file added)

| Fix | File | Change |
|---|---|---|
| 1 | `src/alpha_lab/agents/data_infra/ifvg/search/owner_decision_lock.py` | `LOCK_BODY_FIELDS` (the exact seven fields), `LOCK_PID_MAX = 2**31 − 1`, `LOCK_TOKEN_PATTERN = [0-9a-f]{32}`; `_parse_iso` requires a native `str`, `datetime.fromisoformat`, `tzinfo` and `utcoffset()` present (no `Z` rewriting, no `str()`); `_is_native_int` (`type(v) is int`, so `bool` is refused); `LockBody.parse` rebuilt as a strict validator — exact field set, `lock_schema_version` native `int == 1`, `pid` native `int` in `1 … LOCK_PID_MAX`, `lock_token` `fullmatch`, `host` / `process_start_token` `str` or `None`, both timestamps aware; `json.loads` guarded for `ValueError` and `RecursionError` (review R1); `LockBody.pid` is `int` (never `None`). `_read_raw` returns the file's **bytes**; `_read_state_raw` → `(state, body, raw)`, a non-UTF-8 body is `malformed`; `_read_state` wraps it. `_try_create` opens with `O_BINARY` (the persisted bytes are exactly the body bytes; the Windows CRT no longer rewrites the trailing newline) and records `_created_bytes`; on a failed write it calls `_discard_partial_lock(attempted)`, which removes / quarantines ONLY when the readback is exactly a prefix of the attempted bytes (ownership proven) and otherwise reports "left in place" (readback failure or foreign bytes). New `_prove_created`: the readback must be byte-identical to `_created_bytes` and parse to this writer's token — absent → `lock_lost`, malformed → `lock_body_malformed`, unreadable → `lock_read_failed` (re-raised with the "left in place" disposition), foreign token / altered bytes → `lock_lost`; NOTHING is removed on failure. `acquire()` sets `_held` only after the proof and returns the proven body (the synthesized-body fallback and the RA-04 discard are gone). `refresh()` writes the heartbeat body as bytes. Module docstring updated. |
| R3 | `src/alpha_lab/agents/data_infra/ifvg/search/owner_decision_lock.py` (`_win32_process_times`) | the exact correction: `query_succeeded = GetExitCodeProcess(...)`; `if not query_succeeded: return "unknown"` (the state was NOT determined — never reclaimable); `if exit_code != STILL_ACTIVE: return "dead"` (a SUCCESSFUL query proving exit); previously both conditions returned `dead`. The module docstring names a failed liveness query among the `unknown` cases. |
| 2 | `src/alpha_lab/agents/data_infra/ifvg/search/store_namespace.py` | `store_instance_id_required` registered in `STORE_NAMESPACE_FAILURE_REASONS`; the both-absent branch raises it when no explicit id was supplied — BEFORE `_envelope_for` and before any write (`uuid4` no longer appears in `_initialize_under_mutex`); `initialize_store_namespace` refuses a non-native or non-32-lowercase-hex explicit id with `ValueError` before the mutex; the one-present recovery branches, the class-only / identical-explicit idempotent replay, the mutex serialization and the divergence refusals are unchanged. `initialize_test_namespace` is now the clearly separate DISPOSABLE TEST-ONLY helper: it mints `uuid4().hex` BEFORE calling the real initializer on an unmarked root and replays class-only on a marked root (idempotent reuse of a test store; a marked research root or a half-initialized store is refused by the real initializer exactly as any class-only request). Module docstring updated. |
| 2 | `scripts/ifvg_store_namespace.py` | `--store-instance-id` documented as REQUIRED for the first initialization of an unmarked store (choose it, record it, pass it — never generated); the intent output carries `store_instance_id` and `store_instance_id_required` with an explanatory note; a malformed id is the typed CLI refusal `store_instance_id_malformed` (exit 2) before anything is read or published, with a `ValueError` backstop (`invalid_argument`) around the initializer (review R2); `--confirm` without an id on an unmarked store relays the initializer's `store_instance_id_required` (exit 2, nothing published). |
| 3 | `src/alpha_lab/agents/data_infra/ifvg/ml/regime_oos_assignment.py` | `assert_native_candidate_ids(values)` (exported): every value must be a native `str` (`type(v) is str` — `np.str_`, bytes, numerics, booleans, `None` / NaN / `pd.NA` / `NaT`, dict / list / tuple refused) and non-blank (`strip()`); `candidate_as_of_frame` calls it BEFORE `astype(str)` and before the duplicate check; the construction of the output frame is unchanged (canonical output identical for valid ids). |

### Tests (8 files: 1 added, 7 modified) — 24 new test functions (94 items) in the new module (22 / 92 in `7491769`, + the two Windows-gated R3 tests)

| File | Change |
|---|---|
| `tests/agents/ifvg_search/test_hardening_backend_fix1.py` **(new)** | the targeted proofs of §4; R3: `test_win32_exit_code_query_failure_is_unknown`, `test_stale_lock_is_not_reclaimed_when_win32_exit_query_fails` (through a kernel32 proxy whose only fake export is `GetExitCodeProcess`; every other call reaches the real live process) |
| `tests/agents/ifvg_search/test_owner_decision_lock.py` | the RA-04 test (`…leaves_no_orphan_lock`) rewritten as `test_read_failure_after_exclusive_creation_is_typed_and_removes_nothing`: typed `lock_read_failed`, "left in place", the fresh lock survives with this writer's token, a second writer waits on a live holder (fail closed), operator unlink |
| `tests/agents/ifvg_search/test_store_namespace.py` | first-init refusal without an id (nothing published) then explicit init, explicit and class-only idempotent replay; CLI: `store_instance_id_required` in the intent, refused `--confirm` without an id, initialized with an id, idempotent replay, the research-looking-path refusal now passes an id; the concurrent "class-only initializers converge on a generated instance" section replaced by "all four refuse `store_instance_id_required`, nothing published" |
| `tests/agents/ifvg_search/test_bounded_verification.py`, `test_hardening_fix_round.py`, `test_owner_decisions.py`, `test_seed_production.py`, `test_verification.py` | the research-class (and one test-class) temporary stores initialized through the real initializer now pass an explicit `store_instance_id` (fixed hex constants; one per test) |

### Docs (1 file)

| File | Change |
|---|---|
| `docs/DECISIONS.md` | **D-052** (context, decision, rationale, trade-off of the four corrections and the capacity closure); the reservation note's release parenthetical names FIX.1 as D-052; amended by R3 (the Windows failure path, its correction and its red-first proof) |

Untouched, by design: `scripts/hardening_capacity_benchmark.py` (byte-identical to `a5eee1a`; the ≥ 8 GiB precondition and every numerical limit unchanged), Strategy-Core, Trade-Lab, every fixed M0–M3 source and identity, every immutable artifact and catalog, the user-owned docs.

---

## 4. Required targeted tests → the proving tests

| Required proof | Test(s) (`test_hardening_backend_fix1.py` unless noted) |
|---|---|
| wrong lock schema is malformed and non-reclaimable | `test_wrong_lock_schema_is_malformed_and_non_reclaimable[2/0/-1/"1"/1.0/True/None]`; control: `test_control_the_valid_stale_dead_holder_body_is_reclaimed` (the same body with a valid schema IS reclaimed, so every refusal is attributable to the corrupted field) |
| string, float, bool, zero, negative PIDs refused | `test_string_float_bool_zero_negative_and_out_of_range_pids_are_refused["123"/str(pid)/float/1.5/True/False/0/-1/-pid/LOCK_PID_MAX+1/None]` |
| invalid token shape refused | `test_invalid_token_shape_is_refused[31 chars/33 chars/uppercase/non-hex/trailing newline/int/None/list]` |
| naive or malformed timestamps refused | `test_naive_or_malformed_timestamps_are_refused` (both fields × naive / date-only / text / empty / int / float / None); plus `test_host_and_start_token_must_be_strings_or_null`, `test_missing_or_extra_lock_fields_are_malformed` (missing field, extra field, non-object JSON, non-UTF-8 bytes, 200,000-deep nesting) |
| missing post-create readback does not acquire | `test_missing_post_create_readback_does_not_acquire` (`lock_lost`, not held, nothing left, the next writer acquires) |
| malformed post-create readback does not acquire | `test_malformed_post_create_readback_does_not_acquire_and_removes_nothing` (`lock_body_malformed`, not held, the file left byte-identical, a second writer waits on a live holder) |
| foreign-token post-create readback does not acquire | `test_foreign_token_post_create_readback_does_not_acquire` (`lock_lost`, the foreign lock never unlinked; our token with altered bytes is `lock_lost` too) — plus `test_unreadable_post_create_readback_is_typed_and_leaves_the_lock`, `test_partial_write_cleanup_requires_the_exact_prefix_proof` (removed only on the exact-prefix proof; foreign readback / readback failure → left in place), `test_a_successful_acquire_returns_the_exactly_persisted_body`, and the rewritten RA-04 test in `test_owner_decision_lock.py` |
| **R3** a failed Win32 exit-code query is `unknown`, never `dead` | `test_win32_exit_code_query_failure_is_unknown` (a FALSE `GetExitCodeProcess` → `("unknown", None)`; `process_liveness` → `unknown` even against a "different process" start token; the writer's own start-token query unaffected; a SUCCESSFUL query with exit code 0 → `("dead", None)`) |
| **R3** a stale lock is not reclaimed when the exit query fails | `test_stale_lock_is_not_reclaimed_when_win32_exit_query_fails` (old heartbeat, this live process as the holder, the query failing: the lock file remains present and byte-identical, acquisition does not succeed, `reclaimed_from is None`, typed `lock_holder_liveness_unknown`; control with the real query: `lock_held_by_live_holder`) |
| first real namespace initialization without `store_instance_id` refused | `test_first_real_initialization_without_store_instance_id_is_refused[research/test]` (typed, nothing published — no envelope, no head, no temp files), `test_a_malformed_explicit_instance_id_is_refused_before_anything_is_published`, `test_cli_first_initialization_requires_the_explicit_instance_id`; `test_store_namespace.py::test_initialization_is_explicit_idempotent_and_immutable` and `::test_cli_shows_intent_then_initializes_and_refuses_divergence` |
| explicit-ID crash recovery and idempotent reuse still pass | `test_explicit_id_crash_recovery_and_idempotent_reuse_still_pass`; the unchanged `test_store_namespace.py::test_crash_after_genesis_before_namespace_recovers_exactly`, `::test_crash_after_namespace_before_genesis_recovers_exactly`, `::test_relocated_store_retains_namespace_identity`, `::test_missing_head_is_corruption_not_no_supersessions` |
| concurrent namespace initialization behavior remains correct | `test_concurrent_initialization_remains_serialized_and_fails_closed` (four class-only initializers all refuse with nothing published; four identical explicit + one divergent: one coherent pair, the divergent side refuses closed whichever wins the mutex); the updated `test_store_namespace.py::test_concurrent_identical_initializers_publish_one_coherent_pair` and the unchanged `::test_concurrent_divergent_initializers_one_wins_other_refuses` |
| the disposable helper generates its id before the real initializer | `test_the_disposable_test_helper_generates_its_id_before_the_real_initializer` (the real initializer receives a fresh 32-hex id; a second store gets its own; class-only replay on a marked test store; a marked research root diverges) |
| candidate-as-of rejects non-string IDs before coercion | `test_candidate_as_of_rejects_non_native_or_blank_ids_before_coercion[20 values: int/float/bool/None/NaN/NA/NaT/numpy int/float/bool/np.str_/bytes/empty/whitespace/dict/list/tuple]`, `test_candidate_as_of_native_check_precedes_astype_and_the_duplicate_check` (`1` and `"1"` refuse as non-native before they could collapse into a duplicate) |
| valid canonical IDs and existing golden identities unchanged | `test_candidate_as_of_valid_ids_keep_the_canonical_output` (object and `string` dtype inputs → the pre-FIX.1 construction, frame-equal, native `str` elements); the exact-reuse / golden suite (§7 item 4) |

---

## 5. Targeted evidence (`_targeted_pytest.txt`)

| Run | Result |
|---|---|
| Fast targeted set (new module + lock + namespace + OOS assignment), first green run | 135 passed |
| Wider consumer regression (the new module plus every direct consumer of the three corrected seams and every test file whose research-class initializations gained an explicit id: 18 files) | **297 passed** in 51 s, exit 0, no warnings summary |
| Post-review rerun of the fast set (both reviewer findings applied with their tests) | **135 passed** in 25 s, exit 0 |
| **R3 red-first proof**: the two new tests against the UNCORRECTED source (`7491769`'s lock module restored via `git stash`) | **2 failed as expected** — `('dead', None) != ('unknown', None)` and `DID NOT RAISE OwnerDecisionLockError` (the live holder's lock WAS reclaimed) |
| **R3 green**: the fast set over the corrected tree | **137 passed** in 25 s, exit 0 |

One implementation finding surfaced by the control test and fixed before review: on Windows, `os.open` without `O_BINARY` writes the lock body in CRT text mode (the trailing `\n` persisted as `\r\n`), so an exact byte comparison of the readback could never hold; the writer now opens the lock in binary mode and the heartbeat rewrite persists bytes (§3).

---

## 6. Review (one read-only reviewer, one pass) — `REVIEW.md`

`VERDICT: PASS`; 0 blockers, 0 major, **2 minor** in-scope findings, both **FIXED** in the one permitted correction pass and re-verified (§5, third row): **R1** `LockBody.parse` caught `ValueError` only, so a 200,000-deep nested body escaped as an untyped `RecursionError` (now `malformed`, non-reclaimable; test added); **R2** the CLI caught `StoreNamespaceError` only, so a malformed `--store-instance-id` surfaced as a traceback (now the typed `store_instance_id_malformed` refusal before anything is read or published, with a `ValueError` backstop; test added). The reviewer's checklist confirmed every requirement of §0 items 1–3 as met and the benchmark harness untouched. No carry-forwards. **After the commit, the owner's independent complete lock-safety review returned FAIL on one Windows failure path (R3, §13)** — the one thing the agent reviewer's pass did not catch; corrected and re-gated.

---

## 7. Final gates

| Gate | Command / scope | Result |
|---|---|---|
| 1. Targeted micro-fix tests | §5 | 135 / 297 / 135 passed (first pass); R3 red-first 2 failed as expected; **137 passed** (fast set) and **299 passed** (the consumer regression: the new module + every direct consumer of the lock / namespace / candidate-as-of seams, 18 files) over the corrected tree |
| 2. Complete suite, environment as-is | `python -m pytest -q -p no:cacheprovider` (`_final_pytest.txt`) | **2,287 passed, 0 failed** in 1,609.97 s (0:26:49), exit 0; no warnings summary; both provider keys present — over `7491769`'s tree, which differs from the release head only by the R3 correction (five lines of the Win32 liveness reader); not rerun after R3 by owner decision (§13); the R3 delta is covered by the 299-test consumer regression over the corrected tree |
| 3. Complete suite, provider credentials cleared | `env -u POLYGON_API_KEY -u DATABENTO_API_KEY python -m pytest -q -p no:cacheprovider` (`_final_pytest_keys_cleared.txt`) | **2,287 passed, 0 failed** in 1,656.55 s (0:27:36), exit 0; no warnings summary; both provider keys absent in the run environment — over `7491769`'s tree (same note as above) |
| 4. Exact-reuse / golden suite | pipeline_regime, pipeline_run, orchestrator, evidence-integrity, R6.1-FIX goldens, prop seam, stratification gate, external aggregation (`_exact_reuse_pytest.txt`) | **93 passed, 0 failed** in 353.96 s (0:05:53), exit 0 — over `7491769`'s tree (a liveness verdict cannot alter any persisted identity; same note as above) |
| 5. Warnings as errors | pyproject `filterwarnings = error` (the one narrowly scoped third-party ignore unchanged); the suites above ran under it | PASS — zero warnings-summary sections in the three suite logs and in every targeted / consumer log (warnings are errors); zero new project warnings |
| 6. Ruff | `python -m ruff check src tests scripts` (`_ruff_and_diffcheck.txt`) | `All checks passed!` (exit 0) over the final tree (after R3) |
| 7. `git diff --check` | over `src scripts tests docs` | clean (exit 0) — the R3 changes against `7491769` and the whole release against `a5eee1a` |
| 8. Capacity benchmark with the ≥ 8 GiB precondition | §8 | **PASS over the final tree (after R3)** — `passed = true`; B1 / B2 Overall PASS; min available RAM 9.644 / 10.423 GiB; every RSS, Python-allocation, runtime, artifact-size, determinism and resident-batch limit PASS (§8) |

Collected tests: 2,289 (was 2,195 at `a5eee1a`; +92 items in `7491769`, +2 in R3, all in the new module).

---

## 8. Capacity acceptance closure (`_capacity_benchmark.txt`, `CAPACITY_BENCHMARKS.json`, `CAPACITY_BENCHMARKS.md`)

Run **2026-09-04T04:22:38Z → 04:34:17Z, ALONE on the idle host** (10,974,560 KB ≈ 10.5 GiB free at launch; no other Python process), over the **final tree** (HEAD `7491769` + the R3 working-tree correction = the tree committed as `ffcf39b`; the benchmarked event-detail writer / stratified-summary paths and the harness are untouched by this release). Command: `python scripts/hardening_capacity_benchmark.py --out-dir QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/HARDENING-BACKEND-FIX.1`. `CAPACITY_BENCHMARKS.json`: **`passed = true`** (`policy_id = HARDENING_CAPACITY_POLICY_V1`, `partition_bound_policy_id = event_detail_partition_row_bound_v2`).

| Benchmark | Overall | min available RAM at start (≥ 8 GiB) | 1M-row peak RSS (≤ 1.5 GiB) | 1M-row Python alloc peak | 1M-row wall (≤ 300 s) | growth 500k→1M slope (≤ gate) | projection at the registered max (≤ min(6 GiB, 50 %)) | serialized artifact | determinism |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| B1 — event-detail writer | **PASS** | 9.644 GiB | 0.335 GiB | 35.8 MiB (≤ 512 MiB) | 7.4 s | 114.6 ≤ 349.5 B/row | 1.295 ≤ 4.822 GiB (10,000,000 rows) | 0.329 ≤ 2.000 GiB | identical |
| B2 — regime-stratified summary | **PASS** | 10.423 GiB | 0.482 GiB | 134.6 MiB (≤ 768 MiB) | 37.9 s | 92.4 ≤ 523.0 B/row | 0.826 ≤ 5.212 GiB (5,000,000 rows) | 0.042 ≤ 0.250 GiB | identical |

Worst-lawful-shape proof (B1, 1M rows): skewed 0.302 GiB peak RSS / 217.5 MiB Python peak / 8.6 s, dense 0.337 GiB / 35.7 MiB / 8.2 s; maximum resident writer batch (observed at the writer's Parquet seam) 50,000 ≤ 50,000 rows for every shape and every size; byte-identical repeat hashes — every shape gate PASS.

Output hashes (first 12) are **identical to the prior pass and to both HARDENING-BACKEND-FIX runs**: B1 `97bda11c7c22` / `db0a25a028db` / `2c8f09db151d`, skewed `86f9ef4338d0`, dense `13702335fd42`; B2 `750881699aa3` / `38005c7a1497` / `6c6bdee0c013` — this release changed no benchmarked byte.

The one gate that failed on 2026-09-03 ("minimum available RAM at start ≥ 8 GiB": 7.647 / 7.905 GiB with the owner's applications holding host memory) is closed by this artifact. The harness, its ≥ 8 GiB precondition and every numerical limit are byte-identical to `a5eee1a` — nothing was weakened, bypassed or lowered.

**Prior pass (over `7491769`'s tree, before R3; its raw log superseded by the rerun above — the numbers are recorded here):** run 2026-09-04T01:56:42Z → 02:08:12Z alone on the idle host (≈ 13.0 GiB free at launch); `passed = true`; B1 Overall PASS (min available RAM 13.119 GiB; 1M-row peak RSS 0.339 GiB; Python peak 35.8 MiB; wall 7.5 s; slope 125.2 ≤ 343.1 B/row; projection 1.388 ≤ 6.000 GiB; artifact 0.329 ≤ 2.000 GiB; determinism identical; skewed / dense shapes PASS with the 50,000-row resident batch); B2 Overall PASS (12.161 GiB; 0.474 GiB; 134.6 MiB; 36.8 s; 72.1 ≤ 502.8 B/row; 0.742 ≤ 6.000 GiB; 0.042 ≤ 0.250 GiB; identical). Output hashes identical to both HARDENING-BACKEND-FIX runs.

---

## 9. Preservation

| Constraint | Proof |
|---|---|
| Strategy-Core unchanged | `../Strategy-Core` @ `a4e3303`, 0 dirty files (before and after) |
| Trade-Lab unchanged | `../Trade-Lab` @ `f808b9f`; its 27 dirty files all pre-date this release and were not touched |
| Fixed M0–M3 source and identities unchanged | no file under the fixed stages was edited (the release diff is the 13 files of §3); the exact-reuse / golden suite reproduces every persisted identity (§7 item 4) |
| Immutable artifacts and catalogs unchanged | `data/` gained zero files (`find data -type f -newer <prior COMPLETION_REPORT>` → 0); no `STORE_NAMESPACE.json`, mutex or `SUPERSESSIONS.head` marker under `data/` |
| S11 blocked; MBP-1 offline / research-only; protected and sealed counters at zero | no launch, no activation, no seed production, no verification run; nothing in this release touches those seams |
| No real replay, seed production, bounded verification, model fitting, prop simulation, search, or full pipeline | none executed (the benchmarks use synthetic data under a temporary directory, as designed) |
| Numerical capacity limits and the 8 GiB precondition | `scripts/hardening_capacity_benchmark.py` byte-identical to `a5eee1a` |
| User-owned hunks | the four docs and two data deletions remain exactly the pre-existing worktree changes; none staged or committed |

---

## 10. Packaging

| Item | Value |
|---|---|
| Commits | `74917692a96b725d7e1f6f2c9ec8160e3a18afec` (the four fixes; parent `a5eee1a`; tree `6608a9414080…`; 13 files, +1,092 / −101) and **`ffcf39bdcfb7b4d7ca8995886bb23e682f11e30b`** (R3; parent `7491769`; tree `597c10feb59a…`; 3 files, +133 / −12: `owner_decision_lock.py`, `test_hardening_backend_fix1.py`, `docs/DECISIONS.md`) on `feature/ifvg-prop-robust-config-search-v1`; release head = `ffcf39b`; neither pushed nor merged |
| Patch | `HARDENING-BACKEND-FIX.1.patch` — the two-commit series (`git format-patch -2`; 114,175 B) — sha256 `6ab2a90e0ebceaac590cfd358a9d0dbc75259188932056f8d7c73a80467111da` (`HARDENING-BACKEND-FIX.1.patch.sha256`, verified with `sha256sum -c`) |
| Bundle | `HARDENING-BACKEND-FIX.1.bundle` (`a5eee1a..feature/ifvg-prop-robust-config-search-v1`; requires `a5eee1a`; carries the branch head at `ffcf39b`; 46,283 B; `git bundle verify` OK) — sha256 `f1e83cb924239d8e4f51ac309fac0bec9ec4862aa1525b7d126fac9fddc5b129` (`HARDENING-BACKEND-FIX.1.bundle.sha256`, verified) |
| Worktree after the commits | only the pre-existing user-owned changes (`ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/README.md`, `docs/pipeline_state.yaml`, the two `data/experiment/honest_edge/*.json` deletions) and the untracked working files; nothing pushed, nothing merged |
| Evidence folder (this folder; exactly the items the task lists) | `COMPLETION_REPORT.md`; `_targeted_pytest.txt`; `_final_pytest.txt`; `_final_pytest_keys_cleared.txt`; `_exact_reuse_pytest.txt`; `_capacity_benchmark.txt`; `CAPACITY_BENCHMARKS.json`; `CAPACITY_BENCHMARKS.md`; `_ruff_and_diffcheck.txt`; `REVIEW.md`; `HARDENING-BACKEND-FIX.1.patch` + `.sha256`; `HARDENING-BACKEND-FIX.1.bundle` + `.sha256` |

---

## 11. Deviations and notes (all inside the four-fix scope)

- **RA-04 disposition superseded.** HARDENING-BACKEND-FIX's review round made a persistent read failure right after this writer's own exclusive creation discard the fresh lock so the store was not denied for the life of the process. Under the FIX.1 rule (cleanup only when ownership of the file is proven), a readback that FAILED proves nothing, so the fresh lock is now left in place, typed `lock_read_failed`. Consequence: it carries this writer's live body — other writers wait on a live holder (fail closed) and it becomes reclaimable by liveness once this process exits; an operator may remove it earlier. Recorded in D-052's trade-off.
- **Partial-write cleanup kept, but proven.** A failed body write after exclusive creation still removes (or quarantines) the partial file, now ONLY when the persisted bytes read back as exactly a prefix of the bytes this writer wrote (the interrupted write of a body that embeds this writer's random token); a foreign readback or a readback failure leaves the file in place (typed `lock_create_failed`, "left in place").
- **Persisted lock bytes.** The lock is now written in binary mode and the heartbeat rewrite persists bytes: on Windows the on-disk body ends with `\n` (previously `\r\n` via CRT text mode). Readers parse JSON; nothing compares a lock file byte-for-byte across writers, and no lock file exists under `data/`.
- **`LockBody.pid` is `int`** (no longer `int | None`): the writer always records its pid, and a null pid is now a malformed body. `process_liveness` keeps its `int | None` signature.
- **`OwnerDecisionLock._read_raw` returns bytes** (was text); it remains the one seam every retry and every test monkeypatch goes through.
- **`initialize_test_namespace` semantics.** It mints an id only for an unmarked root; on a marked root it replays class-only. Two concurrent helper calls on one unmarked root therefore diverge and the loser refuses closed (`store_namespace_divergent`) — acceptable for a disposable store and noted in its docstring; production code calls it only from the synthetic fixture, which is guarded by `_under_temp_directory`.
- **Class-only replay of a MARKED store remains accepted** by the real initializer (the "exact idempotent reuse" the task preserves): the refusal is confined to the first initialization of an unmarked store. The one-present recovery branches already required the explicit id and are unchanged.
- **CLI reason codes added:** `store_instance_id_required` (from the initializer, registered in `STORE_NAMESPACE_FAILURE_REASONS`), `store_instance_id_malformed` and `invalid_argument` (CLI-level, like the existing `namespace_class_required`).
- **Independent review document absent** (header): every finding was nevertheless confirmed against the source (§2) before any edit; nothing was taken on trust.
- **R3 (§13).** The Win32 liveness reader's failed-query path was outside the four findings the task enumerated and outside the agent reviewer's checklist (which verified the parse / proof / cleanup seams but not the liveness reader); the owner's independent complete lock-safety review caught it. The exact correction was applied as specified and proven red-first. **Gate scope after R3 (owner decision, 2026-09-04):** the full suites and the golden suite were NOT rerun over the corrected tree — the correction is confined to the Win32 liveness reader (`_win32_process_times`), whose only callers are the lock's stale-holder evaluation (`process_liveness`) and the writer's own start-token query, and the full suites had passed over the tree that differs only by those five lines; the R3 delta is covered by the 137-test fast set and the 299-test consumer regression over the corrected tree; the capacity benchmark, Ruff, `git diff --check` and the collection count WERE rerun over the corrected tree. The three suite logs in this folder are therefore the runs over `7491769`'s tree, as their headers state.
- **No `_PROGRESS_CHECKPOINT.md` in this folder**: the task restricts the folder to the listed items; this report is the resume anchor for FIX.1.

---

## 12. What remains (unchanged owner actions; not part of this release)

Window selection from `../HARDENING-BACKEND/VERIFICATION_WINDOW_SHORTLIST.md`, the `SeedProductionAuthorizationRef`, the seed-only chain, the `VerificationAuthorizationRef`, `init` of the real verification store (now with an operator-chosen, recorded `--store-instance-id`), the program allowlist, the bounded run, and R1 acceptance. The UI/UX implementation branch may rebase onto the release head `ffcf39b` and begin.

---

## 13. R3 — the owner-reported blocker (independent complete lock-safety review of `7491769`)

**Finding (verbatim substance):** in `_win32_process_times`, `if not own and (not GetExitCodeProcess(...) or exit_code != STILL_ACTIVE): return "dead"` treated a query that itself FAILED (the process state not determined → must be `unknown`) exactly like a query that SUCCEEDED with an exit code other than STILL_ACTIVE (demonstrably exited → `dead`). Failure scenario: an old heartbeat, a holder still alive; `OpenProcess` succeeds, `GetExitCodeProcess` fails transiently or operationally; the reader answers `dead`; the stale-lock evaluator authorizes reclamation; the live writer's lock may be removed. The reclaim mutex prevents two reclaimers from racing each other; it does not prevent an incorrect liveness classification from reclaiming a live holder. Windows is the actual development and operator platform.

**Confirmed against the source before editing:** present verbatim at `7491769`.

**Correction (exact, as specified):** `query_succeeded = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))`; `if not query_succeeded: return "unknown", None`; `if exit_code.value != _STILL_ACTIVE: return "dead", None`. The module docstring now lists a failed liveness query among the `unknown` (never reclaimed) cases. Nothing else in the module changed.

**Tests:** `test_win32_exit_code_query_failure_is_unknown` and `test_stale_lock_is_not_reclaimed_when_win32_exit_query_fails` (§4). **Red-first:** both failed against the uncorrected source exactly as the finding predicts (`('dead', None) != ('unknown', None)`; `DID NOT RAISE` — the live holder's lock was reclaimed); both pass with the correction (§5).

**Gates after R3:** the capacity benchmark (PASS, §8), Ruff, `git diff --check`, the fast set (137) and the consumer regression (299) over the corrected tree; the full suites and the golden suite not rerun by owner decision (§11); the R3 commit and the regenerated two-commit patch / bundle in §10.
