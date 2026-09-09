# HARDENING-BACKEND-FIX.1 — Reviewer Report (one read-only pass)

**Scope:** the four fixes of this micro-release only (strict lock-body validation + the post-create
ownership proof; recoverable namespace initialization + CLI; native candidate-id validation; the
capacity harness untouched). One reviewer, one pass, read-only, after implementation and the
targeted tests (135 + 297 passed). No second round, no consensus loop, no unrelated findings.
**Inputs:** the complete source + test diff against `a5eee1a`, the new test module
`tests/agents/ifvg_search/test_hardening_backend_fix1.py`, the current source files.
**Reviewer verdict (verbatim first line):** `VERDICT: PASS`

## Findings (2, both minor, both in scope) and their disposition

| # | Severity | Location | Defect | Failure scenario | Disposition |
|---|---|---|---|---|---|
| FIX.1-R1 | minor | `search/owner_decision_lock.py` `LockBody.parse` | `json.loads` was guarded for `ValueError` only, so a pathologically nested body raised `RecursionError` instead of returning `None` (same at HEAD). | A hostile / accidental `SUPERSESSIONS.lock` of 200,000 × `[` makes `acquire()` / `release()` / `verify_held()` escape with an untyped `RecursionError` rather than the typed `lock_body_malformed`; nothing is coerced or unlinked, so the non-reclaimable guarantee still holds. | **FIXED** — `except (ValueError, RecursionError): return None`; test: `test_missing_or_extra_lock_fields_are_malformed` now writes `b"[" * 200_000` and proves `lock_body_malformed`, file untouched. |
| FIX.1-R2 | minor | `scripts/ifvg_store_namespace.py` `init --confirm` | The handler caught `StoreNamespaceError` only, so the new native-str / 32-hex `ValueError` for a malformed `--store-instance-id` surfaced as a traceback (exit 1) instead of a JSON `refused` record (same class of behavior at HEAD: a pydantic error). | Operator passes `--store-instance-id ABCD…`: nothing is published (correct) but the refusal is untyped. | **FIXED** — the CLI validates the id shape before anything is read or published (`store_instance_id_malformed`, exit 2) and keeps a `ValueError` backstop (`invalid_argument`); test: `test_cli_first_initialization_requires_the_explicit_instance_id` proves three malformed shapes refuse typed with nothing published. |

## Reviewer checklist (as reported)

- §1 strict parse — MET: exact 7-field set; native `int` schema version `== 1`; native `int` pid in `1 … 2**31 − 1` (bool excluded); `fullmatch([0-9a-f]{32})` token (trailing newline and Unicode digits refused); string-or-null host / start token; aware ISO-8601 via `fromisoformat` + `utcoffset()`; no coercion (JSON `NaN` / `Infinity` / huge ints / duplicate keys validated on the final dict). Malformed bodies are never reclaimable (`_evaluate` yields live-holder or `lock_body_malformed`; `_reclaim` re-evaluates under the mutex).
- §1 post-create proof — MET: `_try_create` writes in binary mode and records `_created_bytes`; `_prove_created` requires `raw == _created_bytes` and `token == self.token`; absent → `lock_lost`, malformed / non-UTF-8 → `lock_body_malformed`, persistent read failure → `lock_read_failed`, foreign / altered → `lock_lost`; `_held` set only after the proof; no unlink on any failure path.
- §1 cleanup ownership — MET: only `_discard_partial_lock` unlinks / quarantines, and only when the readback is a prefix of the attempted bytes (which embed this writer's random token); a readback failure or foreign bytes → left in place. (Readback → unlink is not atomic — the same proof-by-readback design as `_reclaim` / `release`.)
- §2 real initializer — MET: `uuid4` no longer appears in `_initialize_under_mutex`; the both-absent branch raises `store_instance_id_required` before `_envelope_for` and before any write; both recovery branches still require the explicit id; class-only / explicit idempotent replay unchanged; mutex serialization unchanged; divergent class / instance → `store_namespace_divergent`.
- §2 disposable helper — MET: `initialize_test_namespace` mints `uuid4().hex` before calling the real initializer and replays class-only on a marked root; the only src caller (`owner_decisions.py`, the synthetic fixture) is guarded by `_under_temp_directory`. Two concurrent helper calls on one unmarked root diverge and the loser refuses closed — acceptable.
- §2 CLI — MET: the intent reports `store_instance_id_required`; `--confirm` without an id on an unmarked root → JSON refused, exit 2, nothing published; an explicit id initializes; class-only replay is idempotent.
- §3 candidate ids — MET: `assert_native_candidate_ids` runs before `astype(str)`; `type(v) is str` refuses int / float / bool / None / NaN / NA / NaT / NumPy scalars / `np.str_` / str subclasses / bytes / dict / list / tuple; `strip()` refuses whitespace-only (NBSP included). Canonical output (object dtype, values, null as-of preserved) unchanged for object, `string`, `string[pyarrow]` and categorical inputs. Zero-width-only strings pass (not whitespace) — consistent with "blank" as specified.
- §4 tests — MET (the new module proves every required item; see the reviewer's enumeration in `COMPLETION_REPORT.md` §4).
- Benchmark — CONFIRMED untouched: `scripts/hardening_capacity_benchmark.py` is absent from the diff; the 8 GiB precondition and every numerical limit are unchanged.

## Carry-forwards (none)

No carry-forwards were recorded: both findings were fixed in the one permitted correction pass and
re-verified by the targeted rerun appended to `_targeted_pytest.txt`.

## R3 — owner-reported blocker after the commit (independent complete lock-safety review)

**Finding (owner, 2026-09-04, against `7491769`): FAIL — one Windows failure path.** In
`_win32_process_times`, `if not own and (not GetExitCodeProcess(...) or exit_code != STILL_ACTIVE): return "dead"`
conflated two different conditions: a query that SUCCEEDS with an exit code other than
STILL_ACTIVE (demonstrably exited → dead) and a query that itself FAILS (the state was not
determined → unknown). Failure scenario: an old heartbeat, a holder that is still alive;
`OpenProcess` succeeds, `GetExitCodeProcess` fails transiently or operationally; the reader
answers `dead`; the stale-lock evaluator authorizes reclamation; the live writer's lock may be
removed. The reclaim mutex prevents two reclaimers from racing each other, not a wrong liveness
classification. Windows is the actual development and operator platform.

**Confirmed against the source before editing:** the quoted logic was present verbatim at
`7491769` (`owner_decision_lock.py`, `_win32_process_times`).

**Correction (exact, as specified):**

```python
query_succeeded = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
if not query_succeeded:
    return "unknown", None
if exit_code.value != _STILL_ACTIVE:
    return "dead", None
```

The module docstring now names a failed liveness query among the `unknown` (never reclaimed)
cases.

**Tests added (`test_hardening_backend_fix1.py`, Windows-gated):**

- `test_win32_exit_code_query_failure_is_unknown` — through a kernel32 proxy whose only fake
  export is `GetExitCodeProcess` (every other call reaches the real live process): a FALSE query
  → `("unknown", None)`, `process_liveness` → `unknown` even with a "different process" start
  token (nothing was determined); the writer's own start-token query is unaffected; a query that
  SUCCEEDS with exit code 0 → `("dead", None)` — the two conditions are distinguished.
- `test_stale_lock_is_not_reclaimed_when_win32_exit_query_fails` — the failure scenario itself
  (old heartbeat, this live process as the holder, the exit-code query failing): the lock file
  remains present and byte-identical, acquisition does not succeed, `reclaimed_from is None`, the
  typed result is `lock_holder_liveness_unknown`; control with the real query: `lock_held_by_live_holder`.

**Red-first proof (recorded in `_targeted_pytest.txt`):** against the uncorrected source the first
test failed with `('dead', None) != ('unknown', None)` and the second with `DID NOT RAISE
OwnerDecisionLockError` — i.e. the live holder's lock WAS reclaimed. With the correction: both
pass; the fast targeted set 137 passed.

**Disposition:** FIXED in a follow-up commit of the same release (hash in `COMPLETION_REPORT.md`
§10); every final gate rerun over the corrected tree (`COMPLETION_REPORT.md` §7 / §13).
