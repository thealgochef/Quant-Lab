# WS-B — capacity implementation and numerical gates (plan §4.4, F-17)

Workstream notes for the main agent (HARDENING-BACKEND). Tests were written
red first (`_red_ws_B.txt`: 12 failed / 2 passed before the code landed —
the two passes are regression guards the R6.1 code already satisfied),
then the code landed, then the targeted suites ran (`_ws_B_pytest.txt`).
The benchmark ran for real on this machine (`CAPACITY_BENCHMARKS.md` /
`.json`; raw driver log `_capacity_benchmark_run.txt`).

## Files touched

| File | Change |
|---|---|
| `src/alpha_lab/propsim/event_detail.py` | `build_account_event_detail(walks, path_records=None, *, …, total_rows=None, path_count=None)`: accepts an ITERABLE of `(path_record, walk_result)` pairs in strictly increasing draw-ordinal order (consumed exactly once; never materialized) with the caller's declared preflight counts (verified exactly after streaming; an under-declared count is a typed `row overrun` refusal at the block, an over-declared count a preflight refusal); the legacy `(walk_results, path_records)` sequence form is kept byte-for-byte. The whole-artifact `np.concatenate` id index (`_event_id_index` / `_assert_unique`) is GONE; the within-block duplicate check stays (bounded by one path block). `_external_uniqueness_check`: DuckDB over the WRITTEN partitions under `SET memory_limit='256MiB'`, `SET threads=2`, `SET temp_directory=<attempt-local tempfile.mkdtemp()>` — `COUNT(*) == COUNT(DISTINCT event_id)` (a duplicate is `duplicate event id across path blocks (<id>…)`) and `COUNT(DISTINCT path_instance_id) == COUNT(DISTINCT path_ordinal) == path_count` (path ids one-to-one with path ordinals); the temp directory is removed in `finally`. `EVENT_ID_CANONICAL_KEY`, `EVENT_ID_UNIQUENESS_CHECK_V1`, `EXTERNAL_CHECK_MEMORY_LIMIT_BYTES`, `EVENT_DETAIL_ROW_GROUP_SIZE` registered; the detail manifest gains `row_group_size` and `event_id_uniqueness` (canonical key, artifact check, memory limit, partitions/rows checked, distinct ids/paths). `numpy` import dropped. |
| `src/alpha_lab/propsim/search_bridge.py` | `persist_account_simulation` feeds the writer through `_walk_pairs()` — a GENERATOR over `run.path_records` / `run.walk_results` in draw-ordinal order (an index permutation, no second all-path event list) — with `total_rows=sum(len(events))` / `path_count=len(path_records)`; a length disagreement is a typed `EventDetailIntegrityError` before the producer runs. |
| `src/alpha_lab/agents/data_infra/ifvg/ml/regime_stratified_prop.py` | External aggregation: per-partition group rows (pandas, bounded by one partition) → typed intermediate Parquet partitions (`INTERMEDIATE_SUMMARY_SCHEMA`) in an attempt-local `tempfile.mkdtemp()` directory → DuckDB (`SET memory_limit='512MiB'`, `SET threads=1` for a deterministic accumulation order, `SET temp_directory=<attempt>/duckdb_tmp`, `preserve_insertion_order=true`): the cross-partition path-repetition refusal (`HAVING COUNT(DISTINCT partition_ordinal) > 1`), the exact aggregation into a per-simulation temporary table, the exact row count (row budget checked cumulatively BEFORE any final row is written), the exact unique-path count, the reasons and the small (cluster, event type) aggregate the strata derive from; the final rows stream in canonical order (`summary_order_sql()` — the ORDER BY form of `_summary_sort_key`; simulations in sorted id order) through `_RowGroupAlignedWriter`, whose row groups are exactly those of `pq.write_table(table, row_group_size=65_536)` so `summary_bytes == summary_parquet_bytes(table)` (test-proven past one row group); the byte budget is checked on the written file before the bytes are read; the attempt directory is removed in `finally` (success and refusal). `StratifiedPropResult.summary` is now a lazily parsed property over `summary_bytes` (`summary_rows` is a field); `build_stratified_prop_body`'s signature, `StratifiedPropBody`, the JSON `detail` keys (plus a new `detail["summary"]["aggregation"]` block) and the loader / panel-assigner seams are unchanged. No per-event or per-group Python structure grows with the artifact (`groups` / `summary_rows` dicts and `seen_paths` sets are gone). |
| `scripts/hardening_capacity_benchmark.py` (new) | `HARDENING_CAPACITY_POLICY_V1` harness: B1 / B2 at 250k / 500k / 1M synthetic rows, each (size, mode) in a fresh subprocess; native RSS via Win32 `GetProcessMemoryInfo` (`PeakWorkingSetSize` − baseline `WorkingSetSize`) / POSIX `getrusage`; `tracemalloc` in a separate run; available RAM via `GlobalMemoryStatusEx`; determinism repeat; slopes, projections and every §4.4 gate; writes `CAPACITY_BENCHMARKS.json` + `.md` (measured and extrapolated tables separated). |
| `tests/propsim/test_event_detail_streaming.py` (new, 8 tests) | one-shot iterable consumed exactly once and never materialized; no whole-artifact index attributes; streaming form reproduces the sequence form byte-for-byte (partitions + manifest); declared preflight counts required and verified (wrong rows / wrong paths / preflight / streaming row overrun); draw-ordinal order required; the external check catches a forged duplicate across blocks and, with the check disabled, NOTHING in memory catches it (the proof lives on disk); a path id reused under another ordinal is refused (`one-to-one`); the external check's temp directory is attempt-local and cleaned on success and refusal, and the publication directory holds only declared files; the bridge passes a generator (not a sequence) with the declared counts. |
| `tests/agents/data_infra/ifvg/test_stratified_prop_external_aggregation.py` (new, 6 tests) | external aggregation equals an in-memory pandas reference (values, order, facts) and is byte-deterministic; `summary_bytes == summary_parquet_bytes(summary)` past one row group (17,000 paths × 4 events = 68,000 summary rows); a path repeated across partitions is refused externally; the row budget is the exact external count (fits at `exact`, refuses at `exact − 1`); the attempt temp directory is cleaned on success and on refusal and nothing is published; the DuckDB connection runs under the explicit memory limit (`512 MiB`), one thread and a spill directory under the attempt root. |
| `tests/propsim/test_account_event_detail.py` | UNCHANGED — every R6.1 D15 test still passes against the streaming writer (byte-identical partitions, budget refusals, uniqueness refusals incl. the cross-block forgery now caught by the external check, store producer semantics). |

## The uniqueness argument (plan §4.4 "prove event-ID uniqueness by the existing deterministic identity projection")

`AccountWalk._emit` (`propsim/account.py`) mints `event_id = sha256(canonical
JSON of {path_instance_id, account_namespace, account_ordinal, event_ordinal,
event_type, ts, payload})` and increments `self._ordinal` per emitted event.
The writer enforces, while streaming, (a) strictly increasing draw ordinals
(hence distinct path records), (b) strictly increasing `event_ordinal` per
path and (c) every event naming its own path. For every production emitter
the key `(path_instance_id, event_ordinal)` is therefore distinct per row and
the ids — a deterministic projection of a superset of that key — are distinct
(under SHA-256 collision resistance, the same assumption every store hash
makes). The writer cannot re-derive a foreign envelope's projection per row
(the emitter's `account_namespace` is not on the envelope), so the
ARTIFACT-level proof is the disk-backed external DuckDB distinct check over
the written partitions (memory-limited, spilling to the attempt-local temp
directory) — it runs for every publication (defense in depth, unconditionally;
the plan's "if any source cannot prove uniqueness … use a disk-backed external
uniqueness check rather than a global in-memory set") and it is what catches
the forged cross-block duplicate of the R6.1 uniqueness test. No whole-artifact
in-memory set exists any more (test-proven: with the external check disabled,
the forgery passes the in-memory checks).

## Deviations from the plan text

1. **The external check is unconditional** (not only "whenever the source cannot prove uniqueness"): the writer has no per-row way to verify that an envelope's id IS the canonical projection, so it always runs the artifact-level check; the canonical-key structure is recorded in the manifest as the emitter-side argument. Cost: one DuckDB pass over the `event_id` / `path_instance_id` / `path_ordinal` columns of the written partitions.
2. **`AccountSimulationRun` still holds every walk result in memory** (`propsim/simulation.py`, the D15-era design): the writer path now builds no second copy and no index, but the simulation's own result object is outside the writer and outside this release (a streaming simulation seam would change `run_account_simulation` and `build_payout_reliability_vector`). Recorded honestly; the B1 benchmark measures the WRITER over a lazy generator, which is what §4.4 gates.
3. **Detail-manifest content grew** (`row_group_size`, `event_id_uniqueness`); the store manifest hashes the detail manifest, so a v2 simulation persisted by R6.1 code would carry a different detail-manifest hash than one persisted now — no real v2 artifact exists (synthetic only) and the partition bytes are unchanged. The simulation IDENTITY (envelope) is unchanged.
4. **Stratified-report detail gained `detail["summary"]["aggregation"]`** (engine, memory limit, threads, row-group size) → `detail_sha256` and therefore the `stratified_prop` report ids re-mint (synthetic only; the summary bytes / hash are unchanged for the existing fixtures, test-proven equal to the single-write form).
5. **Floating-point accumulation order**: the old code summed `amount_total` per partition (pandas) and merged across partitions in Python; the new code sums the per-partition partials in DuckDB single-threaded in partition order. On the existing fixtures the values are identical; for ≥ 3 partitions of one key the last-ulp association order could differ from the R6.1 in-memory code (deterministic run-to-run either way — the determinism gate is on repeat, not on cross-implementation equality; no real artifact exists).
6. **B1 synthetic events are lightweight duck-typed objects** (`__slots__` classes carrying exactly the attributes the writer reads), not Pydantic `PropAccountEventEnvelope`s: the benchmark measures the writer's memory profile; the production emitter's per-event Pydantic overhead belongs to the simulation object (deviation 2).

## Benchmark results

See `CAPACITY_BENCHMARKS.md` (gate table, measured vs extrapolated) — summary
filled in below after the run.

Attempt 2 (`_capacity_benchmark_run.txt`, 2026-09-02T05:29→05:39Z; the
report files are attempt 2's) — **overall PASS, 16/16 gates**. Attempt 1
(`_capacity_benchmark_run_attempt1.txt`) had passed every NUMERICAL gate with
identical output hashes; its only miss was the environmental precondition
"available RAM at start ≥ 8 GiB" on B1 (7.896 GiB while other workstreams'
test suites were running on this shared desktop — 31 GiB total, ~20 GiB held
by the user's applications), and the driver's final console print crashed on
the cp1252 console (`→`) AFTER the report files were written (fixed:
ASCII arrow + UTF-8 stdout).

| gate (plan §4.4) | B1 event detail | B2 stratified summary |
|---|---:|---:|
| minimum available RAM at start ≥ 8 GiB | 8.800 GiB — PASS | 8.755 GiB — PASS |
| 1M-row peak RSS increase ≤ 1.5 GiB | 0.331 GiB — PASS | 0.479 GiB — PASS |
| 1M-row Python allocation peak ≤ 512 / 768 MiB | 35.8 MiB — PASS | 134.6 MiB — PASS |
| 1M-row wall time ≤ 300 s | 7.5 s — PASS | 37.5 s — PASS |
| growth 500k→1M slope ≤ 1.25 × max(slope 250k→500k, 64 B/row) | 133.1 ≤ 268.8 B/row — PASS | 85.3 ≤ 503.4 B/row — PASS |
| projection at the registered maximum ≤ min(6 GiB, 50 % of min available RAM) | 1.446 GiB at 10M rows ≤ 4.400 GiB — PASS | 0.796 GiB at 5M rows ≤ 4.377 GiB — PASS |
| serialized artifact at the registered maximum | 0.329 GiB ≤ 2 GiB — PASS | 0.042 GiB ≤ 256 MiB — PASS |
| determinism (byte-identical output hash on repeat, every size) | identical (also identical across attempts 1 and 2) — PASS | identical (also across attempts) — PASS |

Measured (attempt 2; peak RSS increase = PeakWorkingSetSize after the run −
WorkingSetSize baseline after imports): B1 250k / 500k / 1M rows → +0.219 /
+0.269 / +0.331 GiB, wall 1.9 / 3.7 / 7.5 s, artifacts 8.85 / 17.68 / 35.36
MB (5 / 10 / 20 path blocks of 250 paths), Python allocation peak a flat
35.8 MiB (one path block in memory). B2 250k / 500k / 1M input rows
(= summary rows) → +0.345 / +0.439 / +0.479 GiB, wall 9.6 / 18.7 / 37.5 s,
artifacts 2.04 / 4.43 / 8.92 MB, Python allocation peak 99.1 / 129.0 /
134.6 MiB. The full tables (measured vs extrapolated separated) are in
`CAPACITY_BENCHMARKS.md`; every raw number in `CAPACITY_BENCHMARKS.json`.

The implementation is block/partition bounded (B1: one path block of rows
plus one DuckDB pass with a 256 MiB limit; B2: one partition's group rows in
pandas plus a 512 MiB-limited single-threaded DuckDB aggregation and a
row-group-aligned streaming writer) AND the projected registered ceilings
satisfy every numerical gate, so §4.4 passes without a capacity-policy
change — no ceiling was lowered.

## Test counts

- `_red_ws_B.txt`: 12 failed / 2 passed before the code (collection of the new modules against the R6.1 code).
- `_ws_B_pytest.txt`: the targeted suites on the FINAL WS-B tree (2026-09-02T05:40→05:46Z, with the other workstreams' edits present, including WS-C's `filterwarnings` policy): `tests/propsim` + `test_regime_stratification.py` + `test_regime_stratification_evidence.py` + `test_stratified_prop_external_aggregation.py` + `test_pipeline_regime.py` → **189 passed, 0 failed** (6:27); `ruff check` over the six WS-B files → clean. The 14 new tests (8 + 6) plus the 13 unchanged R6.1 D15 event-detail tests and the 19 pipeline-regime E2Es (the bridge's generator path + the S14 external aggregation through the service) are inside that count.

## Nothing for the main agent to wire

No change to `regime_stratification_service.py`, `regime_stratified_contracts.py`, `search/store.py`, `search/identities.py` or `pyproject.toml` was needed. The main agent should mention in `DEVIATIONS.md` the six deviations above and in `FILES_TOUCHED.md` the seven files (3 src, 1 script, 2 new tests + the unchanged R6.1 D15 test module as the regression guard).
