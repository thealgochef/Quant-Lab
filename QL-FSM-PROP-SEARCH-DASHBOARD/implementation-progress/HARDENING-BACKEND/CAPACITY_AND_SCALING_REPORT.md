# HARDENING-BACKEND — Capacity and Scaling Report (plan §4.4, F-17)

The raw measurements, the formulas and the per-run numbers are in
`CAPACITY_BENCHMARKS.md` / `CAPACITY_BENCHMARKS.json` (attempt 2, the
recorded run; attempt 1 in `_capacity_benchmark_run_attempt1.txt` passed every
numerical gate with identical hashes but missed the ≥ 8 GiB free-RAM
precondition on B1 by 0.1 GiB while other suites ran). This report reads them.

## Verdict

**HARDENING_CAPACITY_POLICY_V1 — PASS (16/16 gates; both benchmarks).**
The implementations are block/partition bounded (the structural requirement)
AND the projections at the registered maxima satisfy every numerical gate
(the numerical requirement). No ceiling was lowered; no capacity policy was
changed.

## What was measured

Synthetic data only, no research simulation, every (benchmark, size, mode) in
a fresh subprocess under a temporary directory, on the release machine (AMD
Ryzen 7 7800X3D, 16 logical CPUs, 31.15 GiB RAM, Windows 11 10.0.26200,
Python 3.13.1, pyarrow 20.0.0, duckdb 1.4.4, pandas 2.3.1). Peak RSS from the
native process monitor (Win32 `GetProcessMemoryInfo`: `PeakWorkingSetSize`
after the run − `WorkingSetSize` baseline after imports and fixture setup —
conservative, because the lifetime peak cannot be reset on Windows);
`tracemalloc` in a separate run as the supplementary Python-allocation
number; available RAM via `GlobalMemoryStatusEx`; wall time; artifact bytes;
byte-identical output hashes on repeat.

| | B1 event-detail writer (+ store publish + production reader) | B2 regime-stratified event summary (external DuckDB aggregation) |
|---|---:|---:|
| 1M-row peak RSS increase (limit 1.5 GiB) | **0.331 GiB** (repeat 0.322) | **0.479 GiB** (repeat 0.479) |
| 1M-row Python allocation peak (limit 512 / 768 MiB) | 35.8 MiB | 134.6 MiB |
| 1M-row wall time (limit 300 s) | 7.5 s | 37.5 s |
| Slope 500k→1M vs gate 1.25 × max(slope 250k→500k, 64 B/row) | 133.1 ≤ 268.8 B/row | 85.3 ≤ 503.4 B/row |
| Projection at the registered maximum vs limit min(6 GiB, 50 % of min available RAM) | 1.446 GiB @ 10,000,000 rows ≤ 4.400 GiB | 0.796 GiB @ 5,000,000 rows ≤ 4.377 GiB |
| Serialized artifact at the registered maximum | 0.329 GiB ≤ 2 GiB | 0.042 GiB ≤ 256 MiB |
| Determinism (byte-identical hash on repeat, every size) | identical (`cd2d99c02c45` / `286d90dd1dbf` / `ab8238fbd5f1`) | identical (`766f2ee4783a` / `2eed4255cefc` / `378d004a281e`) |
| Available RAM at start (limit ≥ 8 GiB) | 8.800 GiB (minimum over the runs) | 8.755 GiB |

## Why the numbers scale the way they do

- **B1** — the writer consumes an ITERABLE of `(path_record, walk_result)`
  pairs exactly once (never materialized), builds ONE path block column-wise,
  writes it as a ZSTD Parquet partition with bounded row groups, hashes it by
  streaming and releases it; the former whole-artifact `np.concatenate` id
  index is gone. Resident memory is therefore proportional to the rows of one
  path block, never to the artifact — the per-row slope FALLS from 215 B/row
  (250k→500k) to 133 B/row (500k→1M) as fixed overhead amortizes. Event-id
  uniqueness is the emitter's canonical-key argument (`(path_instance_id,
  event_ordinal)` under strictly increasing ordinals per path and distinct
  path records; the id is a deterministic projection of a superset of that
  key) PLUS an unconditional disk-backed DuckDB distinct check over the WRITTEN
  partitions (`memory_limit` 256 MiB, attempt-local spill directory, cleaned
  in `finally`) — the only whole-artifact fact is computed externally.
- **B2** — per-partition group rows (pandas, bounded by one partition) go to
  typed intermediate Parquet partitions in an attempt-local temp directory;
  DuckDB (`memory_limit` 512 MiB, one thread, spill directory) performs the
  cross-partition path-repetition refusal, the exact aggregation, the exact
  row and unique-path counts and the strata aggregates; the final rows stream
  in canonical order (the SQL form of `_summary_sort_key`) through a
  row-group-aligned Parquet writer whose bytes equal
  `summary_parquet_bytes(table)`; the row budget is checked cumulatively
  before any final row is written and the byte budget on the written file
  before publication. No Python structure grows with the summary; the slope
  falls from 403 B/row to 85 B/row.

## Shape parameters (adversarial B-02) — what the projections are conditional on

- B1: 200 events per path → 250 paths × 200 = **50,000 rows per path block**
  (20 blocks at 1M rows). `EVENT_DETAIL_BUDGET_V1` caps the total rows
  (10,000,000) and the block size (250 paths) but NOT the events per path, so
  the measured slope and the 10M-row projection hold for that shape; a
  production walk with more events per path scales the per-block resident
  memory linearly (2,000 events/path → 500,000 rows per block, about ten
  times the benchmark's block). The block-bounded STRUCTURE holds for every
  shape; the NUMERICAL claim is shape-conditional. A row-based flush guard
  would split a path block across partitions — outside the current manifest
  contract — and is therefore a future VERSIONED capacity-policy change
  requiring owner approval (DEV-HB-32).
- B2: every (path, event type) is a distinct summary key, so summary rows ≈
  input rows; the external aggregation is bounded by DuckDB's memory limit
  plus its spill directory regardless of the key count.

## Reading for Phase 5 (the operator forecast, plan §7.3)

The measured per-row costs (B1 ≈ 133 B/row resident, ≈ 35 B/row on disk;
B2 ≈ 85 B/row resident, ≈ 9 B/row on disk) and the wall times (B1 ≈ 7.5 µs/row;
B2 ≈ 37 µs/row) are the inputs the Full Authorized Development resource
forecast should use for the D15 detail and summary rows — separately from the
replay throughput benchmark §4.6 requires before that run (not part of this
release). The simulation's own result object (`AccountSimulationRun`) still
materializes every walk result (DEV-HB-11) and is outside these two measured
paths.

## Evidence

`CAPACITY_BENCHMARKS.md`, `CAPACITY_BENCHMARKS.json`,
`_capacity_benchmark_run.txt` (attempt 2), `_capacity_benchmark_run_attempt1.txt`,
`scripts/hardening_capacity_benchmark.py`, the streaming / aggregation tests
(`tests/propsim/test_event_detail_streaming.py` (9),
`tests/agents/data_infra/ifvg/test_stratified_prop_external_aggregation.py` (7)),
`_ws_B_NOTES.md`.
