# CAPACITY_BENCHMARKS — HARDENING_CAPACITY_POLICY_V1 (plan §4.4, F-17)

Run 2026-09-02T05:30:00+00:00 → 2026-09-02T05:39:33+00:00 on Windows-11-10.0.26200-SP0 (AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD, 16 logical CPUs, 31.148 GiB RAM); Python 3.13.1, pyarrow 20.0.0, duckdb 1.4.4, pandas 2.3.1. Synthetic data only; every run in a fresh subprocess under a temporary directory; RSS from the native process monitor (Win32 GetProcessMemoryInfo (PeakWorkingSetSize / WorkingSetSize) via ctypes); `tracemalloc` in a SEPARATE run (supplementary).

Command:

```text
python scripts/hardening_capacity_benchmark.py --out-dir QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/HARDENING-BACKEND
```

## B1 — event-detail writer (production writer over a generator + store publish + production reader)

**Overall: PASS**

### Measured

| input rows | peak RSS increase | repeat | Python alloc peak | wall | repeat wall | artifact bytes | output hash (first 12) | repeat hash | available RAM at start |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 250,000 | 0.219 GiB | 0.223 GiB | 35.8 MiB | 1.9 s | 1.9 s | 8,848,535 | `cd2d99c02c45` | `cd2d99c02c45` | 8.867 GiB |
| 500,000 | 0.269 GiB | 0.289 GiB | 35.8 MiB | 3.7 s | 3.8 s | 17,679,038 | `286d90dd1dbf` | `286d90dd1dbf` | 8.902 GiB |
| 1,000,000 | 0.331 GiB | 0.322 GiB | 35.8 MiB | 7.5 s | 7.5 s | 35,362,544 | `ab8238fbd5f1` | `ab8238fbd5f1` | 8.866 GiB |

| input rows | paths | partitions | build_seconds | read_seconds |
|---:|---:|---:|---:|---:|
| 250,000 | 1,250 | 5 | 1.8 | 0.2 |
| 500,000 | 2,500 | 10 | 3.4 | 0.3 |
| 1,000,000 | 5,000 | 20 | 6.9 | 0.6 |

### Extrapolated (never a pass on its own)

| quantity | value |
|---|---:|
| per-row RSS slope 250k→500k | 215.0 B/row |
| per-row RSS slope 500k→1M | 133.1 B/row |
| slope gate (1.25 × max(slope 250k→500k, 64 B/row)) | 268.8 B/row |
| projection slope used (max(slope 500k→1M, 0)) | 133.1 B/row |
| projected peak RSS increase at 10,000,000 rows | 1.446 GiB |
| projection limit min(6 GiB, 50 % of min available RAM) | 4.400 GiB |
| projected serialized artifact at 10,000,000 rows | 0.329 GiB |
| artifact limit | 2.000 GiB |

### Gates

| gate | measured | limit | result |
|---|---:|---:|---|
| minimum available RAM at start ≥ 8 GiB | 8.800 GiB | 8.000 GiB | PASS |
| 1M-row peak RSS increase ≤ 1.5 GiB | 0.331 GiB | 1.500 GiB | PASS |
| 1M-row Python allocation peak ≤ 512 MiB | 35.8 MiB | 512.0 MiB | PASS |
| 1M-row wall time ≤ 300 s | 7.5 s | 300 s | PASS |
| growth 500k→1M: slope ≤ 1.25 × max(slope 250k→500k, 64 B/row) | 133.1 B/row | 268.8 B/row | PASS |
| projection at the registered maximum (10,000,000 rows) ≤ min(6 GiB, 50 % of min available RAM) | 1.446 GiB | 4.400 GiB | PASS |
| serialized artifact ≤ 2.000 GiB at 10,000,000 rows | 0.329 GiB | 2.000 GiB | PASS |
| determinism: byte-identical output hash on repeat (every size) | identical | identical | PASS |

## B2 — regime-stratified event summary (production builder; external DuckDB aggregation)

**Overall: PASS**

### Measured

| input rows | peak RSS increase | repeat | Python alloc peak | wall | repeat wall | artifact bytes | output hash (first 12) | repeat hash | available RAM at start |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 250,000 | 0.345 GiB | 0.342 GiB | 99.1 MiB | 9.6 s | 9.8 s | 2,042,312 | `766f2ee4783a` | `766f2ee4783a` | 8.755 GiB |
| 500,000 | 0.439 GiB | 0.439 GiB | 129.0 MiB | 18.7 s | 19.1 s | 4,429,418 | `2eed4255cefc` | `2eed4255cefc` | 8.871 GiB |
| 1,000,000 | 0.479 GiB | 0.479 GiB | 134.6 MiB | 37.5 s | 38.3 s | 8,921,104 | `378d004a281e` | `378d004a281e` | 8.806 GiB |

| input rows | summary_rows | events_total |
|---:|---:|---:|
| 250,000 | 250,000 | 250,000 |
| 500,000 | 500,000 | 500,000 |
| 1,000,000 | 1,000,000 | 1,000,000 |

### Extrapolated (never a pass on its own)

| quantity | value |
|---|---:|
| per-row RSS slope 250k→500k | 402.7 B/row |
| per-row RSS slope 500k→1M | 85.3 B/row |
| slope gate (1.25 × max(slope 250k→500k, 64 B/row)) | 503.4 B/row |
| projection slope used (max(slope 500k→1M, 0)) | 85.3 B/row |
| projected peak RSS increase at 5,000,000 rows | 0.796 GiB |
| projection limit min(6 GiB, 50 % of min available RAM) | 4.377 GiB |
| projected serialized artifact at 5,000,000 rows | 0.042 GiB |
| artifact limit | 0.250 GiB |

### Gates

| gate | measured | limit | result |
|---|---:|---:|---|
| minimum available RAM at start ≥ 8 GiB | 8.755 GiB | 8.000 GiB | PASS |
| 1M-row peak RSS increase ≤ 1.5 GiB | 0.479 GiB | 1.500 GiB | PASS |
| 1M-row Python allocation peak ≤ 768 MiB | 134.6 MiB | 768.0 MiB | PASS |
| 1M-row wall time ≤ 300 s | 37.5 s | 300 s | PASS |
| growth 500k→1M: slope ≤ 1.25 × max(slope 250k→500k, 64 B/row) | 85.3 B/row | 503.4 B/row | PASS |
| projection at the registered maximum (5,000,000 rows) ≤ min(6 GiB, 50 % of min available RAM) | 0.796 GiB | 4.377 GiB | PASS |
| serialized artifact ≤ 0.250 GiB at 5,000,000 rows | 0.042 GiB | 0.250 GiB | PASS |
| determinism: byte-identical output hash on repeat (every size) | identical | identical | PASS |

## Reading

- The measured columns are what this machine did; the extrapolated table is a linear projection from the measured 1M-row point with the measured 500k→1M slope and is reported separately (plan §4.4: hardening never passes on extrapolation alone — the implementation must be block/partition bounded AND the projection must satisfy every gate).
- B1 counts rows = events; a walk yields 200 events per path (5,000 paths at 1M rows, 20 path blocks of 250 paths); the writer's input is a GENERATOR, its uniqueness proof is the external DuckDB distinct check, and the reader re-verifies every partition.
- B2 counts rows = input events; each (path, event type) is a distinct summary key, so summary rows ≈ input rows (the registered maximum of 5,000,000 summary rows projects from the 1M-row point).
- `peak RSS increase` = PeakWorkingSetSize after the run − WorkingSetSize baseline after imports and fixture setup (the lifetime peak cannot be reset on Windows, so the number is conservative).

## Shape parameters and the per-block bound (adversarial B-02, recorded after the round)

- **B1 shape:** 200 events per path → 250 paths × 200 events = **50,000 rows per path block**
  (20 blocks at 1M rows). The writer is BLOCK-bounded: its resident memory is proportional to
  the rows of ONE path block (the column lists, the Arrow table and the Parquet write buffer of
  that block) plus fixed overhead — never to the artifact. `EVENT_DETAIL_BUDGET_V1` caps the total
  rows (10,000,000) and the block size (250 paths) but NOT the events per path, so the measured
  per-row slope and the 10M-row projection above hold for the 200 events/path shape; a production
  walk with more events per path scales the per-block resident memory linearly (a 2,000
  events/path walk holds 500,000 rows per block — about ten times the benchmark's block).
- **Reading:** the §4.4 structural requirement (block/partition bounded) holds for every shape;
  the numerical statement is shape-conditional. A row-based flush guard (splitting a path block
  across partitions) is outside the current manifest contract and is therefore a future
  VERSIONED capacity-policy change requiring owner approval — no ceiling was changed here.
- **B2 shape:** every (path, event type) is a distinct summary key, so summary rows ≈ input rows;
  the external aggregation is bounded by DuckDB's 512 MiB memory limit plus its spill directory
  regardless of the key count (the 1M-row peak is dominated by the per-partition pandas group
  step and the reader).
