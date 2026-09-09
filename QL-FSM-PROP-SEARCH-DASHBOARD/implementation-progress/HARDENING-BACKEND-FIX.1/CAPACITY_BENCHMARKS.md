# CAPACITY_BENCHMARKS — HARDENING_CAPACITY_POLICY_V1 (plan §4.4, F-17)

Run 2026-09-04T04:22:38+00:00 → 2026-09-04T04:34:17+00:00 on Windows-11-10.0.26200-SP0 (AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD, 16 logical CPUs, 31.148 GiB RAM); Python 3.13.1, pyarrow 20.0.0, duckdb 1.4.4, pandas 2.3.1. Synthetic data only; every run in a fresh subprocess under a temporary directory; RSS from the native process monitor (Win32 GetProcessMemoryInfo (PeakWorkingSetSize / WorkingSetSize) via ctypes); `tracemalloc` in a SEPARATE run (supplementary).

Command:

```text
python scripts/hardening_capacity_benchmark.py --out-dir QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/HARDENING-BACKEND-FIX.1
```

## B1 — event-detail writer (production writer over a generator + store publish + production reader)

**Overall: PASS**

### Measured

| input rows | peak RSS increase | repeat | Python alloc peak | wall | repeat wall | artifact bytes | output hash (first 12) | repeat hash | available RAM at start |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 250,000 | 0.217 GiB | 0.217 GiB | 35.8 MiB | 2.0 s | 1.9 s | 8,831,474 | `97bda11c7c22` | `97bda11c7c22` | 10.238 GiB |
| 500,000 | 0.282 GiB | 0.295 GiB | 35.8 MiB | 4.0 s | 3.8 s | 17,670,353 | `db0a25a028db` | `db0a25a028db` | 9.675 GiB |
| 1,000,000 | 0.335 GiB | 0.326 GiB | 35.8 MiB | 7.4 s | 7.5 s | 35,378,726 | `2c8f09db151d` | `2c8f09db151d` | 10.379 GiB |

| input rows | paths | partitions | build_seconds | read_seconds | max_partition_rows | max_resident_rows_observed | max_rows_per_partition |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250,000 | 1,250 | 5 | 1.8 | 0.2 | 50,000 | 50,000 | 50,000 |
| 500,000 | 2,500 | 10 | 3.7 | 0.3 | 50,000 | 50,000 | 50,000 |
| 1,000,000 | 5,000 | 20 | 6.9 | 0.6 | 50,000 | 50,000 | 50,000 |

### Extrapolated (never a pass on its own)

| quantity | value |
|---|---:|
| per-row RSS slope 250k→500k | 279.6 B/row |
| per-row RSS slope 500k→1M | 114.6 B/row |
| slope gate (1.25 × max(slope 250k→500k, 64 B/row)) | 349.5 B/row |
| projection slope used (max(slope 500k→1M, 0)) | 114.6 B/row |
| projected peak RSS increase at 10,000,000 rows | 1.295 GiB |
| projection limit min(6 GiB, 50 % of min available RAM) | 4.822 GiB |
| projected serialized artifact at 10,000,000 rows | 0.329 GiB |
| artifact limit | 2.000 GiB |

### Gates

| gate | measured | limit | result |
|---|---:|---:|---|
| minimum available RAM at start ≥ 8 GiB | 9.644 GiB | 8.000 GiB | PASS |
| 1M-row peak RSS increase ≤ 1.5 GiB | 0.335 GiB | 1.500 GiB | PASS |
| 1M-row Python allocation peak ≤ 512 MiB | 35.8 MiB | 512.0 MiB | PASS |
| 1M-row wall time ≤ 300 s | 7.4 s | 300 s | PASS |
| growth 500k→1M: slope ≤ 1.25 × max(slope 250k→500k, 64 B/row) | 114.6 B/row | 349.5 B/row | PASS |
| projection at the registered maximum (10,000,000 rows) ≤ min(6 GiB, 50 % of min available RAM) | 1.295 GiB | 4.822 GiB | PASS |
| serialized artifact ≤ 2.000 GiB at 10,000,000 rows | 0.329 GiB | 2.000 GiB | PASS |
| determinism: byte-identical output hash on repeat (every size) | identical | identical | PASS |

### Worst-lawful-shape proof (HARDENING-BACKEND-FIX §9.3; 1M rows)

| shape | paths | partitions | max partition rows | peak RSS increase | repeat | Python alloc peak | wall | artifact bytes | output hash (first 12) | repeat hash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| skewed | 505 | 21 | 50,000 | 0.302 GiB | 0.324 GiB | 217.5 MiB | 8.6 s | 37,681,054 | `86f9ef4338d0` | `86f9ef4338d0` |
| dense | 250 | 20 | 50,000 | 0.337 GiB | 0.340 GiB | 35.7 MiB | 8.2 s | 35,277,073 | `13702335fd42` | `13702335fd42` |

| gate | measured | limit | result |
|---|---:|---:|---|
| maximum resident writer batch (observed at the writer's Parquet seam) ≤ max_rows_per_partition (every shape, every size) | 50,000 rows | 50,000 rows | PASS |
| skewed: 1M-row peak RSS increase ≤ 1.5 GiB | 0.302 GiB | 1.500 GiB | PASS |
| skewed: 1M-row Python allocation peak ≤ 512 MiB | 217.5 MiB | 512.0 MiB | PASS |
| skewed: 1M-row wall time ≤ 300 s | 8.6 s | 300 s | PASS |
| skewed: byte-identical output hash on repeat | identical | identical | PASS |
| skewed: maximum resident writer batch (observed) ≤ max_rows_per_partition | 50,000 rows | 50,000 rows | PASS |
| dense: 1M-row peak RSS increase ≤ 1.5 GiB | 0.337 GiB | 1.500 GiB | PASS |
| dense: 1M-row Python allocation peak ≤ 512 MiB | 35.7 MiB | 512.0 MiB | PASS |
| dense: 1M-row wall time ≤ 300 s | 8.2 s | 300 s | PASS |
| dense: byte-identical output hash on repeat | identical | identical | PASS |
| dense: maximum resident writer batch (observed) ≤ max_rows_per_partition | 50,000 rows | 50,000 rows | PASS |

## B2 — regime-stratified event summary (production builder; external DuckDB aggregation)

**Overall: PASS**

### Measured

| input rows | peak RSS increase | repeat | Python alloc peak | wall | repeat wall | artifact bytes | output hash (first 12) | repeat hash | available RAM at start |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 250,000 | 0.342 GiB | 0.347 GiB | 99.1 MiB | 9.7 s | 9.6 s | 2,042,314 | `750881699aa3` | `750881699aa3` | 10.680 GiB |
| 500,000 | 0.439 GiB | 0.433 GiB | 129.0 MiB | 18.9 s | 19.0 s | 4,429,400 | `38005c7a1497` | `38005c7a1497` | 10.559 GiB |
| 1,000,000 | 0.482 GiB | 0.483 GiB | 134.6 MiB | 37.9 s | 37.7 s | 8,919,329 | `6c6bdee0c013` | `6c6bdee0c013` | 10.666 GiB |

| input rows | summary_rows | events_total |
|---:|---:|---:|
| 250,000 | 250,000 | 250,000 |
| 500,000 | 500,000 | 500,000 |
| 1,000,000 | 1,000,000 | 1,000,000 |

### Extrapolated (never a pass on its own)

| quantity | value |
|---|---:|
| per-row RSS slope 250k→500k | 418.4 B/row |
| per-row RSS slope 500k→1M | 92.4 B/row |
| slope gate (1.25 × max(slope 250k→500k, 64 B/row)) | 523.0 B/row |
| projection slope used (max(slope 500k→1M, 0)) | 92.4 B/row |
| projected peak RSS increase at 5,000,000 rows | 0.826 GiB |
| projection limit min(6 GiB, 50 % of min available RAM) | 5.212 GiB |
| projected serialized artifact at 5,000,000 rows | 0.042 GiB |
| artifact limit | 0.250 GiB |

### Gates

| gate | measured | limit | result |
|---|---:|---:|---|
| minimum available RAM at start ≥ 8 GiB | 10.423 GiB | 8.000 GiB | PASS |
| 1M-row peak RSS increase ≤ 1.5 GiB | 0.482 GiB | 1.500 GiB | PASS |
| 1M-row Python allocation peak ≤ 768 MiB | 134.6 MiB | 768.0 MiB | PASS |
| 1M-row wall time ≤ 300 s | 37.9 s | 300 s | PASS |
| growth 500k→1M: slope ≤ 1.25 × max(slope 250k→500k, 64 B/row) | 92.4 B/row | 523.0 B/row | PASS |
| projection at the registered maximum (5,000,000 rows) ≤ min(6 GiB, 50 % of min available RAM) | 0.826 GiB | 5.212 GiB | PASS |
| serialized artifact ≤ 0.250 GiB at 5,000,000 rows | 0.042 GiB | 0.250 GiB | PASS |
| determinism: byte-identical output hash on repeat (every size) | identical | identical | PASS |

## Reading

- The measured columns are what this machine did; the extrapolated table is a linear projection from the measured 1M-row point with the measured 500k→1M slope and is reported separately (plan §4.4: hardening never passes on extrapolation alone — the implementation must be block/partition bounded AND the projection must satisfy every gate).
- B1 counts rows = events; a walk yields 200 events per path (5,000 paths at 1M rows, 20 path blocks of 250 paths); the writer's input is a GENERATOR, its uniqueness proof is the external DuckDB distinct check, and the reader re-verifies every partition.
- B2 counts rows = input events; each (path, event type) is a distinct summary key, so summary rows ≈ input rows (the registered maximum of 5,000,000 summary rows projects from the 1M-row point).
- `peak RSS increase` = PeakWorkingSetSize after the run − WorkingSetSize baseline after imports and fixture setup (the lifetime peak cannot be reset on Windows, so the number is conservative).
