# IFVG Lab Restoration Test Matrix

Status: executable acceptance index

Every implementation requirement is represented below. A failed critical gate blocks catalog
activation. Real-permitted-data and performance gates are explicit and are never silently
substituted with synthetic tests.

| ID | Level | Repo | Target/fixture | Assertion | Failure meaning | Severity |
|---|---|---|---|---|---|---|
| SC-CAL-001 | unit | Strategy-Core | exchange schedule maintenance/reopen | closures skipped; first eligible close included | false source gap | critical |
| SC-CAL-002 | unit | Strategy-Core | Friday/Sunday, weekend, holiday | scheduled closures do not invalidate | calendar policy wrong | critical |
| SC-CAL-003 | unit | Strategy-Core | spring/fall DST and rollover | UTC slots map deterministically | schedule drift | critical |
| SC-DSP-001 | unit | Strategy-Core | displacement `(B0,end]` fixture | exact boundary membership | lookback drift | critical |
| SC-DSP-002 | unit | Strategy-Core | available partition missing close | `source_bar_missing` | genuine gap hidden | critical |
| SC-DSP-003 | unit | Strategy-Core | unavailable permitted partition | `source_partition_unavailable` | coverage conflation | critical |
| SC-DSP-004 | unit | Strategy-Core | too few eligible bars | `insufficient_observations` | wrong invalidity | high |
| SC-DSP-005 | unit | Strategy-Core | contradictory coverage/ineligible row | invariant failure | corrupt evidence accepted | critical |
| SC-DSP-006 | unit | Strategy-Core | long/short synthetic bars | normalized signs and paths | direction bug | critical |
| SC-EQL-001 | unit | Strategy-Core | one-tick pool fixture | width <= one tick; distributions retained | policy drift | high |
| SC-EQL-002 | unit | Strategy-Core | long EQL / short EQH | opposing-leg qualification | missing path | critical |
| SC-EQL-003 | unit | Strategy-Core | ties/multiple links | all links; latest cursor then pool ID scalar | nondeterminism | critical |
| SC-EQL-004 | unit | Strategy-Core | saturated pool eviction | active setup keeps evidence | bounded-state corruption | critical |
| SC-ID-001 | characterization | Strategy-Core | frozen v2 goldens | tables/labels/decisions/executions byte-identical | v2 regression | critical |
| SC-ID-002 | characterization | Strategy-Core | ordered emissions and seed hash | exact frozen hashes | ordering/seed drift | critical |
| SC-ID-003 | integration | Strategy-Core | batch/incremental/resume/repeat | identical formula-v2 record bytes | nondeterminism | critical |
| SC-SEED-001 | unit | Strategy-Core | chain digest vectors | portable length-prefixed SHA-256 | platform identity risk | critical |
| SC-CAP-001 | adversarial | Strategy-Core | saturation fixture | state/seed/snapshot under locked bounds | unsafe headroom | critical |
| SC-PERF-001 | benchmark | Strategy-Core | paired repeated replay | p99/median/p95 under contract | replay regression | critical |
| QL-SCH-001 | unit | Quant-Lab | empty/all-null/populated rows | exact same twelve schemas | unstable physical schema | critical |
| QL-SCH-002 | unit | Quant-Lab | missing/extra columns | validation refuses | schema drift accepted | critical |
| QL-MAN-001 | unit | Quant-Lab | tampered file/manifest | hash/size/rows/schema detected | artifact corruption accepted | critical |
| QL-MAN-002 | unit | Quant-Lab | PK/FK/as-of tampering | complete validation detects | linkage corruption accepted | critical |
| QL-MAN-003 | unit | Quant-Lab | mismatched v2 reference | pair rejected | cross-artifact join | critical |
| QL-LNK-001 | unit | Quant-Lab | exact stage/cursor/capture | unique link accepted | exact evidence unavailable | critical |
| QL-LNK-002 | adversarial | Quant-Lab | setup-only/nearest/row-order/duplicate | all rejected | approximate join leakage | critical |
| QL-TIER-001 | unit | Quant-Lab | tier registry | primary excludes 240m/influenced aggregates | experimental leakage | critical |
| QL-TIER-002 | unit | Quant-Lab | formula-v1/superseded input | M2 rejected | defective displacement used | critical |
| QL-TIER-003 | unit | Quant-Lab | M3 no positive variation | descriptive-only status | unsupported modeling | high |
| QL-ACC-001 | unit | Quant-Lab | authorized date enumerator | only permitted paths constructed | protected discovery | critical |
| QL-ACC-002 | integration | Quant-Lab | preparation audit | every June 11/sealed counter equals zero | protected access | critical |
| QL-LBL-001 | unit | Quant-Lab | strict cutoff fixture | post-entry/pre-cutoff only | boundary leakage | critical |
| QL-LBL-002 | unit | Quant-Lab | R1/R1.5/R2/fixed overrides | each recomputes path metrics and identity | label reuse | critical |
| QL-LBL-003 | unit | Quant-Lab | hypothetical protected hit | remains censored boundary | protected outcome leakage | critical |
| QL-WRM-001 | integration | Quant-Lab | ten-day sequential warmup | state retained; warmup candidates excluded | broken occupancy/scope | critical |
| QL-FLD-001 | unit | Quant-Lab | boundary-crossing setup | whole setup excluded | setup leakage | critical |
| QL-FLD-002 | unit | Quant-Lab | overlap and embargo fixture | purge then two-day embargo | label leakage | critical |
| QL-FLD-003 | unit | Quant-Lab | overlapping-candidate fixture | max one OOS prediction | prediction dedupe leakage | critical |
| QL-FLD-004 | unit | Quant-Lab | single/no-class fixtures | invalid reason retained; descriptive completion | hidden invalid folds | high |
| QL-MDL-001 | unit | Quant-Lab | resolved CatBoost contract | exact config/hash/thread/seed | protocol drift | critical |
| QL-MDL-002 | unit | Quant-Lab | missing numeric/categorical values | NaN and `__MISSING__` preserved | silent imputation drift | high |
| QL-MET-001 | unit | Quant-Lab | constant/sparse/undefined AUC fixture | explicit coverage/undefined diagnostics | misleading metrics | high |
| QL-MET-002 | unit | Quant-Lab | reference/calibration/reliability fixture | deterministic complete output | metric drift | high |
| QL-BTS-001 | unit | Quant-Lab | setup/day/paired block fixtures | seed 7, 10k, null under two clusters | uncertainty leakage | high |
| QL-IMP-001 | unit | Quant-Lab | fold/permutation importance | 20 repeats, coverage/variance, descriptive | feature selection leakage | high |
| QL-IMM-001 | integration | Quant-Lab | run/view store | full-ID atomic save and duplicate refusal | mutable research result | critical |
| QL-JOB-001 | integration | Quant-Lab | lock/cancel/restart/crash fixture | boundary cancellation and recovery | unsafe preparation | critical |
| QL-UI-001 | AppTest | Quant-Lab | IFVG Lab shell | Experiments default; three tabs | router not restored | critical |
| QL-UI-002 | AppTest | Quant-Lab | capability/preparation fixtures | status/reasons rendered | unsafe launch | high |
| QL-UI-003 | AppTest | Quant-Lab | exact replay fixture | candidate/decision/trade exact IDs | fallback replay | critical |
| QL-UI-004 | AppTest | Quant-Lab | candidate/execution sentinels | outcomes never exchanged | report conflation | critical |
| QL-UI-005 | AppTest | Quant-Lab | legacy fixture | caveated read-only; no rerun/delete/compare | legacy misuse | high |
| QL-UI-006 | AppTest | Quant-Lab | widget/text inventory | no protected/sealed controls | unsafe UI | critical |
| QL-UI-007 | visual/a11y | Quant-Lab | desktop/narrow/keyboard/empty/failure | readable, navigable, sanitized | unusable/unsafe UI | high |
| E2E-001 | real permitted | both | January preparation twice | identical ID; overwrite refusal | nondeterministic artifact | critical |
| E2E-002 | real permitted | both | full authorized pair | verified reports and zero protected counters | preparation unsafe | critical |
| E2E-003 | real permitted | Quant-Lab | fixed-tier experiment | immutable deterministic reload/compare | research nondeterminism | critical |
| E2E-004 | real permitted | Quant-Lab | exact candidate examples | replay/reconciliation complete | audit gap | high |
| E2E-005 | suites | both | native pytest/Ruff | all pass | integration regression | critical |

Final handoff records the concrete test file and test name for each ID, the command, result,
duration, fixture identity, and measured performance. A skipped critical gate remains open; it is
not reported as passing.

## Final execution status — 2026-07-31

| IDs | Status | Evidence |
|---|---|---|
| SC-CAL-001..003, SC-DSP-001..006 | pass | `test_exchange_calendar.py`, `test_context_displacement.py`; cached-count parity covers weekend and both DST directions |
| SC-EQL-001..004 | pass | `test_equal_level_pools.py`; long-EQL/short-EQH, ties, all links, and saturated eviction |
| SC-ID-001..003, SC-SEED-001 | pass | `test_ifvg_v2_characterization.py`, `test_ifvg_v2_e2e_goldens.py`, `test_ifvg_context_cross_cutting.py` |
| SC-CAP-001 | pass | real January seed 809,170 bytes and transition 23,211 bytes; pool maximum 5 |
| SC-PERF-001 | **fail** | observer/callback p99 pass; median slowdown 199.0982% and repeated p95 203.7889% fail |
| QL-SCH-001..002 | pass | 25 `test_ifvg_context_schemas.py` tests |
| QL-MAN-001..003, QL-LNK-001..002 | pass | v3 manifest/reconciliation/tamper tests and exact-link adversarial fixtures |
| QL-TIER-001..003 | pass | experiment contract and cohort-specific M3 tests |
| QL-ACC-001..002 | pass in synthetic/audit fixtures | explicit-date access tests; protected/sealed event counters are zero |
| QL-LBL-001..003, QL-WRM-001 | pass | strict cutoff, independent barrier, censoring, and sequential warmup fixtures |
| QL-FLD-001..004 | pass | setup-boundary detection precedes censor filtering; purge then embargo; unique OOS rows |
| QL-MDL-001..002, QL-MET-001..002, QL-BTS-001, QL-IMP-001 | pass | fixed model/statistics protocol tests; no adaptive search executed |
| QL-IMM-001, QL-JOB-001 | pass | full-ID/containment/duplicate refusal and persisted preparation fixtures |
| QL-UI-001..006 | pass | 15 UI/AppTest tests and successful local headless server startup |
| QL-UI-007 | open | in-app browser unavailable; viewport/keyboard/visual inspection not substituted |
| E2E-001 | blocked | January verification refuses publication at SC-PERF-001, so repeat ID/overwrite proof cannot complete |
| E2E-002..004 | blocked | no promoted formula-v2 pair; full permitted preparation/model/replay was not started |
| E2E-005 | pass | Strategy-Core 414 unit + 6 validation tests; Quant-Lab 970 tests; both Ruff suites clean |

Commands and durations:

```text
Strategy-Core: python -m pytest tests -q                         414 passed, 140.5 s final run
Strategy-Core: python -m pytest validation -q --durations=10      6 passed, 764.1 s
Strategy-Core: python -m ruff check src tests                     clean
Quant-Lab:     python -m pytest -q                                970 passed, 265.65 s final run
Quant-Lab:     python -m ruff check src tests scripts             clean
Quant-Lab:     python scripts/run_ifvg_context_capture.py --cached-artifacts-only
                                                                  refused at performance gate, 130.3 s
```

The January command used only its explicit 26-date cached allowlist. The Strategy validation
suite used its frozen July 2025 fixtures. Neither run accessed June 11, 2026 or the sealed range.
