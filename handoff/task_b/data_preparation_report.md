# Task B — completed data preparation

The supplied 154-weekday inventory was processed without filling missing days: 151 prepared and three missing. The first ten weekdays, June 2–13, 2025, are warmup. The approved plan has 253 evaluation days from June 16, 2025 through June 10, 2026. Roll days remain included under decision 21.

New cache root: `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\prepared_2025\artifacts`.

The original 2026 cache remains in its original layout. Its separate registration preserves all 138 original preparation dates and their seed chain; it owns 114 weekday replay dates. Each store retains its original cache seed lineage. Reducer and context state continue across the ordered study replay.

| Registration | ID | Prepared receipts | Owned dates |
|---|---|---:|---:|
| `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\prepared_2025\prepared_store.json` | `57f43b0e446c26e01b0e8e8f7a88a52ccf96a78bdaa791ede3474fc0bbbb34ca` | 151 | 151 |
| `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\prepared_2026_registration\prepared_store.json` | `c63f342fe00db7cc5d121f962ca616a8c62bb557a4b02c0948954664875d510b` | 138 | 114 |

Preparation completed at `2026-10-03T21:09:41.948127+00:00`. Process wall clock: 4723.208 seconds; recorded per-day total: 4722.726 seconds. The separate catalog enrichment took 344.560 seconds.

The initial job used selected integer instrument IDs. A separate index-only pass added actual contract symbols from the source's `symbol` column when available, without changing IDs. Original preparation reports and registration bytes were preserved. The enrichment receipt verifies every registered bars/levels file against its original SHA-256 and footer before and after catalog updates, including the earlier 2026 backfill checksums.

Evidence: `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\prepared_2025\preparation_run.json`, `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\prepared_2025\preparation_days.json`, `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\backfill_catalog_receipt.json`, and `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\catalog_symbol_enrichment_receipt.json`.

`partial_bar_count` below counts partial candles, including ordinary session edges. It does not establish a partial exchange session. Special session rules come only from the committed local schedules and are listed independently in `calendar_notes.md`. No structural or holding calendar code was changed.

| Logical day | Status / missing reason | Scope | Special session rule | Selected instrument | Raw symbol | Bars | Partial candles | Seconds | Reader warnings | Last TIME 1m availability (Chicago) | Deadline candle |
|---|---|---|---|---:|---|---:|---:|---:|---|---|---|
| 2025-06-02 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2211 | 8 | 24.270 | prior-day file missing; trading-day window served from a single file | not audited | not audited |
| 2025-06-03 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 23.784 |  | not audited | not audited |
| 2025-06-04 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 24.523 |  | not audited | not audited |
| 2025-06-05 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 37.200 |  | not audited | not audited |
| 2025-06-06 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 27.770 |  | not audited | not audited |
| 2025-06-09 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 19.924 |  | not audited | not audited |
| 2025-06-10 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 25.641 |  | not audited | not audited |
| 2025-06-11 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 32.805 |  | not audited | not audited |
| 2025-06-12 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 22.653 |  | not audited | not audited |
| 2025-06-13 | prepared | warmup | ordinary local schedule | 42005804 | NQM5 | 2421 | 8 | 36.188 |  | not audited | not audited |
| 2025-06-16 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2419 | 8 | 22.781 |  | 2025-06-16T16:00:00-05:00 | present |
| 2025-06-17 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 29.211 |  | 2025-06-17T16:00:00-05:00 | present |
| 2025-06-18 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 28.031 |  | 2025-06-18T16:00:00-05:00 | present |
| 2025-06-19 | prepared | evaluation | Juneteenth: Chicago close 12:00; deadline 11:55 | 42008487 | NQU5 | 2000 | 8 | 7.704 |  | 2025-06-19T12:00:00-05:00 | present |
| 2025-06-20 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 29.302 |  | 2025-06-20T16:00:00-05:00 | present |
| 2025-06-23 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 29.594 |  | 2025-06-23T16:00:00-05:00 | present |
| 2025-06-24 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 22.190 |  | 2025-06-24T16:00:00-05:00 | present |
| 2025-06-25 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 19.975 |  | 2025-06-25T16:00:00-05:00 | present |
| 2025-06-26 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 18.971 |  | 2025-06-26T16:00:00-05:00 | present |
| 2025-06-27 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 23.604 |  | 2025-06-27T16:00:00-05:00 | present |
| 2025-06-30 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 20.069 |  | 2025-06-30T16:00:00-05:00 | present |
| 2025-07-01 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 25.583 |  | 2025-07-01T16:00:00-05:00 | present |
| 2025-07-02 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 20.411 |  | 2025-07-02T16:00:00-05:00 | present |
| 2025-07-03 | prepared | evaluation | Logical-day Chicago close 16:00; deadline 15:55 | 42008487 | NQU5 | 2027 | 8 | 12.705 |  | 2025-07-03T12:15:00-05:00 | absent |
| 2025-07-04 | prepared | evaluation | Prior civil July 3 Chicago close 23:00; deadline 22:55; retained logical day | 42008487 | NQU5 | 2000 | 8 | 4.085 |  | 2025-07-04T12:00:00-05:00 | present |
| 2025-07-07 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 25.699 |  | 2025-07-07T16:00:00-05:00 | present |
| 2025-07-08 | missing / no_source_partition | omitted | ordinary local schedule | null | null | 0 | 0 | 0.000 |  | not audited | not audited |
| 2025-07-09 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2211 | 8 | 21.623 | prior-day file missing; trading-day window served from a single file | 2025-07-09T16:00:00-05:00 | present |
| 2025-07-10 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 19.700 |  | 2025-07-10T16:00:00-05:00 | present |
| 2025-07-11 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 24.438 |  | 2025-07-11T16:00:00-05:00 | present |
| 2025-07-14 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 20.367 |  | 2025-07-14T16:00:00-05:00 | present |
| 2025-07-15 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 29.438 |  | 2025-07-15T16:00:00-05:00 | present |
| 2025-07-16 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 28.357 |  | 2025-07-16T16:00:00-05:00 | present |
| 2025-07-17 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 20.312 |  | 2025-07-17T16:00:00-05:00 | present |
| 2025-07-18 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 19.819 |  | 2025-07-18T16:00:00-05:00 | present |
| 2025-07-21 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 16.110 |  | 2025-07-21T16:00:00-05:00 | present |
| 2025-07-22 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 25.304 |  | 2025-07-22T16:00:00-05:00 | present |
| 2025-07-23 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 24.322 |  | 2025-07-23T16:00:00-05:00 | present |
| 2025-07-24 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 20.618 |  | 2025-07-24T16:00:00-05:00 | present |
| 2025-07-25 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 16.354 |  | 2025-07-25T16:00:00-05:00 | present |
| 2025-07-28 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 18.224 |  | 2025-07-28T16:00:00-05:00 | present |
| 2025-07-29 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 24.114 |  | 2025-07-29T16:00:00-05:00 | present |
| 2025-07-30 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2420 | 8 | 25.558 |  | 2025-07-30T16:00:00-05:00 | present |
| 2025-07-31 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 40.043 |  | 2025-07-31T16:00:00-05:00 | present |
| 2025-08-01 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 52.209 |  | 2025-08-01T16:00:00-05:00 | present |
| 2025-08-04 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 21.849 |  | 2025-08-04T16:00:00-05:00 | present |
| 2025-08-05 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 29.945 |  | 2025-08-05T16:00:00-05:00 | present |
| 2025-08-06 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 26.502 |  | 2025-08-06T16:00:00-05:00 | present |
| 2025-08-07 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 32.625 |  | 2025-08-07T16:00:00-05:00 | present |
| 2025-08-08 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 20.893 |  | 2025-08-08T16:00:00-05:00 | present |
| 2025-08-11 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 20.723 |  | 2025-08-11T16:00:00-05:00 | present |
| 2025-08-12 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 23.394 |  | 2025-08-12T16:00:00-05:00 | present |
| 2025-08-13 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 22.198 |  | 2025-08-13T16:00:00-05:00 | present |
| 2025-08-14 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2420 | 8 | 25.783 |  | 2025-08-14T16:00:00-05:00 | present |
| 2025-08-15 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 21.729 |  | 2025-08-15T16:00:00-05:00 | present |
| 2025-08-18 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 18.609 |  | 2025-08-18T16:00:00-05:00 | present |
| 2025-08-19 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 26.749 |  | 2025-08-19T16:00:00-05:00 | present |
| 2025-08-20 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 37.947 |  | 2025-08-20T16:00:00-05:00 | present |
| 2025-08-21 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 30.136 |  | 2025-08-21T16:00:00-05:00 | present |
| 2025-08-22 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2420 | 8 | 27.116 |  | 2025-08-22T16:00:00-05:00 | present |
| 2025-08-25 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 19.506 |  | 2025-08-25T16:00:00-05:00 | present |
| 2025-08-26 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 22.822 |  | 2025-08-26T16:00:00-05:00 | present |
| 2025-08-27 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 19.500 |  | 2025-08-27T16:00:00-05:00 | present |
| 2025-08-28 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 22.929 |  | 2025-08-28T16:00:00-05:00 | present |
| 2025-08-29 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 25.624 |  | 2025-08-29T16:00:00-05:00 | present |
| 2025-09-01 | prepared | evaluation | Labor Day: Chicago close 12:00; deadline 11:55 | 42008487 | NQU5 | 2000 | 8 | 4.335 |  | 2025-09-01T12:00:00-05:00 | present |
| 2025-09-02 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 36.081 |  | 2025-09-02T16:00:00-05:00 | present |
| 2025-09-03 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 28.358 |  | 2025-09-03T16:00:00-05:00 | present |
| 2025-09-04 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 22.092 |  | 2025-09-04T16:00:00-05:00 | present |
| 2025-09-05 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 39.493 |  | 2025-09-05T16:00:00-05:00 | present |
| 2025-09-08 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 22.043 |  | 2025-09-08T16:00:00-05:00 | present |
| 2025-09-09 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 23.168 |  | 2025-09-09T16:00:00-05:00 | present |
| 2025-09-10 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 27.033 |  | 2025-09-10T16:00:00-05:00 | present |
| 2025-09-11 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2420 | 8 | 21.986 |  | 2025-09-11T16:00:00-05:00 | present |
| 2025-09-12 | prepared | evaluation | ordinary local schedule | 42008487 | NQU5 | 2421 | 8 | 21.945 |  | 2025-09-12T16:00:00-05:00 | present |
| 2025-09-15 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2417 | 8 | 18.766 |  | 2025-09-15T16:00:00-05:00 | present |
| 2025-09-16 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 20.849 |  | 2025-09-16T16:00:00-05:00 | present |
| 2025-09-17 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 35.657 |  | 2025-09-17T16:00:00-05:00 | present |
| 2025-09-18 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 29.147 |  | 2025-09-18T16:00:00-05:00 | present |
| 2025-09-19 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 23.525 |  | 2025-09-19T16:00:00-05:00 | present |
| 2025-09-22 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2420 | 8 | 22.756 |  | 2025-09-22T16:00:00-05:00 | present |
| 2025-09-23 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 30.563 |  | 2025-09-23T16:00:00-05:00 | present |
| 2025-09-24 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 28.160 |  | 2025-09-24T16:00:00-05:00 | present |
| 2025-09-25 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 40.109 |  | 2025-09-25T16:00:00-05:00 | present |
| 2025-09-26 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 32.604 |  | 2025-09-26T16:00:00-05:00 | present |
| 2025-09-29 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 27.604 |  | 2025-09-29T16:00:00-05:00 | present |
| 2025-09-30 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 27.743 |  | 2025-09-30T16:00:00-05:00 | present |
| 2025-10-01 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 24.855 |  | 2025-10-01T16:00:00-05:00 | present |
| 2025-10-02 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 26.932 |  | 2025-10-02T16:00:00-05:00 | present |
| 2025-10-03 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 27.689 |  | 2025-10-03T16:00:00-05:00 | present |
| 2025-10-06 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 25.686 |  | 2025-10-06T16:00:00-05:00 | present |
| 2025-10-07 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 30.535 |  | 2025-10-07T16:00:00-05:00 | present |
| 2025-10-08 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 22.724 |  | 2025-10-08T16:00:00-05:00 | present |
| 2025-10-09 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 30.742 |  | 2025-10-09T16:00:00-05:00 | present |
| 2025-10-10 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 61.744 |  | 2025-10-10T16:00:00-05:00 | present |
| 2025-10-13 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 42.307 |  | 2025-10-13T16:00:00-05:00 | present |
| 2025-10-14 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 53.223 |  | 2025-10-14T16:00:00-05:00 | present |
| 2025-10-15 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 44.742 |  | 2025-10-15T16:00:00-05:00 | present |
| 2025-10-16 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 56.686 |  | 2025-10-16T16:00:00-05:00 | present |
| 2025-10-17 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 58.117 |  | 2025-10-17T16:00:00-05:00 | present |
| 2025-10-20 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 27.636 |  | 2025-10-20T16:00:00-05:00 | present |
| 2025-10-21 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 30.449 |  | 2025-10-21T16:00:00-05:00 | present |
| 2025-10-22 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 48.387 |  | 2025-10-22T16:00:00-05:00 | present |
| 2025-10-23 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 33.278 |  | 2025-10-23T16:00:00-05:00 | present |
| 2025-10-24 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 26.716 |  | 2025-10-24T16:00:00-05:00 | present |
| 2025-10-27 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 23.156 |  | 2025-10-27T16:00:00-05:00 | present |
| 2025-10-28 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 26.762 |  | 2025-10-28T16:00:00-05:00 | present |
| 2025-10-29 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 39.262 |  | 2025-10-29T16:00:00-05:00 | present |
| 2025-10-30 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 53.660 |  | 2025-10-30T16:00:00-05:00 | present |
| 2025-10-31 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 45.935 |  | 2025-10-31T16:00:00-05:00 | present |
| 2025-11-03 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 35.573 |  | 2025-11-03T16:00:00-06:00 | present |
| 2025-11-04 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 51.819 |  | 2025-11-04T16:00:00-06:00 | present |
| 2025-11-05 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 44.457 |  | 2025-11-05T16:00:00-06:00 | present |
| 2025-11-06 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 57.465 |  | 2025-11-06T16:00:00-06:00 | present |
| 2025-11-07 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 58.417 |  | 2025-11-07T16:00:00-06:00 | present |
| 2025-11-10 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 42.335 |  | 2025-11-10T16:00:00-06:00 | present |
| 2025-11-11 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 39.694 |  | 2025-11-11T16:00:00-06:00 | present |
| 2025-11-12 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 50.125 |  | 2025-11-12T16:00:00-06:00 | present |
| 2025-11-13 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 66.665 |  | 2025-11-13T16:00:00-06:00 | present |
| 2025-11-14 | missing / no_source_partition | omitted | ordinary local schedule | null | null | 0 | 0 | 0.000 |  | not audited | not audited |
| 2025-11-17 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 66.001 |  | 2025-11-17T16:00:00-06:00 | present |
| 2025-11-18 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 81.116 |  | 2025-11-18T16:00:00-06:00 | present |
| 2025-11-19 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 69.730 |  | 2025-11-19T16:00:00-06:00 | present |
| 2025-11-20 | missing / no_source_partition | omitted | ordinary local schedule | null | null | 0 | 0 | 0.000 |  | not audited | not audited |
| 2025-11-21 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2316 | 8 | 86.904 | prior-day file missing; trading-day window served from a single file | 2025-11-21T16:00:00-06:00 | present |
| 2025-11-24 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 50.588 |  | 2025-11-24T16:00:00-06:00 | present |
| 2025-11-25 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 52.917 |  | 2025-11-25T16:00:00-06:00 | present |
| 2025-11-26 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 39.514 |  | 2025-11-26T16:00:00-06:00 | present |
| 2025-11-27 | prepared | evaluation | Thanksgiving: Chicago close 12:00; deadline 11:55 | 158704 | NQZ5 | 2000 | 8 | 4.128 |  | 2025-11-27T12:00:00-06:00 | present |
| 2025-11-28 | prepared | evaluation | Ordinary declared Chicago close 16:00; deadline 15:55 | 158704 | NQZ5 | 897 | 8 | 7.518 |  | 2025-11-28T12:15:00-06:00 | absent |
| 2025-12-01 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 49.002 |  | 2025-12-01T16:00:00-06:00 | present |
| 2025-12-02 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 50.489 |  | 2025-12-02T16:00:00-06:00 | present |
| 2025-12-03 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 36.933 |  | 2025-12-03T16:00:00-06:00 | present |
| 2025-12-04 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 41.670 |  | 2025-12-04T16:00:00-06:00 | present |
| 2025-12-05 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 37.152 |  | 2025-12-05T16:00:00-06:00 | present |
| 2025-12-08 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 34.226 |  | 2025-12-08T16:00:00-06:00 | present |
| 2025-12-09 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 30.802 |  | 2025-12-09T16:00:00-06:00 | present |
| 2025-12-10 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 40.323 |  | 2025-12-10T16:00:00-06:00 | present |
| 2025-12-11 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 46.721 |  | 2025-12-11T16:00:00-06:00 | present |
| 2025-12-12 | prepared | evaluation | ordinary local schedule | 158704 | NQZ5 | 2421 | 8 | 58.747 |  | 2025-12-12T16:00:00-06:00 | present |
| 2025-12-15 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 41.427 |  | 2025-12-15T16:00:00-06:00 | present |
| 2025-12-16 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 57.716 |  | 2025-12-16T16:00:00-06:00 | present |
| 2025-12-17 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 60.640 |  | 2025-12-17T16:00:00-06:00 | present |
| 2025-12-18 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 60.745 |  | 2025-12-18T16:00:00-06:00 | present |
| 2025-12-19 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 35.762 |  | 2025-12-19T16:00:00-06:00 | present |
| 2025-12-22 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 23.892 |  | 2025-12-22T16:00:00-06:00 | present |
| 2025-12-23 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 19.868 |  | 2025-12-23T16:00:00-06:00 | present |
| 2025-12-24 | prepared | evaluation | Logical-day Chicago close 16:00; deadline 15:55 | 42002475 | NQH6 | 2025 | 8 | 9.068 |  | 2025-12-24T12:15:00-06:00 | absent |
| 2025-12-25 | prepared | omitted | Existing Christmas full-closure evaluation omission | 42002475 | NQH6 | 0 | 0 | 0.116 |  | not audited | not audited |
| 2025-12-26 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2418 | 8 | 15.037 |  | 2025-12-26T16:00:00-06:00 | present |
| 2025-12-29 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 23.437 |  | 2025-12-29T16:00:00-06:00 | present |
| 2025-12-30 | prepared | evaluation | ordinary local schedule | 42002475 | NQH6 | 2421 | 8 | 18.545 |  | 2025-12-30T16:00:00-06:00 | present |
| 2025-12-31 | prepared | evaluation | Logical-day Chicago close 16:00; deadline 15:55 | 42002475 | NQH6 | 2421 | 8 | 21.918 |  | 2025-12-31T16:00:00-06:00 | present |
| 2026-01-01 | prepared | omitted | Existing New Year full-closure evaluation omission | 42002475 | NQH6 | 0 | 0 | 0.124 |  | not audited | not audited |

Source-access audit is saved in the immutable completion receipt. The completion and enrichment guards assert zero forbidden accesses; no partition after June 10, 2026 was opened.

| Completion audit counter | Value |
|---|---:|
| `path_constructions` | 1203 |
| `metadata_accesses` | 902 |
| `file_opens` | 631 |
| `rows_read` | 3274507467 |
| `protected_path_constructions` | 0 |
| `protected_metadata_accesses` | 0 |
| `protected_file_opens` | 0 |
| `protected_rows_read` | 0 |

The cached-only boundary audit `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\prepared_boundary_audit.json` verified all 578 named bars/levels files before and after projecting timing fields, with zero raw source reads. Its per-day timing coverage remains distinct from the committed local schedule. The full source-access receipts remain external.
