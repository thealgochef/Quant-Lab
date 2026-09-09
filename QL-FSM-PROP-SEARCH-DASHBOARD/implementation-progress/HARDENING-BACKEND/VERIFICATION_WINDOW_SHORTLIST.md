# Verification window shortlist — logical trading days (§5.1; F-22)

**Owner selection: NOT PERFORMED.** This document ranks candidate
windows from already-authorized evidence only; it registers nothing:
`register_program_allowlist` called = `False`;
raw source reads = `False`. No permanent allowlist
was hashed or selected; the owner selects one corrected logical-day window.

- shortlist id (content hash): `1a77a00ed6911f6a158c648349df94a2c37923bf4553eb30bcb90cfdbc08bd6a`
- calendar policy: `cme_globex_18et_weekday_v1`; window length: 5
- evidence: accepted v2 dataset `143b510f8a73896072f44e08f331ef5156e85eb8e5124d25bdf441c4fb6b2ac7` (manifest `b089dfadf44253b7071882cc9577feeacd97e532f4fd7086fe2fcbac39e60c22`), FSM-audit artifact `7e55ee89d9492fa8338cefa3c6389c9e80c4b139f1ec3b3a1a87ab7ac701fe41`
- candidate windows: 110; eligible: 50

## Ranking order (lexicographic, no hidden score)

```text
required_source_and_replay_coverage
→ lifecycle_path_classes_represented
→ audit_day_coverage
→ executed_trades
→ decisions
→ candidates
→ exact_verifier_targets
→ mbp1_scope_evidence
→ panel_control_flow_coverage
→ distance_from_protected_boundary
```

## Required shortlist entries

| Label | Rank | Window | Eligible | Rank key | Seed chain (store days) |
|---|---|---|---|---|---|
| june_proposal | 70 | 2026-06-04 2026-06-05 2026-06-08 2026-06-09 2026-06-10 | False | [5, 1, 5, 0, 0, 0, 0, 0, 5, 1] | 132 (2026-01-01 … 2026-06-03) |
| highest_lifecycle_candidate_decision_trade_coverage | 3 | 2026-02-06 2026-02-09 2026-02-10 2026-02-11 2026-02-12 | True | [5, 4, 5, 9, 9, 30, 9, 0, 5, 119] | 31 (2026-01-01 … 2026-02-05) |
| highest_complete_audit_day_coverage | 1 | 2026-02-04 2026-02-05 2026-02-06 2026-02-09 2026-02-10 | True | [5, 4, 5, 9, 9, 30, 9, 0, 5, 121] | 29 (2026-01-01 … 2026-02-03) |
| top_ranked | 1 | 2026-02-04 2026-02-05 2026-02-06 2026-02-09 2026-02-10 | True | [5, 4, 5, 9, 9, 30, 9, 0, 5, 121] | 29 (2026-01-01 … 2026-02-03) |

## R1 store-day candidate

- as listed: 2026-02-06 2026-02-08 2026-02-09 2026-02-10 2026-02-11 — **provisional_ineligible_as_stated**
- non-trading-day ids: 2026-02-08
- corrected logical form: 2026-02-06 2026-02-09 2026-02-10 2026-02-11 2026-02-12
- the R1 scan scored consecutive STORE days (physical partition dates); 2026-02-08 is a Sunday partition holding the Sunday 18:00 ET open of the following Monday's trading day and is not a trading-day id; the corrected logical form starts at the same first day and takes the next consecutive logical trading days

## Ranking trace (top 12)

| # | Window | Eligible | required_source_and_replay_coverage | lifecycle_path_classes_represented | audit_day_coverage | executed_trades | decisions | candidates | exact_verifier_targets | mbp1_scope_evidence | panel_control_flow_coverage | distance_from_protected_boundary | Failed constraints |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-02-04 2026-02-05 2026-02-06 2026-02-09 2026-02-10 | True | 5 | 4 | 5 | 9 | 9 | 30 | 9 | 0 | 5 | 121 | — |
| 2 | 2026-02-05 2026-02-06 2026-02-09 2026-02-10 2026-02-11 | True | 5 | 4 | 5 | 9 | 9 | 30 | 9 | 0 | 5 | 120 | — |
| 3 | 2026-02-06 2026-02-09 2026-02-10 2026-02-11 2026-02-12 | True | 5 | 4 | 5 | 9 | 9 | 30 | 9 | 0 | 5 | 119 | — |
| 4 | 2026-04-03 2026-04-06 2026-04-07 2026-04-08 2026-04-09 | True | 5 | 4 | 5 | 8 | 8 | 35 | 8 | 0 | 5 | 63 | — |
| 5 | 2026-04-06 2026-04-07 2026-04-08 2026-04-09 2026-04-10 | True | 5 | 4 | 5 | 7 | 7 | 34 | 7 | 0 | 5 | 62 | — |
| 6 | 2026-02-09 2026-02-10 2026-02-11 2026-02-12 2026-02-13 | True | 5 | 4 | 5 | 7 | 7 | 23 | 7 | 0 | 5 | 118 | — |
| 7 | 2026-02-10 2026-02-11 2026-02-12 2026-02-13 2026-02-16 | True | 5 | 4 | 5 | 7 | 7 | 23 | 7 | 0 | 5 | 115 | — |
| 8 | 2026-04-07 2026-04-08 2026-04-09 2026-04-10 2026-04-13 | True | 5 | 4 | 5 | 5 | 5 | 25 | 5 | 0 | 5 | 59 | — |
| 9 | 2026-04-08 2026-04-09 2026-04-10 2026-04-13 2026-04-14 | True | 5 | 4 | 5 | 4 | 4 | 20 | 4 | 0 | 5 | 58 | — |
| 10 | 2026-04-09 2026-04-10 2026-04-13 2026-04-14 2026-04-15 | True | 5 | 4 | 5 | 4 | 4 | 19 | 4 | 0 | 5 | 57 | — |
| 11 | 2026-03-04 2026-03-05 2026-03-06 2026-03-09 2026-03-10 | True | 5 | 4 | 5 | 4 | 4 | 17 | 4 | 0 | 5 | 93 | — |
| 12 | 2026-04-02 2026-04-03 2026-04-06 2026-04-07 2026-04-08 | True | 5 | 4 | 5 | 4 | 4 | 16 | 4 | 0 | 5 | 64 | — |

## Hard constraints of the shortlisted windows

| Window | consecutive_logical_days_1_to_5 | all_source_partitions_present_and_hash_addressable | seed_chain_producible_through_prior_store_day | at_least_one_exact_verifier_target | june_11_and_sealed_excluded |
|---|---|---|---|---|---|
| 2026-06-04 2026-06-05 2026-06-08 2026-06-09 2026-06-10 | True | True | True | False | True |
| 2026-02-06 2026-02-09 2026-02-10 2026-02-11 2026-02-12 | True | True | True | True | True |
| 2026-02-04 2026-02-05 2026-02-06 2026-02-09 2026-02-10 | True | True | True | True | True |

## Exact trading-day mapping of the shortlisted windows

### june_proposal: 2026-06-04 2026-06-05 2026-06-08 2026-06-09 2026-06-10

| Logical day | Session (UTC) | Partition | Key | Kind | Content sha256 |
|---|---|---|---|---|---|
| 2026-06-04 | 2026-06-03T22:00:00+00:00 → 2026-06-04T22:00:00+00:00 | 2026-06-03 | prev_utc_date | mbp1 | `65e679b8a0f1dd963dd12ce3b391088dc982b82615fbb5cef62c5049fcd906e6` |
| 2026-06-04 | 2026-06-03T22:00:00+00:00 → 2026-06-04T22:00:00+00:00 | 2026-06-04 | utc_date | mbp1 | `eb11de9e859a64716cde7c2b2763cac80deab3c0c3bda1ff6de7aee983e99f82` |
| 2026-06-05 | 2026-06-04T22:00:00+00:00 → 2026-06-05T22:00:00+00:00 | 2026-06-04 | prev_utc_date | mbp1 | `eb11de9e859a64716cde7c2b2763cac80deab3c0c3bda1ff6de7aee983e99f82` |
| 2026-06-05 | 2026-06-04T22:00:00+00:00 → 2026-06-05T22:00:00+00:00 | 2026-06-05 | utc_date | mbp1 | `9c464ad078465b29303feb20e8ed15ed076205d2caba8b46e0b11740109fc9c9` |
| 2026-06-08 | 2026-06-07T22:00:00+00:00 → 2026-06-08T22:00:00+00:00 | 2026-06-07 | prev_utc_date | mbp1 | `b79ef03fb15b7222a63d5bd1b42cd16dd04ad97a51a9c431844ff412a5a13e64` |
| 2026-06-08 | 2026-06-07T22:00:00+00:00 → 2026-06-08T22:00:00+00:00 | 2026-06-08 | utc_date | mbp1 | `1bb2f9695515f803ea4b3623732516ac9583f4e4669fc9e2aaf583532b09d263` |
| 2026-06-09 | 2026-06-08T22:00:00+00:00 → 2026-06-09T22:00:00+00:00 | 2026-06-08 | prev_utc_date | mbp1 | `1bb2f9695515f803ea4b3623732516ac9583f4e4669fc9e2aaf583532b09d263` |
| 2026-06-09 | 2026-06-08T22:00:00+00:00 → 2026-06-09T22:00:00+00:00 | 2026-06-09 | utc_date | mbp1 | `0c759eaf00859ac9c6c69c3a0980c76a61d3a5fd46de782abe38e7adc08b69ab` |
| 2026-06-10 | 2026-06-09T22:00:00+00:00 → 2026-06-10T22:00:00+00:00 | 2026-06-09 | prev_utc_date | mbp1 | `0c759eaf00859ac9c6c69c3a0980c76a61d3a5fd46de782abe38e7adc08b69ab` |
| 2026-06-10 | 2026-06-09T22:00:00+00:00 → 2026-06-10T22:00:00+00:00 | 2026-06-10 | utc_date | mbp1 | `a65d6248af45bf19e93635931c0de3d84a1af1a591151d13239ff5a30acbf289` |

### highest_lifecycle_candidate_decision_trade_coverage: 2026-02-06 2026-02-09 2026-02-10 2026-02-11 2026-02-12

| Logical day | Session (UTC) | Partition | Key | Kind | Content sha256 |
|---|---|---|---|---|---|
| 2026-02-06 | 2026-02-05T23:00:00+00:00 → 2026-02-06T23:00:00+00:00 | 2026-02-05 | prev_utc_date | mbp10 | `220b8646fa93a278ad30386fa62cb2f7ac03ab6a41a74586d927375acb60eb85` |
| 2026-02-06 | 2026-02-05T23:00:00+00:00 → 2026-02-06T23:00:00+00:00 | 2026-02-06 | utc_date | mbp10 | `18a8bed25c664f4a6e1da6376e2f57faa0f3abbd624032a548b0825827d5c0bd` |
| 2026-02-09 | 2026-02-08T23:00:00+00:00 → 2026-02-09T23:00:00+00:00 | 2026-02-08 | prev_utc_date | mbp10 | `b68ceafa14fe0f44c9189b967a381bec9eec6f59bbfbc0e7d3e4caa7bc452ced` |
| 2026-02-09 | 2026-02-08T23:00:00+00:00 → 2026-02-09T23:00:00+00:00 | 2026-02-09 | utc_date | mbp10 | `047decaf816467d0ea8bc775980663bdba9aa4e443a55a57db77ec029fa340ce` |
| 2026-02-10 | 2026-02-09T23:00:00+00:00 → 2026-02-10T23:00:00+00:00 | 2026-02-09 | prev_utc_date | mbp10 | `047decaf816467d0ea8bc775980663bdba9aa4e443a55a57db77ec029fa340ce` |
| 2026-02-10 | 2026-02-09T23:00:00+00:00 → 2026-02-10T23:00:00+00:00 | 2026-02-10 | utc_date | mbp10 | `e9f3c10f3de4a5763fc09024491460627e1434d51f71a278c8171624ea77e040` |
| 2026-02-11 | 2026-02-10T23:00:00+00:00 → 2026-02-11T23:00:00+00:00 | 2026-02-10 | prev_utc_date | mbp10 | `e9f3c10f3de4a5763fc09024491460627e1434d51f71a278c8171624ea77e040` |
| 2026-02-11 | 2026-02-10T23:00:00+00:00 → 2026-02-11T23:00:00+00:00 | 2026-02-11 | utc_date | mbp10 | `ebe2a5fbf1dde09df00de214b4d5b53cf3fa65e80391dae642ab733320f78423` |
| 2026-02-12 | 2026-02-11T23:00:00+00:00 → 2026-02-12T23:00:00+00:00 | 2026-02-11 | prev_utc_date | mbp10 | `ebe2a5fbf1dde09df00de214b4d5b53cf3fa65e80391dae642ab733320f78423` |
| 2026-02-12 | 2026-02-11T23:00:00+00:00 → 2026-02-12T23:00:00+00:00 | 2026-02-12 | utc_date | mbp10 | `a9df150a325d5e886c7887a25f50a6ee7801a18a8bcbc3f1e6cc586e53219d7e` |

### highest_complete_audit_day_coverage: 2026-02-04 2026-02-05 2026-02-06 2026-02-09 2026-02-10

| Logical day | Session (UTC) | Partition | Key | Kind | Content sha256 |
|---|---|---|---|---|---|
| 2026-02-04 | 2026-02-03T23:00:00+00:00 → 2026-02-04T23:00:00+00:00 | 2026-02-03 | prev_utc_date | mbp10 | `6e4042ff20b726d85ff9e3dd9412af66d9a78a3a13feac059e3c300113cd3f58` |
| 2026-02-04 | 2026-02-03T23:00:00+00:00 → 2026-02-04T23:00:00+00:00 | 2026-02-04 | utc_date | mbp10 | `6562e76e916d33ee479aa5501eb374f7411ffacff64c585d81ec942fd61613eb` |
| 2026-02-05 | 2026-02-04T23:00:00+00:00 → 2026-02-05T23:00:00+00:00 | 2026-02-04 | prev_utc_date | mbp10 | `6562e76e916d33ee479aa5501eb374f7411ffacff64c585d81ec942fd61613eb` |
| 2026-02-05 | 2026-02-04T23:00:00+00:00 → 2026-02-05T23:00:00+00:00 | 2026-02-05 | utc_date | mbp10 | `220b8646fa93a278ad30386fa62cb2f7ac03ab6a41a74586d927375acb60eb85` |
| 2026-02-06 | 2026-02-05T23:00:00+00:00 → 2026-02-06T23:00:00+00:00 | 2026-02-05 | prev_utc_date | mbp10 | `220b8646fa93a278ad30386fa62cb2f7ac03ab6a41a74586d927375acb60eb85` |
| 2026-02-06 | 2026-02-05T23:00:00+00:00 → 2026-02-06T23:00:00+00:00 | 2026-02-06 | utc_date | mbp10 | `18a8bed25c664f4a6e1da6376e2f57faa0f3abbd624032a548b0825827d5c0bd` |
| 2026-02-09 | 2026-02-08T23:00:00+00:00 → 2026-02-09T23:00:00+00:00 | 2026-02-08 | prev_utc_date | mbp10 | `b68ceafa14fe0f44c9189b967a381bec9eec6f59bbfbc0e7d3e4caa7bc452ced` |
| 2026-02-09 | 2026-02-08T23:00:00+00:00 → 2026-02-09T23:00:00+00:00 | 2026-02-09 | utc_date | mbp10 | `047decaf816467d0ea8bc775980663bdba9aa4e443a55a57db77ec029fa340ce` |
| 2026-02-10 | 2026-02-09T23:00:00+00:00 → 2026-02-10T23:00:00+00:00 | 2026-02-09 | prev_utc_date | mbp10 | `047decaf816467d0ea8bc775980663bdba9aa4e443a55a57db77ec029fa340ce` |
| 2026-02-10 | 2026-02-09T23:00:00+00:00 → 2026-02-10T23:00:00+00:00 | 2026-02-10 | utc_date | mbp10 | `e9f3c10f3de4a5763fc09024491460627e1434d51f71a278c8171624ea77e040` |

Every window above is at most five CONSECUTIVE logical trading days; June 11 and the sealed range are excluded; the seed chain is the canonical STORE-DAY chain from 2026-01-01 through the day before the window (a separately authorized preparation action, never part of the ≤5-day verification evidence footprint).
