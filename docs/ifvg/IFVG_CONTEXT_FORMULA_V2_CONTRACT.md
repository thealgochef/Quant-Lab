# IFVG Context Formula v2 Contract

Status: locked implementation contract  
Formula identity: `ifvg_context_formula_v2`

## Identities

| Contract | Value |
|---|---|
| feature set | `ifvg_context_v1` |
| formula | `ifvg_context_formula_v2` |
| context record schema | `2` |
| observer seed schema | `2` |
| calendar policy | `cme_equity_index_futures_eth_v1` |
| source-gap policy | `exchange_calendar_invalidate_no_impute_v2` |
| equal-level policy | `instrument_tick_grid_one_tick_span_v1` |
| 240m status | `experimental_q40_open` |

The feature-set name stays v1 because its registered measurement vocabulary is unchanged.
Formula-v1 artifacts remain readable. Formula-v1 seeds are rejected under formula v2.
The outer Strategy-Core seed wrapper retains its existing schema when its public shape is
unchanged.

## Exchange-minute schedule

`ExchangeMinuteSchedule` is the shared authority for context TIME-bar construction and
displacement eligibility. It contains normalized sessions, scheduled maintenance, weekends,
holidays, special closures, inclusive coverage bounds, policy ID, and a canonical content hash.
Runtime behavior uses the committed snapshot and cannot depend on a mutable calendar library.

Each minute has a typed `MinuteSlotStatus`. Source availability is separately described by
`ContextSourceCoverage`; a missing row alone can never imply either a scheduled closure or an
available source partition.

For a displacement window `(B0, end]`:

1. The first included close is the first eligible completed 1m close strictly after `B0`.
2. Evidence is read only through the cutoff instant. Candidate label scans retain strict
   `close_ts < cutoff`.
3. Maintenance, weekend/holiday closure, and partition transitions do not invalidate a window.
4. A missing expected close in an audited available partition is `source_bar_missing`.
5. A required permitted but unavailable partition is `source_partition_unavailable`.
6. Too few eligible closes is `insufficient_observations`.
7. Unexpected ineligible bars or contradictory coverage declarations are invariant failures.
8. Genuine gaps use `invalidate_no_impute`; values are never imputed.

Transition ordering, reset/replacement behavior, and no-post-transition consumption do not
change.

## Direction and equal levels

Prices remain integer ticks. Equal-level pools use the existing narrow policy: members may span
at most one tick. Formula v2 names this honestly as
`instrument_tick_grid_one_tick_span_v1`; it is not ATR or discretionary proximity matching.
Member-separation and pool-width distributions are reported.

Long setups qualify opposing-leg EQL sweeps; synthetic short setups qualify opposing-leg EQH
sweeps for normalization coverage only. Synthetic tests do not ratify a canonical short profile.
Every qualifying link is retained. A scalar reference is selected by latest sweep cursor and then
pool ID. Link evidence is immutable on the active setup or pinned until completion, so pool
eviction cannot remove evidence required by that setup.

## Records, seeds, and identity

Canonical record bytes use record schema 2. The observer seed stores stable identity once,
derives duplicate memberships during restore, and omits transient/reconstructible maps. It uses:

```text
H0 = SHA256("ifvg-context-chain-v2\0")
Hn = SHA256(Hn-1 || uint64_be(record_length) || canonical_record_bytes)
```

All integers are fixed-width where specified, timestamps are normalized to UTC, map order is
canonical, and no OpenSSL state serialization is used. Uninterrupted, incremental, batch, and
seed/resume streams must be byte-identical. Existing v2 records, labels, decisions, executions,
and seed hashes remain byte-identical.

## Capacity and performance

Oversized individual records fail explicitly; limits are not raised.

| Gate | Maximum |
|---|---:|
| observer state | 4,194,304 bytes |
| observer seed | 838,860 bytes |
| transition snapshot | 26,214 bytes |
| 1m observer step p99 | 1.6 ms |
| multi-timeframe callback p99 | 8 ms |
| replay median slowdown | 20% |
| repeated-run p95 ceiling | 25% |

Performance protocol is two warmups followed by ten paired alternating baseline/context runs on
the same preloaded source. Reports include machine, Python/dependency versions, sample sizes,
median/p95/p99/CV, and separated source-load, v2, context, normalization, and persistence time.

## Required proof

- Maintenance, reopen, Friday/Sunday, holidays, rollover, both DST directions, and multi-day
  windows.
- Exact `(B0,end]`, strict cutoff, genuine gap, unavailable partition, closure, boundary, and
  insufficient-observation behavior.
- Long and synthetic-short displacement and opposing-leg paths, ties, intervals, no-match,
  all-link retention, scalar selection, and saturated eviction.
- Batch/incremental/uninterrupted/seed-resume/repeat identity.
- Legacy v2 serialization, ordered emission, labels, decisions, executions, and seed hashes.

## Implemented symbols and measured status

The contract is implemented in Strategy-Core by
`candles.exchange_calendar.ExchangeMinuteSchedule`, `MinuteSlotStatus`,
`ContextSourceCoverage`, `structures.displacement.DisplacementAccumulator`,
`structures.equal_levels.EqualLevelPoolTracker`, and
`strategies.ifvg_smc.context_features.IfvgContextObserver`. Calendar partition prefix counts
make long displacement finalization proportional to partition count while retaining exact minute
semantics across weekends, holidays, and both DST directions.

Schema-2 observer seeds now store identity once and reconstruct pool membership, fired-swing
state, active indexes, and other transient maps. Only unreclaimed equal-level links are kept by
the pool tracker; active-setup and qualified links remain independently pinned. Event transport
inherits matching point-in-time provenance from its event envelope. January measurements are:

| Measurement | Observed | Gate | Status |
|---|---:|---:|---|
| state bytes | 809,170 | 4,194,304 | pass |
| seed bytes | 809,170 | 838,860 | pass |
| transition bytes | 23,211 | 26,214 | pass |
| observer p99 | 1.3525 ms | 1.6 ms | pass |
| callback p99 | 0.3990 ms | 8 ms | pass |
| replay median slowdown | 199.0982% | 20% | fail |
| repeated-run p95 slowdown | 203.7889% | 25% | fail |

The repeated report used measurement policy
`preloaded_two_warmup_ten_alternating_pairs_v2`, ten measured pairs, Windows 11, and Python
3.13.1. Capacity, validity, identity, accepted-v2 reconciliation, and point-latency gates pass.
Aggregate replay performance blocks artifact publication. No formula-v2 artifact ID is assigned
until that gate passes and immutable save succeeds.
