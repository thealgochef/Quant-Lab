# Fixed non-MBP geometry bundle

**Historical prefit protocol, superseded before any model fitting.** The initial
inventory missed Strategy-Core's already registered arithmetic ATR14 scale used
by sweep normalization. The user's preference therefore selects
[`B0_GEOMETRY_CORE_ATR14_V1`](IFVG_GEOMETRY_CORE_ATR14_RESEARCH.md) for the actual
batch. ATR20 prefit evidence and readers remain preserved; their successful
formula checks do not justify choosing ATR20 over that existing convention.

Registered 2026-09-09 as `IFVG_GEOMETRY_ATR20_V1`, composed with repaired
`B0_CORE` by `B0_GEOMETRY_ATR20_V1`. This is an offline research feature bundle.
It grants no activation, execution gate, or Strategy-Core serving contract.
The original 28 B0 names, their selected-event definitions and historical bundle
artifacts remain unchanged. The implementation is
`src/alpha_lab/agents/data_infra/ifvg/features/geometry_features.py`.

## Inventory and fixed measurements

None of the three ratios existed among the registered candidate features.
Their geometry numerators and parent-width denominator already existed in B0.
`opposing_size_ticks` and `geometry_opposing_size_ticks` are duplicate raw
measurements in historical B0; the addition uses the latter and adds no second
copy. Existing structure/displacement normalized features measure different
events and are not equivalents.

The available `cbp_realized_volatility_12` is a context-panel measurement: the
population standard deviation of 12 close changes over 13 completed 5m or 15m
bars, with its own panel assignment and regime protocol. It is not a registered
decision-time one-minute ATR convention. `IFVG_VOLATILITY_CONTEXT_V1` remains
planned. However, the original inventory overlooked
`ContextFeatureConfig.atr_scale_period=14` and
`EqualLevelPoolTracker.on_source_bar/atr_at(60)`, which already supply a suitable
registered convention to sweep normalizations. This omission was corrected
before fitting. The ATR20 implementation below remains historical preparation.

| New feature | Exact formula and source |
|---|---|
| `geo_inversion_clearance_atr20` | Repaired `close_through_margin_ticks / decision_atr20_ticks`. The exact selected `InversionRecord` supplies the numerator: LONG inversion close minus selected opposing high; SHORT selected opposing low minus inversion close. This measures close-through clearance at inversion, divided by volatility available at entry decision. |
| `geo_parent_htf_distance_atr20` | Repaired `distance_to_htf_ticks / decision_atr20_ticks`. The final selected `ParentCandidateRecord` supplies the parent-to-tapped-HTF closed-interval nearest-edge separation at selection: `max(0, htf_low-parent_high, parent_low-htf_high)`. Overlap is exactly zero. |
| `geo_opposing_parent_size_ratio` | `geometry_opposing_size_ticks / geometry_parent_size_ticks`, using the selected opposing gap surviving until inversion and the exact locked parent gap. Both widths equal high minus low in ticks. |

All input prices, sizes, distances and ATR are ticks. All three outputs are
dimensionless. The first two share exactly the same decision-time volatility
denominator. That denominator introduces information absent from B0, so the
experiment tests the added bundle; it does not isolate normalization alone.

## ATR20 source and clock

The source is the selected child's verified research context companion
`forward_bars.parquet`, itself bound to its original replay inputs. Only original
60-second bars with `is_complete=True` and `is_partial=False` contribute.
Bars are ordered by saved trading day, availability timestamp and bar ID; duplicate
IDs or availability timestamps are refused. Prices must be finite integer ticks
with consistent OHLC.

Within each saved trading day (18:00 ET boundary), the first completed bar has
`TR = high-low`. Later completed bars have
`TR = max(high-low, abs(high-previous_completed_close), abs(low-previous_completed_close))`.
The seed ATR is the arithmetic mean of the first 20 TR values. Thereafter,
`ATR = (19 * previous_ATR + current_TR) / 20` (Wilder smoothing).

ATR resets each trading day. Asia/London/NY and other named-session boundaries
do not reset it. Original sparse minutes and session breaks receive no fabricated
bars. A price gap to the next observed completed bar enters its true range.
Partial bars contribute neither a TR nor a previous close. This is an observed
completed-bar convention; it does not assert vendor completeness.

The candidate's exact saved entry bar ID must resolve to the same saved day and
availability timestamp as both candidate entry and feature as-of. Its original
OHLC must match. A partial entry bar retains its candidate row with null ATR
ratios; the code does not fall back to an older entry bar. ATR before 20 completed
bars is null with `insufficient_same_day_completed_bars`; a partial entry bar uses
`decision_bar_incomplete`; zero ATR uses `zero_atr_denominator`. A zero parent
width yields a null size ratio with `zero_parent_size_denominator`. No infinity,
epsilon substitute or zero-filled approximation is introduced. These reasons
are audit metadata, not extra model columns. Repaired B0 structural nulls for
retest entry-FVG features remain intact.

## Artifact and validation contract

`materialize_geometry_features(view, bars_1m, source_reference=...)` verifies:

1. The original repaired B0 projection evidence hash, selected parent/opposing
   linkage and projected numerator values.
2. The source context envelope's exact child, v2 artifact and manifest; its v2/v3
   pair must reproduce `view.artifact_pair_hash`. Identical bars from another
   child's context are insufficient.
3. Selected gap IDs, bounds, widths and entry OHLC against the accepted child
   candidate, then the supplied bar values against the verified source file.
4. Every candidate's exact entry decision bar and the shared volatility clock.

The immutable `geometry_feature_artifacts/{geometry_feature_artifact_id}` family
persists `geometry_features.arrow`, `formula_contract.json`, envelope and manifest.
The identity binds the source view, B0 projection evidence, source context,
manifest/file/bar hashes, formula contract, feature schema and resulting table
hash. Audit columns retain raw measurements, ATR, completed count, exact decision
bar ID/availability and selected parent/opposing FVG IDs.

Geometry bundle joins are one-to-one on candidate ID with exact population
equality and no shadowing of existing view columns. A geometry-bearing bundle
must include all three additions and its exact verified feature artifact. The
bundle payload binds that artifact. Absent geometry fields are excluded from
serialization so older bundle-view IDs continue to load unchanged.

Focused checks are in `tests/agents/data_infra/ifvg/test_geometry_features.py`.
They cover Wilder seeding/recurrence, future-bar invariance, partial bars, sparse
clock gaps, trading-day reset, formula values, source drift, selected-event drift,
cross-child refusal, typed missingness and immutable artifact/bundle reload.
Synthetic formula checks establish implementation behavior, not real-data
coverage or predictive value. Real per-child coverage, constants, duplicates,
fold populations and paired model evidence belong to each frozen batch report.
