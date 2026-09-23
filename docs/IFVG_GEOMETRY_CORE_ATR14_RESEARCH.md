# Fixed geometry bundle using the existing Core ATR14 convention

The actual bounded research bundle is `B0_GEOMETRY_CORE_ATR14_V1`, composed of
unchanged repaired `B0_CORE` plus `IFVG_GEOMETRY_CORE_ATR14_V1`. Its implementation
is `src/alpha_lab/agents/data_infra/ifvg/features/geometry_core_atr14.py`.
This is offline candidate-outcome research, with no execution or activation effect.

## Inventory correction made before fitting

The initial inventory found the panel volatility features and planned volatility
block, but missed Core's existing `ContextFeatureConfig.atr_scale_period=14` and
`EqualLevelPoolTracker._scales`. That helper already normalizes registered
`ctx_sweep_sweep_depth_normalized`, `ctx_sweep_reclaim_close_distance_normalized`
and `ctx_sweep_distance_at_lock_normalized`. Its configured period enters the
Core context feature schema hash through `observational_scale_periods`.

It is an appropriate existing tick-based volatility convention available at
entry. The user's preference selects it ahead of the fallback ATR20. This
correction preceded every model fit and used no prediction results. Preserved
ATR20 prefit artifacts and their successful formula checks describe a superseded
preparation protocol; they are not results from the actual research comparison.
The historical ATR20 module, formula identity, bundle and readers remain available
for artifact compatibility. The registered research request uses the actual
Core ATR14 bundle. Neither convention was selected through an ATR-length search.

## Fixed added measurements

| Model feature | Exact formula |
|---|---|
| `geo_inversion_clearance_atr14` | `close_through_margin_ticks / decision_atr14_ticks` |
| `geo_parent_htf_distance_atr14` | `distance_to_htf_ticks / decision_atr14_ticks` |
| `geo_opposing_parent_size_ratio` | `geometry_opposing_size_ticks / geometry_parent_size_ticks` |

The first numerator comes from the exact selected `InversionRecord`: LONG
inversion close minus selected opposing high; SHORT selected opposing low minus
inversion close. The second comes from the last selected `ParentCandidateRecord`
before lock and measures closed-interval distance to the tapped HTF gap:
`max(0, htf_low-parent_high, parent_low-htf_high)`. Interval overlap is a legitimate
zero. All three ratio transformations are new in this task; their raw geometry
inputs existed in B0 before this task. The third keeps the same ratio name and
implementation first introduced in the superseded ATR20 preparation, so it is
not implemented or added twice in the final bundle. Both raw widths equal their
exact selected gap high minus low. These are the
opposing gap surviving until inversion and the parent surviving until lock.

All raw inputs and volatility are ticks; all model additions are dimensionless.
The two ATR ratios use the same entry-decision scale, while their raw numerators
retain their original selected-event clocks. B0 remains 28 raw features. No
denominator or missing-reason audit column becomes another model input. Volatility
introduces information absent from B0; this tests the added bundle and does not
isolate normalization alone.

## Core scale authority and source clock

The numerical implementation calls the existing public Core
`EqualLevelPoolTracker.on_source_bar` and `atr_at(60)` methods directly. It feeds
the selected child's verified `forward_bars.parquet` through the existing
`frame_to_bars` serialization adapter and verifies each resulting bar's exact
availability timestamp. It runs no strategy replay.

Core accepts only TIME bars with `is_complete=True`. The `is_partial` flag is
preserved in evidence but is not a second scale predicate: Core's exact helper
checks `is_complete`, and the new materializer does likewise. Real incomplete
entry bars retain typed-null ratios rather than substituting an older entry bar.

For completed bars in source order, true range is
`max(high-low, abs(high-previous_completed_close), abs(low-previous_completed_close))`.
ATR is the arithmetic mean of the trailing 14 true ranges, not Wilder smoothing.
The first source-chain bar uses its own close as the predecessor, making its TR
equal to high minus low. At least 14 completed source-chain bars are needed.
Subsequent windows include the predecessor immediately before their first bar.

The source chain carries through saved trading days and named sessions. There
is no midnight, 18:00 ET or Asia/London/NY reset. Original sparse minutes and
scheduled breaks receive no fabricated bars; the jump to the next completed
bar enters true range. Original warmup bars initialize the scale without becoming
labeled research observations. The existing context observer snapshots and
restores 21 retained source bars (`max(14,20)+1`) between days, preserving the
same rolling scale. Full source-chain computation is equivalent to that bounded
state because only the last 14 bars and their predecessor affect ATR.

Core updates the one-minute scale in `advance_step` before the reducer runs;
candidate context capture follows the reducer on that same completed step.
Thus the completed entry bar is already in `atr_at(60)` when the candidate is
emitted. The exact candidate entry bar ID, saved day, availability and OHLC must
match the source; no nearest-event or timestamp-only substitution is permitted.

Missing values use `insufficient_source_chain_completed_bars` for the first
13 completed source-chain bars, `decision_bar_incomplete` for an incomplete
entry bar and `zero_atr_denominator` for a zero scale. A zero parent width yields
`zero_parent_size_denominator` for the third ratio. No epsilon or zero-filled
approximation is introduced. Candidate rows remain paired between all arms;
model preprocessing is fitted only on the training partition. Retest B0
entry-FVG structural nulls remain unchanged.

## Provenance, persistence and verification

The shared geometry artifact verifies the original B0 evidence hash, exact
selected-event links, accepted child geometry, entry OHLC, verified context
subject/core/v2 identity, exact v2/v3 artifact pair, context manifest and source
bar file. Matching dates or identical bars cannot substitute another child's
context. It additionally checks the saved context configuration actually uses
ATR14 and complete TIME bars.

The new artifact identity binds formula version
`ifvg_geometry_core_arithmetic_atr14_1m_v1`, all three ordered feature names,
formula contract, immutable source and output hashes. Audit fields retain
`decision_atr14_ticks`, the completed source-chain count, exact entry/ATR bar
ID and time, selected FVG IDs, raw measurements and missing reasons.
`geometry_feature_artifacts` and `bundle_feature_views` remain the immutable
stores; their readers recognize both historical ATR20 and actual Core ATR14.
The bundle refuses mismatched geometry protocols even though the size-ratio
column is intentionally reused across versions.

Focused tests in `tests/agents/data_infra/ifvg/test_geometry_core_atr14.py` compare
the actual Core helper against an independent trailing-14 calculation across
days and gaps, check startup and zero scales, future-bar invariance, exact Core
complete-bar acceptance, source-adapter clock parity and old/new artifact reload.
Existing source-event, B0, registry, bundle and identity tests remain applicable.
Real coverage and paired model evidence are recorded separately under each
child's `geometry_core_atr14_v1` batch report directory.
