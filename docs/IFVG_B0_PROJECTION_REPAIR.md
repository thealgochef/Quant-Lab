# B0 selected-stage projection repair

Implemented and verified 2026-09-08/09. Projection contract:
`ifvg_b0_selected_stage_projection_v2`.

The completed acceptance run is preserved as a partial-feature study. Its ten
empty advertised fields were a projection defect, not proof that the market
evidence was unavailable. The repair reads the original selected Core stage
records from the saved FSM audit companion. No Strategy-Core runtime change or
approximate reconstruction was required.

## Exact sources and preservation

All store-relative IDs below belong to `data/ifvg_datasets/search/v1`.

| Artifact | Exact preserved identity |
|---|---|
| Original completed pipeline | `b19964202ad75b36b2eff1542c1ac6443e702e6b7ac20bb7d8eb7bbb0871dd50` under `data/ifvg_pipeline_jobs` |
| Selected Core child | `fca2cc0d18a99c8382a777a1a244ba2120a5b6228526ee84b92b82728da534f3` |
| Accepted v2 dataset | `f28df982a3d28042365fa18115951c30f764728567ff3134d86d088697afc297` |
| Selected child's FSM audit companion | `d7667632dc795240139d22081bd13642163ea269efbd68a5f43623c742802960` |
| FSM audit contract fingerprint | `c2c01532436fa881cd34a6051aefa9c0f7ac9df92bb909f2a154bffca62edfc9` |
| Original context companion | `caf3d404f7a521b8eef5da2a9e544af7402f3f40bf137b18e102fcbe46ad45ef` |
| Original partial B0 bundle view | `18fbd4ae214a6eb46a5de64821f45dea6a6e363553091111e6c7fe4032312d45` |
| Original completed model run/input | `094337ab156815f129b94b04dc9ba3dd430ad5631f12d30aba06a0d7967ae867` |

The original limitation and its model interpretation remain in
`reports/r5_r6_acceptance/20260908/feature_coverage/B0_FEATURE_COVERAGE_LIMITATION.md`.
The repair never updates any of these artifacts in place. The FSM companion is
an original, neutrality-gated partition of the same Core emission trace; its
`trace_ordinal` values are shared with accepted candidate records. The loader
checks its envelope, file hashes, supported table contract and successful
funnel reconciliation, then binds the Core child, accepted v2 ID/manifest and
each stage table's SHA-256 in projection evidence. The normal research preflight
also verifies the child's saved neutrality and current Core compatibility proof.

## Field-by-field authority and availability

Every repaired field is copied directly from the matched authoritative record.
The formulas below independently verify the copied value; they do not replace
missing records. Intervals and prices are integer ticks. Core's
`structures/fvg.py:277` defines interval distance as zero for overlapping closed
intervals, otherwise their nearest-edge separation: for `[a,b]`, `[c,d]`,
`max(0, c-b, a-d)`.

| Repaired field | Authoritative original Core output and parity formula | Exact event/clock and decision-time availability |
|---|---|---|
| `parent_tf_seconds` | `ParentCandidateRecord.parent_tf_seconds` (`records.py:226`); equals selected `geometry_parent_timeframe_seconds`. | Last **selected** parent emission before the exact parent-lock record, with matching setup and every persisted parent FVG geometry field. Available at that parent's selection emission, before lock and entry. |
| `distance_to_htf_ticks` | Same selected `ParentCandidateRecord`; `interval_distance_ticks(parent_low, parent_high, htf_low, htf_high)` from Core `reducer.py:2501`. | Measured when the final selected parent arrived, against this setup's tapped HTF zone. HTF confirmation must be at/before tap; parent confirmation must be at/before selection. |
| `elapsed_parent_bars_since_tap` | Same selected `ParentCandidateRecord`; exact `s.parent_clocks[parent_tf]`. Verified against the corresponding audit `parent_window_event`'s `parent_clocks` and representative original parent bars. | Core `reducer.py:1241` adds `tf_bar_close_counts[tf]` only while the setup is in S1. Setup opening initializes clocks to zero. Replacement does **not** reset these tap-origin clocks. The value is sampled at parent selection, not entry. “Closed” means a bar delivered by Core's queue; see the partial-bar detail below. |
| `elapsed_1m_bars_since_tap` | Same selected `ParentCandidateRecord`; `selection_step_ordinal - tap_step_ordinal` (`reducer.py:2548`). | Verified against both selected/opened audit step ordinals and ranks in the exact replay's 60-second decision-bar sequence. It stops at final parent selection. Empty minutes and session breaks add no reducer steps. |
| `elapsed_1m_bars_since_selection` | Exact `ParentLockRecord` (`records.py:242`); `lock_step_ordinal - final_parent_selected_step_ordinal` (`reducer.py:1842`). | Join by setup, selected parent FVG ID and exact `geometry_lock_bar_cursor`. A parent replacement resets the selected ordinal. Core requires lock on a later step than selection. Available at lock. |
| `distance_to_parent_ticks` | Final selected `OpposingGapRecord` (`records.py:252`); `interval_distance_ticks(opposing_low, opposing_high, parent_low, parent_high)` (`reducer.py:2596`). | Last selected opposing emission after exact lock and before exact inversion, with all opposing FVG geometry fields equal to the candidate's selected opposing gap. Available at this opposing selection. |
| `elapsed_1m_bars_since_lock` | Same selected `OpposingGapRecord`; `selected_opposing_step_ordinal - lock_step_ordinal` (`reducer.py:2681`). | Sampled at the opposing selection that survives until inversion. It is not elapsed time at candidate entry and is not the first opposing selection if replacement occurred. |
| `close_through_margin_ticks` | Exact `InversionRecord` (`records.py:265`) and Core `close_through_margin_ticks` (`structures/fvg.py:252`). LONG: `inversion_close - opposing_high`; SHORT: `opposing_low - inversion_close`. | Joins exact inversion cursor and opposing FVG ID. The opposing direction must be bearish for LONG, bullish for SHORT. Inversion OHLC is checked against its original completed decision-bar evidence. Available at inversion; entry is later. |
| `bars_armed_to_inversion` | Same `InversionRecord`; `inversion_step_ordinal - most_recent_armed_step_ordinal` (`reducer.py:1901`). | Every selected opposing replacement resets `armed_ordinal` (`reducer.py:2635`). Therefore the anchor is the selected opposing event, **not** the first lifecycle `opposing_armed` transition. Verified against exact decision-bar ordinals; available at inversion. |
| `opposing_size_ticks` | Same `InversionRecord.opposing_size_ticks`; equals `geometry_opposing_size_ticks` and selected opposing interval width. | Available from selected opposing formation, repeated at inversion. This duplicates the already-advertised geometry size; it is not a new independent measurement. |

Core file paths are under
`../Strategy-Core/src/strategy_core/strategies/ifvg_smc/`, except the shared
geometry helpers at `../Strategy-Core/src/strategy_core/structures/fvg.py`.
The selected-row join and checks live in
`src/alpha_lab/agents/data_infra/ifvg/b0_projection.py`.

### Clock details proved by real records

The reducer increments one ordinal for each call to `_step_inner`
(`reducer.py:822`), rather than subtracting timestamps. Candidate
`93f9128c-cdcb-5bca-9a9e-5febc557204c` has **233** tap-to-parent steps over
**293** wall-clock minutes. Candidate
`5e898e6d-e7ac-59f6-9397-b36bde9fc2e9` has **103** steps over **104** minutes.
Both would be wrong under a wall-time approximation.

Core's `replay.py:474-484` drains parent bars whose availability has arrived and
increments `tf_bar_close_counts` for every delivered closed bar. It does not
filter that clock input by `is_complete`. In the first example the selected
1800-second parent clock is **8**, including the delivered terminal partial bar
`1800s:2026-02-10:45` at `2026-02-10T22:00:00Z` (`is_complete=False`). Filtering
that flag would give **7**, incorrectly altering the original Core clock. This
is a contract detail retained by the repair, not a policy change or an inferred
extra bar. The five-record regression fixture retains all 24 relevant original
parent-bar deliveries and independently verifies these parent-clock counts.

### Availability and family-specific absence

Each selected source must precede the candidate in the original trace and be
available no later than candidate entry. Geometry FVG IDs, bounds, timeframe,
direction, A/C bar IDs and confirmation timestamps must agree exactly. Parent
and opposing confirmations cannot be later than their selection emission. HTF
confirmation cannot be later than tap. Continuation FVG confirmation cannot be
later than entry. Tap, lock and inversion bar IDs/timestamps/OHLC must match the
verified decision-bar source. These are additional checks on verified artifacts,
not timestamp-only joins.

`entry_fvg_size_ticks`, `entry_fvg_gap_low_ticks`, and
`entry_fvg_gap_high_ticks` remain null for `ifvg_retest`, which has no entry FVG.
They must be present for `fresh_fvg_continuation`; fabricated retest values fail.
Across the full accepted Core stream the repair validates **204/204** candidates:
152 continuations and 52 retests. The unchanged research scope contains **143**:
111 continuations and 32 retests. All ten repaired fields are populated on every
candidate in both populations.

## Fail-closed validation and versioned persistence

Fresh `build_candidate_feature_view` calls require the exact selected-stage
source and verified decision bars. Missing sources raise a specific unmapped-B0
error. The explicit `allow_legacy_partial_projection=True` compatibility option
labels the result `ifvg_b0_partial_entry_projection_v1` /
`legacy_partial_unverified`; new real research's S06 coverage gate rejects that
mode. Historical readers remain compatible and historical test fixtures opt
into that legacy mode explicitly.

`validate_b0_feature_mapping` distinguishes:

* An absent advertised column or a null without the specified structural reason:
  error, even if an imputer could fit after dropping it.
* An entry-FVG field absent on a retest: mapped, with a structural-null count.
* A mapped feature wholly empty in a particular training fold: a fold-local
  preprocessing fact, separately persisted by the model schema protocol. A
  retest-only fold can legitimately have three empty entry-FVG features.

The ordered raw B0 names remain 28. The baseline feature block's resolution is
version 2 and binds the selected-stage audit schema and projection version.
Feature registry/view identities bind the new projection contract; the view
identity additionally binds source hashes and per-candidate projection evidence.
Persisted candidate-view manifests retain and verify this evidence. Evidence
contains the exact selected stage trace ordinals, event availability timestamps,
parent clock event ID, decision-bar ordinals, and all ten projected values.
S06 compares these values against both the current candidate view and persisted
B0 bundle before the real ladder can run. Model-side schema/reload proofs are
reported separately by the controlled comparison.

The original audit source and accepted v2 tables remain valid and are reused.
The context producer identity includes the versioned projection implementation,
so preparation generates a new context companion and affected downstream
artifacts while retaining historical companions. The controlled run reconciles
the candidate population, unchanged fields, labels, fold definitions and fold
assignments against the original before fitting.

## Verification and limits

Reusable regression evidence is stored at
`tests/agents/data_infra/ifvg/fixtures/b0_real_selected_stages_v2.json` with original
Core/context/table hashes and exact candidate IDs. Fixture SHA-256:
`b5ed9725faea56f7e46d0012198e7b293d02ebc1c596c891002a0820566c505a`.
It exercises both families, parent/opposing replacement, multiple parent
timeframes, a session break, a sparse minute and a delivered terminal partial
parent bar. The tests mutate stage ordinals, parent clocks, source availability,
underlying OHLC, source identity, mapping columns and structural nulls to prove
the relevant guards fail.

The full 204-candidate source audit and its manifest are saved under
`reports/b0_projection_repair/20260909/projection_source_audit/`. Its projection
evidence hash is
`ade034743a1e922a68f2a9bb5ca9b27e5bd8dc25f473590588046a7c756d9fea`.
Commands and actual outcomes are recorded in
`reports/b0_projection_repair/20260909/projection_checks.md`.

There is no remaining reconstruction blocker for any of the ten fields in this
selected study. A different source lacking the required selected-stage audit
cannot pass by replacing clocks with elapsed minutes or substituting geometry
for missing event identity. The five-record fixture and full-source audit prove
this source projection's software correctness; they do not establish adequate
sample size, vendor source completeness, predictive improvement, deployability
or generalization beyond the inspected evaluation period. The controlled
comparison remains exploratory evidence on that already-inspected period.
