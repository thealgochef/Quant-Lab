# IFVG FSM Auditability — Open Decisions Register

Status: all entries are **characterized, unresolved**. This task changes NO
behavior; each entry gets characterization evidence (tests/reports) only. Any
resolution is a future, separately-ratified change.

## D-1 — Same-TF direction conflict is unreachable under `htf_selection_max_per_timeframe = 1`

The conflict rule (reducer.py:1020-1027) computes `top_dirs` over the retained
view of the TOP timeframe; retention keeps the newest
`htf_selection_max_per_timeframe` gaps per TF (reducer.py:986-998). With the
profile's cap = 1, at most one gap per TF is in view, so `len(top_dirs) > 1` is
impossible: `conflicted` can never be true and `taps_conflicted` can never
increment. The doc's conflict-suppression intent is therefore dead code at this
cap. Characterization: reducer fixture with same-TF bull+bear live gaps, cap=1 →
no conflict; `taps_conflicted == 0`; both taps emitted with
`retention_not_selected`/winner semantics. UNRESOLVED: whether conflict should
compare pre-retention gaps or the cap should change (that would be a rule change
— out of scope).

## D-2 — `direction_disabled` on the winner does NOT fall back to the runner-up

`_scan_taps` picks a single global winner (top TF, newest) and only then checks
direction enablement (reducer.py:1027-1042): a bearish 4H winner with shorts
disabled yields `direction_disabled` on the winner and `outranked` on a bullish
1H runner-up — no setup is born at all. Characterization: fixture with bearish
4H winner + bullish 1H runner-up, shorts off → exact drop sequence
(`direction_disabled`, `outranked`), `setups_born` unchanged. UNRESOLVED:
ranked fallback is explicitly out of scope (§16).

## D-3 — `htf_max_age` conflict (doc vs registry memory bound)

The doc's HTF age limit is implemented only as
`FvgRegistry.max_age_days` eviction (fvg.py:404-411, configured
replay.py:388-390 via `section.htf_registry_max_age_days`), a MEMORY bound at
execution-bar granularity — not a selection rule at tap time. `htf_age_seconds`
is emitted as a measurement (reducer.py:1061-1065) but never gates selection.
Age-evicted gaps disappear without any persisted event today (closed by
`fvg_fill_event` kind `evicted_age`). UNRESOLVED: whether the doc intends a
tap-time age gate distinct from registry eviction.

## D-4 — Q-23 doc divergence + BE-off

- Q-23: the strategy doc's wording for the reaction-window semantics diverges
  from the implementation's per-TF clock test
  (`all(clock > window for tf …)` reducer.py:866-871 — i.e., the window is open
  while ANY parent TF is within its own-TF bar budget; the doc reads as a single
  window). The parentless-interval predicate in this task deliberately encodes
  the IMPLEMENTED semantics (`any(clock <= window)`), matching
  `parentless_window_live` (reducer.py:899-911).
- BE-off: break-even management is absent from the reducer (`_walk_trade`
  reducer.py:1642-1678 knows only stop/target); the doc mentions BE handling.
  Persisted trades carry mfe/mae so BE counterfactuals remain derivable
  downstream. UNRESOLVED as a rule decision; no execution change here.

## D-5 — `profile_name` participates in the profile hash + one-per-TF vs single-global wording

- `ifvg_profile_hash` hashes the full section including `profile_name`
  (section.py), so renaming a profile changes identity with zero behavioral
  change. Acknowledged; NOT changed (identity discipline preferred over
  convenience).
- Wording: `htf_selection_max_per_timeframe` retains one gap PER TIMEFRAME in
  view (reducer.py:986-998), but the reducer then selects a SINGLE GLOBAL winner
  (top TF first — reducer.py:1017-1027). The doc's "one per timeframe" wording
  vs the single-global setup slot is a documentation divergence, characterized by
  the emitted `rank`/`outranked` evidence. UNRESOLVED.

## D-6 — Unbounded S1 wait with a SELECTED parent freezes the single slot (found 2026-08-11 via the audit artifact)

The 40-parent-bar reaction window only clocks the PARENTLESS S1 state
(`_apply_expiries` requires `parent is None`); once a parent is selected, the
applicable timeout is `parent_retest_timeout_1m_bars` — which is **None** in
the document profile, as is `parent_reaction_window_1m_bars_max`. A setup
whose selected parent is never wick-retested (lock), never filled, and never
structurally closed therefore waits FOREVER. Observed in the ACCEPTED
baseline `143b510f…` (and identically in the tf-variant): setup
`5230dde3-bf17…` activated 2026-04-13 18:28Z, selected a 10m parent within
22 minutes, then sat in S1 for **two months** until dataset exhaustion on
2026-06-10 — suppressing 9,193 HTF taps (100% `slot_occupied` in May–June)
and every possible trade after 2026-04-09. Invisible before this artifact;
surfaced by the tap drop-reason and parent-window evidence. UNRESOLVED: any
timeout on the selected-parent retest wait is a RULE CHANGE requiring
ratification (candidate knobs: `parent_retest_timeout_1m_bars`,
`parent_reaction_window_1m_bars_max`, or clocking the parent window while a
parent is held).

## Register notes (not decisions)

- **Accepted-v2 pin discrepancy** — `config.py:72` `ACCEPTED_V2_DATASET_ID =
  49902280…` (v3-baseline lineage) ≠ the final-review accepted v2 `143b510f…`
  (manifest payload `b089dfad…`, `IFVG_LAB_FINAL_VERIFICATION.md:18`). The audit
  lane pins `143b510f…`/`b089dfad…` as NEW explicit constants (verified against
  the on-disk manifest before first save); `ACCEPTED_V2_DATASET_ID` is neither
  reused nor changed — the v3 lane keeps its lineage.
- **Authority location** — `IFVG_IMPLEMENTATION_REPAIR_VERIFICATION.md` is
  canonical at `Trade-Lab/docs/ifvg/` (blob
  `ed9030b7b2c5db9338abbd1e2d86fdee2b0da54b`); Trade-Lab is read-only for this
  task; QL references it by path + blob hash, never recreates it.
- **Present-but-empty audit surfaces** — conflict/suppression filters in the
  setup verifier will show zero rows for D-1-suppressed events and structural
  terminal deaths; they are surfaced honestly as "supported by contract,
  zero observed in range", never hidden.
