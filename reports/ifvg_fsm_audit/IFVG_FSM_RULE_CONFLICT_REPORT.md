# IFVG FSM Rule-Conflict Characterization (D-1..D-5)

All entries are characterized, UNRESOLVED, and registered in
`docs/ifvg/IFVG_FSM_AUDITABILITY_OPEN_DECISIONS.md`. No behavior change.

| Decision | Characterization | Evidence in this artifact |
|---|---|---|
| D-1 same-TF conflict unreachable at cap=1 | `taps_conflicted` cannot fire (retention keeps one gap per TF) | conflicted taps observed: **0** (provably zero; SC test `test_d1_same_tf_conflict_unreachable_at_cap_1`) |
| D-2 direction-disabled winner, no fallback | bearish winner + shorts-off drops `direction_disabled`; runner-up drops `outranked`; no setup born | direction_disabled taps observed: **7955**; SC test `test_d2_direction_disabled_winner_no_fallback` |
| D-3 htf_max_age is a memory bound, not a selection gate | age evictions are registry events, never tap-time gates | `evicted_age` fill events persisted; `htf_age_seconds` emitted as measurement only |
| D-4 Q-23 window wording + BE-off | parentless predicate encodes the IMPLEMENTED any-window-open semantics; no break-even management exists | `parentless_step` rows reconcile 1:1 with `parentless_window_live`; resolver walks stop/target only |
| D-5 profile-name-in-hash + one-per-TF wording | renames move identity; retention is per-TF but the slot is single-global | emitted `rank`/`outranked` evidence; identity discipline retained |

Structural parent invalidations observed: **0** (the branch is contract-supported and instrumented even at zero observations).
