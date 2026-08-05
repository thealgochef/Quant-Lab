# IFVG Lab Restoration Open Decisions

Status: defaults locked for the runnable profile

No owner decision blocks `ifvg_v2_doc_default_fresh_static_1r`.

| Topic | Current decision | Consequence |
|---|---|---|
| development cutoff | 2026-06-10 17:00 America/New_York (`2026-06-10T21:00:00Z`) from Strategy-Core `RTH_END` | label closes remain strictly before cutoff |
| 240m/Q-40 | `experimental_q40_open` | excluded from primary tiers; explicit experimental variant only |
| pure retest | blocked: `owner_trigger_not_selected` | no artifact/run until a trigger is ratified |
| ICT-clean | blocked: `same_leg_locality_sweep_semantics_unresolved` | no artifact/run until semantics are ratified |
| canonical short | blocked: `canonical_short_profile_not_ratified` | synthetic short tests validate normalization only |
| broad EQH/EQL tolerance | deferred | formula v2 retains one-tick-span policy |
| BE-managed labels | deferred | separate future label family |
| formula-v1 v3 | readable, `superseded` for M2 | never rewritten; cannot enter M2/M3 |

Any later owner ratification creates a new profile and, where formula/config behavior changes, a
new hashed identity and immutable artifacts. It cannot mutate this profile, formula, run, or
artifact in place.

## Non-decisions

- No adaptive model, threshold, feature, profile, or date search is authorized.
- Development outcomes cannot be called sealed, held-out external validation, or promotion
  evidence.
- Trade-Lab dependency changes and runtime work are not part of this implementation.
- June 11 and June 12 onward cannot be discovered, opened, read, or exposed by preparation or UI
  code.

## Operational gates (not owner decisions)

The runnable profile has no unresolved semantic owner decision, but it is not operationally
ready. January verification passes validity, exact accepted-v2 reconciliation, identity,
capacity, observer p99, and callback p99. It fails the aggregate performance limits:

- median preloaded replay slowdown: 199.0982% versus 20%;
- repeated-run p95 slowdown: 203.7889% versus 25%.

This is a measured engineering blocker, not grounds to relax the contract. The immutable save,
full permitted preparation, experiment run, Strategy-Core pin update, and catalog activation
remain prohibited until a later implementation passes the same fixed protocol. Interactive
viewport/keyboard/visual QA is also pending because the in-app browser was unavailable; AppTest
and local server startup do not close that gate.
