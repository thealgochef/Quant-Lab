# Phase 3 — R1 fixture, seed production and authorization unblock (plan §5)

**Status: contracts authored and synthetically proven; every owner action and
every real replay NOT performed.** This phase closes F-22 (date-domain
mapping), restates F-16 (seed inventory) with what was verified, and
implements F-21 (the separate seed-production authorization) plus the
§5.4/§5.5 unsigned packets. Nothing here selects a window, registers an
allowlist, signs an authorization, or replays real data.

Modules: `search/trading_calendar.py`, `search/verification_window.py`,
`search/seed_production.py`; CLIs `scripts/ifvg_verification_window_shortlist.py`,
`scripts/ifvg_seed_production.py`; tests `tests/agents/ifvg_search/test_trading_calendar.py`
(5), `test_verification_window.py` (4), `test_seed_production.py` (9).
Evidence: `LOGICAL_WINDOW_COVERAGE_SCAN.json` (shortlist id
`1a77a00ed6911f6a158c648349df94a2c37923bf4553eb30bcb90cfdbc08bd6a`),
`VERIFICATION_WINDOW_SHORTLIST.md`, `_red_ws_D.txt`, `_ws_D_pytest.txt`.

## 1. The date-domain finding (F-22) and the corrected semantics

The R1 window scan (`../R1/window_coverage_scan.py`) scored consecutive
**store days** — the physical UTC partition dates of the accepted dataset's
`permitted_source_hashes`. That inventory holds 138 dates: every weekday
AND every Sunday of 2026-01-01 … 06-10 (no Saturday carries a file). The
Sunday file holds the Sunday 18:00 ET Globex open, which belongs to the
following **Monday's** trading day. As a trading-day id, `2026-02-08` would
name the stream `[Sat 18:00 ET, Sun 18:00 ET)` — no market at all — so the
R1 candidate `2026-02-06 … 02-11` mixes a physical partition date with
logical trading days.

Corrected semantics (`trading_calendar.py`, policy `cme_globex_18et_weekday_v1`):

- A **logical trading day** is the Strategy-Core trading-day id `td` whose
  stream is `[td−1 18:00 ET, td 18:00 ET)` (`SESSION_TIMEZONE`,
  `TRADING_DAY_BOUNDARY`); `DatabentoParquetSource.for_trading_day` composes
  it from the physical partitions `td−1` (`prev_utc_date`) and `td` (`utc_date`).
- A logical trading day is a weekday that is not a registered full-closure
  day. **Registered closures inside the window: `2026-01-01` only** — the
  accepted dataset emitted nothing on it and the FSM-audit day funnel does
  not cover it (no bars, no HTF taps). The funnel covers **all 114 other
  weekdays** of `2026-01-02 … 06-10`, so every partial-session holiday is a
  trading day: MLK (`01-19`, htf_taps 18), Presidents' Day (`02-16`),
  **Good Friday `04-03` (9 lifecycle rows, 1 candidate, 1 decision, 1 executed
  trade — a full session; the assumed closure was contradicted by the evidence
  and the policy follows the evidence)**, Memorial Day (`05-25`, htf_taps 19).
  Sundays never emit (0 Sunday emission days); lifecycle rows by weekday:
  Mon 66 / Tue 221 / Wed 200 / Thu 138 / Fri 98.
- `VerificationTradingDayRef{logical_trading_day, session_open_ts_utc,
  session_close_ts_utc, ordered_source_partition_refs[(td−1, prev_utc_date,
  kind, sha), (td, utc_date, kind, sha)]}` maps every logical day to its exact
  ordered physical partitions; the content hashes come from the accepted
  inventory (never re-hashed from disk). The contract refuses a ref whose
  partitions are not exactly `(td−1, td)` or whose session bounds are not the
  18:00 ET roll (DST-aware: `2026-02-09` opens `2026-02-08T23:00Z`,
  `2026-06-08` opens `2026-06-07T22:00Z`).
- **Seed chains follow store days.** The accepted chain replayed the physical
  store-day sequence from the cold start `2026-01-01` (weekdays + Sundays; the
  Sunday days are zero-bar reducer steps). A profile-matching seed must be
  produced over the SAME store-day sequence (`store_day_chain(first, last)` =
  every non-Saturday calendar day), never over a logical-day-only chain,
  because a logical-only replay skips the zero-bar day-roll steps and is not
  proven emission-identical. The verification ALLOWLIST is the logical-day
  sequence; the two domains are never mixed (DEV-D-1 in `_ws_D_NOTES.md`).

## 2. The rebuilt shortlist (§5.1)

Rebuilt by `scripts/ifvg_verification_window_shortlist.py` from
already-authorized evidence only: the FSM-audit accepted v2 dataset
`143b510f…2ac7` (manifest `b089dfad…0c22`, verified by
`load_accepted_v2_tables`), its `permitted_source_hashes` inventory, and the
FSM-audit artifact `7e55ee89…fe41` day funnel (manifest hash verified). No raw
source was discovered, listed or read; `data/` gained zero files;
`register_program_allowlist` was not called; the owner's selection was not
performed.

Per logical day: both partitions present and hash-addressable; lifecycle /
candidate / decision / trade / label counts by `envelope_trading_day`;
**exact verifier targets** = resolved executed trades whose setup id resolves
in the setup-lifecycle table (the setup-verifier deep-link target); audit-day
coverage = the funnel's `source_date` set (which also proves bars were
processed — the measurable control-flow proxy); MBP-1 scope evidence
`not_evaluated` (no MBP-1 evidence artifact exists for these days); panel
coverage `not_evaluated`; distance from the protected boundary = days before
`2026-06-11`.

Hard constraints (all five must hold): 1–5 consecutive logical days; every
partition present and hash-addressable; the canonical store-day seed chain
from `2026-01-01` through the day before the window is complete in the
inventory; ≥ 1 exact verifier target; June 11 and the sealed range excluded.
Ranking is lexicographic in the exact §5.1 order with no hidden score.

| Entry | Rank | Window (logical days) | Eligible | Rank key (§5.1 order) | Seed chain (store days) |
|---|---|---|---|---|---|
| top_ranked / highest_complete_audit_day_coverage | 1 | 2026-02-04 · 02-05 · 02-06 · 02-09 · 02-10 | yes | 5 / 4 / 5 / 9 / 9 / 30 / 9 / 0 / 5 / 121 | 29 (2026-01-01 … 02-03) |
| — | 2 | 2026-02-05 · 02-06 · 02-09 · 02-10 · 02-11 | yes | 5 / 4 / 5 / 9 / 9 / 30 / 9 / 0 / 5 / 120 | 30 |
| highest_lifecycle_candidate_decision_trade_coverage (= the corrected R1 form) | 3 | 2026-02-06 · 02-09 · 02-10 · 02-11 · 02-12 | yes | 5 / 4 / 5 / 9 / 9 / 30 / 9 / 0 / 5 / 119 | 31 (2026-01-01 … 02-05) |
| june_proposal | 70 | 2026-06-04 · 06-05 · 06-08 · 06-09 · 06-10 | **no** — `at_least_one_exact_verifier_target` fails (0 candidates, 0 decisions, 0 trades; lifecycle path classes 1/4) | 5 / 1 / 5 / 0 / 0 / 0 / 0 / 0 / 5 / 1 | 132 (2026-01-01 … 06-03) |

110 candidate windows, 50 eligible. The three February windows tie on every
coverage criterion (9 trades, 9 decisions, 30 candidates, 9 exact targets,
5/5 audit days) and separate only on the last criterion (distance from the
boundary); the "richest" entry breaks that tie on setup-lifecycle rows, as
documented in `verification_window.py`. The R1 store-day candidate is marked
`provisional_ineligible_as_stated` (`2026-02-08` is a Sunday partition); its
corrected logical form `2026-02-06 … 02-12` is rank 3.

**Owner attention:** under the plan's own hard constraints the June proposal
is not an eligible verification window (the doc-default strategy produced no
candidate, decision or trade in June; R1 already recorded May/June as
0/0/0/0). The owner selects one corrected logical-day window; nothing in this
release pre-empts that choice.

## 3. Seed inventory (§5.2, F-16)

No compliant profile-matching seed exists. Verified without listing any
immutable store root: neither `data/ifvg_datasets/search/v1` nor
`data/ifvg_datasets/search_test/v1` exists in this checkout (no `seed_snapshots`
store, no catalog event log), and the existing v2 / replay-chart / FSM-audit
artifacts hold typed tables and evidence, not a restorable `IfvgDaySeed` state
graph (R1's `DRAFT_VERIFICATION_AUTHORIZATION.md` §4: "No v2 per-day seed
cache exists — v2 replays always re-drive the reducer"). Derivation from those
tables is not allowed. The lawful options remain exactly §5.2's: reuse a
verified matching seed (none), derive from a persisted exact terminal state
(unavailable), run a separately authorized seed-production chain (authored
below), or remain blocked.

## 4. The separate seed-production authorization (§5.3, F-21)

`search/seed_production.py`:

- `SeedProductionReplayPolicy` (`seed_production_explicit_chain_v1`) — a
  trusted development-window access class that authorizes exactly the ordered
  store-day chain; a Saturday, a gap, a reordered day, June 11, a sealed date
  or any off-window date is refused BEFORE path construction; an off-chain
  date inside the window is denied and audited.
- `SeedProductionAuthorizationPayload` / `Envelope` / `Ref` (store
  `seed_production_authorizations`) binds: `store_namespace_id` +
  `supersession_head_witness`; baseline profile name + resolved section hash;
  `chain_day_semantics = physical_store_day_chain_v1`,
  `ordered_seed_chain_replay_days` (the store-day chain) and
  `ordered_seed_chain_logical_trading_days` (its logical subset, validated);
  `snapshot_through_day` (= last replay day) and
  `first_intended_verification_day` (= the logical day whose prior physical
  partition is `snapshot_through_day`); the access / calendar / chain policy
  ids; `expected_source_inventory_hash` (over every chain partition's kind and
  content sha); the Quant-Lab replay-source identity and the Strategy-Core
  commit + source identity; `seed_schema_version` (= the installed 2);
  `final_day_exhausts_dataset = False`; the permitted outputs
  (`seed_snapshot`, `access_audit`, `run_receipt`) and the prohibited outputs
  (research/search metrics, candidate/trade performance reports, model
  fitting, feature studies, prop simulation, frontier/insights,
  research-catalog publication); owner decision refs, approver, approval and
  effective instants. An owner-signed authorization must start the chain at
  the canonical cold start `2026-01-01`; placeholders cannot persist.
- `verify_seed_production_authorization` — verified store load (exact id;
  never a listing), namespace equality, current head witness (a moved head
  refuses), synthetic-provenance confinement to `test` namespaces, profile /
  section / chain / inventory / code-identity equality, seed schema,
  effectivity — every check BEFORE any source path (proven with a
  monkeypatched capture that would fail the test if reached).
- `run_seed_production_chain` — verify → policy → `build_ifvg_v2_capture(...,
  final_day_exhausts_dataset=False)` (cached artifacts only by default; the
  provenance read adapter is honoured) → zero forbidden access → the produced
  seed's timestamps canonicalized to stdlib UTC (hash-preserving; see
  DEV-D-3) → `save_seed_snapshot` (profile-bound, content-addressed,
  schema-validated) → the run receipt (`seed_production_runs`, with the
  `access_audit.json` sidecar whose sha the receipt binds) → reload-verify.
  The receipt states `chain_replay_day_count`, `logical_trading_day_count`,
  `verification_evidence_footprint_days = 0`,
  `separately_authorized_preparation = True`, `output_namespace =
  search_test/v1`, `stores_written = (seed_snapshots, seed_production_runs)`.

**Synthetic proof** (`test_seed_production_chain_produces_only_the_permitted_outputs`):
the conftest chain (`2026-01-13`, `01-14` written under the seed-production
policy; `01-15` under the verification policy) in a tmp `test` namespace; a
synthetic-provenance authorization; the run produces a seed hash-equal to the
continuous chain's end-of-day-2 state; the snapshot binds
`snapshot_through_day = 01-14`, `first_replay_day = 01-15`, the entering
`DaySeeds` of day 2, the chain policy and 2 chain days; the access audit shows
zero denied dates and zero protected counters; the store root holds ONLY
`STORE_NAMESPACE.json`, `owner_decisions/` (the genesis head),
`seed_snapshots/`, `seed_production_authorizations/`, `seed_production_runs/`
— no research, pipeline, model, feature, prop, frontier or catalog store gained
an entry; a restart from the reloaded snapshot over day 3 reproduces the
continuous chain's day-3 funnel and end seed exactly; a second run under the
same authorization is verified reuse. Divergent code identity, inventory or
profile refuse before any path; synthetic provenance refuses in an unmarked
root and in a `research` namespace; a foreign namespace id refuses at persist.

## 5. Unsigned packets (§5.4 B/G, §5.5)

- `build_seed_production_packet` (step B) — the canonical store-day chain
  from `2026-01-01` through the physical day before the first intended
  verification day, every binding filled from evidence, owner fields
  `<OWNER_TO_FILL>`; `SeedProductionAuthorizationPayload.model_validate`
  refuses it. For the June proposal the chain is **132 store days**
  (`2026-01-01 … 06-03`); for the top-ranked February window it is **29**.
- `build_verification_authorization_packet` (step G / §5.5) — buildable only
  after a verified seed exists (exact-loaded, profile-bound, continuous with
  the window; the run receipt, when supplied, binds the seed's provenance and
  authorization id): the corrected logical allowlist and its hash (computed
  locally for display only — nothing registered), shortlist and
  coverage-matrix ids, profile + section hash, seed id / hash / schema /
  provenance, `verification_fixed_allowlist_max5_v1`, namespace + head
  witness, the pipeline/verification semantic identity placeholder, the
  expected `ReplayInputBundle` source-inventory hash, zero protected/sealed
  requirements, the three stamps, blank owner fields.
  `VerificationAuthorizationRef.model_validate` refuses the draft.
- `scripts/ifvg_seed_production.py`: `packet` (unsigned), `verify-authorization`,
  `run` (a persisted signed authorization id is required; synthetic provenance
  refuses outside test namespaces; exit 2 on every typed refusal). Importing
  launches nothing.

## 6. The remaining owner steps (§5.4) — NONE performed here

| Step | Owner / real action | Status | What it requires |
|---|---|---|---|
| A. Select a provisional logical verification window | owner | **NOT PERFORMED** | one window from `VERIFICATION_WINDOW_SHORTLIST.md` (the June proposal is ineligible as stated) |
| B. Create the unsigned seed-production packet | `scripts/ifvg_seed_production.py packet` | authored; runnable after A (needs the marked `search_test/v1` namespace + the current QL/SC identities) | — |
| C. Owner signs `SeedProductionAuthorizationRef` | owner | **NOT PERFORMED** | the packet's owner fields; persisted via `persist_seed_production_authorization` |
| D. Run only the authorized seed-production chain | real replay (29–132 store days) | **NOT RUN** | C; cached day artifacts or `--allow-artifact-rebuild` under the authorized policy |
| E. Publish and verify one seed snapshot + access audit | part of D | **NOT RUN** | — |
| F. Owner reviews the concrete seed id and provenance | owner | **NOT PERFORMED** | the run receipt |
| G. Build the final unsigned `VerificationAuthorizationRef` packet | `build_verification_authorization_packet` | authored; runnable after E | the seed id + run receipt id + shortlist / coverage ids |
| H. Owner signs the final verification authorization | owner | **NOT PERFORMED** | G |
| I. Register the one program allowlist and run Phase 4 | real run | **NOT RUN** | H; `register_program_allowlist` is called by the slice runner only |

A failed or rejected seed-production run does not authorize verification.
