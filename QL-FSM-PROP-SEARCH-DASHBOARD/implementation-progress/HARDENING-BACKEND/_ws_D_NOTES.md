# WS-D notes — Phase 3 (§5): logical calendar, shortlist, seed production, packets

Closed 2026-09-02. Red-first evidence `_red_ws_D.txt` (three collection errors
before the code landed); final run `_ws_D_pytest.txt` (63 passed: the 18 new
tests + `test_identities` 19 with the two new identity pairs registered in the
same process + the untouched `test_child_replay` 10 / `test_verification` 9 /
`test_verification_slice` 7); `ruff check` clean on every file below. No owner
action, no allowlist registration, no real replay, no real source path.
`data/` gained zero files.

## Files touched (all NEW; no existing source module edited)

| File | Content |
|---|---|
| `src/.../ifvg/search/trading_calendar.py` | policy `cme_globex_18et_weekday_v1` (`TradingCalendarPolicy`, `CME_GLOBEX_18ET_WEEKDAY_V1`; registered full closures = `2026-01-01`; evidence-verified holiday sessions), `is_logical_trading_day`, `logical_trading_days`, `next_/previous_logical_trading_day`, `assert_consecutive_logical_days`, `consecutive_logical_windows` (1–5), `physical_partitions_for` (`td−1 prev_utc_date`, `td utc_date`), `session_bounds_utc` (DST-aware 18:00 ET roll), `store_day_chain` (every non-Saturday calendar day), `SourcePartitionRef`, `VerificationTradingDayRef` (validator: exactly `(td−1, td)` + the exact session bounds), `trading_day_ref_from_inventory`, `inventory_from_permitted_source_hashes`, `CANONICAL_CHAIN_START_DAY = 2026-01-01` |
| `src/.../ifvg/search/verification_window.py` | `LOGICAL_WINDOW_RANKING_ORDER` (the §5.1 order verbatim), `HARD_CONSTRAINT_IDS`, `LogicalDayCoverage`, `LogicalWindowScore` (validator: constraints complete, `eligible` = conjunction, `rank_key` = the registered-order projection), `ShortlistEntry`, `R1CandidateAssessment`, `VerificationWindowShortlist` (content id property; `owner_selection = "NOT PERFORMED"`, `register_program_allowlist_called = False`, `no_raw_source_reads = True`), `build_logical_day_coverage`, `rank_logical_windows`, `build_verification_window_shortlist`, `render_shortlist_markdown`, `shortlist_document` |
| `src/.../ifvg/search/seed_production.py` | `SeedProductionReplayPolicy`, `SeedProductionAuthorizationPayload/Envelope/Ref`, `SeedProductionRunPayload/Envelope`, `SeedProductionAuthorizationError` (18 typed reasons), `seed_chain_source_inventory_hash`, `persist_seed_production_authorization`, `synthetic_seed_production_authorization`, `verify_seed_production_authorization`, `run_seed_production_chain` (+ `_canonical_utc`, `_entering_day_seeds`), `build_seed_production_packet`, `build_verification_authorization_packet`, the two markdown renderers; registers the identity pairs `SeedProductionAuthorization` and `SeedProductionRun` |
| `scripts/ifvg_verification_window_shortlist.py` | rebuilds the shortlist from the accepted dataset + audit funnel (manifest hashes verified) into the evidence folder; importing launches nothing |
| `scripts/ifvg_seed_production.py` | `packet` / `verify-authorization` / `run` (exit 2 on typed refusals); importing launches nothing |
| `tests/agents/ifvg_search/test_trading_calendar.py` (5) | policy = weekday minus registered closures (Good Friday is a trading day by evidence); consecutive-day semantics; partitions + DST session bounds; store-day chain; refs bind both partitions from the inventory and refuse malformed refs |
| `tests/agents/ifvg_search/test_verification_window.py` (4) | ranking order is the plan order; coverage rows on logical days with both partitions (a missing partition → not hash-addressable; a Sunday is never a row); lexicographic ranking + hard constraints + seed chain; the required shortlist entries, the R1 candidate assessment, JSON/markdown, and no allowlist marker |
| `tests/agents/ifvg_search/test_seed_production.py` (9) | the policy fails before path; the payload binds the chain semantics (13 refusals); verify refuses on every divergence with typed reasons incl. a moved head; synthetic provenance confined to test namespaces; the synthetic seed-production proof (permitted outputs only, emission-identical restart, verified reuse); refusals before any path (monkeypatched capture never called); unsigned packets fail validation; the CLI imports launch nothing and `packet`/`run` behave |
| evidence | `LOGICAL_WINDOW_COVERAGE_SCAN.json` (1.8 MB; 110 windows with their trading-day refs), `VERIFICATION_WINDOW_SHORTLIST.md`, `PHASE3_SEED_PRODUCTION_AND_WINDOW.md`, `_red_ws_D.txt`, `_ws_D_pytest.txt`, this file |

## Hooks the main agent must add (existing modules I did not edit)

1. `search/identities.py::registered_identity_pairs()` — add `seed_production`
   to the `from . import (...)` list so the identity-projection audit
   enumerates `SeedProductionAuthorization` and `SeedProductionRun`
   (`trading_calendar` and `verification_window` register no envelope pair —
   the shortlist is a content-id contract, not a store entry). With the two
   pairs registered in-process, `test_identities.py` passes (19).
2. `docs/pipeline_state.yaml` / `ARCHITECTURE.md` (main-agent staging): the
   Phase 3 summary — the calendar policy, the store-day seed-chain semantics,
   the two new stores (`seed_production_authorizations`, `seed_production_runs`
   — already in `SEARCH_STORE_NAMES`), the CLIs, and the June-proposal
   ineligibility finding for the owner.
3. OPTIONAL hardening of an R1 latent defect (`search/child_replay.py`,
   not mine): `_SeedSandboxUnpickler.find_class` refuses `pytz._UTC`, but a
   seed produced from day artifacts LOADED FROM PARQUET carries `pytz` tzinfo
   on every timestamp (pyarrow→pandas→`to_pydatetime()`), so a seed snapshot
   saved directly from a disk-chain replay (the real R1 seed path) cannot be
   reloaded by the sandbox. The seed-production runner canonicalizes the seed
   graph to stdlib `timezone.utc` before persisting (hash-preserving —
   `isoformat` is identical — and self-checked), so seeds produced through
   THIS lane reload. The R1 slice's own `save_seed_snapshot` path (if ever
   used directly on a disk-chain seed) still has the defect; recommended fix:
   canonicalize inside `save_seed_snapshot` (or allow `pytz.UTC`/`_UTC` in
   the sandbox safe-list). Recorded as DEV-D-3.

## Deviations / scoping notes

- **DEV-D-1 — seed chains are STORE-day chains, the allowlist is logical.**
  The plan's "ordered logical seed-chain trading days" is bound as BOTH
  `ordered_seed_chain_replay_days` (the physical store-day chain: every
  non-Saturday calendar day from the cold start — the sequence the accepted
  chain replayed, Sundays being zero-bar reducer steps) and
  `ordered_seed_chain_logical_trading_days` (its validated logical subset).
  A logical-only replay would skip the Sunday day-roll steps and is not
  proven emission-identical to the accepted chain, so it is refused by the
  policy's store-day-chain rule. `snapshot_through_day` is the physical day
  before the first verification day (a Sunday for a Monday start).
- **DEV-D-2 — Good Friday 2026-04-03 is a trading day.** The directive's
  provisional closure list named it; the accepted dataset shows a full
  session (9/1/1/1 rows, funnel htf_taps 78). The policy registers only
  `2026-01-01`. The other holidays' sessions are evidence-verified and
  recorded in `EVIDENCE_VERIFIED_SESSIONS`.
- **DEV-D-3 — timestamp canonicalization before persisting the seed** (see
  hook 3): `_canonical_utc` rewrites aware datetimes carrying a non-stdlib
  tzinfo to `timezone.utc` (same instant); the runner refuses to persist if
  the seed hash changed (it cannot: `isoformat` is identical).
- **DEV-D-4 — "exact verifier target"** is defined as a resolved executed
  trade whose setup id resolves in the setup-lifecycle table (the
  setup-verifier deep-link target). Under that definition the June proposal
  has zero targets and fails the §5.1 hard constraint; recorded as an owner
  finding, not a selection.
- **DEV-D-5 — measurable proxies.** "Panel/control-flow coverage" is scored
  as the number of window days present in the FSM-audit day funnel (bars
  processed); MBP-1 scope evidence is `not_evaluated` (0) for every window
  (no MBP-1 evidence artifact exists for these days); panel coverage is
  `not_evaluated`. Both are stated on every coverage row.
- **DEV-D-6 — the "highest lifecycle/candidate/decision/trade coverage"
  entry** maximizes the tuple (executed trades, decisions, candidates,
  setup-lifecycle rows, then the better rank) over the eligible windows — the
  §5.1 order for the first three, lifecycle rows as the documented tie-break;
  the three February windows tie on the first three and separate on
  lifecycle rows / boundary distance.
- **DEV-D-7 — an owner-signed authorization must start at the canonical
  cold start** `2026-01-01`; the synthetic proof's chain starts at
  `2026-01-13` (the conftest days) under the synthetic provenance only.
- **DEV-D-8 — reuse detection replays first.** A second run under the same
  authorization re-replays the chain and then finds the identical snapshot
  and receipt (`reused = True`); the receipt id is not derivable before the
  seed exists. Acceptable for the preparation action (idempotent, no new
  artifact); an early exit would need a receipt index, which would be a
  store listing.
- **DEV-D-9 — verification-packet provenance** comes from the run receipt
  (`seed_production_run_id`), which the builder loads and cross-checks
  against the seed; without a receipt id the packet states
  `unknown_no_run_receipt_supplied`.

## Test counts

New: 18 (5 + 4 + 9). Combined final run: 63 passed (`_ws_D_pytest.txt`).
Not run by this workstream: the full suite (main agent at the gate) and every
module outside the seven listed above.

## Shortlist headline (evidence, not a selection)

110 candidate 5-day logical windows, 50 eligible. Rank 1 `2026-02-04 … 02-10`
(9 trades / 9 decisions / 30 candidates / 9 exact targets / 5 audit days;
seed chain 29 store days); rank 3 = the corrected R1 form `2026-02-06 … 02-12`
(31 store days); the June proposal `2026-06-04 … 06-10` is rank 70 and
INELIGIBLE as stated (no exact verifier target; seed chain 132 store days).
The R1 store-day candidate is `provisional_ineligible_as_stated`
(`2026-02-08` is a Sunday partition).
