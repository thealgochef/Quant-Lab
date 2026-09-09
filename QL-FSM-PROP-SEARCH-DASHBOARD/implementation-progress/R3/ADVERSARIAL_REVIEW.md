# R3 — Adversarial review (two independent read-only reviewers)

Two independent read-only reviewers ran against the working tree BEFORE any
finding-driven fix (2026-08-19; the R3 seam tests and the deferred-objective
fix from DEV-R3-2 were already in place; every other propsim module was as
authored on 2026-08-18). Reviewer 1: contract fidelity + gate coverage lens.
Reviewer 2: access safety + identity + immutability + regression lens. Both
executed reproductions against the tree. Verbatim verdicts:

- Reviewer 1: **3 blockers · 6 majors · 5 minors** — "The R3 gate as defined
  in PHASED_DELIVERY.md is **not met** … the suites test beside the gaps,
  not across them."
- Reviewer 2: **1 blocker · 4 majors · 9 minors**.

Deduplicated across both reports: **4 blockers, 7 majors, 11 minors** (see
`ADVERSARIAL_REVIEW_RESOLUTION.md` for the dedup map and per-finding
resolution). Every finding was resolved in code/tests before the R3 commit.

---

## Reviewer 1 — contract fidelity and gate coverage (verbatim summary)

- F1 (blocker): Two intrabar scenario policies change the simulation identity
  but can never change the simulated result — `_breach_mode_for` ignored the
  policy id; the walk had no favorable-first mode; the artifact builder's
  "adverse" (far-from-close, direction-agnostic) disagreed with the walk's
  MAE-first semantics under the same policy id. Repro: two runs identical
  except the scenario policy id → ids differ, results identical.
- F2 (blocker): The PHASED R3 gate end-to-end (compile → contract → simulate
  on the R1 baseline stream) did not exist — `adapters.py` and both
  `prop_metrics` builders were dead code (zero callers), no production
  `prop_simulator` existed, no test built a `PropFirmContractEnvelope` from a
  compilation.
- F3 (blocker): Day-count bases outside TRADING_DAY were silently
  reinterpreted or dropped by the engine — calendar-basis recurring fees
  charged NOTHING (no refusal), calendar-basis max_eval_days never expired,
  min_days silently reinterpreted, withdrawal fixed_cadence ignored its
  basis; the clock-level test masked it and contained a dead assertion.
  Repro: calendar-month recurring fee, 45 funded days over 2 months → 0 fees,
  no error.
- F4 (major): `max_days_per_path` was a result-changing constructor-only
  argument outside the simulation identity; the audit test exempted it with a
  false justification. Repro: same id, different results.
- F5 (major): A mixed-fidelity bundle reported "supported" for a
  chronology-sensitive rule at bundle grain even when one trade's only
  evidence was an assumed scenario and scenario use was forbidden.
- F6 (major): `run_account_simulation` trusted caller-asserted evidence — no
  hash check of the capability report vs `path_capability_report_id`, an
  EMPTY report passed `_require_mode_support`, `bundle_fidelities` was a free
  side channel, the scenario policy id was never validated, and the shipped
  test fixtures exploited all of it.
- F7 (major): Supersession never retired the old contract id (definition-only
  record; missing `effective_at`; no refusal path; no test).
- F8 (major): Result-changing firm-contract fields silently inert in the
  walk: `account_expiration`, `payout_processing`, funded-phase
  `consistency_pct`, `breach_observation_policy` /
  `unrealized_equity_counts_for_breach`.
- F9 (major): `build_evaluation_fitness` miscounted passes (a pass-then-
  funded-breach path counted as NOT passing; the `"funded"` verdict check was
  dead), days-to-pass used total days walked, breach probability mixed
  phases; both builders untested and unwired.
- F10 (minor): Literal types degraded to `str` (`risk.instrument`,
  `copied_market_path_policy`); `FirmCalendarPolicy.holidays` undeclared
  extension; `PropContractSupersession.effective_at` missing.
- F11 (minor): Intraday trail floor raises emitted no `threshold_ratchet`
  events.
- F12 (minor): §16.4/P0 coverage gaps — recurring fee never charged in any
  test; 3 skip reasons untested; withdrawal `max_allowed_each_period`
  untested; P0-15 proven only as payload hashes (no two-run payout-stream
  diff); no mixed-leg portfolio fixture; no portfolio bootstrap runner (the
  shared-sequence property untested under bootstrap); positive path-event
  linkage never exercised; wording scan covered only 2 of 12 modules; the
  `prop_simulator is None` deferred-objective branch untested.
- F13 (minor): Adapter stream hash silently dropped absent columns
  (fail-open) and its docstring overclaimed reorder sensitivity.
- F14 (minor): The first account's fee event carried the literal
  `"account-open"` in `event_ts_utc`.

Reviewer 1 §16.4 coverage table verdict: 14 rows covered, fees row partial
(recurring MISSING), risk row partial (3 skip reasons), adapter row MISSING.
Reviewer 1 also verified CORRECT: 1m-OHLC unordered truthfulness; fidelity
vocabulary; per-rule OHLC refusal; ordered-fills-≠-chronology; bundle
completeness/identity; firm/scenario separation; P0-13 total order;
P0-17/P1-A ladder; P0-21 bootstrap semantics; P0-14 at the clock layer;
stress registry; portfolio common path (replay); prop gates rows; frontier
seam (worst-firm flip, ALL-legs, scoping, typo-objective unreachability);
identity hygiene incl. the audit force-imports; July-walker compatibility +
parity.

## Reviewer 2 — safety, identity, immutability, regression (verbatim summary)

- F1 (blocker): `max_days_per_path` result-shaping outside the identity AND
  bootstrap mode never required `bootstrap_protocol_id`. Repro: same
  simulation id, `expected_net_payout_90d` −122.2 vs +167.5 (sign flip that
  would flip `evaluate_prop_gates`).
- F2 (major): the nine propsim identity pairs were invisible to the CANONICAL
  audit enumeration (`registered_identity_pairs()` imported no propsim
  module; the required-name assert named none of them; coverage was a
  collection-order side effect). When force-registered, all nine PASS — the
  gap was enumeration, not compliance. The `extra_envelope_fields` diff
  itself was verified sound.
- F3 (major): the intrabar scenario policy id inert in the walk (same root as
  reviewer 1 F1) + `_require_mode_support` never checked the payload's
  scenario id against the bundle artifacts' policy ids.
- F4 (major): the runner never revalidated resolved artifacts against the
  identity fields pinning them (policy set, firm, report id, bundle-id
  binding); the shipped fixtures ran with `firm_contract_id="a"*64` against a
  firm hashing elsewhere.
- F5 (major): the `prop_simulator` call was the only per-child seam without
  exception containment/sanitization — one child's exception aborted the
  whole search, froze the state file at `underlying_edge_passed`, and leaked
  a raw traceback; the `result is None` (reused child) contract was
  undocumented and untested.
- F6 (minor): the ladder's synthetic cap was not bound to the compilation's
  document set (`documents=()` laundered past the cap).
- F7 (minor): `AccountWalkResult.final_state` exposed the LIVE mutable state
  (continuing the walk mutated an already-returned result).
- F8 (minor): the portfolio's only integrity check was a dead self-comparison
  (`pragma: no cover` on a check that audits nothing); the real leg-id ↔
  policy-hash binding was absent; account ids collided across copies.
- F9 (minor): spec `Literal`s loosened to `str` in three payload fields.
- F10 (minor): `manifest_payload_sha256` synthesized from the payload hash at
  construction — a pre-materialization pseudo-manifest presented as a
  post-materialization fact; the spec's two-factor pin degenerates to one.
- F11 (minor): an empty trade-path bundle fabricated `core_replay_id="0"*64`.
- F12 (minor): `gross_trade_stream_hash` was column-subset-dependent
  (positional rows; schema collisions possible).
- F13 (minor): dead assertion in the calendar test.
- F14 (minor): `fixed_cadence` withdrawal silently reinterpreted any cadence
  basis as trading days.

Reviewer 2 date-safety grep (23 files): `2026-06-11` **0 hits**; sealed range
**0 hits**; real store paths/listings/opens in new code **0**; network
primitives **0**; all date literals classified SAFE (synthetic 2026-01-*
family, the 2020-01-06 synthetic bootstrap anchor, metadata timestamps, and
pre-existing R1 string fixtures). Verified correct: July evaluation-only API
untouched (zero diffs, 87 propsim tests passing, no monkeypatching, no
stdlib-calendar shadowing); engine parity treats the old API as ground truth;
charter refuses unregistered objectives (KeyError unreachable); no-simulator
regression path preserved; identity kernel behavior (re-hash on
construction, ImmutableMap, deep-freeze attacks); determinism/test hygiene
(seeded rng only, no wall-clock, tmp-only writes); capability doctrine
enforced.
