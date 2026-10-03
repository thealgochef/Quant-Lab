# Strategy-Core v3 Compatibility Matrix

## Current Quant-Lab main pairing (2026-09-22, unified runtime)

The owner approved using one current Core for both the normal Quant-Lab
installation and the IFSM research screen, superseding the two-pin isolation
used during research. Quant-Lab's `pyproject.toml` and `research/core/current.json`
must identify the same exact published Core commit. Both workflows verify that
commit's prepared source; package version `0.1.0` is not a compatibility proof.
See Quant-Lab's `research/core/README.md` for installation and verification.

The installed wheel resolves its matching managed source checkout outside the
Quant-Lab repository, independently of an older sibling checkout used by other
work. IFSM uses the same current checkout, with no implicit historical fallback.
This updates Quant-Lab only; Trade-Lab's environment and pin remain separate.

Historical `a4e3303` and `38825ed` sources remain available for exact reproduction.
The original IFSM manifest, bundle, study identities and results stay immutable.
Old configurations retain their legacy defaults, but corrected execution cannot
resume schema-2 checkpoints that omit required gap tracking and pending bars.
New work uses fresh compatible state; a source change never silently rewrites
an old study or substitutes corrected results for its historical evidence.

The dated sections below retain their original scope and historical claims.

Quant-Lab IFVG B0 projection addendum (2026-09-09):
`ifvg_b0_selected_stage_projection_v2` consumes original verified Core
`ParentCandidateRecord`, `ParentLockRecord`, `OpposingGapRecord` and
`InversionRecord` emissions through the exact saved FSM audit companion. Quant-Lab
checks interval-distance and close-through formulas against Core helpers,
selected/replaced stage identity, completed decision-bar ordinal differences,
parent-window clocks and as-of emission order. Retest entry-FVG nulls remain
structural. See `../Claude-Quant-Lab/docs/IFVG_B0_PROJECTION_REPAIR.md` for the ten
field contracts. Core strategy mechanics and saved Core source identity are
unchanged; new Quant-Lab feature/evidence identities version the repair. This
software contract does not establish sample sufficiency, predictive improvement,
R5B completeness, S11 authorization or activation.

> Historical snapshot: June 2026. Status clarified on **2026-09-08**; the original
> matrix below is retained as migration evidence. Columns labelled "current"
> describe the source inspected at that time and have not been reverified
> against current consumer repositories. For current Core source, start with
> [README.md](README.md), [package version stamps](src/strategy_core/__init__.py),
> [contract schema](src/strategy_core/contract/schema.py), and
> [IFVG context features](docs/IFVG_CONTEXT_FEATURES.md). This matrix does not
> establish current Trade-Lab compatibility or deployment status.

Updated: 2026-06-08. Scope: current code state only. Bundle file presence/checksums are intentionally deferred until a candidate bundle is selected for promotion.

| Field / semantic | Strategy-Core v3 canonical | Quant-Lab current | Trade-Lab current | Compatibility verdict |
|---|---|---|---|---|
| Platform stamp | `PLATFORM_VERSION = strategy_core_platform_v1` (the engine axis renamed at E1) | Emits `platform_version` from `strategy_core.PLATFORM_VERSION` in `strategy.json`; dataset cache hash includes the platform version. | Backend imports Strategy-Core and exposes `strategy_core_platform_version` in runtime metadata, but model registry activation still needs Strategy-Core contract-loader fail-close. | **Market-data runtime aligned; model serving still blocked.** |
| Contract schema | `trade_lab_contract_v2` with the two-axis binding (required `platform_version` + `strategy_version`); `LabelPolicy` includes `decision_offset_minutes`; optional `research_session_experiment` audits Quant-Lab train/eval/gate scope. | Emits full dashboard-utility contract with `decision_offset_minutes`, v3 sessions, level scheme, feature windows, inference block, and research session scope. | `LabelPolicy` lacks `decision_offset_minutes`; `StrategyContract` lacks the version binding (now `platform_version`) and research-session audit metadata. | **Trade-Lab loader rejects/does not understand v3 bundle.** |
| Bars / price source | Trade-price tick bars on 0.25 grid; default decision bar `147t`; deterministic side-signed ordering. | `_build_bars_for_date()` uses trade-price bars; engine adapter threads 0.25. Book-mid mode retained only for regression/historical comparison. | `ApplicationRuntime` routes trade bars through `StrategyCoreService`/`StrategyRuntime`; legacy `domain/candles.py` remains compatibility/test code. | **Market-data runtime aligned.** |
| Sessions | ET-native: boundary 18:00; `asia` 19:00→02:45, `london` 03:00→08:00, `ny` 09:00→17:00; gaps are `none`. | `dashboard_utility_builder` imports `RESEARCH_SESSION_SCHEME`; contract emits `eligible_session="ny"`. | Runtime session/trading-day state comes from Strategy-Core snapshots and is mapped to Trade-Lab DTO labels. Legacy Chicago classifier remains in compatibility modules and must not be used for new runtime strategy paths. | **Runtime aligned; stale module quarantining remains.** |
| PDH/PDL source | Full prior trading-day high/low over `[18:00,18:00)` ET (`prior_day_full`). | Builder carries `prev_full_hl`; contract emits `pdh_pdl_source=prior_day_full`. | Level state is owned by the `touch_reversal` plugin (`StrategyLevelState`, plugin-owned since S-B3a): the runtime seeds prior-day summaries and v3 session levels through its lifecycle methods and reads levels/zones back via the plugin accessors. | **Runtime aligned; bundle parity still required.** |
| Level availability | Enforced `available_from` guard; self-touches before session close cannot consume a zone. | Levels carry availability; engine adapter passes `with_availability=True` in production/honest mode. | Runtime zones/touches come from Strategy-Core and enforce `available_from`; tests were updated away from legacy exact-level merged-zone assumptions. | **Runtime aligned.** |
| Touch rule | Bar-range intersects merged zone representative; first touch per zone/day. | Uses `strategy_core.build_zones()` / `detect_touches()` for production utility path. | Runtime touch updates come from Strategy-Core and are mapped to Trade-Lab `TouchEvent` DTOs/observations; legacy exact-touch engine is not used by `ApplicationRuntime`. | **Runtime aligned.** |
| Features | 3 trade-print interaction features + 3 runtime approach features: `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`, `app_large_trade_vol_pct`, `app_avg_trade_size`, `app_max_spread`. | Engine adapter computes the 6 live features; `strategy.json` feature order is contractual. | `int_absorption_ratio` uses trades, but dwell features still use quote-mid quotes; feature registry is local. | **Trade-Lab feature drift.** |
| Label entry | `realistic_at_decision`: decision time = touch close + 5m; entry price supplied at decision instant; label window starts after decision. | `engine_decision` calls `resolve_honest_outcome()` and injects trade-price lookup. | `outcome_tracker.register()` still uses `level_price_ticks` as entry reference. | **Trade-Lab outcome drift.** |
| Label/cutoff | MAE-first; TP 15 / SL 30 / trap MFE 5 defaults; flatten 16:40 ET; forward cutoff 17:00 ET. | Contract emits constants from Strategy-Core; tests assert v3 values. | Outcome tracker parses contract cutoffs, but schema lacks decision offset and runtime entry remains level-based. | **Partial only; not v3-safe.** |
| Inference gate | `tradeable_reversal`, `ny`, confidence 0.70. | Emits `eligible_session="ny"`. | Runtime session labels now come from Strategy-Core, but model activation/gating still needs v3 bundle parity before serving. | **Runtime session aligned; model-serving gate still blocked.** |
| Test evidence | Strategy-Core unit suite: 135 collected, passing under Python 3.14 and Trade-Lab Python 3.13. | Contract/no-drift tests assert v3 stamp, `147t` default, decision offset, full prior day, `eligible_session=ny`; training-report tests assert failed-gate save blocking and OOS gate/report artifacts. | Trade-Lab backend full suite passes; `test_strategy_core_acceptance.py` verifies direct Strategy-Core touch output matches Trade-Lab runtime DTO/observation output. | **Market-data runtime path tested; model-serving parity still required.** |

Bottom line: **Quant-Lab and Strategy-Core are v3-aligned for research/training/contract emission, and Trade-Lab's market-data runtime now consumes Strategy-Core for bars/sessions/levels/touches.** The next engineering step is model-serving parity: Strategy-Core contract-loader fail-close, Strategy-Core feature/outcome paths, and an end-to-end parity run on one v3 bundle plus one raw-data slice before any paper serving.


### HTF selection-cap experiment support (2026-09-12)

The registry now offers cap 2 as a pending value requiring exact strategy-search
approval; cap 1 stays the default. Evaluate One fixed settings expose this cap.
`search/htf_cap_experiment.py` creates the four fixed-profile drafts for the
240/90-wait comparison, preventing automatic default expansion. Per-timeframe
selection ranks both directions before taps and conflict/direction handling.
The new `selection_audit.py` companion preserves complete per-bar inventory
observations, pre-cap universes and HTF creations when the imported research
Core supports them, with strict coverage and tap reconciliation. These source
files enter replay identity; old artifacts and canonical tables stay immutable.

The task-local Core branch ports only active-selected-HTF physical tracking
after registry eviction, including schema-3 day seeds, and adds audit-only
selection observations. The installed/live Core pin is unchanged. Required
full replay uses the original 117-date bundle, four separately approved fixed
profiles and source-comparable controls; no other research policies apply.


### Sequence-supply research support (2026-09-15, isolated task runtime)

`opposing_min_gap_ticks` defaults to null (inherit the global floor). A non-null
research override changes only the newly arriving 1m opposing stream. The 1m
detector captures the union of required floors and persists its actual floor in
schema-4 registry seeds; HTF, parent and fresh-entry consumers retain the global
floor. No old replay/detection cache is reused for a changed profile.

The pending registry values add an opposing floor of one tick and the parent
timeframe set [1m, 3m, 5m, 10m, 15m, 30m]. Every 1m decision is delivered to the
parent clock/structural-close input when 1m parent selection is configured; it
counts after activation and is eligible through count 40 inclusive. Existing
higher-TF/newest/id ranking, later-bar retest, physical-before-transition ordering,
strict post-tap/post-lock confirmation, strict inversion and execution guards
remain in force. The TF union already includes 1m, so source bar/level panels
remain identical; profile capture/replay/config identities remain distinct.

`search/sequence_supply_experiment.py` builds exactly six separately approved
Evaluate One profiles: B0/D0, B2/D2 (opposing floor only), B3/D3 (parent TF only).
There are no combinations or H1 profiles. This is exploratory same-period
research; temporal-origin changes, fitting, new data and live promotion remain
outside this authorization. Exact source census/hypotheses reside in the task's
review package. Existing studies retain their original identities and artifacts.
# One-hour / four-hour gap invalidation choice (2026-09-18)

Research application development Core adds `htf_gap_invalidation_policy` with
original `execution_wick_full_fill_v1` default and explicit
`own_timeframe_close_v1`. The latter uses separate starting-role validity with
strict own-timeframe finalized closes, retaining physical wick evidence and
active selected tracking after discovery age expiration. Day seed schema 5
rejects older and different-policy resumes; pending-bar repair is retained.
See `docs/IFVG_GAP_INVALIDATION.md` for finalization and role isolation. Installed
and live Core pins remain unchanged. Existing optional lifecycle controls retain
their original defaults for saved application profiles.


## Scheduled holding close and entry sessions (2026-09-18 development study)

The isolated `Strategy-Core-daily-close` checkout adds opt-in
`holding_policy=scheduled_daily_close_v1`. The research configuration is 15:55
America/Chicago, with an earlier market closure minus five minutes governing.
A distinct frozen holding calendar corrects April 3, 2026 to 08:15 Chicago close;
structural calendars, anchors, gap finalization and document sessions remain
unchanged. Entries use an independent explicit Chicago window or all-open mode.
The existing New York morning preset is historical; corrected Chicago morning
is a separately versioned preset.

Both the reducer and replay path enforce actual execution-time locks. A finalized
one-minute deadline close resolves any remaining position after stop-first/target
checks; missing coverage and unresolved mandatory-close positions fail explicitly.
The partial signed return and exit price are emitted as `scheduled_close`, with
supplemental forced-exit evidence. Pending analytic setups and physical/own-chart
gap tracking persist. Day seed 6 / reducer 4 carries the close clock and lock;
old unrestricted day seed 5 / reducer 3 has an explicit compatibility reader.

Quant-Lab's source-bound 32-profile session/distance/opposing-width/parent-timeframe
matrix must use the same frozen final source, calendar and full price bundle.
This is historical research with no live broker cancel/ack/heartbeat integration.
See `docs/IFVG_DAILY_CLOSE_SESSIONS.md` for execution and operational boundaries.

## IFSM MenthorQ level context — Task A1 (2026-10-03)

Core's pure `strategies/ifvg_smc/menthorq_levels.py` supplies immutable snapshots,
Chicago cash slots, EOD nearest-level/normalization values and final-entry gates.
Quant-Lab owns CSV parsing, content-hash receipts and timestamp availability.
The existing replay `levels_for` accepts `IfvgLevelInputs(levels, menthorq)` while
ordinary level tuples remain compatible. The plugin uses that same orchestrator
via its extended `set_static_levels(..., menthorq_for=...)` handoff.

| Section field | Neutral value | Active behavior |
|---|---|---|
| `menthorq_context_version` | `None` | `menthorq_eod_v1` enables runtime lookup/export. |
| `regime_gate_policy` | `off` | `positive_only` or `negative_only` at final entry. |
| `regime_unknown_policy` | `allow` | `block` rejects unavailable context inside the availability interval for an enabled regime gate. |
| `nearest_support_gex1_block` | `False` | Blocks when GEX 1 is among tied nearest levels strictly below entry. |

The four neutral values are explicitly excluded from profile hashes; changed
values enter them. Gate overrides require the named context version. These fields
are not observer configuration or feature-schema inputs. Existing record schemas,
context families and seeds retain their contracts.

Availability is `[06:00,17:00)` America/Chicago with DST. Outside, both gates pass,
flags are null and status is `not_applicable_outside_hours` even under unknown
policy `block`. Enabling shorts bypasses both strategy gates. Reasons follow the
existing session/schedule reasons; blocked candidate evidence remains visible.
Explicit Chicago entry windows supply the slot gate without a new section field.
The per-run `context_export.csv` and reconciled session × regime × slot reports
are Quant-Lab review outputs outside immutable dataset manifests and identities.
Record family, ML plumbing, intraday HVL and target caps are deferred. See Core
decision D-P-18 and `C:/tasks/ifsm_task_a1_level_context/decisions.md`.
