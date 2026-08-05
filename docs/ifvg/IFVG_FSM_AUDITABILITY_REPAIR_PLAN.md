# IFVG FSM Auditability Repair — Architecture Plan

Status: planning authority for the `ifvg_fsm_audit_v1` companion artifact and the
setup-level visual verifier. Behavior-neutral: **no strategy rule, profile, or label
change; no model training; no ablation; existing artifacts immutable; sealed/June-11
untouched; Trade-Lab unmodified.**

## Authority

- Strategy doc: `Trade-Lab/docs/ifvg-strat.md` — last touching commit `5f37a05655c1b76ebf85df137658be7c7f72cec5`, git blob `9b5f6f163ae060030c5695dbc0aede94e0ebebcd` (matches `verification._verify_authoritative_blob`).
- Prior repair verification: `Trade-Lab/docs/ifvg/IFVG_IMPLEMENTATION_REPAIR_VERIFICATION.md` (canonical location; Trade-Lab is read-only for this task), git blob `ed9030b7b2c5db9338abbd1e2d86fdee2b0da54b`. Referenced, never recreated.
- Accepted final-review v2 dataset: `143b510f8a73896072f44e08f331ef5156e85eb8e5124d25bdf441c4fb6b2ac7`, manifest payload `b089dfadf44253b7071882cc9577feeacd97e532f4fd7086fe2fcbac39e60c22` (per `reports/ifvg_final_review/2314066…/IFVG_LAB_FINAL_VERIFICATION.md:18`).
- Companion documents: `IFVG_FSM_EVIDENCE_CONTRACT.md` (traceability matrix), `IFVG_FSM_AUDITABILITY_OPEN_DECISIONS.md` (D-1..D-5 register), `IFVG_SETUP_VERIFIER_IMPLEMENTATION_PLAN.md`.

## 1. Current replay/capture flow and the exact discard points

Flow: SC `run_day` (`strategy_core/strategies/ifvg_smc/replay.py`) drives
`DayOrchestrator.on_decision_bar` per 1m close → typed `IfvgEmission` stream →
QL `capture_single_date` (`capture_driver.py`) flattens via `_flat`
(capture_driver.py:85-100) stamping `entering_seed_hash` + capture schema →
`build_ifvg_v2_capture` (dataset.py:461-513) chains days, assigns
`trace_ordinal = range(len(trace))` over the whole flattened non-funnel trace
(dataset.py:513) → `partition_capture_tables` (contracts.py:295-317) keeps ONLY the
7 `RecordTable` kinds → `save_v2_dataset_immutable` (manifest.py:226-349).

Evidence destroyed or never emitted today:

| # | Evidence | Where it dies |
|---|---|---|
| 1 | `htf_tap` / `parent_candidate` / `opposing` / `parent_lock` / `inversion` / `setup_resolution` records (all considered/rejected events, with full drop/penetration/rank/distance/causality payloads) | Emitted by the reducer, flattened by QL, then **dropped by `partition_capture_tables`** (contracts.py:295-317 keeps only the 7 executable tables); `capture_driver.py:107` additionally drops the `funnel` emission from rows |
| 2 | Registry fill/eviction events (`first_touch`, `filled`, `evicted_age`, `evicted_cap`) | Produced by `FvgRegistry.on_execution_bar` / `.add` but consumed only as invalidation triggers; `add()` returns discarded at replay.py:553-555; never emitted as records |
| 3 | Entry-joint causality counterfactual (`confirmed_after`, `fully_formed_after` for the entry family) | Computed then deleted — `del confirmed, fully` at reducer.py:1287 |
| 4 | `parentless_window_live` structure | Only a scalar day counter (reducer.py:899-911); per-interval structure never persisted |
| 5 | Parent slot deaths (S1 provisional deaths at reducer.py:783-795/:819-831; S2-S4 terminal deaths via `_terminate_pretrade` reducer.py:913-936) | Only lifecycle rows + funnel counters; no dedicated record with fill depth/clock/window evidence |

Consequence (FSM restrictiveness audit, decision A): the verifier is
candidate-keyed, so 170/215 setups — including all 91 S1 HTF-fill deaths — cannot
be displayed, and guard restrictiveness cannot be distinguished from rule
over-restriction.

## 2. Central parity constraints (drive the whole architecture)

1. **`trace_ordinal` is QL-side and global** (dataset.py:513 for v2, the v3 twin for
   context builds): any new in-band emission shifts every later row's ordinal →
   parity break. Therefore NO new emission kind may enter the existing
   `IfvgDayResult.emissions` tuple.
2. **The per-day flattened frame is a column union across kinds** and partition
   keeps all union columns: any new field on an in-band kind adds an all-NA column
   to every v2 table → `_table_content_hash` (dataset.py:636-657) mismatch with zero
   row changes. Therefore NO field additions to any in-band record dataclass.
3. **`entering_seed_hash` hashes the full seed graph** (`seed_hash`, state.py:85-87,
   over `_canon` of all dataclass fields of `IfvgDaySeed` → registry snapshots →
   reducer snapshot → nested `Fvg`/`Bar`/`SweepTracker`/`GeometryEvidence`).
   Therefore ZERO changes to any snapshotted dataclass
   (`FvgStateSnapshot`, `FvgRegistrySnapshot`, `IfvgSetupSnapshot`,
   `IfvgReducerSnapshot`, `IfvgDaySeed`, …). A seed-shape freeze test pins this.

## 3. Architecture: parallel audit channel + QL reuse of in-band dropped kinds

**(a) Reuse.** The six already-emitted kinds (`htf_tap`, `parent_candidate`,
`opposing`, `parent_lock`, `inversion`, `setup_resolution`) already ride the shared
trace with `trace_ordinal` + `entering_seed_hash`. QL partitions them OUT of the
same trace into audit tables instead of discarding them — zero SC change, exact
interleaving with v2 rows preserved for free. The evidence contract
(`IFVG_FSM_EVIDENCE_CONTRACT.md`) PROVES column-by-column that these records carry
every required field; any gap is an explicit new-emission requirement.

**(b) New SC audit channel — per-step drained, opt-in, never spanning a replay
boundary.** The reducer appends audit records to an internal buffer during
`step()`; the orchestrator drains via `drain_audit()` from `on_decision_bar`
immediately after `step()` (and after `finalize_day`/`finalize_dataset`). The
buffer is provably empty at every externally observable boundary — intra-day
checkpoint, interruption, incremental mid-day end, exception recovery, arbitrary
seed/resume. Opt-in at the replay/orchestrator level:
`audit_capture_mode = "disabled" | "fsm_audit_v1"` (constructor / `run_day`
parameter — NOT a section/profile field; profile hash untouched; `disabled`
short-circuits all audit work). `step()` return unchanged; `on_decision_bar`
return unchanged (protects plugin.py:237, the context observer capture at
replay.py:556-573, and `ContextReplayTape` comparisons replay.py:776-816).
`IfvgDayResult` gains one defaulted field `audit_emissions: tuple = ()` — not
hashed anywhere; tape-playback regenerates it identically because the core
reducer runs live under playback.

**(a2) Canonical cross-channel ordering contract.** Every audit record carries:
`source_step_ordinal` (reducer 1m step ordinal), `source_bar_id`,
`source_bar_cursor`, `reducer_substep` (fixed enum:
`01_context_fill_maintenance`, `02_pretrade_invalidation`, `03_fsm_transition`,
`04_candidate_intake`, `05_parentless_instrumentation`, `06_resolution`),
`reducer_substep_ordinal` (sequence within the substep),
`core_trace_ordinal_before` / `core_trace_ordinal_after` (count of core
emissions already appended this step before/after the audit emission — QL
resolves these against the day's flattened trace to give exact interleaving vs
v2 rows), and `audit_seq` (monotone per day). Substep codes are stamped at the
emission site (orchestrator stamps 01; reducer stamps 02-06). Verifier ordering:
`source_step_ordinal → reducer_substep → reducer_substep_ordinal → audit_seq →
timestamp (final fallback only)`. This single total ordering across both
channels is a hard contract; same-minute ambiguity is a test failure.

### New SC record types (`records.py`, additive; `IFVG_AUDIT_RECORD_SCHEMA_VERSION = 1`; all carry the (a2) ordering fields)

1. `FvgFillEventRecord` (kind `fvg_fill_event`) — with exact setup linkage stamped
   at emission time from live state: fvg + event_kind
   (`first_touch|filled|evicted_age|evicted_cap`) + `BarEvidence` of `bar_1m` +
   prior/new `reached_ticks` + prior/new penetration + `far_boundary_ticks` +
   `fill_depth_ticks` + `remaining_fraction_after` + `wick_crossed_far_boundary` /
   `body_closed_through_far_boundary` evidence flags + ages +
   `registry_live_count_after`, PLUS `setup_id` (nullable),
   `fvg_role ∈ {htf|parent|opposing|entry|registry_only}`, `selected_for_setup`,
   `setup_phase_before`, `setup_phase_after`, `linked_slot_death_event_id`,
   `linked_setup_resolution_event_id` — resolved from `_Setup` state at the
   emission moment; NEVER recovered later by timestamp/nearest matching. Enabled by
   default-`None` field additions to `FvgFillEvent` (fvg.py:320-326 — NOT
   snapshotted; consumers read only `.fvg_id`/`.kind`; prior extreme readable
   pre-mutation at fvg.py:418-424). Emission points: fill maintenance
   replay.py:507-509; capture of currently-discarded `add()` returns
   replay.py:553-555. Role/phase linkage supplied by the orchestrator/reducer,
   which hold both the events and the setup state in scope.
2. `EntryCausalityRecord` (kind `entry_joint_causality`) — candidate_id-keyed
   (emitted inside `_emit_candidate` after `make_candidate_id`, reducer.py:1396 —
   exact join; v2 `entry_candidate` schema untouched); stop the
   `del confirmed, fully` at reducer.py:1287 and thread through.
3. `ParentSlotDeathRecord` (kind `parent_slot_death`) — dedicated, covering S1
   provisional-parent death AND S2/S3/S4 terminal parent fills AND
   structural-close variants at every pre-entry stage: event_id, setup_id,
   parent_fvg_id, phase, death_reason, death_ts_utc, death bar
   id/cursor/OHLC, physical_fill / structural_close flags, far_boundary_ticks,
   prior/new reached_ticks, fill_depth_ticks, wick_crossed_far_boundary,
   body_closed_through_far_boundary, parent_clocks snapshot,
   remaining_window_bars_by_tf, open_window_timeframes,
   parentless_interval_started, event_cursor + (a2) fields. Emitted at the S1
   candidate-death branch (reducer.py:783-795), the structural S1 branch
   (reducer.py:819-831), `_terminate_pretrade` (reducer.py:913-936),
   `finalize_dataset`, and `_resolve_trade` (slot_freed). Makes the 36
   provisional deaths and 83 terminal parent-fill deaths reviewable consistently.
4. `FvgInvalidationEventRecord` (kind `fvg_invalidation_event`) — structural
   invalidation evidence separate from physical fill: invalidation_kind
   (`physical_full_fill | structural_body_close`), source_timeframe,
   source_bar_id/OHLC (the parent-TF bar for structural), boundary_ticks,
   close_through_margin_ticks, strict_comparison_result, setup_id,
   parent_fvg_id, phase. Emitted at both invalidation branches even though the
   current artifact shows zero structural terminal deaths — the contract supports
   the active rule.
5. `ParentWindowEventRecord` (kind `parent_window_event`) — `opened` /
   `parent_selected` / `parent_cleared` with clocks snapshot (activation
   reducer.py:1080-1101; selection reducer.py:1862-1864; cleared
   reducer.py:794-795 / :830-831).
6. **Parentless intervals — semantics LOCKED**: start = first completed 1m
   reducer step whose POST-INTAKE state is `S1 ∧ parent is None ∧ any reaction
   window open`; end = first later step whose post-intake state no longer
   satisfies that predicate. Persisted per interval: first/last_counted_bar_id,
   first/last_counted_cursor, bars_count, start/end_ts_utc, end_reason, open TFs
   at start/end, successor_parent_fvg_id, eventual_lock,
   eventual_terminal_reason. Hard reconciliation:
   `sum(bars_count) == parentless_window_live` per day AND full run (counter at
   reducer.py:899-911 unchanged). Mandated tests: the S1 provisional-death step
   itself counts (its post-intake state is already parentless — exact counter
   mirror; terminal deaths early-return and never count); ends on successor
   selection; ends on all-windows-expired; ends on HTF fill; crosses a
   trading-day boundary; survives arbitrary intra-day seed/resume; a parent
   selected on the current intake step prevents that bar from counting.
7. Day funnel: QL-only — `CaptureDayResult.funnel` already carries
   `funnel_counters()`; materialize as a long table. NO new `_funnel` keys.

Audit ordering authority is the (a2) fields; per-day `audit_seq` + QL-assigned
chain-wide `audit_trace_ordinal` are secondary conveniences. Per-step draining
(not per-day) means the buffer never spans any boundary →
batch/incremental/seed-resume/mid-day-interrupt identical by construction, with
tests to prove it.

## 4. Identity and versioning

- `IFVG_AUDIT_RECORD_SCHEMA_VERSION = 1` (new constant, records.py).
- `IFVG_RECORD_SCHEMA_VERSION` stays 2; `IFVG_CAPTURE_SCHEMA_VERSION` stays 2
  (contracts.py requires == 2); `FVG_SNAPSHOT_SCHEMA_VERSION`,
  `REDUCER_SNAPSHOT_SCHEMA_VERSION`, `IFVG_SEED_SCHEMA_VERSION` all unchanged.
- QL `FsmAuditIdentity` (new, manifest.py): repositories (incl. NEW SC commit +
  source-tree hash) + accepted v2 dataset id `143b510f…` + manifest payload sha
  `b089dfad…` + resolved profile hash + evaluation hash + date allowlist +
  permitted source hashes + `audit_schema_version`. Content-addressed,
  overwrite-refusing, atomic publish under
  `data/ifvg_datasets/fsm_audit/v1/<id>/exploration/`. NO outcome/performance
  fields.
- **Pin discrepancy resolved explicitly**: `config.py:72`
  `ACCEPTED_V2_DATASET_ID = 49902280…` is the v3-baseline lineage pin and is NOT
  reused or changed. The audit lane pins `143b510f…`/`b089dfad…` as NEW explicit
  constants, verified against the on-disk manifest before first save. Registered
  in the open-decisions doc.

## 5. Capacity estimates and performance gates

Estimates (validated by measurement before publication; measured values recorded
in the artifact capacity report):

- Fill events dominate: every live gap × wick-overlap bar. Upper bound ≈ live-set
  size × touched fraction × ~1380 1m bars/day; with LTF registry `max_live`
  bounds and HTF age bounds, expected O(10³-10⁴) audit rows/day, well under
  Arrow/parquet comfort. Audit buffer is structurally bounded by per-step
  draining (asserted in tests).
- Hard gates (all measured, all blocking publication): max audit record bytes;
  max audit rows per trading day; max artifact bytes for the full 138-day
  authorized range; audit-enabled replay slowdown bound (dedicated audit replay:
  bounded absolute wall time + memory); `audit_capture_mode="disabled"` replay
  shows **no material regression** — threshold declared here BEFORE measuring:
  **median per-day wall-time delta ≤ 5% and paired p95 ≤ 10% vs the pre-change
  baseline**, mirroring the v3 performance-gate method (dataset.py performance
  measurements); setup-verifier load time and chart-build p95 budgets (declared
  in the verifier plan).

## 6. Migration strategy

- Accepted artifacts untouched: v2 `143b510f…`, v3 `09ef35d0…`, candidate view,
  M0-M3 runs are immutable and remain valid; nothing rewrites or supersedes them.
- The audit artifact is ADDITIVE: a companion keyed to the accepted v2 id.
- The replay-chart artifact is superseded by a NEW id (schema/policy version
  bump); the old artifact is retained and its catalog entry preserved.
- Cached per-day artifacts (bars/levels, `artifacts_tag`) are reused unchanged —
  the audit build replays with the same day-artifact cache; zero
  protected/sealed access (`ExplorationDataPolicy` counters must be zero).
- SC workflow: implement → commit (new HEAD X) → push → QL `pyproject.toml:37`
  repin `@X` → reinstall (direct_url must record X; verification.py:92-93
  HEAD-match) → only NEWLY-saved artifact identities embed X.

## 7. File-level change list

Strategy-Core:
- `src/strategy_core/structures/fvg.py` — default-`None` evidence fields on
  `FvgFillEvent` (prior/new reached ticks, penetration, flags, ages,
  live-count); populate in `add()`/`on_execution_bar`. No snapshot change.
- `src/strategy_core/strategies/ifvg_smc/records.py` — audit ordering mixin
  fields + 5 new record dataclasses + `IFVG_AUDIT_RECORD_SCHEMA_VERSION` +
  `IfvgAuditEmission`.
- `src/strategy_core/strategies/ifvg_smc/reducer.py` — audit buffer +
  `drain_audit()`; emissions at the sites in §3; parentless interval tracker;
  entry-causality threading (reducer.py:1287→record); substep stamping.
- `src/strategy_core/strategies/ifvg_smc/replay.py` — `audit_capture_mode`
  parameter on `DayOrchestrator`/`run_day`; orchestrator-side fill-event capture
  (replay.py:507-509, :553-555) with substep 01; per-step drain;
  `IfvgDayResult.audit_emissions` defaulted field.
- Tests: `tests/test_fvg_registry.py`, `tests/test_ifvg_reducer.py`,
  `tests/test_ifvg_v2_e2e_goldens.py` (goldens unchanged + channel-separation +
  seed-shape freeze), `tests/test_ifvg_replay_parity.py` (chained vs continuous
  vs plugin audit identity + seed_hash golden hex digests recorded BEFORE the
  change), new parentless-interval + drain-boundary tests.

Quant-Lab:
- `src/alpha_lab/agents/data_infra/ifvg/audit_contracts.py` (new) — audit table
  enum + Arrow schemas + PK/FK/as-of validation + reconciliation.
- `capture_driver.py` — `CaptureDayResult.audit_rows` (defaulted); flatten audit
  emissions on a separate frame via `_flat`.
- `dataset.py` — `build_ifvg_fsm_audit_v1` (clone of the chain loop
  dataset.py:461-513 retaining full trace + audit rows) + in-builder parity gate
  vs accepted `143b510f…`.
- `manifest.py` — `FsmAuditIdentity` + `save_fsm_audit_immutable` (modeled on
  save_v2_dataset_immutable, manifest.py:226-349).
- `config.py` — new audit-lane accepted-v2 pin constants (NOT touching
  `ACCEPTED_V2_DATASET_ID`).
- `fsm_audit_parity.py` (new) — exact parity gate + report.
- `scripts/prepare_ifvg_fsm_audit.py` (new) — persisted job runner (job-dir/
  lock/state pattern; `DevelopmentDataAccess`; same 138 authorized days;
  cached day-artifacts reused; zero protected/sealed).
- `replay_chart_store.py` / `replay_chart_provider.py` /
  `scripts/ifvg_verifier_*.py` / `visual_review_store.py` — per the setup
  verifier plan.

## 8. Acceptance criteria

1. Parity gate: exact identity on all 7 v2 tables vs accepted `143b510f…`
   (row counts, ordered canonical row hashes, PK/FK sets); expected funnel
   215 activations / 132 candidates / 33 executions / 30 post-warmup resolved.
   ANY mismatch fails the task.
2. Funnel ⇔ audit-event reconciliation exact, including
   `sum(parentless bars_count) == parentless_window_live` per day and full run.
3. Cross-channel total-ordering contract tested; same-minute ambiguity = failure.
4. The 7 mandated parentless-interval tests green.
5. Per-step drain boundary tests (intra-day interrupt / seed-resume) green.
6. Performance/capacity gates measured and passed (`disabled` mode: no material
   regression per the declared threshold).
7. 215/215 setups selectable and renderable in the setup verifier.
8. D-1/D-2 characterization tests green; D-1..D-5 registered as OPEN decisions —
   no behavior change.
9. QL publication gate satisfied (committed, pushed, commit hash recorded in
   manifests, catalogs updated last).
10. All reporting numbers produced from the artifacts; the three prior
    overstatements (cross-setup opportunity cost, session impact,
    ranked-fallback benefit) re-reported as descriptive, from persisted
    evidence only.
11. Out of scope enforced: no rule/threshold/session/timeout/direction/
    entry-family/label change; no ranked fallback; no conflict-rule fix; no
    ablation; no training; no Trade-Lab writes; no sealed access.
