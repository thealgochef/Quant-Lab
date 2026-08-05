# IFVG Setup-Level Visual Verifier — Implementation Plan (amended)

Extends the SHIPPED candidate-level verifier (`replay_chart_store.py`,
`replay_chart_provider.py`, `scripts/ifvg_verifier_charts.py`,
`scripts/ifvg_verifier_tab.py`, `visual_review_store.py`) to setup-level review.
Depends on the `ifvg_fsm_audit_v1` artifact (see
`IFVG_FSM_AUDITABILITY_REPAIR_PLAN.md`). Motivation: the shipped verifier is
candidate-keyed — 170/215 setups (including all 91 S1 HTF-fill deaths) cannot be
displayed today.

## 1. Replay-chart artifact v2 (setup-aware; NEW identity, old artifact retained)

`replay_chart_store.py` changes:

- Version bumps → new content identity: `REPLAY_CHART_SCHEMA_VERSION` 1→2,
  `CANDIDATE_RANGE_POLICY` → `candidate_range_v2` (adds setup ranges),
  `STAGE_GATING_POLICY` → `stage_gate_ordinal_cursor_ts_v2` (adds audit-stage
  gates). `RESAMPLE_RULE_ID` and bar schema unchanged.
- Identity now ALSO pins: `fsm_audit_artifact_id` + `fsm_audit_manifest_hash`.
- New table `setup_bar_range.parquet` — one row per setup (expected 215):
  activation/lock/arm/inversion/candidate/terminal/display-end ts + source
  columns for each, HTF fvg id/tf from audit evidence, parent tf, session at
  activation, end reason + phase-at-death, parentless interval refs
  (interval ids), Q-40 exposure flag (240m shown ⇒ watermark), candidate_ids
  (JSON list; empty for candidate-less setups).
- Corroboration stays OUTSIDE canonical identity (existing decision, unchanged).
- Old v1 artifact + catalog entry retained; new artifact published beside it.

**Decided: frozen `VerifierBundleRef`** — fields: `profile_name`,
`v2_dataset_id`, `v2_manifest_hash`, `v3_dataset_id`, `v3_manifest_hash`,
`fsm_audit_artifact_id`, `fsm_audit_manifest_hash`, `replay_chart_artifact_id`,
`replay_chart_manifest_hash`. `ArtifactPairRef` stays exactly as-is (pair
identity only; never overloaded). Provider/tab open by `VerifierBundleRef`;
catalog entries updated additively — and NO catalog points to the new bundle
until the final QL commit hash is recorded (publication gate, plan §Phase 6.5).

## 2. Provider extension (`replay_chart_provider.py`)

- `open_replay_context` accepts a `VerifierBundleRef`, loads + verifies the
  fsm-audit artifact alongside the pair.
- `list_setups(ctx)` — 215 rows with filter columns (§3 below).
- `resolve_selection` accepts `setup_id` (existing candidate/decision/trade ids
  preserved).
- `setup_evidence(ctx, setup_id, mode, stage)` — returns, ordered by the (a2)
  cross-channel contract (`source_step_ordinal → reducer_substep →
  reducer_substep_ordinal → audit_seq → ts` last-resort): tap candidates incl.
  drops, parent candidates/replacements, slot deaths, parentless intervals,
  lock/opposing/inversion evidence, fill events w/ OHLC + depth, terminal event.
- Stages extended: `htf_tap → activation → parent_candidate → parent_lock →
  opposing_selected → inversion → entry_candidate → trade_opened → terminal /
  trade_resolution`. `StageGate` ordering rule unchanged (ordinal > cursor > ts;
  ungateable evidence hidden, counted in the omission report).

## 3. Tab (`scripts/ifvg_verifier_tab.py`)

- Selection mode: setup / candidate / decision / trade.
- Setup filters: candidate-less; terminal reason; phase-at-death; HTF timeframe;
  parent timeframe; sessions; Q-40 exposure; parentless (had ≥1 interval);
  conflict/suppression flags — the latter two will be PRESENT-BUT-EMPTY if the
  audit shows zero events (D-1); surfaced honestly with a zero-count caption,
  never hidden.
- Ranges: activation→terminal default; separate start/end-ts filter fields.
- Warmup default OFF; model statuses unchanged (visible at decision time only).

## 4. Charts (`scripts/ifvg_verifier_charts.py`)

- Setup-mode figure: activation→terminal default range; fill/death bar overlays
  incl. depth + prior/new extreme hover; tap-candidate markers with drop
  reasons; parentless-interval vrects; terminal marker with reason.
- Candidate/trade mode preserves ALL existing overlays byte-for-byte.
- Q-40 watermark iff 240m shown (existing rule).
- Budgets/omission reports (existing) extended to the new layers with declared
  p95 budgets (measured in the capacity report; gate blocks publication).
- Styling grammar: **no candidate-less setup may look executable** — death/
  expiry styling is visually distinct from entry/trade styling at every stage.

## 5. Review ledger (`visual_review_store.py`)

Additive extension of `ifvg_visual_review_v1`: `setup_id` +
`fsm_audit_artifact_id` + new verdict fields (`htf_verdict`, `parent_verdict`,
`opposing_verdict`, `inversion_verdict`, `fill_verdict`) + new tags for
setup-level findings. CSV export updated; append-only immutability tests
re-verified. Existing rows remain readable unchanged.

## 6. Deterministic review queue

`IFVG_SETUP_VISUAL_REVIEW_SAMPLE.csv`: 13 S4 + 15 S3 + stratified S2 +
slot-death setups + stratified rapid-HTF-fill by tf/age/size/session +
suppression/conflict cases (if any) + parentless-over-threshold. Deterministic
ordering (setup_id sort within strata; no RNG).

## 7. Tests

- Provider: all-215-selectable walk; candidate-less setup renders; PIT
  fill-hiding (no post-stage evidence leaks through a stage gate); stage-gate
  ordering on audit records; pair-identity isolation (bundle refuses mismatched
  artifact ids); budgets.
- Tab: AppTest selection modes + filters incl. present-but-empty; range
  filters; warmup toggle.
- Ledger: immutability, new-field round-trip, CSV export.
- Golden real-data cases: the 13 S4 setup ids as a deterministic fixture +
  representative S1 HTF-fill death + parentless setup.
- Browser viewport/keyboard QA recorded as OPEN (no approved browser session).
