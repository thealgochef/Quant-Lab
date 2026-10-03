# Decisions — answers to `handoff/discovery/questions_for_luis.md`

Ratified by Luis, October 3, 2026. Binding for Task A; cite this file in the design register.

| # | Discovery question | Decision |
|---|---|---|
| 1 | Q1, Q2 — June 11 – July 10, 2026 | Stays sealed. Rules are developed on Jun 2025 – Jun 10, 2026. Jun 11 remains the buffer; Jun 12 – Jul 10 (21 weekdays) is the final test. No guard is changed. |
| 2 | Scope | Task A1 = level lookup + regime gate + GEX 1 nearest-support gate + slot gate via existing explicit windows + grouped reports + per-run export. Task A2 (later) = the manifest-bound `menthorq_context_v1` record family and ML allowlisting. GEX 3 target cap deferred further (as a counterfactual label family first). |
| 3 | Q7 — intraday HVL | Out of scope. EOD HVL only. |
| 4 | Q11 — availability | The level row for trading date D is usable from 06:00 to 17:00 America/Chicago on D. Outside that, context is unavailable and no gate applies. `source_eod_date` must be earlier than `trading_date`. |
| 5 | Q8 — nearest support | Universe = every EOD level column strictly below the honest entry price; nearest = largest such value; ties keep all names; the gate fires only if GEX 1 is among the tied names; no level below → no block, recorded unknown. Longs only. |
| 6 | Q10 — opening move, HVL side | Prior cash close = last 1-minute close with Chicago time ≤ 15:10 on the prior trading day with bars. Opening move = (bar open − prior close) ÷ ((1D Max − 1D Min) ÷ 2), signed and absolute. HVL side = above / below / at. Implied move ≤ 0 → normalized values null. |
| 7 | Q11 — rolls / price scale | Provider records the source-selected instrument per logical day and a roll flag when it changes; roll-flag days are reported separately and excluded from rule comparisons. No back-adjustment, no calendar-inferred rolls. |
| 8 | Q12 — warmup, 2021–22 | First ten store days of the 2025 block are warmup; evaluable from mid-June 2025 (Task B). The 2021–22 block is dropped. |
| 9 | Q3, Q4 — baseline | `ifvg_v2_doc_default_fresh_static_1r` with `holding_policy = scheduled_daily_close_v1`, costed, through the real research pipeline under a new charter created for approval. Every switch is compared against that one baseline. |
| 10 | Q13 — reports | Executed-trade metrics and funded net cash, grouped session × regime × slot with unknown buckets and reconciled totals. Model evaluation is a later step. |
| 11 | Q5, Q6, Q14 — housekeeping | Existing uncommitted lab UI work is committed and merged to `main` before branching (Task A, Step 0). The five known test failures stay documented and must not grow. Missing-source windows keep their current invalid/null handling. The async-rithmic/protobuf conflict is irrelevant to this work and is not touched. The August package is not required; the reproduced 6,074 / 77% figures stand. |
| 12 | Task A1 §3 — availability precedence | Decision 4 governs. Outside 06:00–17:00 America/Chicago (`before_0600` or `after_1700`), neither gate evaluates: the candidate passes both; the export records `regime_gate_blocked = null`, `nearest_support_gate_blocked = null`, and `gate_status = "not_applicable_outside_hours"`. `regime_unknown_policy` applies only inside the availability window, when the level row is missing (`no_level_row`) or the regime row is missing (`regime = "unknown"`): `allow` passes; `block` blocks with `context_unavailable`. Include an Asia-session entry test with `regime_unknown_policy = "block"` that is not blocked. Ratified by Luis in the Task A1 implementation chat, October 3, 2026. |
| 13 | Task A1 §7 — parity scope | Byte-for-byte parity covers the seven v2 execution tables plus `label_source_1m`, and `ifvg_profile_hash`. For v3/context tables, compare content excluding every column carrying Core source identity, pinned commit, source hash or run identity; list the excluded columns per table and reasons in `parity_report.md`. Any difference in a non-excluded column is a parity failure. Ratified by Luis in the Task A1 implementation chat, October 3, 2026. |

## Additional Task A1 implementation rulings (October 3, 2026)

- Scratch store: if the preparation CLI cannot target scratch, add an explicit output-root option or environment variable; defaults unchanged; record in `CHANGES.md`.
- Slot window values mirror the complete existing `daytime_chicago_0700_1555_v1` shape: `enabled_entry_sessions`, `entry_schedule_policy`, `entry_schedule_timezone`, and `entry_schedule_windows` together.
- If selected instrument identity is unreachable, `roll_flag` is null, no comparison rows are excluded, and the comparison table has note column `roll_flag_unavailable`; record in `questions.md` and continue.
- If funded net cash is reachable only through the approval-gated research pipeline, produce executed-trade metrics and export; mark comparison net-cash columns `not_produced_in_a1`, note it, and continue.
- Otherwise choose the closest existing pattern, record it in `CHANGES.md`, and continue. Stop only if a required name, value or policy is covered by none of these rulings and none of `TASK_A1.md`.
