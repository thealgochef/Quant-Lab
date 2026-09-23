# IFSM daily close and Chicago entry schedules

Implemented September 18, 2026 for the bounded 32-profile research request in
`../Claude-Quant-Lab-Research-Artifacts/archived-reports/ifvg_daily_close_sessions_20260918/owner_prompt.txt`. This is a historical
research execution contract, not a broker integration or live deployment.

## Owner rules and research buffer

The owner requires every position to close **before 4:00 PM Chicago time**, with
no holding through a daily or weekend market closure. Holding through midnight
inside an open overnight session is permitted. The research uses **3:55 PM
Chicago time**, a proposed five-minute operational buffer. It is not a separately
specified owner deadline or a universal prop-firm rule. A scheduled earlier
market close takes precedence, with the same five-minute lead.

`holding_policy=scheduled_daily_close_v1` uses `daily_close_timezone=America/Chicago`,
`daily_close_time=15:55`, and `daily_close_buffer_minutes=5`. The selectable
`legacy_unrestricted_v1` preserves historical behavior; it is forbidden in the
new 32-profile comparison. Existing saved studies and immutable results retain
their identities and original holding semantics.

## Execution and state

The actual one-minute execution path first checks protective stops and targets,
with stop-first precedence when both are touched. At the planned deadline, any
remaining position exits at the recorded close of the complete one-minute bar
ending exactly at that deadline. No later candle, next-session price, inferred
last row, or synthetic watchdog fill is used. Missing or overdue boundary prices
fail closed. A time exit has `resolution=scheduled_close`, actual `exit_ticks`,
`scheduled_exit_deadline_ts_utc`, and `scheduled_exit_schedule_id`. Its signed
partial gain or loss and the unchanged 0.514-point round-trip cost enter metrics
exactly once. A resolved setup is consumed.

Entry execution is locked from the deadline inclusive to the legal reopening
exclusive. Signal and execution times must both pass the entry window and lock.
Pending analytical setups, their clocks, own-chart validity, detector history,
and continuous context remain available outside entry windows. Reopening does
not replay a canceled signal. This executor fills synchronously at confirmation
bar close and has no outstanding broker entry-order queue. Forced-exit evidence
records zero pending entry orders and retains conceptual protection until the
position is resolved. The simulated clock watchdog detects missing overdue
execution evidence; it does not prove delivery of a production broker flatten.

Core day seeds use schema 6/reducer 4 and preserve the daily-close clock, last
completed deadline and reopening lock. Historical schema 5/reducer 3 seeds remain
readable for legacy holding only. Mandatory-close resumption without its required
clock state is rejected. Existing pending logical-close delivery remains intact.

## Entry presets and compatibility

The corrected selectable preset is **Morning - 7:00 AM to 10:30 AM Chicago time**
(`morning_chicago_0700_1030_v1`). It uses timezone-aware `America/Chicago` conversion
and accepts new entries from 7:00 AM inclusive to 10:30 AM exclusive. Protective
management continues after 10:30 AM; the common afternoon deadline still applies.

The old `ny_0700_1030` identifier continues to mean 7:00 AM–10:30 AM New York,
displayed as **Legacy morning - 6:00 AM to 9:30 AM Chicago time (historical)**.
Opening a legacy draft does not change or auto-save its meaning. The explicit
corrected-copy action creates a new draft. Corrected presets use distinct
configuration/result identities. Neither structural chart anchors, source day
labels, nor the global structural session scheme moves.

New policies are `entry_schedule_policy=explicit_windows_v1` with
`entry_schedule_timezone=America/Chicago` and explicit `entry_schedule_windows`,
or `all_open_market_v1` for all legally open hours. `legacy_doc_sessions_v1` keeps
the original native New York windows. A full-day schedule never uses equal start
and end values. The 32-profile matrix has exactly eight members for each of:

- Original Chicago windows: 3:00 PM–12:45 AM, 1:00 AM–6:00 AM, 7:00 AM–1:00 PM.
- All open-market hours, intersected with the mandatory lock.
- Daytime: 7:00 AM–3:55 PM Chicago.
- Corrected morning: 7:00 AM–10:30 AM Chicago.

Each schedule crosses opposing distance 80/160 ticks (20/40 points), opposing-only
minimum width 1/4 ticks (0.25/1 point), and supporting charts excluding/including
one minute while retaining 3-, 5-, 10-, 15- and 30-minute charts. The control is
`S0_D160_W1_P0`. Every profile keeps own-chart-close gap invalidation, cap 2 and
all other recovered baseline settings, costs, source dates and current charter.

## Calendar and evidence

The separate holding calendar preserves the original structural calendar and
chart finalization identities. Official calendar evidence corrects Good Friday,
April 3, 2026 to an 8:15 AM Chicago market close, an 8:10 AM research deadline,
and Sunday 5:00 PM reopening. The exact source URLs and amendment are bundled in
`shared/source_manifest.json` and `shared/trading_schedule.csv`. Planned times
are derived before looking up prices. All 117 saved dates remain: ten warmup and
107 evaluation dates, with 114 applicable closure events. No additional price
dates, fitting, holdout or June 11 access is authorized.

The new priced execution projection `core_executed_trade_priced_exit_v2` retains
all three exit-evidence fields even for an empty mandatory-close result. The
original projection and historical bytes remain readable. Selection of the new
projection depends on the frozen holding policy, never observed profitability.
`daily_close_evidence.py` independently checks every position against every lock
interval, source-bar prices, stop-first precedence, exactly-once costs, and full
forced-event coverage. A Friday-to-Monday position fails even if its exit is
before 4:00 PM. All eight morning profiles receive 107 explicit daily audit rows.

The process-local research launcher `scripts/run_ifsm_research_ui.py` selects
`../Strategy-Core-daily-close` with a verified source identity. Installed/live
Core pins remain unchanged. The prior 154-trade own-chart-close study is a
historical unrestricted-holding reference, not a compliant new control.
