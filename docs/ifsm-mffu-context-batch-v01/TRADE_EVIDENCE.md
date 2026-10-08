# Funded trade evidence fields

The saved funded trade table reports `gross_r` and `net_r` separately. Each is
the full trade's exact cent P&L divided by `initial_risk_cents`, the original
stop distance in ticks times the entry quantity and per-contract tick value.
For a partial exit, gross P&L includes both the first fill and the remaining
quantity's final fill; net P&L also includes every actual entry, partial and
final fill fee. The table retains the exact cent numerators and denominator so
the rounded R values can be checked.

Price excursions use the already consumed ordered print path or the declared
minute adverse-first approximation. `price_min_ticks`/`price_max_ticks` and
their `_ns`, `_utc` and `_chicago` event times describe the complete held price
path. `pre_target_price_*` ends at the actual first target fill. When a print
jumps beyond a limit target, that segment stops at the target fill; the print
is still part of the open remainder's `post_target_price_*` segment after a
partial exit. A stop-market gap uses the actual gap-through fill. A scheduled
close includes its closing fill. Approximate candle timestamps are synthetic
and identified by `price_excursion_fidelity`; they are not exchange print times.
`favorable_excursion_ticks` and `adverse_excursion_ticks` use the full held
price path relative to the entry price and trade direction.

`min_equity_*` and `max_equity_*` are account equity extrema in cents and their
times. They are distinct from price excursions. `price_excursion_status` is
`available`, `available_entry_fill_only`, `available_fills_only`,
`unavailable_legacy_reuse`,
`unavailable_source_record`, or `unavailable_checkpoint_history`. Reused old
trades receive an unavailable status without a market replay. The exact Core
`geometry.htf.fvg_id` captured at eligible entry is saved as `htf_zone_id` for
trade concentration only; it does not change entry or exit decisions. A
missing historical zone ID remains explicitly unknown in analysis.

The saved-result Trade review displays both R values, held-price min/max and
their Chicago event times, the first-target segments, fidelity, HTF zone ID and
the separate account equity range. Point-in-time review hides these outcome
lines until the trade has exited. The existing Trades tab equity chart now says
"Account equity range" so its cent-based account extrema are not presented as
price excursions.

The saved MFFU Lab matrix displays all 64 immutable intent rows, dispositions,
and resolved entry gate, exit, cap, size, overhead, geometry, and context policy
values. Completed-result review shows saved matched cash effects, factor
interactions, and receipt waiting intervals with calendar and evaluated-trading-day
counts; unavailable financial values stay blank. The result ZIP includes the
same saved analysis, compact account and HTF-zone trade concentration tables,
and a `TRADING_RULES.md` section describing every frozen MFFU axis. That section
is generated from and checked against the approved plan. Each exported Core
decision retains its UTC decision time in the nested context receipt in
`decision_context.csv`. Reused controls have separately labeled posthoc context
annotations because their v02 policy decisions were not executed.
