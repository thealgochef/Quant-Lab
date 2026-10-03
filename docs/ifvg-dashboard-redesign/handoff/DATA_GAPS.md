# Data gaps — placeholders still showing

> **Updated in part — September 25, 2026 analytical corrections** (`docs/ifvg-redesign-fixes/closeout/CALCULATION_DEFINITIONS.md`, section 17). The
> rows below keep their original wording; these changed since: the price-evidence minute is now
> linked from the published review folder's hash-checked `approximated_minutes.csv` (Trade
> review shows April 13, 2:03 AM for the leader's April 12 trade); the risk boxes are now
> "Average payouts among accounts that failed within the tested horizon" and "Pooled net cash
> per purchased account"; the simulator version is conditional resampling for BOTH firms
> (MyFundedFutures is not "exact"); zones from another configuration's record are labelled
> "Related context …; exact setup identity for this configuration is not established"; and the
> time-under-water message no longer says "the data points to 18–20".

Every placeholder the redesigned screens still show for the reference study (funded variation study, September 23, 2026 result), why it shows, and what would fill it. A placeholder always says what is missing; no screen shows a zero or an estimate in its place.

## Showing on the reference study

| Where | What shows | Why | What would fill it |
|---|---|---|---|
| Funded results · Unseen window | "Not yet run. The same columns fill in here once a confirmation run is approved." The "Plan a confirmation run" button is disabled with "Confirmation runs aren't available yet". | No run has used June 11, 2026 onward. Planning a confirmation run is outside this task. | An owner-approved confirmation run over the protected window, and a planning screen for it. |
| Summary · Verdict · Sample | "Limited … June 11 onward not yet tested." | Same as above. The rule also marks any window under 250 trading days as Limited, and this one has 107. | A confirmation run, or a multi-year study. |
| Summary · Quality gates · Days with a trade | "Not in export", with "63 on funded trades" shown as a reference only | Gates are evaluated on the strategy replay without accounts. For the variation configurations the saved result stores only that replay's totals (trades, win rate, Net R, profit factor, drawdown, days under water), not its trades or trading days. | Save the no-account replay's trades, or its per-day results, in the funded result. |
| Summary · Quality gates · Best day's share of profit | "Not in export", with "14% on funded days" shown as a reference only | Same as above. The study's saved definition is also different (largest day ÷ sum of all days, in R). | Same as above. |
| Summary · Quality gates · Session stability, time-block consistency, best setup's share | "Not in export" | Same as above. These need the replay's per-session, per-period and per-setup results. | Same as above. |
| Funded results · Checks on the leader · Quality gates | "4 of 5 pass … 2 not in this study's export" (the mock shows "6 of 7") | The two gates above cannot be evaluated. | Same as above. |
| Trade review · setup zones for trades no verified configuration executed | "Setup zones weren't recorded for this study". Candles, entry, stops and exits are still drawn. | The funded comparison replays each variant, but saves setup geometry only for the verified study's own configurations. Only 2 of the 64 variants are members of that study. A trade links to a verified record only when a verified configuration made the same execution. Every short trade, and some long entries of variants with other parent charts, have no such record. | Save the setup geometry of each variant's replay with the funded result. |
| Trade review · 2R and 3R configurations | Zones from the 1R record of the same entry and stop, labeled "its target differs from this configuration's" | Same as above: no verified configuration used a 2R or 3R target. | Same as above. |
| Trade review · price-evidence note | "… 1 approximated from the one-minute bar (which minute, and why, isn't in this study's export)" | The saved trade row counts approximated minutes but doesn't say which minute or why. | Per-minute approximation records in the funded result. |
| Trade review · "+ New tag" | Visible but disabled: "New tags can't be saved yet …" | The review ledger accepts only its fixed tag list. The mock's four other tags were added to that list without a format change; free-form tags would change the ledger's format. | An owner decision to extend the ledger format for free-form tags. |
| Risk and simulation · "Expected payouts before an account dies" and "Expected cash per account bought" | "Not run yet · choose Run with <firm>'s own rules" until the button is pressed. It then shows the values: TakeProfitTrader 1.84 payouts and $5,328 per account bought (1,000 paths, seed 20260923, about 8 seconds). | By design: the simulator version runs on request because it takes seconds, not milliseconds (rule 15). | Nothing missing. Press the button. |
| Risk and simulation · simulator version, TakeProfitTrader | Labeled approximate | Each saved trade keeps its lowest and highest equity but not the path between them. TakeProfitTrader's floor trails the running high inside a trade, so the in-trade order matters. The stored one-minute bars settled the order of the high and low for all 13,923 saved trades. MyFundedFutures' floor moves only at the close, so its results are exact. | Save the time of each trade's highest equity, or its in-trade equity path. |
| Risk and simulation · trades an account loss ended | Resampled with their recorded, shortened result | The funded rows don't store the strategy's own exit for trades that an account's loss limit cut short. | Store the no-account exit for those trades. |
| New funded comparison / Review and approve · withdrawal triggers other than $500 | Saved in the draft; approval blocked: "Comparing withdrawal triggers needs the funded simulator to accept a trigger per plan …" | The simulator uses one trigger: each firm's $500 minimum request above the retained $2,100. | A per-plan trigger in the funded simulator and plan builder (engine work, outside this task). |
| New funded comparison / Review and approve · gap rule other than the base configuration's | Saved; approval blocked with a plain sentence | The gap rule is not a funded variation setting, so every configuration keeps its base configuration's rule. | Add it to the plan builder's variation settings, with engine support. |
| Review and approve · pass/fail checks changed from the saved ones | Saved; approval blocked | A funded plan can't carry its own checks. The results screens use the source study's saved thresholds. | Plan-carried thresholds. |
| New funded comparison · dates other than January 13 – June 10, 2026 | Saved; approval blocked with both ranges and day counts | A funded comparison replays a completed strategy study's saved dates. | A completed strategy study over the new dates (days before 2026 also need the preparation authorization recorded in the repair task). |
| New funded comparison · legacy baseline | Warning; approval blocked | The legacy baseline is not a configuration of the verified study the comparison replays. | A verified study that includes it. |
| Review and approve · Most trading days under water | "Needs your decision. Nothing in the last two studies passes 3; the data points to 18–20." | The owner's open decision (saved value 3; weighing 18–20). The page never picks a value. | The owner's chosen limit. |

## Filled since the mocks were drawn

- `[earliest stored date]` and `[resolved count]`: "Earliest start: December 14, 2021 …" and the resolved trading days (107 for the saved dates, with 10 warmup days), from repair R8's research period.
- `[needs the trailing-floor solver]`: the simulator-backed payout race (Risk and simulation).
- `[your limit]`: the saved limit (3 days), marked as awaiting your decision.

## Shown only when a record is missing (not showing for the reference study)

- My studies leader columns: "Not finished" / "No saved result" / "None completed" / "Not in this study" / "Saved result unreadable (…)".
- Settings and evidence: any settings row whose saved field is absent reads "Not in this study's export". The approximation clause of the Verification list is left out when no published review folder with a matching file exists. The download button is disabled when nothing has been published.
- Market conditions: stored one-minute bars or the study calendar unavailable; a condition with no trades ("No trades"); fewer than three measurable trades for a correlation.
- Summary and overview: the verified strategy package unavailable (beta, buy and hold and the gates then say so).
