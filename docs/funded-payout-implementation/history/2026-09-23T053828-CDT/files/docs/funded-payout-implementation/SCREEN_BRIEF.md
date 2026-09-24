# Completed-results screen

This brief applies the owner's presentation requirements; it does not change financial rules.

## Main decision

Which tested strategy and account policy produced the most received cash after all modeled account costs, and what risk, spending and waiting were involved?

Present **TakeProfitTrader** and **MyFundedFutures** as separate side-by-side comparisons or clearly labeled tabs. Never sum them into a jointly financed ten-account operation. Use plain English, readable dollars/points, explicit chart timeframes, and Chicago dates/times with AM/PM.

## Visual hierarchy

Start with the question, actual period and a visible simulated-result/approximation status. Primary cards: net cash earned; payouts received after split; largest single payout; account costs; accounts lost before versus after paying. Pending cash is not received cash. Keep usable cash, unused credits, current account count, processing timers and next-five funding threshold accessible without crowding the main row.

Use three purposeful views: cumulative receipts/costs/net cash over time; monthly receipts and spending including zero/partial months; account journeys from creation through payments, pauses, losses and replacement. Include units, useful hover labels and accessible non-color status cues. Avoid inventing probabilities from copied accounts on one market path.

Select an account to see its balance versus the loss threshold and payout history. Select a trade to see quantity, entry/exit Chicago times, full price points, dollars, initial exposure and closure reason. Separate voluntary payout-protection inactivity from missing signals, trading-hour restrictions, processing waits and missing replacement credits.

## Design boundaries

Keep the established application shell and installed charting components. Avoid decorative gradients, oversized empty metric cards, marketing claims, monospace business labels, excessive badges, redundant charts and raw developer tables. Make keyboard access, legible contrast, responsive layout and no-payout/failure states usable. Material limitations are not debug clutter and must stay visible.

No internal hashes, paths, serialized settings, schema/reducer labels, stack traces or implementation counters in the business view or tooltips. Developer evidence stays separate. All shown numbers and automatic exports come from the same immutable verified result.

Test real screen selection, save/reload and result navigation. Inspect actual rendered screenshots/browser evidence when available. Sample fixtures are clearly labeled; an empty or sample screen is not evidence a historical simulation completed. Record any unavailable visual validation in the final report.
