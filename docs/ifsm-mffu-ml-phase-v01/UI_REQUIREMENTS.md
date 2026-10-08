# Existing Lab integration — no separate ML application
Publish one discoverable study in the ordinary Quant-Lab → IFVG Lab → My studies workflow, with a clear development-only name such as "IFSM ML phase 01 — entry value and continuation". Source execution may be isolated; user viewing must not require task-specific environment variables, a dedicated port, or manual source/store switching.

## Reuse the existing shell, selections and business definitions
Keep the study/reference/job/feature/model/operation selection consistent across Summary, predictive results, funded results, account journeys and Trade review. Opening a detail and returning, refresh and restart must retain the correct scope. No old 26-card study block or copied external app.

## Three useful views
1. Predictive evidence: all 24 cells, target definition, actual eligible/scored observations, training/test chronology, model coverage, mean-reference error comparison, feature-group differences and invalid-fold reasons.
2. Funded policies: 24 named policies plus two matched no-ML controls, actual net cash and deltas, activity, replacement spending, payment waits and large payouts. Reuse all six repaired lenses and targets on the correct 171-date denominator. Do not compare a fresh October operation with full-year June-start cash.
3. Decision review: feature values known at entry/checkpoint, their source/availability and imputation origin, earlier-trained model version, signed prediction with R units, fixed action threshold and actual action. Separate subsequent label/outcome; hide future decisions/features/fills until the point-in-time cursor reaches them.

A regression score is expected R or incremental R, not "probability of winning". Do not display it as percent confidence. Distinguish zero score from no prediction, and unchanged baseline from model fallback. The first-partial model does not decide whether to enter; the entry model does not change the remainder.

## Visual/reporting semantics
Show human policy descriptions with IDs available for traceability, dollars as dollars, original risk units as R, micros separate from NQ proxy, and real Chicago AM/PM timestamps. Preserve exact machine event timestamps in evidence. A small chart can compare score and eventual outcome, but label that outcome retrospective; a trade-review checkpoint must not reveal it early.

A feature-importance chart, when valid, is descriptive and not a causal explanation or a new selected feature rule. No automatic live activation, "strong alpha" badges, forecast payout promises or silent winner-only view. Insufficient-data and unfavorable cells are normal report outcomes.

Same authoritative computations feed UI and exports. No manual edited CSV/annotation as a substitute for fixing shared presenters. Precompute/cache small summaries so normal navigation does not fit models or replay accounts.

## Compact acceptance proof
Show normal discovery after fresh launch; full 24-cell table including status; one predictive feature comparison; matched funded result with correct dates and baseline; one entry rejection and one continuation action with before/at point-in-time; account/lens drilldown then Back. Use actual completed evidence or explicitly label a synthetic boundary example. A missing real action is not license to fabricate a winner or select a different threshold.

Inspect readable desktop layout plus one narrow view. Keep necessary coverage, not a separate screenshot project for every cell. Preserve all currently accepted gamma/geometry and legacy saved-result screens.
