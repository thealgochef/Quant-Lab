# Addendum to the running IFVG redesign-fixes task

## Apply to the existing task, not a second implementation session

Continue `docs/ifvg-redesign-fixes/TASK.md` and preserve its useful interface repairs. This addendum supplies the analytical and evidence corrections omitted from that task. Reconcile it into the existing task ledger and working instructions before further affected edits. Do not reset completed work or overwrite the owner's instruction files.

This request corrects reporting, derived analyses, evidence linkage, and review behavior. It does not authorize new trading studies, new execution rules, expanded data access, engine promotion, a different interface design, or a commit/push. No historical financial rerun is required merely to correct these reports.

## 1. Resolve conflicting instructions first

- The old `CALCULATIONS.md` and `DECISION_RULES.md` are NOT authoritative where they conflict with these corrections. Update the relevant maintained definitions, presentation code, help text, and tests together. Archive or mark superseded definitions without rewriting immutable historical exports. A test matching an incorrect old formula or interpretation is not acceptance evidence.
- Original F2 may identify which diagnostic produced a figure, but neither a fixed closed-profit boundary nor the compressed firm-ledger simulation is an established fresh-account payout probability. Supersede wording that still claims otherwise.
- Original F3's “lucky side,” “unlucky side,” and “wasn't a lucky ordering” rules are superseded. A percentile can be reported descriptively under the specified model; it does not measure luck or prove a causal explanation.
- `SOURCE_OBSERVATIONS.md`'s “Checked and correct (don't change)” statements do not settle market-label availability, benchmark clock consistency, setup causality, or the interpretation of historical-order parity. Reconcile those issues rather than treating them as forbidden to examine.
- Test the actual final code and report its real failures. Do not engineer the outcome to be exactly five failures because original F1 names that number. Record verified pre-existing failures separately from new failures, unexpected passes, and tests not executed. Likewise, before original F6 attributes the $134 versus $126 interval difference to random draws, check the exact seed, sample count, source population, quantile convention and method; document the supported cause rather than assume it.

## 2. Add these analytical and evidence requirements

### A1 — Separate retrospective market labels from information known at entry

The old daily labels use the entry day's eventual close and a full-study volatility threshold. Preserve that calculation only as retrospective description. An entry-known view must use completed observations available by the entry timestamp and thresholds derived solely from earlier permitted information. Document the source cutoff, logical trading-day mapping, and insufficient-history behavior. Insufficient history is not a zero-valued condition.

Test that appending future observations, changing later observations in the same entry day, and altering later volatility leave every earlier entry-known label unchanged. Include an evening entry and a daytime entry. Do not assume an unrelated feature using only completed pre-entry minute bars has the same defect.

A valid repair may retain the retrospective view and mark the entry-known analysis unavailable until implemented correctly. State that outcome explicitly; do not claim the new entry-known calculation exists unless tested. New derived outputs must be versioned separately from frozen economic results.

### A2 — Distinguish a trading-profit drawdown from account failure

The 10/20/40/100-trade chart counts sampled closed-profit paths declining at least $2,000 from a previous high. Rename it accordingly. Its approximately 94% figure is not the share of funded accounts failing. Remove explanations claiming it proves early withdrawals prevent those failures.

Likewise, the simplified +$2,600 versus -$2,000 race measures first crossings of fixed closed-profit boundaries, not received payouts under firm rules. Preserve useful diagnostics with correct names and methodology.

Negative tests must include $0 -> +$6,000 -> +$3,500: a $2,500 trading decline does not imply breach after the relevant account floor has locked at $0 or +$100. Include an open-position failure that would be hidden by looking only at a later unrestricted closing result. Do not alter the actual frozen account mechanics to make the diagnostic agree.

### A3 — Keep the firm-ledger race explicitly conditional

Name it conditional resampling of recorded trades using the selected firm's ledger rules. Disclose that its observations already reflect the historical account's entry selection, skipped opportunities, fixed trade slots, shortened liquidation outcomes, and compressed intratrade paths. These limitations concern both firms; do not describe the entire MyFundedFutures scenario as exact.

Historical-order agreement remains a distinct integration check. It does not prove correctness for every sampled order or reconstruct the full strategy on a fresh account.

Add this counterexample using synthetic marked-profit paths, excluding fees only for clarity:

- A: $0 -> -$1,000 -> +$2,500 -> -$100 -> +$4,000 -> +$3,000.
- B: $0 -> -$1,000 -> +$2,500 -> +$2,000 -> +$4,000 -> +$3,000.

They share their global minimum, maximum, ordering, and final value, and both compress to $0 -> -$1,000 -> +$4,000 -> +$3,000. Under the selected intraday floor that locks at $0, A fails on the intermediate reversal and B survives. The compressed representation cannot distinguish them. Merely supplying the high's timestamp is insufficient. Do not invent missing intermediate observations or claim universal best/worst bounds without proving them.

If a full-path, regenerated-strategy model requires new research, record a separate proposal; do not implement it by launching an unapproved study. Preserve the useful conditional model now. Carry its qualification into the Summary finding, action label, comparison table, cached output, and export—not only a footnote.

When comparing models, match their horizons and relevant inputs, or disclose the differences. Bind cached calculations to source result/revision, configuration, firm, model version, sampling method, seed, path count, horizon, and account assumptions. A cached conditional figure must not inherit an exactness label from a different model.

### A4 — Correct population definitions and payout clocks

Rename the average excluding surviving accounts as “Average payouts among accounts that failed within the tested horizon.” Export the numerator and denominator and show ongoing accounts, unresolved outcomes, and pending receipts separately. It is not a general lifetime expectation.

Label pooled net cash divided by all accounts purchased as that pooled ratio. Remove any assertion that it must be lower than an average of scenario-specific ratios. Define which money is received and which costs are deducted.

Distinguish reaching the simplified profit threshold, actual eligibility while alive and flat, requesting a payout, and receiving it. Label time/trades to each endpoint separately, retaining the saved two-business-day receipt convention. No requested-but-unreceived amount counts as cash received. Changing an analysis goal must not change the greedy withdrawal policy.

### A5 — Correct statistical interpretation, including original F3

A resampled ending-profit percentile cannot establish that ordering was or was not lucky. Permutations of a fixed additive trade multiset have an unchanged final total even when drawdowns and account outcomes differ. Sampling with replacement changes membership/frequency as well as order.

A descriptive sentence such as “The recorded cash result exceeds X% of outcomes under this conditional model; ties count half” is permitted with its assumptions and horizon. Do not require a predetermined percentile or classify it as luck using the 35–65% thresholds.

Explain the deflated-Sharpe statistic using its actual series, formula, and stated candidate-search adjustment. It is not an 81% probability of genuine alpha or future payouts, and the current 64-candidate adjustment does not automatically account for all prior research. Do not hide the weakest assessment when both supporting tests fail.

Describe low daily linear association with market movement as an observation, not proof that the strategy did not benefit from rising markets. Revise any dependent generated findings, templates, and acceptance tests.

### A6 — Establish setup identity before describing borrowed geometry as causal evidence

Prefer the selected configuration's own setup record. Cross-configuration equivalence must establish the relevant input/source binding, starting/supporting/opposing zones, activation and confirmation sequence, and relevant setup history. Matching entry time, direction, family, entry price, stop, or target is not enough by itself.

When equivalence is unavailable, label the overlay “Related context from [source]; exact setup identity not established.” Keep actual funded fills visible, but do not use the related overlay to populate verified causal judgments or call it the selected configuration's proven formation history.

Test same-entry/different-zone and same-entry/different-prior-history cases. Missing causal evidence must not remove known execution facts or cause fallback to a different account's trade.

### A7 — Strengthen original F5 with an early-history negative test

Test the first January account, before later accounts exist. The April example alone cannot test exclusion of future accounts because all six accounts already existed then.

Cover final trade counts, future account options, future event/moment choices, outcome-containing labels, links, tooltips, default selections, visible statistics, chart bounds, and cached full-history state. Changing future records must not change the future-hidden representation at the earlier cursor. Keep full history explicitly separate; already-known scheduled deadlines can remain.

Keep the verified April distinction: actual entry April 12 at 7:07 PM Chicago; cursor 7:10:00 PM; partial exit 7:10:27.251840803 PM. The partial must remain hidden at that cursor. Do not move any actual fill timestamp to match a mock.

### A8 — Preserve independent reviewer judgments

Round-trip a saved mixed judgment: entry correct, stop incorrect. Opening it or saving notes only must preserve both independent values. Also test missing/unknown values and unchanged saves. An explicit combined edit must not be inferred from displaying the combined control.

Preserve result/source/configuration/firm/account/trade namespaces and existing saved notes. Use isolated test copies, never the owner's actual annotations for a destructive reproduction.

### A9 — Correct partial-exit contribution accounting

The thirteen daily-deadline trades contain net contributions of $4,490.68 from the first halves and $40,278.18 from the remaining halves, totaling $44,768.86 in the frozen reference. Reconstruct from the actual quantities and allocated costs; use those values as reference checks, not hardcoded output.

Either use the remaining-half amount for remainder attribution or label the full amount “Total profit from trades held to the daily deadline.” The two legs must reconcile to each trade and the aggregate. Do not subtract this reporting difference from payouts or change the frozen trading record.

### A10 — Use one benchmark clock and restore existing minute-level evidence

Benchmark profit and any benchmark failure claim must share the entry instant, source path, quantity, costs, holding convention, and failure rules, or be clearly identified as different benchmarks. A failure observed in the morning cannot describe a benchmark entered at that afternoon's close. Original “already correct” wording does not settle this consistency test.

Connect the existing hash-verified `approximated_minutes` companion to Trade review using exact run/revision/configuration/firm/account/trade identity. The April trade's identified interval is April 13 at 2:03 AM Chicago; the companion also records supporting context. Missing a minute field on the trade row does not mean the companion evidence is absent. Handle conflicts and unavailable companions honestly. No new raw-data reads are needed to make this link.

### A11 — Preserve research meaning while keeping the interface cleanups

Keep original F4's shared count and F11's equivalent-approval tests. Counts must come from the same authoritative saved plan/membership; do not merely hardcode an identical count in three screens. Reopening an incompatible approved experiment stays read-only and unchanged. Explain excluded unsupported combinations before approving a genuinely new plan; do not fabricate half exits at unsupported targets.

Keep unavailable quality measurements distinct from zero/fail/pass and distinguish available checks from missing required inputs. Do not substitute funded observations for strategy-only populations. Preserve the three-day historical preference; do not select 18–20 or introduce survival/activity filters as part of this correction.

Retain original help-text, navigation, and readability repairs. Do not remove meaningful risk/error colors or hide instrument and proxy-price identity when replacing technical wording. Preserve complete settings in accessible details rather than truncate distinguishing values simply to achieve a two-line table row.

## 3. Verification, preservation, and handoff

Maintain one checklist mapping every original F1–F11 item and A1–A11 item to its status, regression, current-app check, and remaining dependency. State Fixed, Qualified/relabelled, Deferred, or Not verified accurately. A qualification is not implementation of a fully exact replacement model.

Use task-owned source and isolated store copies. Preserve the existing financial runs, approved plans, source bindings, research notes, fees, account rules, calendars, and engine pins. Preserve one funded account at a time per configuration/firm with paid replacements; no copying, monthly credit limit, or expansion in this mode. A large received payment still counts when the account later fails.

Do not broaden searches across raw-data directories. Use explicit authorized code/documentation/artifact paths and guarded readers. A before/after file hash proves no writes, not no reads. Document access scope and any uncertainty separately. June 11, 2026 onward stays protected.

After the final functional edits, run the full suite and lint as the existing task requests. Record the exact source snapshot, imported engine, commands, exit codes, passed/failed/skipped counts, and verified baseline comparisons. If any code or test changes afterward, rerun the affected checks and describe the sequence; do not call it a full-suite run of a different final tree. Do not weaken tests to force expected old numbers.

Obtain a fresh read-only review of the final task changes against the recorded pre-task source, including existing uncommitted work. Supply task-only diffs and actual test evidence internally. Retest and obtain review of any material final corrections.

In the actual apps, demonstrate corrected market-label modes; both risk-model labels; conditional payout statistics and clocks; borrowed-context classification; early-January and April future-hidden states; mixed-judgment preservation; correct leg attribution; and the approximated-minute link. Use readable actual screenshots, not mocks. Preserve firm/configuration/account context on navigation. Keep the old feature-and-model cold-load/356-unavailable issue explicitly open unless that particular route is measured and diagnosed; do not infer its resolution from unrelated routes.

Publish a new compact correction handoff, separate from the immutable prior export. Include:

- `FIX_REPORT.md`: complete mapping of original and added corrections, implemented/relabelled/deferred status, remaining limits, and exact test sequence.
- `CALCULATION_DEFINITIONS.md`: corrected populations, clocks, horizons, formulas, sampling methods, interpretation boundaries, and analysis version/source references.
- `COUNTEREXAMPLES.json`: the synthetic negative inputs, expected outcomes, actual outcomes, and test references needed to inspect the findings.
- `validation_summary.json`: source/import identifiers, command results, final test sequence, browser checks, baseline-failure verification, and what was not tested.
- `preservation_and_access_checks.json`: preservation scope and digest results; data-access evidence scope separately; no claim of a no-read audit from hashes alone.
- `historical_order_parity.json`: reuse the existing compact per-pair receipt with exact comparison fields and source identity, or mark it reported/unavailable. Do not launch an unapproved historical study to recreate a missing receipt.
- `REVIEW_FINDINGS.md`: the completed independent review, not its instruction template.
- `SCREEN_EVIDENCE.md` and a small `screenshots/` folder: actual corrected application states and exactly what they demonstrate.
- `manifest.json`: every actual final payload, size, and digest except the manifest itself; explicit file-count convention.

Verify the completed ZIP by extracting it and checking its final manifest and local links. Keep raw market data, full stores, full path simulations, source trees, patches, scripts, and long logs in the internal engineering area. Include only the concise input/output evidence needed for external review. Preserve an internal reproducible execution record of sampled inputs and outputs for any reported simulation; do not substitute only screenshots for numerical evidence.

Stop at this corrected handoff. No further interface redesign, economic study, data preparation, or strategy optimization is authorized here.
