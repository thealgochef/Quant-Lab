# Decision rules

> **Note — September 25, 2026 analytical corrections.** Where these rules conflict with
> `docs/ifvg-redesign-fixes/ANALYTICAL_CORRECTIONS_ADDENDUM.md`, the addendum wins. In
> particular rule 8 ("pick the stricter reading") never licenses calling a convention a proven
> worst case, or dropping a population (accounts still open, requests still processing) from a
> reported statistic: conventions are labelled as conventions and populations are shown.
> Maintained definitions: `docs/ifvg-redesign-fixes/followup-1/CALCULATION_DEFINITIONS.md` (since
> follow-up 1; the closeout's copy is the delivered version).

Use these whenever the mocks, TASK.md, SCREENS.md or CALCULATIONS.md don't settle something. Decide, log it in `handoff/DECISIONS_LOG.md` with the rule number, and keep going. Stopping to ask is reserved for rule 16.

## Order of authority

1. **Precedence.** The owner's latest instruction, then TASK.md boundaries, then these rules, then CALCULATIONS.md, then SCREENS.md, then the mock images, then existing app patterns. When two conflict, the higher one wins; log the conflict.

## Reading the mocks

2. **North star, not pixels.** Match each screen's content, order, grouping, labels and interactions. Spacing, exact sizes and chart rendering can follow what the application's framework does well. A panel may reflow on narrow screens; its order may not change.
3. **Numbers in the mocks are examples.** Never copy a number from an image into code. Every figure comes from data through CALCULATIONS.md. If your computed value differs from the mock, trust the computation and note the difference in the report. Resampled figures will differ slightly between runs.
4. **Screens or states the mocks don't show.** Build them from the nearest mock pattern: a new tab looks like the existing tabs, a new table looks like the ranking table, a new chart follows DESIGN_SYSTEM.md chart rules. Empty states say what's missing and why, in one sentence.
5. **Bracketed text in the mocks** (`[not yet built]`, `[resolved count]`, `[earliest stored date]`, `[needs the trailing-floor solver]`) marks values that didn't exist when the mocks were drawn. Build the real value if CALCULATIONS.md defines it and the data exists. Otherwise keep the placeholder, with the style in DESIGN_SYSTEM.md.

## Data

6. **Never invent a value.** No zeros, blanks or estimates standing in for missing data. Use a placeholder that names what's missing: "Not in this study's export", "Not yet built", "Setup zones weren't recorded for this study".
7. **Use stored records first.** Look for an existing saved record before computing anything. Link records by their stored identifiers (for trades, `strategy_trade_id` and the funded trade reference). Never re-run the strategy, engine or funded simulator on historical data to fill a gap. The one exception is P5: feeding resampled trade orders to the funded simulator.
8. **Unclear definitions.** Pick the stricter reading — the one that makes the result look worse or the check harder to pass — and write the definition you used next to the number (a short "how this is measured" note) and in the log.
9. **Randomness.** Every simulation uses a fixed default seed and shows its path count. "Run again" picks a new seed and says it did. Cache results by study, configuration, firm, settings, seed and path count.
10. **Firms stay separate.** Every funded number belongs to one firm. Never total across firms.
11. **Time.** Every displayed time is Chicago time, 12-hour, like "April 12, 2026, 7:07 PM". A trading day runs from the 5:00 PM open to the 4:00 PM close; label trades by trading day. Stored times stay as stored.

## Existing features

12. **Nothing disappears.** Every existing feature gets a place in the new layout. If the mocks don't show where, put it in the closest tab under a "More" section at the bottom and list it in FEATURE_MAP.csv. Removing a feature entirely needs the owner.
13. **Reuse before rebuild.** Where an existing panel already computes a figure — block bootstrap in the context studies, the market regime tab in the feature-and-model studies, the four-part verdict, cash per $1 of accounts in the payout simulator, the 95% interval in search results — reuse that code and adapt its display. Only write a new calculation when the existing one's definition differs from CALCULATIONS.md; then follow CALCULATIONS.md and log the difference.
14. **Two apps.** Share components between both entry points instead of copying them. Don't merge the two apps. My studies lists studies from both study stores.

## Building

15. **Dependencies and speed.** Prefer libraries already in the project. Add a new, widely used, maintained library only when no existing one can draw a required chart (for example candlesticks with shaded zones), and log it. Each screen should open in about 3 seconds from cache. Anything slower runs on request with a progress message and is cached afterward.

## When to stop and ask the owner

16. **Only these:** anything that would change a money figure in a stored result, study membership, a saved approval or review; anything touching June 11, 2026 onward; removing a feature with no home; or a boundary in TASK.md section 2 that seems impossible to keep. Everything else is decided by these rules and logged.

## Wording

17. **Plain English.** Spell things out; no acronyms or code names on screen. "Net R" and "1R" are allowed because the owner uses them. Setting values are shown in words ("All open-market hours", "Half at the target, rest held to break-even or 3:55 PM"), never as internal keys. A selected value is always readable in full without hovering.
18. **Placeholders and warnings are sentences a trader understands.** Say what's missing and what it affects, like "Needs the version that supports half exits. Your saved settings haven't been changed."
