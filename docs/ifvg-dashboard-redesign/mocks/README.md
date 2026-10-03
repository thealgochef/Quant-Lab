# Mock screens

Designed September 23, 2026 with the owner. They're the north star for content, order, grouping, wording and interactions. Values were computed from the funded variation study `5fa65149843484b1` (export v4), the stored one-minute E-mini bars, and an earlier study's saved setup record for the April 12 trade. Resampled values come from one fixed-seed draw and will differ slightly from yours.

Bracketed values (`[resolved count]`, `[earliest stored date]`, `[needs the trailing-floor solver]`, `[not yet built]`) didn't exist yet when the mocks were drawn. See DECISION_RULES.md rule 5.

| Image | Screen | What to notice |
|---|---|---|
| `01_my_studies.png` | My studies | Tabs by study type; leader per firm; only "Open results" and "Continue draft" |
| `02_funded_results.png` | Funded results | One status line; selection vs unseen windows; merged ranking; checks on the leader |
| `02b_funded_results_state_myfundedfutures.png` | Same, other firm | The firm switch changes the table and the leader together |
| `03_detail_summary.png` | Detail · Summary | Verdict, findings with next steps, key measures, concentration, gates |
| `03b_detail_summary_state_68_percent_range.png` | Same, 68% range | The range switch changes the interval text |
| `04_detail_payouts_and_accounts.png` | Detail · Payouts and accounts | Existing panels kept; one account's balance against its trailing limit |
| `05_detail_risk_and_simulation.png` | Detail · Risk and simulation | Payout race, fan with sample paths, end distribution, streaks, drawdown growth |
| `05b_detail_risk_state_shuffle_every_trade.png` | Same, other method | Fan and labels switch to "shuffle every trade" |
| `06_detail_trades.png` | Detail · Trades | Distribution, summary, excursions, trade list with both fills |
| `07_detail_market_conditions.png` | Detail · Market conditions | Condition cards, shaded profit, transition table, entry-condition scatter |
| `08_detail_settings_and_evidence.png` | Detail · Settings and evidence | Full settings with sources, verification, corrections, limits, decisions |
| `09_trade_review.png` | Trade review | Real candles, zones, numbered setup markers, review form |
| `09b_trade_review_state_point_in_time.png` | Same, point in time | Everything after 7:10 PM hidden, including the result |
| `10_new_funded_comparison_setup.png` | New funded comparison | Named baseline, date range, readable chips, new settings, plan count |
| `10b_setup_state_legacy_baseline_warning.png` | Same, legacy baseline | Warning appears when the legacy baseline is picked |
| `11_review_and_approve.png` | Review and approve | Changing settings first, validated checks, approval |
| `11b_review_state_blocked_engine_and_bad_threshold.png` | Same, blocked | Engine mismatch locks approval; an impossible threshold is flagged, not trimmed |

## Clicking through them

`source/` holds the mock files. They need a local web server:

```text
cd docs/ifvg-dashboard-redesign/mocks/source
python -m http.server 8765
```

Then open `http://localhost:8765/Main.dc.html`. Links between screens work. The review screen's blocked state is set by the `engineState` and `thresholdError` values in the script tag at the bottom of `ConfigureReview.dc.html`.

`support.js` is the small viewer runtime the mocks were drawn with (it bundles React, MIT licensed). It's only for viewing the mocks — don't import it into the application.
