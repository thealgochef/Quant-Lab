# IFVG Lab — calculation definitions (corrected September 25, 2026)

This file is the maintained definition of every figure the redesigned IFVG Lab shows for a
funded comparison. It supersedes `docs/ifvg-dashboard-redesign/CALCULATIONS.md` wherever the
two differ; the older file keeps its text and carries a notice that lists what was replaced.
Section 17 maps every superseded definition to its replacement.

- **Reference case.** Funded variation study `5fa65149843484b1…` (published review export v4),
  configuration S1-T1-H14-P1-L-SO ("All open-market hours · Long only · Half at 1R"), at
  TakeProfitTrader unless stated. Full result id
  `5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b`; plan id
  `78406b15cdd9c1363b7cf626f7e45d48e58f6aa4e634a25dc00a3e57bdb28336`.
- **Dates.** Research period January 13 – June 10, 2026: 107 trading days. The saved run
  starts at the January 12, 5:00 PM open and its cutoff is the June 10, 4:00 PM close
  (Chicago). A trading day runs from the 5:00 PM open to the 4:00 PM close and is named by its
  closing date, so an entry at 5:00 PM or later belongs to the next day's trading day. June 11,
  2026 onward is never read.
- **Source.** The definitions describe the code in the delivered source tree named in
  `validation_summary.json`. Every calculation reads saved records only. No figure here
  reruns a strategy, a funded study or a new data preparation.
- **Money.** Stored money figures are exact cents read from the saved result. Derived figures
  are computed in dollars and shown rounded as stated.

## 1. Populations and shared series

| Series | Population | Reference |
|---|---|---|
| Funded trades | The saved funded trade records of ONE configuration at ONE firm, in execution order, across every account that configuration bought at that firm. Each record's `net_pnl_usd` is after all fill costs. | 114 trades; sum $37,730.58; mean $330.97 |
| Daily results | Sum of funded trade results per study trading day, $0 on a day without a trade. | 107 days |
| Trade path | Cumulative trade result from $0, trade by trade, across accounts (trading profit before payouts; not one account's balance). | ends at +$37,730.58 |
| Strategy measures | The study's stored no-account replay totals (Net R, win rate, drawdown in R, profit factor, trades, longest days under water). Never recomputed from funded trades. | Net R 73.45 · win rate 58.4% |
| Firm terms | Frozen in the saved result (`settings.firm_profiles`). | TakeProfitTrader: $102 per account, 80% share, $2,000 loss allowance, floor trails peak equity inside a trade and locks at $0 (touch fails), $2,100 retained cushion, $500 minimum gross request. MyFundedFutures: $125, 90%, floor moves at the session close and locks at +$100. |
| Processing clock | Frozen in the saved result: payout received two business days after the end-of-day request. | |

Firms are never added together.

## 2. Stored money figures

Net cash (payouts received after the split − every account purchase), received after the
split, account costs, payouts, accounts, largest and median payout are read from the saved
summary. Reference: $30,781.88 net cash; $31,393.88 received in 13 payouts; $612.00 for 6
accounts; cash per $1 of accounts $50.30 (net cash ÷ account costs).

## 3. Series measures

- **Worst drawdown** — largest fall of the trade path from its running high (the path starts
  at $0). It is a closed-trade drawdown across accounts, not one account's balance against its
  floor; the account-risk verdict line names it that way and shows the per-account loss
  allowance beside it. Reference $3,661.
- **Sharpe** — mean(daily) ÷ sample standard deviation(daily) × √252 over all 107 days.
  Reference 4.01. **Sortino** — mean(daily) ÷ √(mean(min(daily, 0)²)) × √252. Reference 18.82.

## 4. Result-per-trade ranges and the $133.51 versus $126 difference

- **Method.** Resample the 114 funded trade results with replacement, 114 per resample,
  20,000 resamples, and take each resample's mean (`funded_measures.bootstrap_mean_ranges`).
  Generator `numpy.random.default_rng(20260923)` (PCG64) through `rng.integers`; quantiles by
  `numpy.percentile` with linear interpolation: 68% = 16th–84th, 90% = 5th–95th, 95% =
  2.5th–97.5th percentile. numpy 2.3.1.
- **Result.** Mean $330.97; 68% $203.44 to $458.69; 90% $133.51 to $550.16; 95% $99.59 to
  $596.41. These describe the recorded trades under i.i.d. resampling; they are not a forecast.
- **Why $133.51 and not the reference $126.** Checked one factor at a time
  (`interval_investigation.json`, reproducible internally):
  - *Population:* identical — 114 trades, mean $330.97 to the cent, as in the reference.
  - *Quantile rule:* no effect — all eight other numpy percentile rules give $133.51 on the
    same draw.
  - *Seed:* across 200 seeds of the same method the 90% range's low end averages $131.03
    (standard deviation $1.62; lowest $126.36, highest $136.12); the application's seed
    20260923 is +1.5 standard deviations. The reference's bounds are whole dollars, so each
    seed was also compared on all six bounds rounded to the dollar: **seed 11 reproduces all
    six** — 68% $200.99 to $457.83, 90% $126.36 to $550.67, 95% $95.29 to $600.78, which round
    to the reference's $201/$458, $126/$551 and $95/$601 (the application's own function with
    seed 11 gives the same). It is the only one of the 200 seeds that does.
  - *Path count and method:* other path counts (tried at seed 20260923 only), the legacy
    `RandomState` generator, blocks of 10,
    the normal and Student t approximations and scipy's percentile, basic and BCa bootstraps
    do not reproduce all six bounds.
  - *Conclusion (supported, not assumed):* the difference is a seed difference within the same
    method — same population, 20,000 resamples, i.i.d. resampling of means, linear 5th–95th
    percentiles. The reference's seed was never recorded; its six bounds are exactly what seed
    11 produces, and $133.51 is what the application's fixed seed 20260923 produces. The 90%
    low end varies from seed to seed by about $1.62 (standard deviation), so a 5% tolerance on
    a single seed's low end can be exceeded by seed choice alone. (An interim draft of this
    closeout called the cause "not established" after comparing only the one bound; the
    independent review corrected that — `REVIEW_FINDINGS.md`.)

## 5. Probabilistic and deflated Sharpe ratio

- **Series.** The 107 daily funded results (days without a trade count as $0). Daily Sharpe
  SR = mean ÷ sample standard deviation = 0.2525; sample skew g3 = 3.16; kurtosis g4 = 13.81
  (not excess); T = 107.
- **Probabilistic Sharpe ratio (PSR).** Φ((SR − SR*) × √(T − 1) ÷ √(1 − g3·SR + (g4 − 1)/4 ·
  SR²)) with SR* = 0. Reference > 0.99. It is Φ of a z-score — a normal-approximation
  confidence score, like one minus a one-sided p-value — comparing the observed daily Sharpe
  ratio with 0, **assuming the days are independent draws** with the observed skew and
  kurtosis. It is not the chance that some event happens.
- **Deflated Sharpe ratio (DSR).** The same formula with SR* = SR0 = √V × ((1 − γ)·Φ⁻¹(1 − 1/N)
  + γ·Φ⁻¹(1 − 1/(N·e))), γ = 0.5772, N = 64 = the completed configurations compared in this
  study at this firm (a configuration whose days never vary enters with daily Sharpe 0), V = the
  sample variance of their daily Sharpe ratios. SR0 = 0.1987 daily (3.15 annualized); DSR 0.81.
- **Interpretation boundary.** Neither number is a probability that the edge is genuine, that
  alpha exists, or that payouts will continue. The 64-configuration adjustment covers only the
  configurations compared in this study at this firm; it does not account for earlier research
  or other studies. The screens say so beside the numbers ("Probabilistic Sharpe ratio",
  "Deflated for 64 configurations compared").

## 6. Quality gates and the pending time-under-water decision

Thresholds come from the source strategy study's saved gate settings and are evaluated on the
stored no-account replay. A gate whose input the replay did not store shows "Not in export": it
is unavailable, which is different from fail, pass or zero. A funded-trade figure shown under
such a gate is labelled "for reference only" and is never used as the gate's input (funded
observations are not a substitute for the strategy-only population). The saved time-under-water
limit stays 3 trading days; the leader's replay shows 40 and fails it. The screens mark that
limit as the owner's pending decision and propose no other value.

## 7. Verdict and findings

| Verdict part | Rule |
|---|---|
| Data integrity | Pass when the money reconciles, every checked position minute was on recorded trades or a labelled approximation, and no price day is missing. |
| Edge | Each check passes, fails or is unavailable (95% range's low end above $0; deflated Sharpe ratio at least 0.5). "Holds" when both pass; "Weak" when at least one fails and not both; **"Not supported" when both fail** (the weakest state is shown, never hidden); "Partly checked" when one passes and the other is unavailable (the text names the missing check); "Not available" when neither can be made. Unavailable is never counted as a failure. |
| Account risk | "Watch" when the worst closed-trade drawdown across accounts exceeds the per-account loss allowance, or an account was lost before its first payout; else "OK". |
| Sample | "Limited" under 250 trading days or without an unseen-window result. |

Findings (default thresholds unchanged): concentrated result (largest account ≥ 75% of cash
received, or five largest trades ≥ 60% of trading profit); early losses (section 12 or 14,
whichever is shown, ≥ 15%); held halves (section 8); low daily linear association with the
index (R² < 0.2, section 10).

## 8. Profit of trades held to the daily deadline (half-exit legs)

- **Population.** The configuration's funded trades at one firm whose last part closed at the
  scheduled daily deadline (3:55 PM Chicago; `exit_kind = scheduled_close`). Reference: 13.
- **Legs.** For each trade, with sign +1 long / −1 short, the saved tick value (50 cents for a
  micro) and cost per contract per fill ($0.514):
  - first half = (half-exit price − entry)·sign·tick value·half quantity − the entry fill's cost
    allocated to the half − the half-exit fill's cost;
  - remainder = (final exit − entry)·sign·tick value·remaining quantity − the entry fill's cost
    allocated to the rest − the final fill's cost.
  - The entry fill's cost is split by quantity in exact tenths of a cent; the half takes the
    whole-cent floor of its share and the rest takes the remainder, so the two always add up
    to the recorded entry cost. A trade whose half never filled has first half $0 and
    remainder = its whole result.
- **Reconciliation.** For every trade the two legs add up to its recorded net result, and the
  three fill costs to its recorded costs; otherwise the legs are not used and the finding
  speaks of whole trades only.
- **Reference (frozen, recomputed, not hardcoded).** First halves $4,490.68; remainders
  $40,278.18; whole trades $44,768.86 (13 trades). The finding "Held halves carry the profit"
  quotes the remainders ($40,278) against total trading profit ($37,731) and names the other
  two amounts. This is attribution of recorded results only: nothing is subtracted from
  payouts or cash, and the trading record is unchanged.

## 9. Buy-and-hold benchmark (version `buy_and_hold_one_emini_first_open_v2`)

One E-mini (Nasdaq-100, $20 a point), no costs, held continuously (nights and weekends)
from **one entry instant**: the open of the first stored one-minute bar of the first study
trading day (January 12, 2026, 5:00 PM Chicago, 25,953.50), to the last one-minute close of
the last study trading day (June 10, 2026, 4:00 PM, 28,472.00). Result = (exit − entry) × $20 =
**+$50,370**. The first $2,000 fall is the first one-minute close at least 100 points below the
running high, where the entry price counts as the first high: **January 12, 2026, 7:41 PM**
(the bar closing then, at 25,851.25). A fall inside a minute that recovers by its close is not
seen. Both claims use the same entry, path, quantity, costs and holding convention.

Superseded v1: profit from the first trading day's 4:00 PM close (+$51,475, still reproducible
from the daily closes) while the fall was measured from that day's 5:00 PM open, so the
"January 13, 9:03 AM" fall described a position that, by the profit's own definition, was
entered only that afternoon.

## 10. Tie to the index

Least-squares slope (beta) and R² of daily results on the daily E-mini change × $20 over the
same 107 days (each day's change is its close minus the previous stored trading day's close).
Reference beta 0.067, R² 0.112. **Interpretation:** a low R² is an observation that a straight
line of daily co-movement explains little of the daily variation; it does not show that the
result was independent of market direction (for example, a long-only configuration can depend
on rising prices through which days it trades).

## 11. Resampled paths (Risk and simulation)

- **Sampling.** Both methods draw WITH replacement from the recorded funded trade results:
  "Keep streaks together" draws blocks of 10 consecutive trades at random start positions and
  joins them; "Draw single trades" draws one trade at a time. A path can repeat some trades and
  omit others, so its total differs from the recorded total. (A pure permutation of the same
  trades would always end at the recorded total while its drawdowns and account outcomes
  varied.) Default 20,000 paths, seed 20260923; "Run again" draws a new seed and says so.
- **Fan, end distribution, losing streaks.** Percentiles 5/25/50/75/95 of the path at each
  trade up to the recorded trade count; ends in $5,000 bins; the longest run of losing or
  break-even trades. Reference (keep streaks together · draw single trades): 5th $14,258 ·
  $15,221; median $38,893 · $37,229; 95th $65,399 · $62,719.
- **Where the recorded result sits.** Share of path ends below the recorded +$37,731, ties
  counting half: 47% (blocks). The sentence says it describes the recorded result under this
  sampling model and does not measure luck. No lucky/unlucky classification and no 35–65%
  threshold exist any more.

## 12. Fixed closed-profit boundary diagnostic (formerly "payout race, fixed floor")

Each path starts at $0 and adds resampled closed trade results (blocks of 10, up to 200 trades)
until cumulative closed profit reaches the upper boundary (default +$2,600 = the firm's $2,100
cushion + $500 minimum request) or the lower boundary (default −$2,000 = the loss allowance).
Reference: 78% upper first, 22% lower first, typical 8 and 10 trades. **It is a diagnostic of
fixed closed-profit boundaries:** it leaves out the firm's trailing and locking floor, losses
inside open trades, payout requests, processing and receipts. It is not a payout model and not
a probability of account failure.

## 13. Sampled closed-profit drawdown from a previous high (formerly "drawdown growth")

For trades 0–100 of 20,000 resampled paths (blocks of 10, no withdrawals, no account floor): the
median, 75th and 95th percentile of each path's largest fall so far below a previous high of
cumulative closed profit, and the share of paths whose fall has reached $2,000. Reference at
10 / 20 / 40 / 100 trades: 14% / 37% / 65% / 94%. **The 94% is not the share of funded accounts
that fail.** A funded account's floor can lock (TakeProfitTrader at $0, MyFundedFutures at
+$100), so a $2,000 fall from a higher peak need not end it (counterexample: $0 → +$6,000 →
+$3,500 inside one trade — a $2,500 fall that neither firm's ledger treats as a breach); and a
loss inside an open trade can end an account even though the closed result recovers
(counterexample: $0 → −$2,100 → +$500). No statement that early withdrawal prevents these falls
remains.

## 14. Conditional resampling of recorded trades with the firm's ledger rules

Model id `conditional_firm_ledger_resampling_v1` (`presentation/lab/firm_race.py`), formerly
"payout race — full version" / "the firm's own rules". It runs only when the owner presses
"Run conditional resampling with <firm>'s rules"; its result is kept in the application's
process memory, never in a store.

- **Inputs.** The configuration's recorded funded trades at the firm (the trade population
  of section 1), the study's own 114 trade slots (each keeps its recorded entry instant, exit
  instant and trading day), the saved firm terms, processing clock, quantity, tick value and
  cost per fill, the verified strategy package's trading calendar, and the package's stored
  one-minute bars (only to order each trade's lowest and highest point).
- **Sampling.** Default 1,000 paths, seed 20260923, blocks of 10 recorded trades drawn with
  replacement. The first min(114, 200) slots of path i use exactly the trade indices of path
  i of the fixed-boundary diagnostic; slots past 200 would continue from a derived extension
  seed (not reached here).
- **Adapter.** Each drawn trade is compressed to entry, lowest, highest and exit (plus the
  recorded half exit at 1R). The order of lowest and highest comes from the record or the
  stored bars (all 114 were pinned); otherwise the highest point is placed first **by
  convention**, which is not a proven best or worst case. A slot the account refuses
  (finished for the day after eligibility, or payout processing) is skipped. The real
  funded ledger (`propsim/funded/pair_ledger.PairLedger`, public calls only) applies the
  floor, lock, greedy full-surplus request at the day's end, processing clock and a paid
  replacement ($102 / $125) when an account fails — one live account at a time.
- **Horizon.** The study's 114 trade slots, January 13 – June 10, 2026 (cutoff June 10,
  4:00 PM); the last slot is June 5. Its fixed-boundary counterpart runs up to 200 resampled
  trades, so the two tables' shares are not directly comparable (the screen says so).
- **Endpoints and clocks (first account of each path).** Eligibility = the ledger's
  `eligibility_secured` event (realized balance ≥ $2,100 cushion + $500 minimum right after a
  closed trade, account alive and flat; no more entries that day); request = `requested`
  (that trading day's end); receipt = `received` (after the saved two-business-day clock);
  failure = the account's loss-limit failure. "Received a first payout first" = a receipt
  before any failure; "failed before any payout" = failure first. Trades to an endpoint =
  the first account's trades closed by then (eligibility = request = receipt in trades,
  because entries are refused while secured or processing; the days differ). The
  eligibility, request and receipt medians cover paths whose first account reached that
  event; the failure median covers paths whose first account failed before any payout.
- **Populations.**
  - *Average payouts among accounts that failed within the tested horizon* = payouts received
    by every account that failed before the cutoff ÷ those accounts, pooled over all paths.
    Not a lifetime expectation. Shown with its numerator and denominator, and beside it the
    accounts still open at the cutoff (with their payouts) and payout requests still
    processing (with their after-split amount, never counted as received).
  - *Pooled net cash per purchased account* = (all paths' payouts received after the split −
    all account purchases) ÷ all accounts purchased. A pooled ratio; it is not a mean of
    per-path ratios and is not claimed to be above or below one (it can be either; tested).
  - *Net cash percentiles* over paths, and the recorded net cash's percentile with ties
    counting half.
- **Reference (default draw; `race_evidence/`).** TakeProfitTrader: 58.4% of first accounts
  received a first payout first, 41.6% failed first, 0% neither; typical 6 trades to
  eligibility, request, receipt and to failure; typical days from purchase 3.5 to
  eligibility, 4.0 to the request, 9.0 to receipt, 3.5 to failure; 6,240 payouts ÷ 3,395
  failed accounts = 1.84; 1,000 accounts open at the cutoff with 4,616 payouts; 0 requests
  unresolved; pooled ($23,863,727.91 − $448,290.00) ÷ 4,395 accounts = $5,327.74; net cash 5th ·
  median · 95th percentile $9.8k · $23.1k · $38.1k; the recorded $30,781.88 is above 81.8%
  of paths ("82%"). MyFundedFutures: 67.9% / 32.1%; 6,165 ÷ 2,488 = 2.48; pooled ($26,743,318.86 − $436,000.00) ÷ 3,488 = $7,542.24;
  recorded $34,818.11 above 82.3%. `outcomes.csv` (one row per path) reproduces every one of
  these figures.
- **Limitations (shown on screen before and after a run, in the Summary finding, carried in
  the cached result's `model_id` and in the export `race_evidence/definition.json`).**
  (1) Source selection: the recorded trades already reflect the historical accounts' entry
  selection and skipped opportunities. (2) Fixed slots: trades are placed in the study's
  recorded slots, not where a fresh account would have traded. (3) Shortened liquidation
  outcomes: a trade cut short when a historical account failed keeps its shortened result.
  (4) Compressed intratrade path: a reversal between the stored points is not represented.
  These apply to **both** firms; for MyFundedFutures (floor moves only at the session
  close) the stored lowest point decides a loss inside a trade, but (1)–(3) still apply, so
  its results are conditional, not exact.
- **Counterexample (tested).** With TakeProfitTrader's terms (floor trails peak equity,
  locks at $0, touch fails): A = $0 → −$1,000 → +$2,500 → −$100 → +$4,000 → +$3,000 fails at
  the −$100 reversal; B = $0 → −$1,000 → +$2,500 → +$2,000 → +$4,000 → +$3,000 survives. Both
  compress to ($0, −$1,000, +$4,000, +$3,000), also with the high's timestamp. The compressed
  path in time order survives for both; with the highest point first it fails for both — so
  neither placement bounds the full-path outcome.
- **Historical-order parity** (the saved order replayed through the adapter reproduces the
  saved summary and every trade) is a separate integration check: exact for both leader pairs
  (net cash, receipts, payouts, accounts, costs, 114 of 114 trades); the all-128-pair check is
  the existing test `tests/agents/ifvg_lab/test_firm_race.py` (reported by the final
  full-suite run; no saved per-pair receipt exists). It does not show that other sampled
  orders, or a fresh account running the full strategy, are modelled exactly.
- **Cache binding.** Key = (store root, result id, configuration, firm, model id, method,
  seed, paths, slot count, cutoff, SHA-256 of the saved firm terms and processing clock). A
  result under another model id or terms digest is never returned; the typed diagnostic
  boundaries are not part of the key and never change this run (the greedy withdrawal
  policy is the ledger's own).

## 15. Market conditions: two separately versioned label sets

The Market conditions tab has a "Labels" switch. Both sets use the same daily closes (the last
stored one-minute close of each trading day: 4:00 PM, or earlier on a shortened day such as
January 19, February 16, May 25 at 12:00 PM and April 3 at 8:15 AM) and the same label names
(Rising/Falling × volatile/quiet).

- **Retrospective (`retrospective_daily_close_v1`, the default view, numbers unchanged).** Day
  D: Rising when D's own close is above the close 10 stored trading days earlier; volatility =
  sample standard deviation of the last 10 daily percentage changes including D's own; volatile
  when above the median of that measure over the whole study. Because it uses D's own close
  (after any entry on D) and a whole-study threshold, it describes the day with hindsight and
  was not known when a trade was entered. Reference days: Rising · quiet 39, Falling · volatile
  34, Rising · volatile 18, Falling · quiet 13, not enough history 3.
- **Known at entry (`entry_known_prior_closes_v1`).** Completed observations for day D = stored
  trading days before D whose recorded close instant is before D's 5:00 PM Chicago open (a
  Monday opens Sunday at 5:00 PM; the open stays 5:00 PM across daylight-saving changes). An
  entry at 5:00 PM or later belongs to the next day's trading day, so an evening entry and a
  daytime entry of the same trading day share one label. Trend(D) = last completed close
  against the completed close 10 trading days before it; measure v(D) = sample standard
  deviation of the last 10 daily changes among completed closes (both need 11 completed
  closes); threshold(D) = median of v(d) for every stored day d up to and including D (each
  v(d) uses only closes completed before d's open, so all are known at D's open), needing at
  least 10 measures. Otherwise the day is "Not enough history" — its own state, never a zero
  and never folded into a condition. Trades are labeled by their entry's trading day only (the
  exit day's label would use later information; the tab replaces the entry/exit switch with a
  note in this mode). Reference: first labeled day January 30, 2026; Rising · quiet 31, Rising
  · volatile 23, Falling · quiet 5, Falling · volatile 35, not enough history 13 (January
  13–29); 73 of 107 days carry the same label in both sets. Leader at TakeProfitTrader: Rising ·
  quiet 32 trades +$21,451.04; Rising · volatile 18, +$5,809.96; Falling · quiet 6, +$7,985.82;
  Falling · volatile 42, +$4,885.74; not enough history 16, −$2,401.98.
- **Invariance (tested).** Appending later days, changing D's own close, or changing later
  closes (later volatility) leaves the entry-known label of D and of every earlier day
  unchanged; a previous day whose close instant is at or after D's open is excluded.
  Negative control: changing D's own close changes D's retrospective label (Rising · quiet →
  Falling · volatile in the test), and changing later closes moves the retrospective
  whole-study threshold, which can relabel other days; appending later days leaves D's
  retrospective label unchanged in the test case.
- **Entry conditions** (the scatter under the cards) are unchanged: each measure uses only
  one-minute bars that had closed by the entry, so they do not share the retrospective labels'
  defect.
- The stored bars begin with trading day January 2, 2026 (seven stored days precede the
  January 13 study start). CALCULATIONS.md's "stored bars start January 1" named the calendar
  day of that first session's 5:00 PM open.

## 16. Trade review: setup identity, point in time, reviews and the minute record

- **Setup identity.** A configuration that is a member of the verified strategy study uses
  its own saved setup record by trade id ("Zones come from this configuration's own saved
  setup record."; identity established). Any other configuration can only be linked to
  another configuration's record of an execution with the same entry instant, direction,
  entry family, entry price and stop (plus target when equal). That is **related context,
  not setup identity**: the settings (hours, gap charts, parents) differ, so zones,
  activation and history may differ. The screens say "Related context from configuration
  S1_D80_W1_P1's saved setup record …; exact setup identity for this configuration is not
  established.", title the setup cards "Related setup context", never name a review step
  from the related record ("Higher-timeframe gap is valid"), never offer its steps as
  point-in-time moments, and disable the two formation judgments ("Higher-timeframe gap is
  valid", "Parent gap and retest are right"), which save nothing then (entry, stop and exit
  stay reviewable: they are execution facts). Recorded fills, stops, exits, results and balances are shown the
  same with or without a record; nothing falls back to another account's or trade's record.
  The leader (S1-T1-H14-P1-L-SO) has no own record in the package, so every leader trade
  shows related context.
- **Point in time.** At a chosen moment the page shows only what was known then: bars that
  had closed, fills and setup steps at or before it, accounts with a trade entered at or
  before it (the reviewed trade's own account always), trades entered at or before it, no
  trade total, no
  result or balance after, no five-largest-trades note, no earlier reviews, no loss-limit
  check or replacement after it, and the scheduled 5:00 PM → 4:00 PM day as the chart's range
  (never the stored bars' last minute). Moment choices: the entry, every 5 minutes for an
  hour, then hourly to the 4:00 PM close (plus the own record's setup steps when identity is
  established). January 12, 2026, 10:31 PM entry (trading day January 13): cursor 10:35 PM,
  accounts offered "All accounts" and "Account 1" only, one trade. April 12, 7:07 PM entry:
  cursor 7:10:00 PM; the half exit filled at 7:10:27.251840803 PM (recorded instant
  unchanged) and stays hidden. Choosing another trade or account rebuilds every picker from
  that trade's own moment. "›" and "Save and next trade" move the clock forward: to the next
  recorded trade, or — at the last trade — to the end of the study, where everything is shown
  with a note saying so. Opened in point in time without a link, the page starts on the plan's
  first configuration; Full history starts on the ranking leader, and switching an open page
  from Full history to Point in time keeps the configuration already on screen (the study,
  firm and configuration are the reviewer's context; the point-in-time guarantee covers what
  the page shows about that configuration's trades and accounts).
- **Review judgments.** The ledger is append-only. A save writes the overall verdict and only
  the steps the reviewer set; "Not reviewed" writes nothing, and the combined "Entry and stop
  are right" step writes both fields only when chosen. A missing value is unknown, never
  "incorrect". Earlier reviews (with independent entry and stop values) are listed unchanged;
  namespaces are the result, the configuration-and-firm pair, the account and the trade.
- **Approximated-minute record.** Trade review links the published review folder's
  `approximated_minutes.csv` (the latest export whose manifest names this exact result;
  bytes accepted only when their SHA-256 matches the manifest) by configuration, firm,
  account, entry minute, exit minute, exit kind and net result (Chicago texts parsed, never
  re-formatted), and checks uniqueness within the pair, that each minute lies within the
  trade, that Chicago and UTC times agree, no duplicates, and that the count equals the
  trade's `minutes_approximated`; a row with this trade's account, entry and exit minutes but an exit
  or result no trade of the pair recorded is a conflict. Otherwise it says the record
  conflicts or is unavailable, and a missing approximated-minute count is said to be not
  recorded, never read as zero.
  Reference: the leader's April 12 trade at TakeProfitTrader (Account 6) links exactly one
  minute, April 13, 2026, 2:03 AM Chicago (two trades shared the minute's first timestamp; no
  stop, target, break-even stop or loss limit lay inside the minute's range, so its order
  cannot change an exit, survival or payout), from export v4 of result `5fa65149…`. It stays
  hidden in point in time until the exit.

## 17. Superseded definitions (old → current)

| Where it was | Old statement | Current |
|---|---|---|
| CALCULATIONS.md, Checks · deflated Sharpe; overview card | "about an 81% chance the leader's edge isn't just the best of 64 draws"; "Chance the true Sharpe is above zero" | Probabilistic and deflated Sharpe ratio with series, independence assumption and trial count; not a probability of genuine edge or payouts (§5) |
| CALCULATIONS.md, Summary · Edge | "Not shown if both fail" | "Not supported" (§7) |
| CALCULATIONS.md, Findings | "Early account losses" from the flat race; "Held half carries the profit" with whole trades $44,769; "Little tied to market direction" | Named model per basis (§12, §14); remainders $40,278 (§8); low daily linear association, an observation (§10) |
| CALCULATIONS.md, Tie to the index | Buy and hold +$51,475 close to close; fall "January 13, 9:02 AM" from a different start | One entry instant: +$50,370; January 12, 7:41 PM (§9) |
| CALCULATIONS.md, Risk · payout race (fixed floor) | A payout race of fresh accounts | Fixed closed-profit boundary diagnostic (§12) |
| CALCULATIONS.md, Risk · payout race (full version) | "Expected payouts before an account dies"; "Expected cash per account bought" | Conditional resampling; average payouts among accounts that failed within the horizon; pooled net cash per purchased account (§14) |
| CALCULATIONS.md, Risk · resampling | "reorders", "other orders of the same trades", "Shuffle every trade" | Sampling with replacement; "Draw single trades" (§11) |
| CALCULATIONS.md, Risk · drop from a high | "reached $2,000" read as the loss limit | Sampled closed-profit drawdown, not account failure (§13) |
| CALCULATIONS.md, Market conditions · labels | One label set "from the previous 10 trading days" | Retrospective v1 (numbers unchanged, named) and known at entry v1 (§15) |
| CALCULATIONS.md, Quality gates | "the owner is weighing 18–20" (screen: "the data points to 18–20") | Neutral pending decision; saved 3 unchanged (§6) |
| DECISION_RULES.md rule 8 | "Pick the stricter reading", used to call best-first "stricter" and to leave open accounts out of a statistic | A convention is labelled a convention; populations are shown, not dropped (§14) |
| Redesign DECISIONS_LOG F4, F6, F18, P4b.1, P4b.2, P4b.8, P5.4, P5.5, P5.6, P5.7, P5.10, P5.11, P5.14, P7b.7, M16, FX1, FX2, FX6 | see each entry's supersession mark | decisions AC1–AC13 in the same log |
| Fix report F6 | "$133.51 against $126 … a random-draw difference" (asserted, not checked) | Checked: a seed difference within the same method; seed 11 reproduces all six reference bounds (§4) |
| DATA_GAPS | "MyFundedFutures' … results are exact"; "which minute, and why, isn't in this study's export" | Conditional for both firms (§14); linked from the published record (§16) |
