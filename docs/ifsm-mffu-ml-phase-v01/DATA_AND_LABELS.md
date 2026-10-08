# Data, event identity, point-in-time features and labels
## 1. Reuse verified sources; create only missing research derivatives
Read the b8db binding, exact effective sections, final local reporting source and prepared-input registrations. Current user-side roots are hints, not permission to reset an older checkout. Verify against the local current source map before making a task copy. Follow supported catalog/registration access; never scan raw directories to discover arbitrary extra dates.

Both unchanged MenthorQ v02 ZIPs are included. Prefer an already verified local extraction; otherwise extract to task-local inputs and verify its manifest. Do not re-collect or re-reconstruct vendor data. Loading canonical tables may encounter metadata outside the model dates; only eligible rows for the authorized scope may enter feature/label calculations. No fitting aggregate, future-report fallback or data inspection from June 11, 2026 onward.

## 2. Primary opportunity population
Create one fixed, account-independent reference opportunity stream per baseline, using the corrected engine, original ten warmup dates and full authorized 253-date chronological chain. Preserve the single-setup/single-position lifecycle, intended entries, five-at-1R/entry-stop remainder, scheduled daily close and price/fee conventions. Disable only funded-account limits, acquisition/payout effects and ML decisions in this LABEL SHADOW. It is not a bank account and does not create cash receipts.

Use an existing verified equivalent market-only replay when available. Otherwise one causal shadow replay per reference is in scope. Capture the actual parent/opposing/entry state and source timestamps during that replay. Do not invent an alternate pattern detector, reconstruct geometry from chart pixels or borrow another configuration's setup evidence. Count and explain differences from the old ordinary stop-first table; its 387/351 examples are not required exact populations under a different exit-order contract.

The primary entry dataset contains every actual eligible ENTRY of that fixed shadow process once. It is not a collection of all retest diagnostics, all rejected taps, every repeated candidate, nor only the winning/paid funded trades. A shadow position's first reached 1R checkpoint supplies one continuation observation. Both retain losers and flat/deadline exits.
Create typed status rows for unresolved/censored/price-invalid labels and exclude these from numeric training targets, not from coverage reporting. Never label an inactive retest or a missing label as a loss or zero.

The PRIMARY label resolver uses ordered recorded print evidence with the same causal signal/bar boundary as the funded engine, with account constraints removed. Label-only paths must be distinguishable from ordinary candle approximations. Reuse old label values only on an exact identity/execution/cost match; do not pool stop-first candle labels with print-ordered ones.

If a label path needs the baseline's minute approximation, retain its observation and reason but exclude it from the primary exact-label fit/metrics. Do not cherry-pick based on its return. Economic evaluation retains inherited labeled approximations and reports their impact scope. Current-time model fallback must NEVER inspect whether a FUTURE part of a trade will later have deficient coverage.

## 3. Two label contracts
Let E be intended/filled entry under the declared resolver, S the original structural stop, q0=10 micros, point_value=$2 per micro, and R0=abs(E-S)*q0*2. R0 is positive gross initial price risk, excluding fees; it stays fixed after partial fills. Financial posting uses exact cents; label ratios use float64 from those reconciled amounts.

### ENTRY: entry_partial_net_r_v1
y_entry = (sum of all realized leg price profits minus all actually posted fill fees) / R0.
It is the market-only partial-policy outcome through entry-stop, structural stop or mandatory close. It is not account loss, net cash per account, or the later result under a learned exit. Keep label_start, label_end, label_available_at, event/source/fidelity IDs and all leg arithmetic.

### CONTINUATION: continuation_incremental_net_r_v1
At the first target event, the baseline first five sell at the recorded target limit; only the remaining q=5 is discretionary. Store a feature snapshot through that event, independent of both future branches.

HOLD branch: leave the remaining five under the inherited entry-price stop or mandatory close.
CLOSE branch: retain all protection and close the remaining five at the first permissible strictly later ordered print after the checkpoint decision. Use event ordinal/sequence to disambiguate same-nanosecond prints. Do not use the trigger print retroactively. If an inherited stop/liquidation/deadline occurs first, its priority stands. In the market-only label shadow, there is no account liquidation; protective stop/deadline still applies.

y_continuation = (net value of remaining five in HOLD - net value of the same five in CLOSE) / R0.
Sunk entry cost and the realized first-half result are shared and cancel. Allocate remaining entry fee identically on both branches if shown; future exit fee is included once in each. Do not allocate the first half's secured profit to continuation. Entry-time and checkpoint-time features are distinct snapshots.

label_available_at is at least the later of the two branch resolutions and every source availability used in either label. Never train this label before both branches could be known. These are local market-price counterfactual labels; they do not capture subsequent account/slot effects. The FULL financial replay answers that separate question.

If exact checkpoint or later execution evidence cannot be established, mark the label unavailable. Do not substitute a whole trade's final MFE or the ordinary candle's finished high. The new voluntary close assumes zero computation delay but at least the next event; broker latency/liquidity remain unmodeled.

## 4. Features
The complete ordered definitions and lists are in contracts/FEATURE_DEFINITIONS.json and FEATURE_LISTS.json. These are fixed research fields, not claims all are already saved.

Map every field to its actual source/event/formula. Structural dimensions are from the owning setup; current clock, completed-price and vendor fields are at the job's decision. Required risk/deadline and event identity failures invalidate that prediction; legitimate optional missingness stays explicit and uses the frozen preprocessing. A programming failure is not legitimate missingness.

The current completed entry candle may supply an entry feature because its availability defines that entry decision under the inherited close-fill model. For a target reached mid-minute, only completed earlier bars and observed prints through the checkpoint are known. Never borrow the eventual high/low/close of that target candle.

No final trade outcome, eventual exit kind, future full-day trend/volatility class, realized future account loss, last day of a gamma run, source-end coverage, report path/ID, raw timestamp, account number, setup ID or configuration rank is a model input. Feature `mae_so_far_r` is only observed adverse movement through the checkpoint; final MAE/MFE remains label/audit data.

F0 excludes the dynamic allowance and all vendor-derived transforms. F1 adds range/allowance information; F2 adds gamma and levels. MCB062 is still a vendor-conditioned population because its rule already uses expected movement. Model missingness indicators belong only to the features in that cell, so F0 does not receive F2 vendor-availability flags.

Gamma source availability remains the inherited nominal 10 PM Chicago policy, independent clocks for levels and aggregate gamma, distinct-report positive age and declared staleness. Exact event ns stay exact; use canonical decision/source identity, not a rounded nearest-time join. Annotated historical context is acceptable only when causally reconstructible and separately marked; it must not be relabeled as an original executed context choice.

## 5. Grouping and training isolation
Keys include reference, stream, exact causal setup/activation, decision event/cursor, instrument/contract, schema and source identity. Account integer or timestamp alone is not a unique key. Preserve a cross-reference economic-episode key for detecting shared market events. A repeated HTF zone on a later independent activation is not automatically the same episode.

Each baseline trains separately. All features/models within a reference/job use identical labels and outer test row IDs. Purge any training label interval or exact setup episode overlapping the test partition and require label_available_at <= frozen training cutoff. Do not move a test event into training to make a fold fit.

The training population is the predetermined reference shadow, not each learned policy's selected trades. Later walk-forward fits may incorporate only earlier shadow labels that have resolved by their cutoff. New on-policy economic decisions are scored online but are not inserted retrospectively into prior training. Report new-event and out-of-training-range feature frequencies.

Do not train a market model on funded survivors alone. Funded records are retained for account evaluation, not used as duplicate outcome labels.

## 6. Dataset outputs
One versioned dataset per reference/job, with label components and feature availability/provenance; observation status/coverage table; exact schema; source and policy identities; before/after feature-cache isolation checks. Prefer bounded columnar artifacts, with compact CSV/JSON review copies. Full decision rows for this small research sample belong in the review delivery; raw full-market tape does not.
