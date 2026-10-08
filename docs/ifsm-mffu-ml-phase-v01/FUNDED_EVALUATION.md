# Matched chronological MyFundedFutures evaluation
## 1. Operations and initialization
Evaluate each of the 24 declared learned policies, plus NO_ML_MCB062 and NO_ML_MCB025, over the same 171 scored dates. A model with some invalid folds still has an honest fallback operation; if all folds are invalid, identify it as baseline-fallback-only, not a successful ML model. Deduplicate computation only when identical economic behavior is proved; retain all named outcomes and fallback coverage.

For each reference, causally run its fixed market-only shadow through the ten warmup dates and all pre-score history. At the scored boundary, clone its canonical STRATEGY/MARKET state into every policy of that reference and its unchanged control. There must be no open shadow position carried through the previous mandatory close. Preserve surviving setup, HTF registry, locks/clocks and pending completed-bar state; do not reset a setup just to make the fit boundary convenient.

Create exactly ONE fresh funded account at the scheduled session open for trading date October 9, 2025 (normally Oct 8 at 5 PM Chicago), with $50,000 nominal / zero earned balance, initial -$2,000 relative floor, no inherited payouts or wallet, and charge $125. Use the supported fresh-account initializer, retaining only the shared causal strategy/market state described above. Do not import June-funded gains, failures or account counters. Verify this seed once per reference and persist it; all its policies begin identically.

Then every operation carries its own full strategy, model, account and payout state continuously through June 10 cutoff. Daily/window/model changes do not reset balances or setups. A failure buys a fresh paid same-policy account using the inherited replacement rules; it does not reset the model schedule. No monthly credit constraint, growth, copy or cross-policy wallet.

Old full-year results are preservation anchors. They cannot be sliced into these fresh-account scored controls. Reuse a scored result only if source/seed/dates/effective policy/fees match exactly.

## 2. ML policy ENTRY
Ask the scorer once for an otherwise executable active fresh-continuation opportunity. Preserve strategy causality, nonpolicy geometry, session, account admission and existing invalidation priorities. Inactive/unratified retests and already-blocked diagnostics remain observational.

- finite prediction < 0: refuse that executable setup with explicit `ml_entry_negative_return`, no fill and no fee; consume/terminate the refused setup using the tested executable-refusal lifecycle, not a wait-for-a-later-better-score mechanism.
- prediction >= 0: use unchanged baseline entry sizing/fill and exit policy.
- legitimate unavailable model or required current-time feature: baseline action with typed fallback, no fabricated zero prediction.
- model/feature integrity error: fail the affected operation clearly; do not hide a bug as ordinary fallback.

No repeated candidates may score their way into a reset, extra quota, ratified retest or same-bar resurrection. Account fee/breach ordering remains inherited after actual admission. In particular, the new scorer does not borrow future knowledge that a forthcoming price move will liquidate the account.

## 3. ML policy CONTINUATION
All entries remain baseline. First five exit at the inherited 1R target fill and costs; the remaining five receive the normal entry-price protective stop.

At that checkpoint, build features only through the observed event. A finite score <0 elects a voluntary close of the remaining five at the FIRST permissible STRICTLY LATER ordered print. Preserve event order when equal-ns prints exist. No retrospective fill at the target's triggering price and no finalized developing-candle fields. No later rescoring of the same first checkpoint.

Protective stop, account failure and mandatory close stay active and keep the inherited priority. If any closes the remaining exposure before the voluntary fill, cancel the latter and charge the actual exit only once. At an approximated checkpoint whose causal event/feature ordering cannot be established, apply the declared baseline fallback based on that CURRENT limitation. Never use future tape-quality knowledge to decide whether to change the trade.

Prediction >=0 or legitimate unavailability retains the original remainder. No gamma-sign default, smaller leverage, additional trailing stop or discretionary profit-protection rule is introduced.

Pending close intent, exact triggering event, score, model identity, protective orders and account state must survive checkpoint/restart. Once flat, notify the normal strategy lifecycle so later setups respond to the actual changed occupancy. No trade-row deletion or replay of a fixed precomputed trade list is an adequate economic evaluation.

## 4. Actual inference at new opportunities
Policy changes can create or displace later entries. Compute live replay features and apply the model active at that chronological event even when its trade ID never occurred in the training/reference dataset. Store source/version and training-range diagnostics. Do not treat "not in saved prediction CSV" as no signal.

Each model is retrained only on earlier eligible fixed-shadow labels. All cells' training sets remain predefined and independent of their own economic selection. Do not let the model's later winners choose its next training sample.

The economic test span is contiguous. Freeze each model before its block; switch at the defined session boundary without resetting the strategy/account. Every event must prove its score came from an earlier-only fit. Model preparation may be performed offline in bulk, but later block models must never be used by earlier replay events.

## 5. Economics and uncertainty
Keep actual per-fill cents, account-floor comparators and EOD ratchet, request/receipt clock, processing pause, early-close calendar and NQ-price proxy. A model still must flatten by the mandatory deadline. Pending receipts beyond cutoff are not cash. A no-trade campaign still paid for its initial funded account; do not show a free zero-cost account.

Produce actual trades, fills, accounts, losses, payouts and cash records. Primary objective = received trader share minus every acquisition cost. Keep all unfavorable operations, fallback fractions and approximated events visible.

Report initial wait, later receipt gaps and trailing waiting at cutoff separately; net cash, large receipts, account usage, time to flat +$2,000, entry coverage and average activity use the same saved-record definitions as the repaired lenses, but the denominator/start must be THIS 171-date operation.

Compare every policy with its matched no-ML reference and show predictive versus economic contribution separately. A local continuation label ignores later setup/account effects; the operation measures them. Do not describe a subset label gain as payout gain or sum separate hypothetical campaigns.

## 6. Reuse, completion and publication
Prove model-disabled and always-baseline scoring do not alter applicable reference event transitions or transaction fields. Compatible cached sources/shadows/control campaigns are reusable; changing feature-only/reporting code need not rerun all economics when nonimpact is recorded. Recovered jobs validate identity and resume; a failed chart export must not relaunch a completed worker.

The new experiment is stored separately and published in ordinary My studies. Research source isolation must not recreate a special-port viewer. The production pin and every old result remain unchanged.
