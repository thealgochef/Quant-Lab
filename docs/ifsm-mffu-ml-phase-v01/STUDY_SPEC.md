# Phase-01 study specification
## Questions
1. Can features available at an otherwise-valid IFSM entry predict its fixed-partial-policy net return better than a training-only constant?
2. At the first 1R checkpoint, can those available features predict the incremental value of retaining the remaining half versus closing it?
3. Do the corresponding fixed actions improve modeled MyFundedFutures cash, rather than merely improve a trade statistic?
4. What is the incremental contribution of expected movement and gamma/levels beyond structure and clock?

The 24 cells are two references × two jobs × three feature sets × two models. Four training-mean prediction baselines share rows/folds by reference/job; two unchanged funded controls supply the matched economic comparison. There are at most 26 named scored-period financial operations, not 64 old configurations or pooled copy-trading. Invalid models may produce explicit baseline-fallback operations, but must not be called successful ML trials.

## Reference retention and comparison
Use the complete saved effective sections in references/. Preserve all fields not explicitly affected by the model decision. ID-like and inherited `planned_state` fields are audit metadata, not indicators of current completion.

The old full-period cash anchors ($48,102.16 for MCB062 and $39,102.26 for MCB025) are preservation checks only. They are NOT targets for the shorter scored-period model comparison. Reuse full-history references when compatibility is established; score-period controls usually require new initialization/replay because an October fresh account is not a slice of the earlier June-funded campaign.

Train each reference separately. Shared entries across references do not multiply independent market evidence. Predictive comparisons F1-F0, F2-F1 and CatBoost-Ridge use identical valid outer rows/labels within reference/job. Constant comparisons use the exact same eligible training and test rows. If validity differs, give the common-row comparison plus unmatched coverage; never quietly compare different favorable samples.

Economic comparisons use identical scored dates, prices, fee/firm rules and initial funded states. Compare every ML policy with its own unchanged control; model/feature contrasts also remain within the same reference/job. No direct pooled predictive metric from unlike target populations. No sum of alternative configurations' dollars.

## Feature-bundle names are not trading-policy names
F0/F1/F2 in this task are ML input bundles. They must NOT be mapped to the old batch's similarly named entry-admission filters. Every reference retains its original no-extra-gamma-rejection policy; only the explicitly declared learned action can alter admission or continuation. Switching an ML feature bundle must not directly change a strategy parameter.

## Selection and claims
Models are explanatory/scoring research, not an instruction to deploy. No hyperparameter, target, threshold, feature-subset or seed search. Preserve heavy-tail outcomes; no outcome-based deletion or label winsorization. Do not repeatedly extend training until a profitable model appears.

The original period and baseline have been inspected repeatedly. Predictions can be forward-in-time relative to each fit while the strategy-selection process remains development-exposed. Do not call this an untouched holdout, forecast payout probability or live certification. Positive and negative research findings both satisfy the research question when methods and evidence are sound.

Adaptive geometry versus fixed wider distance remains a future question. No fixed-30/40-point arms, new entry family, daily cap, gamma rule, leverage search, MLP, HMM, reinforcement learning or qscore is added to this first phase.
