# Chronological fit protocol
## Schedule (fixed before fitting)
contracts/DATE_AND_FOLD_PLAN.json contains the exact inherited dates and nine test blocks. Indices below are zero-based:
- Initial training date candidates E[0:80], ending October 6, 2025.
- Separation E[80:82], October 7–8, 2025.
- First test E[82:102], October 9–November 5, 2025.
- Each next test begins 20 evaluated dates later; the last block has 11 dates.
- Total scored dates: 171, October 9, 2025–June 10, 2026.

Training candidates expand from E[0] through the date TWO evaluated dates before the test opens. Labels and event/episode overlaps are then purged. The gap constrains fitting, not trading: economic test blocks are contiguous, and a later fold's two separation dates were already evaluated under its previous fold. Do not remove two trading dates at every model switch or skip zero-entry dates.

Fit cutoff is the scheduled session close of the final training date; no unresolved label is eligible. The fitted model becomes available at the first scored session open under a disclosed offline/zero historical training-latency assumption. Models cannot backfill predictions before their activation. Respect DST, early closes and original known missing dates.

The original ten warmup dates initialize market state only and supply no training labels or economic profits. Feature calculations may use authorized historical sources available by each decision. Exact dates are not regenerated with a generic business-day function.

## Fit validity
Read MODEL_PROTOCOL.json. Minimum entry training population is 30 labels across 15 dates; continuation 20 labels across 10 dates. These are declared engineering minima, NOT proof of statistical adequacy. Require finite labels and nonconstant training target. Regression does not require both binary classes.

Keep insufficient folds and error reasons. An unavailable fold uses the unchanged baseline in economic replay; no invented score, pooled later data, shifted boundaries or silently substituted model. Missing whole feature groups must be exposed and cannot be claimed as evidence those features were useful.

## Models and preprocessing
Fixed Ridge and CatBoost parameters are in MODEL_PROTOCOL.json. Resolve versions and exact imported paths locally before fit; use the same environment and definitions for offline and replay features. Do not upgrade production to match a website. A task-only compatible dependency installation is allowed if necessary; record it.

Numeric imputers/scalers and CatBoost quantization are trained ONLY on the eligible training partition. Categorical domains are predeclared structural domains; missing/unseen tokens are distinct. No future vocabulary, labels or preprocessing statistics. Keep all-null/constant columns and their coverage explicit, not silently replace them with other indicators.

Do not use the outer test set as eval_set, early stopping, target transform selection, feature selection, calibration fitting or threshold tuning. No model/seed search. Targets and predictions are not clipped; outliers remain economic observations. Record all resolved parameters and actual feature order.

A mean predictor for each reference/job/fold uses the SAME resolved training rows and labels. It is a predictive benchmark only; unchanged IFSM is the funded control. The learned score acts at fixed zero without calibration or percentile search.

## Diagnostics
For every valid fold and cell report row/date counts, censoring/coverage, RMSE, MAE, MSE relative to the training-mean predictor, signed mean prediction error, and score/outcome association when defined. Pooled skill = 1 - sum(model squared errors)/sum(training-mean squared errors), with a null reason when denominator is zero.

Give predicted<0 versus >=0 outcome groups, counts and after-cost R; any illustrative bins must be fixed in advance or fitted only on training data. Do not report AUC/Brier as metrics of these regressors. Confidence statements should use paired day blocks/setup clustering, not independent-trade normal errors.

Use a fixed seed 7 and 2,000 paired five-evaluated-date moving-block bootstrap repetitions for predictive-loss differences; keep empty dates and the same sampled blocks across compared cells. Label these development uncertainty summaries. Do not bootstrap shuffled cash events and present them as new valid funded paths. Chronological cash and per-period differences are observed histories, not model-independent forecast intervals.

For both tasks show the contribution of the largest baseline continuations and how many negative-score decisions target them. This is posthoc outcome diagnosis, not an oracle feature or a filter tuned on those winners. Do not remove large outcomes to improve error metrics.

## Identity and audit
Each fit binds task protocol, exact source, data/label/feature schema, row IDs, source-date policy, fold schedule, train cutoff, parameters, Python/library versions and seed. Every prediction carries model/fold/event/as-of identity, finite score or typed unavailability, and action semantics.

Old classifiers and M0–M3 keep their existing contracts; use a new return-regression protocol. The old 40/5-day, zero-cost static-barrier contract is NOT this phase's label/fold policy. Reuse implementation utilities after verifying compatibility, not their old scientific settings.
