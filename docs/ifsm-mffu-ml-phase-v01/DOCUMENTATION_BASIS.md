# Documentation basis — design, not measured performance
Checked for this handoff on October 8, 2026. Use compatible task-local installed library versions and freeze them; do not upgrade production merely because a `stable` documentation URL is newer.

- scikit-learn Ridge reference: regularized linear least squares, alpha and solver options.
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- scikit-learn common pitfalls: consistent train-only preprocessing and avoiding test information in fitting.
  https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn TimeSeriesSplit: temporal order and separation; this study splits an exact trading-date allowlist rather than treating irregular trades as equally spaced.
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- CatBoostRegressor reference: regression estimator, native categorical treatment and configurable training parameters.
  https://catboost.ai/docs/en/concepts/python-reference_catboostregressor

These sources support API/methodology choices, not the selected alpha, feature formulas, sample minima, exact folds, or profitability of an ML policy. Those are explicit authored research defaults in contracts/. Prior project facts come from references/ and their linked review scopes. No current live-firm term refresh or new market research was substituted for the owner's saved simulation profiles.
