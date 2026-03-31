# Phase 3 — Model Inference & Signal Generation

**Scope:** CatBoost model loading, real-time inference, prediction output, outcome tracking
**Prerequisite:** Phase 2 complete (level detection, observation windows, feature computation)
**Produces:** A prediction engine that runs CatBoost inference on completed observation windows and tracks prediction outcomes
**Does NOT include:** Paper trading, API server, frontend

---

## 1. What This Phase Builds

**1. Model Manager** — Loads CatBoost model files, manages model versions (active, historical), handles model uploads and rollbacks. Only one model is active at a time.

**2. Prediction Engine** — Receives completed observation windows from Phase 2, runs CatBoost inference, produces predictions with class probabilities. Determines whether predictions should trigger auto-execution (NY RTH only).

**3. Outcome Tracker** — Monitors price after a prediction to determine whether the prediction was correct. Tracks MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion) for each prediction event. Updates resolved prediction markers for chart display.

---

## 2. Model Details

**Model type:** CatBoost 3-class classifier
**Classes:** `tradeable_reversal`, `trap_reversal`, `aggressive_blowthrough`
**Input features:** 3 floats — `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`
**Output:** Predicted class + probability for each class
**Model file format:** CatBoost native format (`.cbm` file saved via `model.save_model()`)

**Performance from experiment (top-3 feature model):**
- Reversal precision: 86.1%
- Blowthrough recall: 71.4%
- Cross-fold accuracy std: 0.04 (very stable)
- Pooled accuracy: 53.4% (3-class)

---

## 3. Relevant PRD Sections

- **Section 6** (Model Behavior) — Prediction classes, observation window, prediction display scope, historical replay, resolved outcomes
- **Section 8** (Auto-Execution) — Trigger conditions (reversal during NY RTH), account groups
- **Section 11** (Dashboard Layout) — Model Management tab
- **Section 14** (Edge Cases) — Model upload while trades open

---

## 4. Directory Structure (Phase 3 additions)

```
src/alpha_lab/dashboard/
  model/
    __init__.py
    model_manager.py         # Model loading, versioning, activation
    prediction_engine.py     # CatBoost inference on observation results
    outcome_tracker.py       # MFE/MAE tracking, prediction resolution

tests/dashboard/
  model/
    __init__.py
    test_model_manager.py
    test_prediction_engine.py
    test_outcome_tracker.py
```

---

## 5. Component Specifications

### 5.1 Model Manager (`model_manager.py`)

**Purpose:** Manage CatBoost model files, versions, and activation.

**Interface:**
```python
class ModelManager:
    """Manages CatBoost model lifecycle — loading, versioning, activation.

    Stores model files on disk and metadata in PostgreSQL. Only one model
    is active at a time. Supports upload of new models with a confirmation
    step (review metrics before activating) and rollback to prior versions.
    """

    def __init__(self, model_dir: Path, db_session) -> None

    async def load_active_model(self) -> CatBoostClassifier | None
        """Load the currently active model from disk. Returns None if
        no model has been activated yet."""

    async def upload_model(self, file_path: Path, metrics: dict | None = None) -> ModelVersion
        """Upload a new model file. Does NOT activate it.
        Copies file to model_dir, creates DB record."""

    async def activate_model(self, version_id: int) -> None
        """Activate a model version. Deactivates any previously active model.
        Fails if trades are currently open (defer to next flat period)."""

    async def rollback(self, version_id: int) -> None
        """Rollback to a specific previous model version."""

    async def get_active_version(self) -> ModelVersion | None
    async def get_all_versions(self) -> list[ModelVersion]

    @property
    def model(self) -> CatBoostClassifier | None
        """The currently loaded model instance."""
```

**Model storage:**
```
data/models/
  v1_2026-03-05.cbm          # Model files named by version + date
  v2_2026-03-15.cbm
  ...
```

**Model version metadata** (stored in `model_versions` table from Phase 1):
- version label
- file path
- is_active flag
- metrics (accuracy, precision, fold results — JSONB)
- uploaded_at, activated_at timestamps

**Activation constraint:** A new model cannot activate while paper trades are open. The system defers activation until all positions are flat.

### 5.2 Prediction Engine (`prediction_engine.py`)

**Purpose:** Run CatBoost inference on completed observation windows.

**Interface:**
```python
class PredictionEngine:
    """Runs CatBoost inference on observation window features.

    Receives completed ObservationWindow objects from the observation
    manager, extracts the 3 features, runs model.predict() and
    model.predict_proba(), and produces a Prediction object.
    """

    def __init__(self, model_manager: ModelManager) -> None

    async def predict(self, observation: ObservationWindow) -> Prediction | None
        """Run inference on a completed observation.
        Returns None if no model is loaded."""

    def on_prediction(self, callback: Callable[[Prediction], None]) -> None
        """Register callback for new predictions."""

@dataclass
class Prediction:
    event_id: str                       # Links back to TouchEvent
    timestamp: datetime                 # When prediction was made
    observation: ObservationWindow      # Full observation context
    predicted_class: str                # 'tradeable_reversal', 'trap_reversal', 'aggressive_blowthrough'
    probabilities: dict[str, float]     # {'tradeable_reversal': 0.85, 'trap_reversal': 0.10, ...}
    features: dict[str, float]          # The 3 input features
    is_executable: bool                 # True only if reversal AND during NY RTH
    trade_direction: TradeDirection     # LONG or SHORT
    level_price: Decimal                # The level that was touched
    model_version: str                  # Which model version made this prediction
```

**Execution eligibility:**
```
is_executable = (
    predicted_class == "tradeable_reversal"
    AND current_session == "ny_rth"  # 09:30–16:15 ET
    AND current_time < 15:49 ET      # Before observation cutoff
)
```

**Prediction display:** Predictions are generated for ALL sessions (Asia, London, Pre-market, NY RTH). Only reversal predictions during NY RTH are flagged as executable for paper trading (Phase 4).

**Feature array construction:**
```python
features = np.array([[
    observation.features["int_time_beyond_level"],
    observation.features["int_time_within_2pts"],
    observation.features["int_absorption_ratio"],
]])
predicted_class = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]
```

### 5.3 Outcome Tracker (`outcome_tracker.py`)

**Purpose:** Track price after predictions to determine if they were correct.

**Interface:**
```python
class OutcomeTracker:
    """Tracks prediction outcomes by monitoring price after signals.

    For each prediction, monitors MFE (how far price moved in the
    predicted reversal direction) and MAE (how far against). Resolves
    predictions as correct or incorrect based on defined thresholds.
    Updates the observation_events table with resolution data.
    """

    def __init__(self, db_session) -> None

    def start_tracking(self, prediction: Prediction) -> None
        """Begin tracking price for this prediction."""

    def on_trade(self, trade: TradeUpdate) -> list[ResolvedOutcome]
        """Process a trade against all active trackers.
        Returns list of any newly resolved outcomes."""

    def on_session_end(self) -> list[ResolvedOutcome]
        """Resolve all remaining unresolved predictions at session end."""

    def on_outcome_resolved(self, callback: Callable[[ResolvedOutcome], None]) -> None
        """Register callback for resolved outcomes."""

    @property
    def active_trackers(self) -> int

@dataclass
class ResolvedOutcome:
    event_id: str
    prediction: Prediction
    mfe_points: float               # Maximum favorable excursion
    mae_points: float               # Maximum adverse excursion
    resolution_type: str            # 'tp_hit', 'sl_hit', 'session_end', 'flatten'
    prediction_correct: bool        # Did the predicted class match reality?
    actual_class: str               # What actually happened
    resolved_at: datetime
```

**Resolution logic:**

Outcome tracking uses the same thresholds as the experiment's labeling:

```
For each active prediction, track running MFE and MAE from level_price:

For LONG predictions:
    mfe = max(trade.price - level_price) for all trades since prediction
    mae = max(level_price - trade.price) for all trades since prediction

For SHORT predictions:
    mfe = max(level_price - trade.price)
    mae = max(trade.price - level_price)

Resolution:
    IF mfe >= 25.0 points:
        actual_class = "tradeable_reversal"
        prediction_correct = (predicted_class == "tradeable_reversal")
        resolution_type = "tp_hit"

    ELSE IF mae >= 37.5 points:
        IF mfe >= 5.0:
            actual_class = "trap_reversal"
        ELSE:
            actual_class = "aggressive_blowthrough"
        prediction_correct = (predicted_class == actual_class)
        resolution_type = "sl_hit"

    On session end or hard flatten:
        Resolve based on current mfe/mae using same logic
        resolution_type = "session_end" or "flatten"
```

**Note on 25-point vs 10-point threshold:** The experiment used 10 points as the reversal threshold for labeling. For outcome tracking on the dashboard, we use 25 points because that matches the Group B take profit (the larger target). The chart markers show ✓/✗ based on whether the 25-point reversal target was reachable. This can be made configurable later but 25 points is the default — it represents a meaningful, tradeable reversal rather than a minor bounce.

**MFE/MAE update:** Store running MFE/MAE in the `observation_events` table, updated on each trade. This allows the dashboard to show live P&L tracking for each prediction.

---

## 6. PostgreSQL Updates (Phase 3)

No new tables needed — Phase 2 already created `observation_events` with prediction and outcome columns. Phase 3 populates:
- `prediction`, `prediction_probabilities` — filled when prediction fires
- `outcome_resolved`, `outcome_correct`, `mfe_points`, `mae_points` — filled when outcome resolves

---

## 7. Test Specifications

### 7.1 `test_model_manager.py` (9 tests)

```
"""
Phase 3 — Model Manager Tests

Tests CatBoost model loading, versioning, and activation. The model
manager ensures only one model is active, supports rollback, and
prevents model swaps while trades are open.

Business context: The trader can upload new model versions as they
retrain on accumulating data. Bad model versions can be rolled back
immediately. Model activation is deferred if positions are open to
avoid mid-trade model changes.
"""
```

1. `test_upload_model` — Uploading a model file creates a DB record and copies the file
2. `test_activate_model` — Activating a model sets is_active=True and deactivates others
3. `test_load_active_model` — `load_active_model()` returns the correct CatBoostClassifier
4. `test_no_active_model_returns_none` — Before any activation, returns None
5. `test_rollback` — Rolling back activates a previous version
6. `test_only_one_active` — After activation, exactly one model has is_active=True
7. `test_get_all_versions` — Returns all uploaded versions with metadata
8. `test_model_metrics_stored` — Metrics dict is stored in JSONB and retrievable
9. `test_model_file_persisted` — Model file exists at the expected path after upload

### 7.2 `test_prediction_engine.py` (10 tests)

```
"""
Phase 3 — Prediction Engine Tests

Tests CatBoost inference on observation window features. The prediction
engine is the bridge between feature computation and trading decisions.

Business context: When the observation window completes, the model has
~0ms to produce a prediction (no latency requirement, but should be
near-instant). The prediction determines whether paper trades fire on
all 5 simulated accounts. 86.1% reversal precision means roughly 1 in
7 reversal calls is wrong — the paper trading system must handle this.
"""
```

1. `test_predict_returns_prediction` — Valid observation with loaded model produces a Prediction
2. `test_predict_no_model_returns_none` — No active model returns None, not an error
3. `test_prediction_class_is_valid` — Predicted class is one of the 3 valid classes
4. `test_probabilities_sum_to_one` — Class probabilities sum to ~1.0
5. `test_features_passed_correctly` — The 3 features from the observation appear in the Prediction
6. `test_executable_reversal_during_rth` — Reversal during NY RTH has is_executable=True
7. `test_not_executable_reversal_outside_rth` — Reversal during London session has is_executable=False
8. `test_not_executable_trap` — Trap prediction during RTH has is_executable=False
9. `test_not_executable_blowthrough` — Blowthrough during RTH has is_executable=False
10. `test_callback_fires_on_prediction` — Registered callback receives the Prediction

### 7.3 `test_outcome_tracker.py` (12 tests)

```
"""
Phase 3 — Outcome Tracker Tests

Tests prediction outcome resolution based on price movement after
signals. The outcome tracker determines if predictions were correct
by monitoring MFE and MAE thresholds.

Business context: Outcome tracking enables the resolved prediction
markers on the chart (✓ or ✗ border) and feeds the performance
analytics in the Analysis tab. Accurate outcome tracking is essential
for evaluating model performance in live conditions.
"""
```

1. `test_mfe_tracking_long` — LONG prediction correctly tracks max favorable (price above level)
2. `test_mfe_tracking_short` — SHORT prediction correctly tracks max favorable (price below level)
3. `test_mae_tracking` — Correctly tracks max adverse excursion
4. `test_resolve_reversal_on_tp` — MFE >= 25 points resolves as tradeable_reversal
5. `test_resolve_blowthrough_on_sl` — MAE >= 37.5 with MFE < 5 resolves as blowthrough
6. `test_resolve_trap_on_sl` — MAE >= 37.5 with MFE >= 5 resolves as trap
7. `test_prediction_correct_true` — Predicted reversal that resolves as reversal is correct=True
8. `test_prediction_correct_false` — Predicted reversal that resolves as blowthrough is correct=False
9. `test_session_end_resolves_all` — Unresolved predictions resolve at session end
10. `test_db_updated_on_resolve` — observation_events table updated with mfe, mae, outcome
11. `test_callback_fires_on_resolve` — Registered callback receives ResolvedOutcome
12. `test_multiple_concurrent_trackers` — Multiple predictions tracked independently

---

## 8. Integration with Phase 2

```python
# Wire prediction engine into the observation completion callback:
observation_manager.on_observation_complete(prediction_engine.predict)
prediction_engine.on_prediction(outcome_tracker.start_tracking)

# Wire outcome tracker into the tick stream:
pipeline.register_trade_handler(outcome_tracker.on_trade)
```

**Data flow:**
```
Observation Complete → Prediction Engine → Prediction
                                        ↓
                                    Outcome Tracker ← live trades
                                        ↓
                                    ResolvedOutcome → DB update + [Phase 4]
```

---

## 9. Acceptance Criteria

Phase 3 is complete when:

1. **Model loads and predicts** — A CatBoost .cbm file loads successfully and produces predictions on feature inputs
2. **Predictions fire on observation completion** — Every completed observation produces a prediction (or None if no model loaded)
3. **Execution eligibility is correct** — Only reversal predictions during NY RTH are flagged executable
4. **Outcome tracking resolves correctly** — MFE/MAE thresholds produce correct actual_class labels
5. **DB records updated** — Predictions and outcomes are stored in observation_events table
6. **All tests pass** — Every test in Section 7 passes

---

## 10. Notes for Claude Code

- **CatBoost is already a project dependency.** Check `pyproject.toml` — it should be installed.
- **The model file comes from the experiment.** The trained model from the hypothesis test should be saved as a `.cbm` file and placed in `data/models/`. If it doesn't exist yet, create a test fixture model for development.
- **Prediction latency is not a concern.** CatBoost inference on 3 features is < 1ms. No optimization needed.
- **The outcome tracker runs continuously.** It processes every trade tick against all active (unresolved) predictions. With typically 0-3 active predictions, this is negligible load.
- **Don't conflate resolution thresholds with paper trading TP/SL.** The outcome tracker uses 25pt MFE / 37.5pt MAE for prediction resolution. Paper trading (Phase 4) uses configurable TP/SL per account group. These are separate systems.
