"""QL-UI-PARITY P5 acceptance (headless).

Builds the D-036 dataset/config through the SAME constructor code path the UI
build/train buttons use (ml_training_tab.py UI build :2150-2168 equivalent,
train cfg :2314-2344 equivalent, with the D-036 widget values), resolves the
train kwargs via resolve_training_kwargs (the 5 contract features in contract
order, purged-days 40/5/5/2/30, RFECV checkbox False), trains on the warm
7850272e caches, saves via save_trained_model(allow_failed_gates=True) under a
temp models dir, and verifies against models/NQ_W3_20260617T220752Z:

REQUIRED: selected_features == the 5 in order; strategy.json feature_set
identical; evaluation.json fold dates (label_purge.folds) identical; OOS row
count == 42.
TARGET: model.cbm sha256 identical.

Evidence -> QLUI_ACCEPTANCE.log (QL root, untracked). On full pass the temp
bundle dir is deleted; on any failure it is kept for adjudication.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

QL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(QL_ROOT / "src"))

LOG_PATH = QL_ROOT / "QLUI_ACCEPTANCE.log"
REF_DIR = QL_ROOT / "models" / "NQ_W3_20260617T220752Z"
TMP_MODELS_DIR = QL_ROOT / "models_qlui_acceptance_tmp"
BUNDLE_NAME = "QLUI_ACCEPTANCE"

# The 5 contract features in contract order (NQ_W3_20260617T220752Z
# strategy.json feature_set.names).
PINS = [
    "int_time_within_2pts",
    "int_absorption_ratio",
    "app_avg_trade_size",
    "app_large_trade_vol_pct",
    "app_max_spread",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("qlui_acceptance")


def _load_ml_training_tab():
    module_name = "_ml_training_tab_acceptance"
    module_path = QL_ROOT / "scripts" / "ml_training_tab.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import subprocess

    import catboost
    import numpy
    import pandas as pd

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=QL_ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "-uno"],
        cwd=QL_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    log.info("QL HEAD: %s (tracked tree %s)", head, "DIRTY" if dirty else "clean")
    log.info(
        "versions: python=%s catboost=%s numpy=%s pandas=%s",
        sys.version.split()[0],
        catboost.__version__,
        numpy.__version__,
        pd.__version__,
    )

    tab = _load_ml_training_tab()
    from alpha_lab.agents.data_infra.ml.config import (
        DashboardUtilityConfig,
        MLPipelineConfig,
        ModelConfig,
        SessionExperimentConfig,
        WalkForwardConfig,
        session_experiment_from_preset,
    )
    from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
        build_utility_dataset,
    )

    # ── D-036 widget values ─────────────────────────────────────────
    ml_symbol = "NQ"
    data_dir = tab._DEFAULT_DATA_DIR
    ml_start, ml_end = date(2025, 11, 21), date(2026, 2, 13)
    ml_tp, ml_sl = 15, 15
    ml_bar_type = "147t"
    ml_int_window = 5
    ml_approach, ml_approach_window = True, 15
    ml_iterations, ml_depth = 1000, 6
    ml_session_preset = "all_to_ny"

    # Walk-forward sliders left at UI defaults: replicate the UI derivation
    # (span of the FULL available store, not the selected window).
    _avail = tab.get_available_dates(ml_symbol, data_dir)
    _first, _last = date.fromisoformat(_avail[0]), date.fromisoformat(_avail[-1])
    _span = (_last - _first).days
    ml_train_days = min(30, max(5, _span // 4))
    ml_test_days = min(7, max(2, _span // 12))
    ml_gap_days = 1
    log.info(
        "store range %s..%s span=%dd -> UI-default walk_forward train/test/gap = %d/%d/%d",
        _avail[0], _avail[-1], _span, ml_train_days, ml_test_days, ml_gap_days,
    )

    preset_scope = session_experiment_from_preset(ml_session_preset)
    session_experiment = SessionExperimentConfig(
        training_sessions=preset_scope.training_sessions,
        evaluation_sessions=preset_scope.evaluation_sessions,
        production_gate_sessions=preset_scope.production_gate_sessions,
        report_session_breakdowns=preset_scope.report_session_breakdowns,
    )

    # ── UI build config (:2150-2168 equivalent) ─────────────────────
    build_config = MLPipelineConfig(
        training_mode="dashboard_utility",
        dashboard_utility=DashboardUtilityConfig(
            tp_points=float(ml_tp),
            sl_points=float(ml_sl),
            bar_type=ml_bar_type,
            interaction_window_minutes=ml_int_window,
            include_approach_features=ml_approach,
            approach_window_minutes=ml_approach_window,
        ),
        session_experiment=session_experiment,
        tick_size=0.25,
        instrument=ml_symbol,
    )
    cache_tag = build_config.dataset_config_hash()
    log.info("build config cache tag: %s (expect 7850272e)", cache_tag)
    if cache_tag != "7850272e":
        log.error("REQUIRED FAIL (precondition): cache tag %s != 7850272e", cache_tag)
        return 2

    dates_in_range = [
        d for d in _avail if ml_start.isoformat() <= d <= ml_end.isoformat()
    ]
    log.info(
        "dates in range: %d (%s..%s)",
        len(dates_in_range), dates_in_range[0], dates_in_range[-1],
    )

    dataset = build_utility_dataset(
        dates_in_range,
        data_dir,
        build_config,
        progress_fn=lambda frac, text: log.info("build %.0f%% %s", frac * 100, text),
    )
    log.info("dataset built: %d rows, %d cols", len(dataset), len(dataset.columns))
    feature_universe = [c for c in dataset.columns if c.startswith(("int_", "app_"))]
    log.info("feature universe: %d features", len(feature_universe))

    # ── resolve kwargs (P1 seam, exactly as the UI train button) ────
    train_kwargs = tab.resolve_training_kwargs(
        PINS,
        "purged-days",
        {
            "train_days": 40,
            "test_days": 5,
            "step_days": 5,
            "purge_days": 2,
            "min_train_events": 30,
        },
        False,
    )
    log.info("resolved train kwargs: %s", train_kwargs)

    # ── UI train config (:2314-2344 equivalent) ─────────────────────
    train_config = MLPipelineConfig(
        training_mode="dashboard_utility",
        dashboard_utility=DashboardUtilityConfig(
            tp_points=float(ml_tp),
            sl_points=float(ml_sl),
            bar_type=ml_bar_type,
            interaction_window_minutes=ml_int_window,
            include_approach_features=ml_approach,
            approach_window_minutes=ml_approach_window,
        ),
        walk_forward=WalkForwardConfig(
            train_days=ml_train_days,
            test_days=ml_test_days,
            gap_days=ml_gap_days,
        ),
        model=ModelConfig(
            iterations=ml_iterations,
            depth=ml_depth,
            rfecv_enabled=train_kwargs["rfecv_enabled"],
            loss_function="MultiClass",
        ),
        session_experiment=session_experiment,
        tick_size=0.25,
        instrument=ml_symbol,
    )

    result = tab.run_walk_forward_training(
        dataset,
        train_config,
        "label_encoded",
        day_folds=train_kwargs["day_folds"],
        pinned_features=train_kwargs["pinned_features"],
    )
    ev = result["eval_result"]
    log.info(
        "trained: n_valid_folds=%s n_skipped=%s selected_features=%s",
        result.get("n_valid_folds"),
        result.get("n_skipped_folds"),
        result["trained_model"].selected_features,
    )
    log.info("session_filter: %s", result.get("session_filter"))

    gates = tab.check_quality_gates(ev)
    log.info("quality gates all_passed=%s", gates["all_passed"])

    out_dir = TMP_MODELS_DIR / BUNDLE_NAME
    saved_path = tab.save_trained_model(
        result["trained_model"],
        ev,
        train_config,
        out_dir,
        training_result=result,
        dates_used=dates_in_range,
        allow_failed_gates=True,
    )
    log.info("saved bundle: %s", saved_path)

    # ── comparisons ─────────────────────────────────────────────────
    new_eval = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    ref_eval = json.loads((REF_DIR / "evaluation.json").read_text(encoding="utf-8"))
    new_strat = json.loads((out_dir / "strategy.json").read_text(encoding="utf-8"))
    ref_strat = json.loads((REF_DIR / "strategy.json").read_text(encoding="utf-8"))

    failures: list[str] = []

    # R1 — selected features == the 5 pins, in order (model + evaluation.json).
    got_model = list(result["trained_model"].selected_features)
    got_eval = list(new_eval.get("selected_features", []))
    if got_model != PINS:
        failures.append(f"R1 selected_features (model) {got_model} != {PINS}")
    if got_eval != PINS:
        failures.append(f"R1 selected_features (evaluation.json) {got_eval} != {PINS}")
    log.info("R1 selected_features == pins in order: %s", got_model == PINS and got_eval == PINS)

    # R2 — strategy.json feature_set identical.
    if new_strat.get("feature_set") != ref_strat.get("feature_set"):
        failures.append(
            f"R2 feature_set differs: new={new_strat.get('feature_set')} "
            f"ref={ref_strat.get('feature_set')}"
        )
    log.info(
        "R2 strategy.json feature_set identical: %s",
        new_strat.get("feature_set") == ref_strat.get("feature_set"),
    )

    # R3 — evaluation.json fold dates identical (label_purge.folds date fields;
    # this is where per-fold dates are persisted in the bundle).
    def fold_dates(ev_json):
        folds = (ev_json.get("label_purge") or {}).get("folds") or []
        return [
            {k: f.get(k) for k in ("fold", "test_start", "max_label_window_end")}
            for f in folds
        ]

    new_folds, ref_folds = fold_dates(new_eval), fold_dates(ref_eval)
    if new_folds != ref_folds:
        failures.append(f"R3 fold dates differ: new={new_folds} ref={ref_folds}")
    log.info("R3 fold dates identical: %s (%s)", new_folds == ref_folds, new_folds)
    if new_eval.get("fold_metrics") != ref_eval.get("fold_metrics"):
        log.warning(
            "INFO-only: fold_metrics differ: new=%s ref=%s",
            new_eval.get("fold_metrics"), ref_eval.get("fold_metrics"),
        )
    else:
        log.info("fold_metrics identical (informational)")

    # R4 — OOS row count == 42.
    new_oos = pd.read_parquet(out_dir / "oos_predictions.parquet")
    if len(new_oos) != 42:
        failures.append(f"R4 OOS row count {len(new_oos)} != 42")
    if new_eval.get("oos_predictions_rows") != 42:
        failures.append(
            f"R4 evaluation.json oos_predictions_rows {new_eval.get('oos_predictions_rows')} != 42"
        )
    log.info(
        "R4 OOS rows: parquet=%d eval_json=%s (expect 42)",
        len(new_oos), new_eval.get("oos_predictions_rows"),
    )

    if failures:
        for f in failures:
            log.error("REQUIRED FAIL: %s", f)
        log.error("VERDICT: REQUIRED FAILED — first divergence: %s", failures[0])
        log.error("temp bundle KEPT for adjudication: %s", out_dir)
        return 2

    log.info("VERDICT: all REQUIRED equalities PASS")

    # TARGET — model.cbm sha256 identical.
    new_sha = hashlib.sha256((out_dir / "model.cbm").read_bytes()).hexdigest()
    ref_sha = hashlib.sha256((REF_DIR / "model.cbm").read_bytes()).hexdigest()
    log.info("TARGET sha256: new=%s ref=%s", new_sha, ref_sha)
    if new_sha != ref_sha:
        new_meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        ref_meta = json.loads((REF_DIR / "metadata.json").read_text(encoding="utf-8"))
        new_imp = new_meta.get("feature_importances", {})
        ref_imp = ref_meta.get("feature_importances", {})
        first_diff = None
        for feat in PINS:
            if new_imp.get(feat) != ref_imp.get(feat):
                first_diff = (feat, new_imp.get(feat), ref_imp.get(feat))
                break
        log.error(
            "VERDICT: TARGET FAILED (REQUIRED passed) — cbm sha mismatch. "
            "First differing feature importance: %s (new=%s ref=%s). "
            "Nondeterminism vs wiring must be adjudicated at review.",
            *(first_diff if first_diff else ("NONE — importances identical", None, None)),
        )
        log.error("temp bundle KEPT for adjudication: %s", out_dir)
        return 3

    log.info("VERDICT: TARGET PASS — model.cbm sha256 identical")
    shutil.rmtree(TMP_MODELS_DIR)
    log.info("temp bundle dir deleted: %s", TMP_MODELS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
