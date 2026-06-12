#!/usr/bin/env python3
"""Run dashboard-utility ML session-scope experiments from the CLI.

This is the non-Streamlit path for the same configuration surfaced in the ML
Training Workbench. Saved bundles carry ``supported_by_runtime=true`` since E2
(Trade-Lab is repointed onto Strategy-Core and gates activation itself).

W3a riders: ``--fold-scheme purged-days`` swaps the calendar-day
``WalkForwardSplitter`` for the purged TRADING-day scheme
(train/test/step/purge days + min-train-events, the train_dashboard_model
fold logic); ``--pin-features`` trains/serves an exact feature list
(disables RFECV selection).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ml_training_tab import (
    _DEFAULT_DATA_DIR,
    _DEFAULT_MODEL_DIR,
    get_available_dates,
    run_walk_forward_training,
    save_trained_model,
)

from alpha_lab.agents.data_infra.ml.config import (
    SESSION_EXPERIMENT_PRESETS,
    DashboardUtilityConfig,
    MLPipelineConfig,
    ModelConfig,
    SessionExperimentConfig,
    WalkForwardConfig,
    session_experiment_from_preset,
)
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import build_utility_dataset


def _parse_sessions(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _date_slice(available: list[str], start: str | None, end: str | None) -> list[str]:
    if not available:
        return []
    start_value = start or available[0]
    end_value = end or available[-1]
    return [d for d in available if start_value <= d <= end_value]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a dashboard-utility session experiment on local Databento parquet.",
    )
    parser.add_argument("--preset", choices=sorted(SESSION_EXPERIMENT_PRESETS), default="all_to_ny")
    parser.add_argument("--train-sessions", help="Comma-separated override, e.g. ny or asia,london")
    parser.add_argument("--evaluation-sessions", help="Comma-separated override")
    parser.add_argument("--gate-sessions", help="Comma-separated confidence-gate override")
    parser.add_argument("--symbol", default="NQ")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--bar-type", default="147t", choices=["147t", "987t", "2000t", "1m"])
    parser.add_argument("--tp", type=float, default=15.0)
    parser.add_argument("--sl", type=float, default=30.0)
    parser.add_argument("--interaction-window", type=int, default=5)
    parser.add_argument("--include-approach-features", action="store_true")
    parser.add_argument("--approach-window", type=int, default=90)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument(
        "--fold-scheme",
        choices=["calendar", "purged-days"],
        default="calendar",
        help=(
            "calendar = WalkForwardSplitter over --train/test/gap-days; "
            "purged-days = purged TRADING-day folds over --fold-*-days (W3a)"
        ),
    )
    parser.add_argument("--fold-train-days", type=int, default=40)
    parser.add_argument("--fold-test-days", type=int, default=5)
    parser.add_argument("--fold-step-days", type=int, default=5)
    parser.add_argument("--fold-purge-days", type=int, default=2)
    parser.add_argument("--min-train-events", type=int, default=30)
    parser.add_argument(
        "--pin-features",
        help=(
            "Comma-separated EXACT feature list to train/serve "
            "(disables RFECV selection)"
        ),
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--rfecv", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save a model bundle after training")
    parser.add_argument("--allow-failed-gates", action="store_true")
    parser.add_argument("--model-name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config/date range without building/training",
    )
    return parser


def _resolve_session_scope(args: argparse.Namespace) -> SessionExperimentConfig:
    preset = session_experiment_from_preset(args.preset)
    payload = preset.model_dump()
    train_override = _parse_sessions(args.train_sessions)
    eval_override = _parse_sessions(args.evaluation_sessions)
    gate_override = _parse_sessions(args.gate_sessions)
    if train_override is not None:
        payload["training_sessions"] = train_override
    if eval_override is not None:
        payload["evaluation_sessions"] = eval_override
    if gate_override is not None:
        payload["production_gate_sessions"] = gate_override
    return SessionExperimentConfig(**payload)


def _resolve_config(args: argparse.Namespace) -> MLPipelineConfig:
    return MLPipelineConfig(
        training_mode="dashboard_utility",
        dashboard_utility=DashboardUtilityConfig(
            tp_points=args.tp,
            sl_points=args.sl,
            bar_type=args.bar_type,
            interaction_window_minutes=args.interaction_window,
            include_approach_features=args.include_approach_features,
            approach_window_minutes=args.approach_window,
        ),
        session_experiment=_resolve_session_scope(args),
        walk_forward=WalkForwardConfig(
            train_days=args.train_days,
            test_days=args.test_days,
            gap_days=args.gap_days,
        ),
        model=ModelConfig(
            iterations=args.iterations,
            depth=args.depth,
            rfecv_enabled=args.rfecv,
            loss_function="MultiClass",
        ),
        tick_size=0.25,
        instrument=args.symbol,
    )


def main() -> int:
    args = _build_parser().parse_args()
    config = _resolve_config(args)
    available = get_available_dates(args.symbol, args.data_dir)
    dates = _date_slice(available, args.start, args.end)
    day_folds = None
    if args.fold_scheme == "purged-days":
        day_folds = {
            "train_days": args.fold_train_days,
            "test_days": args.fold_test_days,
            "step_days": args.fold_step_days,
            "purge_days": args.fold_purge_days,
            "min_train_events": args.min_train_events,
        }
    pinned_features = _parse_sessions(args.pin_features)
    summary = {
        "phase": "resolved_config",
        "symbol": args.symbol,
        "data_dir": str(args.data_dir),
        "available_dates": len(available),
        "selected_dates": len(dates),
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "session_experiment": config.session_experiment.model_dump(),
        "bar_type": config.dashboard_utility.bar_type,
        "include_approach_features": config.dashboard_utility.include_approach_features,
        "approach_window_minutes": config.dashboard_utility.approach_window_minutes,
        "fold_scheme": args.fold_scheme,
        "day_folds": day_folds,
        "pinned_features": pinned_features,
        "walk_forward": config.walk_forward.model_dump(),
        "model": config.model.model_dump(),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0
    if not dates:
        raise SystemExit("No selected dates available; check --data-dir/--symbol/--start/--end")

    dataset = build_utility_dataset(dates, args.data_dir, config)
    if dataset.empty:
        raise SystemExit("No touch events detected for selected dates")

    result = run_walk_forward_training(
        dataset,
        config,
        "label_encoded",
        day_folds=day_folds,
        pinned_features=pinned_features,
    )
    ev = result["eval_result"]
    report = {
        "phase": "trained",
        "n_total": result.get("n_total"),
        "n_training_samples": result.get("n_training_samples"),
        "n_evaluation_candidate_samples": result.get("n_evaluation_candidate_samples"),
        "n_valid_folds": result.get("n_valid_folds"),
        "n_skipped_folds": result.get("n_skipped_folds"),
        "precision": ev.precision,
        "recall": ev.recall,
        "f1": ev.f1,
        "roc_auc": ev.roc_auc,
        "brier_score": ev.brier_score,
        "gated_oos": result.get("gated_oos"),
        "session_filter": result.get("session_filter"),
    }
    print(json.dumps(report, indent=2, default=str))

    if args.save:
        model_name = args.model_name or (
            f"{args.symbol}_{args.preset}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        output_dir = _DEFAULT_MODEL_DIR / model_name
        saved = save_trained_model(
            result["trained_model"],
            ev,
            config,
            output_dir,
            training_result=result,
            dates_used=dates,
            allow_failed_gates=args.allow_failed_gates,
        )
        print(json.dumps({"phase": "saved", "path": str(saved)}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
