"""Tests for the one-command Databento acceptance CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SESSION_SCOPE = {
    "training_sessions": ["asia", "london", "ny"],
    "evaluation_sessions": ["asia", "london", "ny"],
    "production_gate_sessions": ["ny"],
    "report_session_breakdowns": True,
}


def _load_acceptance_module():
    module_name = "_databento_acceptance_cli_under_test"
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "run_databento_acceptance.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_quant_focused_tests_include_acceptance_cli_regressions():
    cli = _load_acceptance_module()

    assert "tests/agents/test_databento_acceptance_cli.py" in cli.QUANT_FOCUSED_TESTS


def test_build_training_command_defaults_to_saved_research_approach_run():
    cli = _load_acceptance_module()
    args = cli.AcceptanceConfig(
        model_name="acceptance_model",
        start="2025-07-09",
        end="2025-08-15",
        train_days=5,
        test_days=2,
        gap_days=1,
        iterations=20,
        depth=4,
    )

    command = cli.build_training_command(args)

    assert command[:2] == [sys.executable, "scripts/run_dashboard_session_experiment.py"]
    assert "--include-approach-features" in command
    assert "--allow-failed-gates" in command
    assert command[command.index("--model-name") + 1] == "acceptance_model"
    assert command[command.index("--approach-window") + 1] == "90"


def test_audit_bundle_summarizes_fail_closed_artifact(tmp_path):
    cli = _load_acceptance_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.cbm").write_text("fake model")
    (model_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "n_samples": 2,
                "precision": 0.5,
                "recall": 0.25,
                "f1": 0.333,
                "roc_auc": 0.55,
                "brier_score": 0.30,
                "quality_gates": {
                    "all_passed": False,
                    "allow_failed_gates": True,
                    "gates": {
                        "Test samples >= 200": {
                            "passed": False,
                            "value": 2,
                            "threshold": 200,
                        },
                    },
                },
                "gated_oos": {"trade_count": 1, "precision": 1.0},
                "session_filter": {
                    "rows_before_session_filter": 3,
                    "rows_after_session_filter": 2,
                },
                "session_experiment": SESSION_SCOPE,
                "oos_predictions_file": "oos_predictions.parquet",
                "oos_predictions_rows": 2,
            },
        ),
    )
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "selected_features": ["int_time_beyond_level"],
                "session_experiment": SESSION_SCOPE,
            },
        ),
    )
    (model_dir / "strategy.json").write_text(
        json.dumps(
            {
                "contract_version": "trade_lab_contract_v2",
                "platform_version": "strategy_core_platform_v1",
                "strategy_id": "touch_reversal",
                "strategy_version": "1",
                "supported_by_runtime": True,
                "touch_rule": {"bar_type": "147t"},
                "label_policy": {
                    "decision_offset_minutes": 5,
                    "forward_bar_type": "147t",
                },
                "inference": {"eligible_session": "ny", "confidence_gate": 0.7},
                "research_session_experiment": SESSION_SCOPE,
            },
        ),
    )
    pd.DataFrame(
        {
            "session": ["ny", "london"],
            "label": ["tradeable_reversal", "trap_reversal"],
            "prob_tradeable_reversal": [0.8, 0.2],
            "gate_0_70_ny": [True, False],
        },
    ).to_parquet(model_dir / "oos_predictions.parquet")

    summary = cli.audit_bundle(model_dir, validate_contract=False)

    assert summary["artifact"]["required_files_present"] is True
    assert summary["quality_gates"]["all_passed"] is False
    assert summary["quality_gates"]["allow_failed_gates"] is True
    assert summary["runtime"]["supported_by_runtime"] is True
    assert summary["runtime"]["platform_version"] == "strategy_core_platform_v1"
    assert summary["runtime"]["strategy_id"] == "touch_reversal"
    assert summary["runtime"]["strategy_version"] == "1"
    assert summary["oos_predictions"]["rows"] == 2
    assert summary["oos_predictions"]["gate_true_counts"] == {"gate_0_70_ny": 1}


def test_audit_bundle_unservable_flag_is_rejected(tmp_path):
    # E2 flip: the audit now asserts the bundle IS servable; False fails closed.
    cli = _load_acceptance_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ["model.cbm", "metadata.json", "evaluation.json", "strategy.json"]:
        (model_dir / name).write_text("{}")
    pd.DataFrame({"prob_tradeable_reversal": [0.1]}).to_parquet(
        model_dir / "oos_predictions.parquet",
    )
    (model_dir / "strategy.json").write_text(
        json.dumps(
            {
                "platform_version": "strategy_core_platform_v1",
                "supported_by_runtime": False,
            },
        ),
    )

    with pytest.raises(ValueError, match="supported_by_runtime=true"):
        cli.audit_bundle(model_dir, validate_contract=False)
