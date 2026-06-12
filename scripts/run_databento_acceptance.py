#!/usr/bin/env python3
"""One-command Databento/Quant-Lab/Strategy-Core acceptance check.

W2 P3c: THE W3 ACCEPTANCE ENTRYPOINT. The W3 greenlight runs a fresh
train -> emit -> discover -> activate -> replay acceptance through this
script: bundle and data paths come from ``QL_ACCEPTANCE_BUNDLE`` and
``QL_ACCEPTANCE_DATA_DIR`` (CLI flags override; the script exits loudly when
neither names a bundle), and the bundle's contract is validated through the
Strategy-Core REGISTRY SECTION HOOK (``validate_section_via_registry=True``)
— the typed plugin SectionModel, never a hand-built section.

This wraps the dashboard session experiment CLI, audits the saved bundle, and
optionally runs the focused Quant-Lab + Strategy-Core contract tests. It is a
research-readiness check: weak models may be saved with an explicit failed-gate
override. As of E2 the emitter stamps ``supported_by_runtime=true`` (Trade-Lab
is repointed + parity-proven since the C/D windows), and the audit asserts the
flag is True — Trade-Lab activation refuses False.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRATEGY_CORE_ROOT = REPO_ROOT.parent / "Strategy-Core"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
REQUIRED_BUNDLE_FILES = (
    "model.cbm",
    "metadata.json",
    "evaluation.json",
    "strategy.json",
    "oos_predictions.parquet",
)
QUANT_FOCUSED_TESTS = (
    "tests/agents/test_databento_acceptance_cli.py",
    "tests/agents/test_ml_training_reporting.py",
    "tests/agents/test_strategy_contract_nodrift.py",
    "tests/agents/test_strategy_contract_repoint.py",
)


@dataclass(frozen=True)
class AcceptanceConfig:
    """Resolved knobs for a Databento acceptance training run."""

    model_name: str
    preset: str = "all_to_ny"
    symbol: str = "NQ"
    data_dir: str = "data/databento"
    start: str | None = None
    end: str | None = None
    bar_type: str = "147t"
    tp: float = 15.0
    sl: float = 30.0
    interaction_window: int = 5
    include_approach_features: bool = True
    approach_window: int = 90
    train_days: int = 30
    test_days: int = 7
    gap_days: int = 1
    iterations: int = 500
    depth: int = 6
    allow_failed_gates: bool = True


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_training_command(config: AcceptanceConfig) -> list[str]:
    """Build the saved dashboard-session training command."""

    command = [
        sys.executable,
        "scripts/run_dashboard_session_experiment.py",
        "--preset",
        config.preset,
        "--symbol",
        config.symbol,
        "--data-dir",
        config.data_dir,
        "--bar-type",
        config.bar_type,
        "--tp",
        str(config.tp),
        "--sl",
        str(config.sl),
        "--interaction-window",
        str(config.interaction_window),
        "--approach-window",
        str(config.approach_window),
        "--train-days",
        str(config.train_days),
        "--test-days",
        str(config.test_days),
        "--gap-days",
        str(config.gap_days),
        "--iterations",
        str(config.iterations),
        "--depth",
        str(config.depth),
        "--save",
        "--model-name",
        config.model_name,
    ]
    if config.start:
        command.extend(["--start", config.start])
    if config.end:
        command.extend(["--end", config.end])
    if config.include_approach_features:
        command.append("--include-approach-features")
    if config.allow_failed_gates:
        command.append("--allow-failed-gates")
    return command


def _run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S603 - command is built from fixed local CLI paths/args.
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _load_strategy_contract_summary(
    strategy_json: Path,
    strategy_core_root: Path,
) -> dict[str, Any]:
    """Validate strategy.json through Strategy-Core's fail-closed loader."""

    strategy_src = strategy_core_root / "src"
    if strategy_src.exists():
        sys.path.insert(0, str(strategy_src))
    # W2 P3c: register the production plugin so the loader's registry hook can
    # type the section — the acceptance path consumes the plugin SectionModel,
    # never a hand-built section.
    import strategy_core.strategies.touch_reversal  # noqa: F401, PLC0415
    from strategy_core import CONTRACT_VERSION, PLATFORM_VERSION  # noqa: PLC0415
    from strategy_core.contract.loader import load_strategy_contract  # noqa: PLC0415

    contract = load_strategy_contract(
        strategy_json,
        expected_platform_version=PLATFORM_VERSION,
        validate_section_via_registry=True,
    )
    return {
        "loader": "Strategy-Core",
        "section_via_registry_hook": True,
        "expected_contract_version": CONTRACT_VERSION,
        "expected_platform_version": PLATFORM_VERSION,
        "loaded_type": type(contract).__name__,
        "feature_count": contract.feature_count,
        "feature_names": list(contract.feature_set.names),
    }


def audit_bundle(
    model_dir: Path | str,
    *,
    strategy_core_root: Path | str = DEFAULT_STRATEGY_CORE_ROOT,
    validate_contract: bool = True,
) -> dict[str, Any]:
    """Audit a saved model bundle and return machine-readable acceptance stats."""

    model_dir = Path(model_dir)
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (model_dir / name).exists()]
    if missing:
        msg = f"Missing required model bundle files: {missing}"
        raise FileNotFoundError(msg)

    evaluation = _read_json(model_dir / "evaluation.json")
    metadata = _read_json(model_dir / "metadata.json")
    strategy = _read_json(model_dir / "strategy.json")

    if strategy.get("supported_by_runtime") is not True:
        msg = "Acceptance artifacts must be servable with supported_by_runtime=true (E2)"
        raise ValueError(msg)

    oos_file_name = evaluation.get("oos_predictions_file", "oos_predictions.parquet")
    oos_path = model_dir / oos_file_name
    oos = pd.read_parquet(oos_path)
    gate_columns = [column for column in oos.columns if column.startswith("gate_")]
    probability_columns = [column for column in oos.columns if column.startswith("prob_")]

    contract_summary: dict[str, Any] | None = None
    if validate_contract:
        contract_summary = _load_strategy_contract_summary(
            model_dir / "strategy.json",
            Path(strategy_core_root),
        )

    quality_gates = evaluation.get("quality_gates", {})
    failed_gates = [
        name
        for name, payload in quality_gates.get("gates", {}).items()
        if isinstance(payload, dict) and not payload.get("passed", False)
    ]

    summary: dict[str, Any] = {
        "artifact": {
            "path": str(model_dir),
            "required_files_present": True,
            "file_sizes": {
                name: (model_dir / name).stat().st_size for name in REQUIRED_BUNDLE_FILES
            },
        },
        "evaluation": {
            "n_samples": evaluation.get("n_samples"),
            "precision": evaluation.get("precision"),
            "recall": evaluation.get("recall"),
            "f1": evaluation.get("f1"),
            "roc_auc": evaluation.get("roc_auc"),
            "brier_score": evaluation.get("brier_score"),
        },
        "quality_gates": {
            "all_passed": quality_gates.get("all_passed"),
            "allow_failed_gates": quality_gates.get("allow_failed_gates"),
            "failed": failed_gates,
            "gates": quality_gates.get("gates", {}),
        },
        "session_filter": evaluation.get("session_filter"),
        "session_experiment": {
            "evaluation": evaluation.get("session_experiment"),
            "metadata": metadata.get("session_experiment"),
            # v3: the research scope rides the strategy-owned section subtree.
            "strategy": strategy.get("section", {}).get("research_session_experiment"),
        },
        "gated_oos": evaluation.get("gated_oos"),
        "runtime": {
            "contract_version": strategy.get("contract_version"),
            "platform_version": strategy.get("platform_version"),
            "strategy_id": strategy.get("strategy_id"),
            "strategy_version": strategy.get("strategy_version"),
            "supported_by_runtime": strategy.get("supported_by_runtime"),
            # v3: touch_rule is section-bound; forward_bar_type stays envelope.
            "bar_type": strategy.get("section", {}).get("touch_rule", {}).get("bar_type"),
            "decision_offset_minutes": strategy.get("label_policy", {}).get(
                "decision_offset_minutes",
            ),
            "forward_bar_type": strategy.get("label_policy", {}).get("forward_bar_type"),
            "eligible_session": strategy.get("inference", {}).get("eligible_session"),
            "contract_loader": contract_summary,
        },
        "oos_predictions": {
            "file": oos_file_name,
            "rows": len(oos),
            "columns": list(oos.columns),
            "session_counts": oos["session"].value_counts(dropna=False).to_dict()
            if "session" in oos.columns
            else {},
            "label_counts": oos["label"].value_counts(dropna=False).to_dict()
            if "label" in oos.columns
            else {},
            "gate_true_counts": {column: int(oos[column].sum()) for column in gate_columns},
            "probability_null_counts": {
                column: int(oos[column].isna().sum()) for column in probability_columns
            },
        },
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run/audit the Quant-Lab Databento acceptance workflow.",
    )
    # W2 P3c: bundle + data paths come from env vars (CLI flags override).
    parser.add_argument("--model-name", default=os.environ.get("QL_ACCEPTANCE_BUNDLE"))
    parser.add_argument("--preset", default="all_to_ny")
    parser.add_argument("--symbol", default="NQ")
    parser.add_argument(
        "--data-dir", default=os.environ.get("QL_ACCEPTANCE_DATA_DIR", "data/databento")
    )
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--bar-type", default="147t")
    parser.add_argument("--tp", type=float, default=15.0)
    parser.add_argument("--sl", type=float, default=30.0)
    parser.add_argument("--interaction-window", type=int, default=5)
    parser.add_argument("--no-approach-features", action="store_true")
    parser.add_argument("--approach-window", type=int, default=90)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--disallow-failed-gates", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--strategy-core-root", type=Path, default=DEFAULT_STRATEGY_CORE_ROOT)
    parser.add_argument("--skip-quant-tests", action="store_true")
    parser.add_argument("--skip-strategy-tests", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> AcceptanceConfig:
    return AcceptanceConfig(
        model_name=args.model_name,
        preset=args.preset,
        symbol=args.symbol,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        bar_type=args.bar_type,
        tp=args.tp,
        sl=args.sl,
        interaction_window=args.interaction_window,
        include_approach_features=not args.no_approach_features,
        approach_window=args.approach_window,
        train_days=args.train_days,
        test_days=args.test_days,
        gap_days=args.gap_days,
        iterations=args.iterations,
        depth=args.depth,
        allow_failed_gates=not args.disallow_failed_gates,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.model_name:
        parser.error(
            "no bundle named: set QL_ACCEPTANCE_BUNDLE or pass --model-name "
            "(the W3 acceptance entrypoint refuses to guess a bundle)"
        )
    config = _config_from_args(args)
    model_dir = DEFAULT_MODEL_DIR / config.model_name
    acceptance: dict[str, Any] = {"config": asdict(config), "commands": {}}

    if not args.audit_only:
        train_command = build_training_command(config)
        train_result = _run_command(train_command, cwd=REPO_ROOT)
        acceptance["commands"]["train"] = train_result
        if train_result["returncode"] != 0:
            print(json.dumps(acceptance, indent=2, default=_json_default))
            return train_result["returncode"]

    acceptance["bundle_audit"] = audit_bundle(
        model_dir,
        strategy_core_root=args.strategy_core_root,
        validate_contract=True,
    )

    if not args.skip_quant_tests:
        acceptance["commands"]["quant_focused_tests"] = _run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                *QUANT_FOCUSED_TESTS,
                "-q",
                "-W",
                "error",
            ],
            cwd=REPO_ROOT,
        )
    if not args.skip_strategy_tests:
        strategy_python = args.strategy_core_root / ".venv" / "bin" / "python"
        acceptance["commands"]["strategy_contract_tests"] = _run_command(
            [
                str(strategy_python) if strategy_python.exists() else sys.executable,
                "-m",
                "pytest",
                "tests/test_contract.py",
                "-q",
                "-W",
                "error",
            ],
            cwd=args.strategy_core_root,
        )

    failed_commands = {
        name: result
        for name, result in acceptance["commands"].items()
        if isinstance(result, dict) and result.get("returncode") != 0
    }
    acceptance["passed"] = not failed_commands
    acceptance["failed_commands"] = list(failed_commands)

    summary_path = model_dir / "acceptance_summary.json"
    summary_path.write_text(
        json.dumps(acceptance, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(acceptance, indent=2, default=_json_default))
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
