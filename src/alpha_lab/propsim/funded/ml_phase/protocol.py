"""Resolve the owner's finite phase-01 contracts without fitting or market access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "ifsm_ml_return_phase_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def load_contracts(root: Path) -> dict:
    names = (
        "MODEL_PROTOCOL",
        "MODEL_MATRIX",
        "FEATURE_LISTS",
        "FEATURE_DEFINITIONS",
        "DATE_AND_FOLD_PLAN",
        "REFERENCE_BINDINGS",
    )
    result = {
        name: json.loads((Path(root) / "contracts" / f"{name}.json").read_text(encoding="utf-8"))
        for name in names
    }
    validate_contracts(result)
    return result


def validate_contracts(contracts: dict) -> None:
    schedule = contracts["DATE_AND_FOLD_PLAN"]
    dates = schedule["original_evaluation_dates"]
    warmup = schedule["original_warmup_dates"]
    if len(dates) != 253 or len(warmup) != 10 or dates != sorted(set(dates)):
        raise ValueError("phase requires the exact 253/10 inherited date scope")
    source = contracts["REFERENCE_BINDINGS"]["approved_market_source"]
    if dates != source["evaluation_dates"] or warmup != source["warmup_dates"]:
        raise ValueError("dates differ from inherited source")
    if max(dates + warmup) > "2026-06-10" or set(dates) & set(warmup):
        raise ValueError("invalid or protected date scope")
    folds = schedule["folds"]
    if len(folds) != 9 or schedule["scored_dates"] != dates[82:]:
        raise ValueError("phase requires nine folds and 171 contiguous scored dates")
    for index, fold in enumerate(folds):
        start = 82 + index * 20
        if (
            fold["train_candidate_dates"] != dates[: start - 2]
            or fold["separation_dates"] != dates[start - 2 : start]
            or fold["test_dates"] != dates[start : start + 20]
        ):
            raise ValueError("fold differs from frozen expanding schedule")
    matrix = contracts["MODEL_MATRIX"]
    expected = {
        (r, j, f, m)
        for r in ("MCB062", "MCB025")
        for j in ("ENTRY", "CONTINUATION")
        for f in ("F0", "F1", "F2")
        for m in ("RIDGE", "CATBOOST")
    }
    cells = matrix["cells"]
    actual = {(c["reference"], c["job"], c["feature_set"], c["model"]) for c in cells}
    if len(cells) != 24 or actual != expected or len({c["cell_id"] for c in cells}) != 24:
        raise ValueError("not the declared 24-cell matrix")
    if any(c["threshold"] != 0 or c["combined_entry_and_exit_policy"] for c in cells):
        raise ValueError("only the separate fixed-zero actions are authorized")
    definitions = contracts["FEATURE_DEFINITIONS"]
    for job in ("ENTRY", "CONTINUATION"):
        for group in ("F0", "F1", "F2"):
            expected_names = [
                r["name"]
                for r in definitions["rows"]
                if job in r["jobs"] and r["group"] in definitions["feature_groups"][group]
            ]
            if contracts["FEATURE_LISTS"][job][group] != expected_names:
                raise ValueError("feature ordering or bundle isolation changed")


def action_for(score: float | None) -> str:
    import math

    if score is None or not math.isfinite(score):
        return "baseline_unavailable"
    return "change" if score < 0 else "baseline"
