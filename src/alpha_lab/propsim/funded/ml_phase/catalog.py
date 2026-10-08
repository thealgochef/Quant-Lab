"""Additive ordinary-library discovery of immutable saved phase reports."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .protocol import digest
from .runtime import sha_file


def catalog_path(store):
    return Path(store) / "catalog/ifsm_ml_phase_results.json"


def load_report(pointer):
    path = Path(pointer["report_path"])
    stat = path.stat()
    return _load_report(str(path), pointer["report_sha256"], pointer["report_id"],
                        stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2)
def _load_report(filename, expected_sha, report_id, mtime_ns, size):
    path = Path(filename)
    if sha_file(path) != expected_sha:
        raise PermissionError("saved ML report hash differs from its catalog binding")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["report_id"] != report_id:
        raise PermissionError("saved ML report identity differs")
    body = {k: v for k, v in report.items() if k != "report_id"}
    if digest(body) != report["report_id"]:
        raise PermissionError("saved ML report content identity differs")
    return report


def registered_reports(store):
    path = catalog_path(store)
    if not path.exists():
        return [], []
    pointers = json.loads(path.read_text(encoding="utf-8"))["studies"]
    rows, issues = [], []
    for pointer in pointers:
        try:
            load_report(pointer)
            rows.append(pointer)
        except (OSError, ValueError, KeyError) as error:
            issues.append(f"ML phase publication unavailable: {error}")
    return rows, issues


def publish(store, pointer):
    load_report(pointer)
    path = catalog_path(store)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"studies": []}
    if any(p["report_id"] == pointer["report_id"] for p in saved["studies"]):
        return
    saved["studies"].append(pointer)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def point_in_time(row, cursor_ns, *, label=None, after_outcome=False):
    """Strict separation of decision-known evidence from eventual outcomes."""
    if cursor_ns < row["decision_ns"]:
        return {"status": "decision_not_yet_available"}
    provenance = row.get("feature_provenance", {})
    features = {name: row.get(name) for name, source in provenance.items()
                if source.get("known_at_ns", row["decision_ns"]) <= cursor_ns}
    view = {"status": "decision_available", "decision_ns": row["decision_ns"],
            "event_ordinal": row["event_ordinal"], "features": features,
            "feature_provenance": {name: provenance[name] for name in features},
            "prediction": row["prediction"]}
    if label and (cursor_ns > label["label_available_ns"] or
                  after_outcome and cursor_ns >= label["label_available_ns"]):
        view["retrospective_fixed_shadow_outcome"] = {key: label.get(key) for key in (
            "label", "label_status", "label_available_ns", "label_components")}
    return view
