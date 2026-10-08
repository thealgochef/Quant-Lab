"""Read-only reporting, causal review and normal-library contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.lab.library import study_row
from alpha_lab.propsim.funded.ml_phase.catalog import (
    load_report,
    point_in_time,
    publish,
    registered_reports,
)
from alpha_lab.propsim.funded.ml_phase.diagnostics import paired_diagnostics
from alpha_lab.propsim.funded.ml_phase.protocol import digest
from alpha_lab.propsim.funded.ml_phase.runtime import sha_file


def test_paired_blocks_keep_empty_dates_and_known_loss_difference():
    cells = [{"cell_id": f"{f}_RIDGE", "reference": "MCB025", "job": "ENTRY",
              "feature_set": f, "model": "RIDGE"} for f in ("F0", "F1")]
    dates = [f"date{i}" for i in range(20)]
    rows = [{"cell_id": c["cell_id"], "row_id": "same", "trading_day": dates[10],
             "prediction": 0 if c["feature_set"] == "F1" else 2, "label": 1}
            for c in cells]
    benchmark = {"cells": cells, "predictions": rows}
    result = paired_diagnostics(benchmark, dates)
    assert result == paired_diagnostics(benchmark, dates)
    assert result[0]["evaluated_dates_including_empty"] == 20
    assert result[0]["paired_rows"] == 1
    assert result[0]["development_95pct_interval"] == [0, 0]


def test_cursor_hides_future_decision_features_and_label():
    row = {"decision_ns": 100, "event_ordinal": 3, "x": 5, "poison": 999,
           "prediction": {"score": -0.2, "action": "change"},
           "feature_provenance": {"x": {"known_at_ns": 90},
                                  "poison": {"known_at_ns": 200}}}
    label = {"label": -1, "label_available_ns": 300, "label_status": "exact"}
    assert point_in_time(row, 99, label=label) == {"status": "decision_not_yet_available"}
    at = point_in_time(row, 100, label=label)
    assert at["features"] == {"x": 5}
    assert "retrospective_fixed_shadow_outcome" not in at
    same_stamp = {**label, "label_available_ns": 100}
    assert "retrospective_fixed_shadow_outcome" not in point_in_time(row, 100, label=same_stamp)
    after = point_in_time(row, 300, label=label, after_outcome=True)
    assert after["retrospective_fixed_shadow_outcome"]["label"] == -1


def test_additive_catalog_verifies_report_and_library_routes_normally(tmp_path):
    body = {"test": "synthetic"}
    report = {**body, "report_id": digest(body)}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    pointer = {"report_id": report["report_id"], "report_path": str(path),
               "report_sha256": sha_file(path)}
    publish(tmp_path, pointer)
    publish(tmp_path, pointer)
    rows, issues = registered_reports(tmp_path)
    assert rows == [pointer] and not issues
    study = SimpleNamespace(state=pointer, kind="ml_phase", draft=None, key=report["report_id"],
                            archived=False, updated="", status="Completed", name="ML study",
                            question="24 cells", dates="2025-10-09 to 2026-06-10")
    row = study_row(study, "main", current="main")
    assert row.tab == "model" and row.route == "ml_phase"
    path.write_text('{"changed": true}')
    with pytest.raises(PermissionError, match="hash differs"):
        load_report(pointer)
    assert registered_reports(tmp_path)[1]
