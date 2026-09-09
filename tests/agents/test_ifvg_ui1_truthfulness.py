"""UI-1 truthfulness units (plan F-05, F-07, §6.7):

* minimize metrics use the REVERSED colorscale with the direction labelled;
* the reconciliation banner derives from evaluated gates only;
* no-runs / not-selected situations never render as artifact_unavailable;
* the publication gates are namespace- and state-bound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("developer_presentation")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: E402
    OBJECTIVE_DIRECTIONS,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (  # noqa: E402
    FIRM_MATRIX_METRICS,
    HEATMAP_METRICS,
)

_CELLS = [
    {"row_value": "a", "col_value": "x", "value": 1.0, "cell_class": "stable_plateau",
     "sample_count": 30},
    {"row_value": "b", "col_value": "x", "value": 9.0, "cell_class": "stable_plateau",
     "sample_count": 30},
]


def _scale(figure) -> list[str]:
    return [color for _position, color in figure.data[0].colorscale]


@pytest.mark.parametrize("metric_key", sorted(set(HEATMAP_METRICS.values())))
def test_heatmap_colorscale_follows_the_registered_direction(metric_key) -> None:
    from ifvg_results_charts import build_sensitivity_heatmap

    figure, _ = build_sensitivity_heatmap(
        _CELLS, row_axis="A", col_axis="B", metric_label="m", metric_key=metric_key
    )
    maximize, _ = build_sensitivity_heatmap(
        _CELLS, row_axis="A", col_axis="B", metric_label="m", metric_key="net_expectancy_r"
    )
    title = figure.data[0].colorbar.title.text
    if OBJECTIVE_DIRECTIONS[metric_key] == "minimize":
        assert "lower is better" in title
        assert _scale(figure) == list(reversed(_scale(maximize)))  # worst is never green
    else:
        assert "higher is better" in title
        assert _scale(figure) == _scale(maximize)


@pytest.mark.parametrize("metric_key", sorted(set(FIRM_MATRIX_METRICS.values())))
def test_firm_matrix_colorscale_follows_the_registered_direction(metric_key) -> None:
    from ifvg_results_charts import build_firm_matrix_figure, direction_colorscale

    cells = [
        {"config_name": "c", "firm_label": "f", "value": 0.1, "universal": True},
        {"config_name": "d", "firm_label": "f", "value": 0.9, "universal": True},
    ]
    figure, _ = build_firm_matrix_figure(cells, metric_label="m", metric_key=metric_key)
    scale, note = direction_colorscale(metric_key)
    assert note in figure.data[0].colorbar.title.text
    assert (scale == "RdYlGn_r") == (OBJECTIVE_DIRECTIONS[metric_key] == "minimize")
    assert direction_colorscale("not_a_metric") == ("RdYlGn", "direction unregistered")
    assert direction_colorscale(None)[1] == "direction unregistered"


def test_every_displayed_metric_has_a_registered_direction() -> None:
    for key in (*HEATMAP_METRICS.values(), *FIRM_MATRIX_METRICS.values()):
        assert key in OBJECTIVE_DIRECTIONS, key


def test_reconcile_banner_derives_from_evaluated_gates() -> None:
    """Plan F-07: green only for evaluated passing gates; an unevaluated
    report renders the UNAVAILABLE warning, never the success banner."""

    apptest = pytest.importorskip("streamlit.testing.v1")
    import ifvg_lab_tab as tab

    def _render(report: dict) -> object:
        tab._APPTEST_REPORT = report  # type: ignore[attr-defined]

        def _app() -> None:
            import ifvg_lab_tab
            import streamlit as st

            ifvg_lab_tab._render_reconciliation_report(st, ifvg_lab_tab._APPTEST_REPORT)

        at = apptest.AppTest.from_function(_app, default_timeout=60)
        at.run()
        assert not at.exception
        return at

    passing = _render(
        {
            "passed": True,
            "evaluated": True,
            "evaluated_gate_count": 2,
            "unevaluated_reports": ["data_access_audit.json"],
            "reports": {
                "validity_report.json": {"passed": True, "violations": {}},
                "identity_report.json": {"passed": True, "violations": {}},
                "data_access_audit.json": {"protected_file_opens": 0},
            },
        }
    )
    assert passing.success and "2 evaluated gate(s)" in str(passing.success[0].value)
    assert not passing.warning
    unevaluated = _render(
        {
            "passed": None,
            "evaluated": False,
            "unevaluated_reports": ["data_access_audit.json"],
            "reports": {"data_access_audit.json": {"protected_file_opens": 0}},
        }
    )
    assert not unevaluated.success
    assert "UNAVAILABLE" in str(unevaluated.warning[0].value)
    failing = _render({"passed": False, "reports": {}})
    assert failing.error and not failing.success


def test_reconciliation_report_passed_is_derived_from_the_pair_reports(tmp_path) -> None:
    """The producer never hard-codes ``passed``: it folds the evaluated
    report flags and lists the unevaluated reports."""

    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.context_contracts import ContextRecordTable
    from alpha_lab.agents.data_infra.ifvg.context_reporting import (
        build_context_reconciliation_audit_report,
    )

    def _pair(reports: dict) -> SimpleNamespace:
        reference = SimpleNamespace(
            artifact_id="a" * 64,
            manifest_payload_sha256="b" * 64,
            dataset_schema_version=3,
            feature_formula_version="v3",
            preparation_status=SimpleNamespace(value="complete"),
        )
        tables = {table: [] for table in ContextRecordTable}
        return SimpleNamespace(
            v2=SimpleNamespace(reference=reference),
            v3=SimpleNamespace(
                reference=reference, manifest={}, tables=tables, reports=reports
            ),
        )

    passing = build_context_reconciliation_audit_report(
        _pair(
            {
                "validity_report.json": {"passed": True},
                "identity_report.json": {"passed": True},
                "data_access_audit.json": {"protected_file_opens": 0},
            }
        )
    )
    assert passing["passed"] is True and passing["evaluated"] is True
    assert passing["evaluated_gate_count"] == 2
    assert "data_access_audit.json" in passing["unevaluated_reports"]
    failing = build_context_reconciliation_audit_report(
        _pair({"validity_report.json": {"passed": False}, "identity_report.json": {"passed": True}})
    )
    assert failing["passed"] is False
    empty = build_context_reconciliation_audit_report(_pair({}))
    assert empty["passed"] is None and empty["evaluated"] is False
    assert json.dumps(empty["gate_evaluations"])  # JSON-serializable (nulls)


def test_empty_states_never_use_artifact_unavailable_for_no_runs() -> None:
    from alpha_lab.agents.data_infra.ifvg.study_status import (
        EMPTY_STATE_PRESENTATIONS,
        EmptyStateKey,
    )

    for state_id in ("no_runs", "pipeline_no_runs"):
        assert EMPTY_STATE_PRESENTATIONS[state_id].key is EmptyStateKey.NO_RUNS
    for state_id in ("not_selected", "not_applicable", "not_configured"):
        assert EMPTY_STATE_PRESENTATIONS[state_id].key is not EmptyStateKey.ARTIFACT_UNAVAILABLE
    # the UI sources never render artifact_unavailable for a no-runs listing
    for script in ("ifvg_active_runs_tab.py", "ifvg_results_tab.py", "ifvg_pipeline_tab.py"):
        source = (_REPO / "scripts" / script).read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            if '"artifact_unavailable"' in line:
                window = "\n".join(source.splitlines()[max(0, line_number - 8) : line_number + 6])
                assert "no runs" not in window.lower() or "no_runs" in window, (script, line_number)
