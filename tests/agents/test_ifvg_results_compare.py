"""FUX-RES-007..009 + FUX-DRILL-001 AppTests: dimension ribbon with
match_basis, incompatibility suppression, seven insight categories with
exact evidence, account-timeline order and linked drill-down."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("developer_presentation")

apptest = pytest.importorskip("streamlit.testing.v1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_results_compare as compare  # noqa: E402
import ifvg_study_tab as study_tab  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.search.insights import (  # noqa: E402
    InsightCategory,
)
from tests.agents.ifvg_search.study_ui_fixture import (  # noqa: E402
    build_completed_search,
)


@pytest.fixture(scope="module")
def completed_search(tmp_path_factory) -> dict:
    return build_completed_search(tmp_path_factory.mktemp("compare_ui"))


def _patch(monkeypatch, fixture: dict) -> None:
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", fixture["store_root"])
    monkeypatch.setattr(study_tab, "STATE_ROOT", fixture["state_root"])
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", fixture["draft_root"])


def _comparison_app() -> None:
    # AppTest re-executes this source in a fresh namespace: every module
    # must be imported HERE (module-level imports are not visible).
    import ifvg_results_compare as compare
    import ifvg_results_tab as results  # noqa: F811
    import ifvg_study_tab as study_tab
    import streamlit as st

    roots = study_tab.workspace_roots(st)
    runs = __import__(
        "alpha_lab.agents.data_infra.ifvg.study_providers",
        fromlist=["list_search_runs"],
    ).list_search_runs(roots["state_root"], roots["store_root"])
    bundle = results.load_results_bundle(
        store_root=roots["store_root"],
        state_root=roots["state_root"],
        search_id=runs[0].search_id,
    )
    compare.render_comparison(st, roots=roots, bundle=bundle)
    compare.render_insights_for_bundle(st, bundle)


def _timeline_app() -> None:
    import ifvg_results_compare as compare
    import ifvg_study_tab as study_tab
    import streamlit as st

    compare.render_account_timeline(st, roots=study_tab.workspace_roots(st))


def _text(at) -> str:
    return "\n".join(
        [str(b.value) for b in at.markdown]
        + [str(c.value) for c in at.caption]
        + [str(h.value) for h in at.subheader]
        + [str(w.value) for w in at.warning]
    )


def test_dimension_ribbon_precedes_metrics_with_match_basis(
    monkeypatch, completed_search
) -> None:
    """FUX-RES-007: ribbon fields incl. match_basis; four panels; the
    membership panels stay honest without a lineage-gated delta build."""

    _patch(monkeypatch, completed_search)
    at = apptest.AppTest.from_function(_comparison_app, default_timeout=120)
    at.run()
    assert not at.exception
    tables = str([t.value.to_dict() for t in at.table]) + str(
        [frame.value for frame in at.dataframe]
    )
    joined = _text(at) + tables
    for field in (
        "delta type",
        "changed dimensions",
        "frozen dimensions",
        "computation path",
        "compatibility",
        "uncertainty method",
        "development status",
        "match_basis",
    ):
        assert field in joined, field
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels[:4] == [
        "Parameter Diff",
        "Funnel Delta",
        "Strategy Delta",
        "Prop Delta",
    ]
    assert "paired_cell_bootstrap_10000_seed7_v1" in joined


def test_incompatible_pair_suppresses_membership_claims(
    monkeypatch, completed_search
) -> None:
    """A lineage-breaking changed axis renders parameter diff only plus the
    explicit incompatibility notice — no membership approximation."""

    _patch(monkeypatch, completed_search)

    def _app() -> None:
        import ifvg_results_compare as compare
        import ifvg_study_tab as study_tab
        import streamlit as st

        bundle = {
            "children": [
                {
                    "core_replay_id": "a" * 64,
                    "axis_value_ids": {
                        "min_gap_ticks_capture": "min_gap_ticks_capture.4"
                    },
                    "comparison_role": "baseline",
                    "state": "completed",
                    "failure_reason": None,
                    "explanation": "",
                    "ordinal": 0,
                },
                {
                    "core_replay_id": "b" * 64,
                    "axis_value_ids": {
                        "min_gap_ticks_capture": "min_gap_ticks_capture.other"
                    },
                    "comparison_role": "challenger",
                    "state": "completed",
                    "failure_reason": None,
                    "explanation": "",
                    "ordinal": 1,
                },
            ],
            "names": {"a" * 64: "Baseline", "b" * 64: "Gap variant"},
            "metrics_by_child": {},
            "prop_vectors": {},
            "frontier": None,
        }
        compare.render_comparison(
            st, roots=study_tab.workspace_roots(st), bundle=bundle
        )

    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.run()
    at2 = at
    # select two DIFFERENT configurations
    boxes = [w for w in at2.selectbox if w.key and w.key.startswith(compare._CMP)]
    boxes[1].set_value(boxes[1].options[1]).run()
    assert not at2.exception
    text = _text(at2)
    assert "not comparable" in text.lower()
    assert "disabled, never approximated" in text
    headings = " ".join(str(h.value) for h in at2.subheader)
    assert "not comparable" in headings.lower() or "Populations" in text
    # §25 rule 1 / DT §3.1 (adversarial F4c): the metric-delta panels are
    # SUPPRESSED for a config-diff-only pair — no populated delta tables
    warnings = [str(w.value) for w in at2.warning]
    assert (
        sum("config-diff-only" in warning for warning in warnings) == 2
    )  # Strategy Delta + Prop Delta both suppressed
    frames = [frame.value for frame in at2.dataframe]
    for frame in frames:
        assert "delta" not in [c.lower() for c in frame.columns] or (
            "axis" in frame.columns
        )  # only the parameter diff renders


def test_insight_panel_renders_the_seven_categories_with_evidence(
    monkeypatch, completed_search
) -> None:
    """FUX-RES-008: seven verbatim categories, neutral wording, evidence."""

    _patch(monkeypatch, completed_search)
    at = apptest.AppTest.from_function(_comparison_app, default_timeout=120)
    at.run()
    assert not at.exception
    text = _text(at)
    for category in InsightCategory:
        assert f"**{category.value}**" in text, category
    lowered = text.lower()
    for banned in ("winner", "production ready", "because of the change"):
        assert banned not in lowered
    assert "derived deterministically from persisted artifacts" in lowered


def test_account_timeline_orders_by_envelope_and_links_evidence(
    monkeypatch, completed_search
) -> None:
    """FUX-RES-009 + FUX-DRILL-001: total event order, marker/table
    synchronization, exact-id drill-down only."""

    _patch(monkeypatch, completed_search)
    at = apptest.AppTest.from_function(_timeline_app, default_timeout=120)
    at.run()
    assert not at.exception
    picker = next(
        w for w in at.selectbox if w.key == f"{compare._CMP}timeline_pick"
    )
    labeled = next(
        option for option in picker.options if "synthetic_fixture_firm" in option
    )
    picker.set_value(labeled).run()
    assert not at.exception
    frames = [frame.value for frame in at.dataframe]
    events_table = next(frame for frame in frames if "ordinal" in frame.columns)
    assert list(events_table["ordinal"]) == [0, 1, 2, 3, 4]  # envelope order
    assert "payout" in set(events_table["type"])
    codes = " ".join(str(c.value) for c in at.code)
    assert completed_search["simulation_id"] in codes
    text = _text(at)
    assert "Result scope:" in text
    assert "event_ordinal" in text  # the order provenance is stated
    row_picker = next(
        w for w in at.selectbox if w.key == f"{compare._CMP}timeline_row"
    )
    row_picker.set_value("2").run()  # the payout event with trade linkage
    assert not at.exception
    codes = " ".join(str(c.value) for c in at.code)
    assert "trade: trade-0001" in codes
    assert any(b.label == "Open exact" for b in at.button)


def test_unresolved_exact_id_terminates_sanitized(
    monkeypatch, completed_search
) -> None:
    _patch(monkeypatch, completed_search)
    at = apptest.AppTest.from_function(_timeline_app, default_timeout=120)
    at.run()
    typed = next(
        w for w in at.text_input if w.key == f"{compare._CMP}timeline_typed"
    )
    typed.set_value("0" * 64).run()
    assert not at.exception
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "Artifact unavailable" in headings
    everything = headings + " ".join(str(c.value) for c in at.caption)
    assert str(completed_search["store_root"]) not in everything
