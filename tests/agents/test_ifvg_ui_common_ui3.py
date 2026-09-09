"""UI-3 §6.6 / §6.4 — the shared UI primitives: detail levels (Summary /
Research details / Technical identity & audit) over the persisted
disclosure vocabulary, the identity reveal, the metric and roll-up cards
and the glossary expander (AppTest)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("developer_presentation")

apptest = pytest.importorskip("streamlit.testing.v1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_ui_common as common  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.help_registry import (  # noqa: E402
    GLOSSARY_TERMS,
)
from alpha_lab.agents.data_infra.ifvg.study_status import DisclosureLevel  # noqa: E402


def _app() -> None:
    import ifvg_ui_common as common
    import streamlit as st

    from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import (
        evaluate_metric,
    )
    from alpha_lab.agents.data_infra.ifvg.presentation.rollups import (
        RollupSection,
        rollup_section,
    )

    level = common.detail_levels(st, key="ifvg_study_v1_ui3_detail")
    st.session_state["ifvg_study_v1_ui3_level"] = level
    legacy = common.disclosure_level(st, key="ifvg_study_v1_ui3_legacy")
    st.session_state["ifvg_study_v1_ui3_legacy_level"] = legacy
    common.identity_reveal(st, "search_id", "a" * 64, human="Baseline study")
    reading = evaluate_metric("profit_factor", 1.4, gate=1.1, proposed=True)
    common.metric_card(st, reading)
    missing = evaluate_metric("brier_score", None)
    common.metric_card(st, missing)
    rollup = rollup_section(RollupSection.STRATEGY_QUALITY, [reading])
    common.rollup_card(st, rollup)
    common.glossary_expander(st)


def test_detail_levels_render_the_three_tiers_over_the_persisted_vocabulary() -> None:
    assert common.DETAIL_LEVEL_LABELS == {
        "summary": "Summary",
        "analyst": "Research details",
        "audit": "Technical identity & audit",
    }
    assert set(common.DETAIL_LEVEL_LABELS) == {level.value for level in DisclosureLevel}
    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    radio = next(r for r in at.radio if r.key == "ifvg_study_v1_ui3_detail")
    assert radio.options == ["Summary", "Research details", "Technical identity & audit"]
    assert radio.help and "Summary" in radio.help and "Technical identity" in radio.help
    assert at.session_state["ifvg_study_v1_ui3_level"] == "summary"
    # the R4 seam delegates: the same labels, the same persisted values
    legacy = next(r for r in at.radio if r.key == "ifvg_study_v1_ui3_legacy")
    assert legacy.options == radio.options
    assert at.session_state["ifvg_study_v1_ui3_legacy_level"] == "summary"
    radio.set_value("Research details").run()
    assert not at.exception
    assert at.session_state["ifvg_study_v1_ui3_level"] == "analyst"
    legacy = next(r for r in at.radio if r.key == "ifvg_study_v1_ui3_legacy")
    legacy.set_value("Technical identity & audit").run()
    assert at.session_state["ifvg_study_v1_ui3_legacy_level"] == "audit"


def test_identity_reveal_metric_and_rollup_cards_and_glossary() -> None:
    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    codes = [str(block.value) for block in at.code]
    assert "a" * 64 in codes  # the full id, copyable
    captions = "\n".join(str(block.value) for block in at.caption)
    assert "Baseline study" in captions and "search_id" in captions
    metrics = at.metric
    labels = [metric.label for metric in metrics]
    assert "Profit factor" in labels and "Brier score" in labels
    profit = next(metric for metric in metrics if metric.label == "Profit factor")
    assert profit.value == "1.40"
    assert profit.help and "profit_factor" in profit.help and "higher is better" in profit.help
    assert "✓ Pass" in captions and "proposed" in captions.lower()
    assert "∅ Unavailable" in captions  # a missing value is never green
    markdown = "\n".join(str(block.value) for block in at.markdown)
    assert "Strategy quality" in markdown and "▲ Warning" in markdown
    assert "Inspect next" in captions
    glossary = next(e for e in at.expander if e.label == "Glossary")
    rendered = "\n".join(str(block.value) for block in glossary.markdown)
    for term in GLOSSARY_TERMS:
        assert f"**{term}**" in rendered


def test_paginate_controls_carry_help() -> None:
    def _paged() -> None:
        import ifvg_ui_common as common
        import streamlit as st

        common.paginate_controls(st, 120, key="ifvg_study_v1_ui3_pages")

    at = apptest.AppTest.from_function(_paged, default_timeout=60)
    at.run()
    assert not at.exception
    size = next(s for s in at.selectbox if s.key == "ifvg_study_v1_ui3_pages_page_size")
    page = next(n for n in at.number_input if n.key == "ifvg_study_v1_ui3_pages_page")
    assert size.help and page.help
