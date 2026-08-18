"""CohortSpec interpretation tests (P0-10; TEST_MATRIX §3.6; DT §8).

Each ``InterpretationMode`` maps 1:1 onto its comparison class; only
``sequential_strategy_profile`` constructs an executable counterfactual — via
a NEW child profile built from registered axis values, never a filter over an
existing candidate table; descriptive cohorts can never select execution or
prop delta families.
"""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    resolve_axis_overrides,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_profile_id_for,
    canonicalize_section,
)
from alpha_lab.agents.data_infra.ifvg.study.cohort import (
    BASELINE_COHORT,
    CohortEnvelope,
    CohortPayload,
    InterpretationMode,
    interpretation_to_comparison_class,
)
from alpha_lab.agents.data_infra.ifvg.study.delta_outputs import (
    DELTA_FAMILY_SELECTION,
    DeltaOutputFamily,
)

_EXECUTION_FAMILIES = {
    DeltaOutputFamily.EXECUTION_DELTA,
    DeltaOutputFamily.PROP_DELTA,
    DeltaOutputFamily.PAYOUT_DELTA,
    DeltaOutputFamily.PORTFOLIO_DELTA,
    DeltaOutputFamily.RISK_DELTA,
}


def test_interpretation_modes_map_one_to_one() -> None:
    assert interpretation_to_comparison_class(
        InterpretationMode.DESCRIPTIVE_SLICE
    ) == "cohort_descriptive"
    assert interpretation_to_comparison_class(
        InterpretationMode.SPECIALIZED_MODEL
    ) == "cohort_model"
    assert interpretation_to_comparison_class(
        InterpretationMode.SEQUENTIAL_STRATEGY_PROFILE
    ) == "strategy_counterfactual"
    assert len(InterpretationMode) == 3  # no fourth option exists


def test_descriptive_and_model_cohorts_never_select_execution_families() -> None:
    for cohort_class in ("cohort_descriptive", "cohort_model"):
        families = set(DELTA_FAMILY_SELECTION[cohort_class])
        forbidden = families & _EXECUTION_FAMILIES
        assert not forbidden, (
            f"{cohort_class} may never select execution/prop deltas: {forbidden}"
        )


def test_strategy_counterfactual_selects_population_and_funnel() -> None:
    families = set(DELTA_FAMILY_SELECTION["strategy_counterfactual"])
    assert DeltaOutputFamily.POPULATION_DELTA in families
    assert DeltaOutputFamily.FUNNEL_DELTA in families


def test_sequential_strategy_profile_constructs_a_child_not_a_filter() -> None:
    """The third mode builds a NEW canonical child profile from registered
    axis values (full sequential replay semantics) — the cohort filter fields
    play no role in its identity."""

    overrides = resolve_axis_overrides(
        {"parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.240"}
    )
    assert overrides == {"parent_retest_timeout_1m_bars": 240}
    resolved = resolve_profile_config(
        {
            "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "section_overrides": overrides,
        }
    )
    section = canonicalize_section(resolved.section)
    # a genuinely NEW profile identity, canonical and study-independent
    assert section.profile_name == canonical_profile_id_for(section)
    assert section.profile_name.startswith("ifvg_search_profile_")
    from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

    baseline = resolve_profile_config(
        {"profile_name": "ifvg_v2_doc_default_fresh_static_1r"}
    )
    assert ifvg_profile_hash(section) != baseline.section_config_hash
    # the cohort contract itself carries filters but NO section overrides —
    # a cohort cannot smuggle a strategy change
    assert "session_filter" in CohortPayload.model_fields
    assert not any("override" in field for field in CohortPayload.model_fields)


def test_cohort_identity_is_stable_and_filter_sensitive() -> None:
    assert BASELINE_COHORT.payload.interpretation_mode is (
        InterpretationMode.DESCRIPTIVE_SLICE
    )
    filtered = CohortEnvelope.from_payload(
        BASELINE_COHORT.payload.model_copy(update={"session_filter": ("ny_am",)})
    )
    assert filtered.cohort_id != BASELINE_COHORT.cohort_id
    rebuilt = CohortEnvelope.from_payload(BASELINE_COHORT.payload)
    assert rebuilt.cohort_id == BASELINE_COHORT.cohort_id


def test_tampered_cohort_envelope_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not hash its payload"):
        CohortEnvelope(
            cohort_id="0" * 64,
            payload=BASELINE_COHORT.payload,
        )
