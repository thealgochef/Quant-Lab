"""Comparison contract suites (§7A.15; TEST_MATRIX §3.1/§3.9)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.study.comparison_contracts import (
    REQUIRED_EQUALITIES,
    ComparisonClass,
    ComparisonClassMismatchError,
    build_comparison,
)
from alpha_lab.agents.data_infra.ifvg.study.computation_path import ComputationPath
from alpha_lab.agents.data_infra.ifvg.study.delta_outputs import (
    DELTA_FAMILY_SELECTION,
    DeltaOutputFamily,
    select_delta_families,
)
from alpha_lab.agents.data_infra.ifvg.study.study_cell import (
    BASELINE_STUDY_CELL,
    StudyCellIdentity,
    _baseline_cell_payload,
)


def _replay_path() -> ComputationPath:
    return ComputationPath(
        full_strategy_replay=True,
        feature_materialization=False,
        label_recomputation=False,
        model_refit=False,
        model_gated_sequential_replay=False,
        cost_recomputation=True,
        prop_resimulation=True,
        bootstrap_resimulation=True,
        reuse_trade_stream_hash=False,
    )


def _challenger_cell() -> StudyCellIdentity:
    base = _baseline_cell_payload()
    changed = base.model_copy(
        update={
            "strategy_profile": base.strategy_profile.model_copy(
                update={
                    "strategy_profile_id": "ifvg_search_profile_0123456789abcdef",
                    "profile_hash": "d" * 64,
                    "section_config_hash": "d" * 64,
                }
            ),
            "data_lineage": base.data_lineage.model_copy(
                update={"core_replay_id": "e" * 64, "v2_artifact_id": "f" * 64}
            ),
        }
    )
    return StudyCellIdentity.from_payload(changed)


def _evidence(cell_id: str, **overrides) -> dict:
    base = {
        "authorized_date_set_id": "dev_dates_v1",
        "cost_policy_id": "gross_zero_cost_v1",
        "label_policy_id": "candidate_static_r1_v2",
        "execution_policy_id": "confirmation_close_next1m_stop_first_v1",
        "fold_protocol_id": "ifvg_context_walkforward_40_5_5_2_v1",
    }
    base.update(overrides)
    return base


def test_all_twenty_classes_have_equalities_and_families() -> None:
    assert len(ComparisonClass) == 20
    assert set(REQUIRED_EQUALITIES) == set(ComparisonClass)
    assert set(DELTA_FAMILY_SELECTION) == {cls.value for cls in ComparisonClass}
    assert len(DeltaOutputFamily) == 23


def test_declared_class_must_match_observed_changes() -> None:
    challenger = _challenger_cell()
    with pytest.raises(ComparisonClassMismatchError, match="requires changed"):
        build_comparison(
            baseline=BASELINE_STUDY_CELL,
            challengers=(challenger,),
            comparison_class=ComparisonClass.COST_POLICY,
            computation_path=_replay_path(),
            interpretation="strategy_counterfactual",
            equality_evidence={},
        )
    with pytest.raises(ComparisonClassMismatchError, match="does not permit"):
        build_comparison(
            baseline=BASELINE_STUDY_CELL,
            challengers=(challenger,),  # data_lineage changed AND strategy changed
            comparison_class=ComparisonClass.DATA_LINEAGE,
            computation_path=_replay_path(),
            interpretation="descriptive",
            equality_evidence={},
        )
    with pytest.raises(ComparisonClassMismatchError, match="identical"):
        build_comparison(
            baseline=BASELINE_STUDY_CELL,
            challengers=(BASELINE_STUDY_CELL,),
            comparison_class=ComparisonClass.STRATEGY_COUNTERFACTUAL,
            computation_path=_replay_path(),
            interpretation="strategy_counterfactual",
            equality_evidence={},
        )


def test_compatible_comparison_selects_families_automatically() -> None:
    challenger = _challenger_cell()
    evidence = {
        BASELINE_STUDY_CELL.cell_id: _evidence(BASELINE_STUDY_CELL.cell_id),
        challenger.cell_id: _evidence(challenger.cell_id),
    }
    envelope, compatibility = build_comparison(
        baseline=BASELINE_STUDY_CELL,
        challengers=(challenger,),
        comparison_class=ComparisonClass.STRATEGY_COUNTERFACTUAL,
        computation_path=_replay_path(),
        interpretation="strategy_counterfactual",
        equality_evidence=evidence,
    )
    assert compatibility.compatibility_status == "compatible"
    assert compatibility.required_equalities_satisfied is True
    assert "population_delta" in envelope.payload.metric_registry
    assert "funnel_delta" in envelope.payload.metric_registry
    assert envelope.payload.changed_dimension_ids == ("data_lineage", "strategy_profile")
    # determinism: same inputs, same comparison id
    envelope_b, _ = build_comparison(
        baseline=BASELINE_STUDY_CELL,
        challengers=(challenger,),
        comparison_class=ComparisonClass.STRATEGY_COUNTERFACTUAL,
        computation_path=_replay_path(),
        interpretation="strategy_counterfactual",
        equality_evidence=evidence,
    )
    assert envelope_b.comparison_id == envelope.comparison_id


def test_failed_equalities_degrade_to_config_diff_only() -> None:
    challenger = _challenger_cell()
    evidence = {
        BASELINE_STUDY_CELL.cell_id: _evidence(BASELINE_STUDY_CELL.cell_id),
        challenger.cell_id: _evidence(challenger.cell_id, cost_policy_id="verified_nq_v1"),
    }
    envelope, compatibility = build_comparison(
        baseline=BASELINE_STUDY_CELL,
        challengers=(challenger,),
        comparison_class=ComparisonClass.STRATEGY_COUNTERFACTUAL,
        computation_path=_replay_path(),
        interpretation="strategy_counterfactual",
        equality_evidence=evidence,
    )
    assert compatibility.compatibility_status == "config_diff_only"
    assert compatibility.compatible_for_metric_delta is False
    assert "cost_policy_id" in compatibility.differing_fields
    # NO delta families beyond identity + compatibility
    assert set(envelope.payload.metric_registry) == {
        "identity_delta",
        "compatibility_delta",
    }


def test_cohort_classes_never_select_execution_or_prop_families() -> None:
    for cls in ("cohort_descriptive", "cohort_model"):
        families = {family.value for family in select_delta_families(cls)}
        assert not families & {
            "execution_delta",
            "cost_delta",
            "prop_delta",
            "payout_delta",
            "portfolio_delta",
            "risk_delta",
        }, cls


def test_everything_else_equal_classes_are_structural() -> None:
    base = _baseline_cell_payload()
    lineage_changed = StudyCellIdentity.from_payload(
        base.model_copy(
            update={
                "data_lineage": base.data_lineage.model_copy(
                    update={"v2_artifact_id": "f" * 64}
                )
            }
        )
    )
    envelope, compatibility = build_comparison(
        baseline=BASELINE_STUDY_CELL,
        challengers=(lineage_changed,),
        comparison_class=ComparisonClass.DATA_LINEAGE,
        computation_path=_replay_path(),
        interpretation="descriptive",
        equality_evidence={},  # structural — no caller-fabricated tokens needed
    )
    assert compatibility.compatibility_status == "compatible"
    assert compatibility.required_equalities_satisfied is True
    # identical cells cannot claim a data-lineage comparison (exemption removed)
    with pytest.raises(ComparisonClassMismatchError, match="identical"):
        build_comparison(
            baseline=BASELINE_STUDY_CELL,
            challengers=(BASELINE_STUDY_CELL,),
            comparison_class=ComparisonClass.DATA_LINEAGE,
            computation_path=_replay_path(),
            interpretation="descriptive",
            equality_evidence={},
        )


def _gated_cell_outside_validation() -> StudyCellIdentity:
    """A model-gated cell CANNOT lawfully validate in V1 (S11 is blocked and
    execution-affecting decision policies are planned-only), so the comparison
    layer that R5+ will use post-ratification is exercised on a deliberately
    validation-bypassed construction. The cell-level refusal itself is covered
    in test_study_cell."""

    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        canonical_contract_sha256,
    )

    base = _baseline_cell_payload()
    payload = type(base).model_construct(
        **{
            **{name: getattr(base, name) for name in type(base).model_fields},
            "decision_policy": base.decision_policy.model_copy(
                update={
                    "decision_policy_key": "fixed_probability_threshold_v1",
                    "resolved_decision_policy_id": "9" * 64,
                }
            ),
        }
    )
    return StudyCellIdentity.model_construct(
        cell_id=canonical_contract_sha256(payload), payload=payload
    )


def test_model_gate_execution_requires_a_new_trade_stream() -> None:
    gated = _gated_cell_outside_validation()
    same_stream = {
        BASELINE_STUDY_CELL.cell_id: {
            "frozen_model_fit_id": "1" * 64,
            "gross_trade_stream_hash": "2" * 64,
        },
        gated.cell_id: {
            "frozen_model_fit_id": "1" * 64,
            "gross_trade_stream_hash": "2" * 64,
        },
    }
    _, compatibility = build_comparison(
        baseline=BASELINE_STUDY_CELL,
        challengers=(gated,),
        comparison_class=ComparisonClass.MODEL_GATE_EXECUTION,
        computation_path=_replay_path(),
        interpretation="strategy_counterfactual",
        equality_evidence=same_stream,
    )
    assert compatibility.compatibility_status == "config_diff_only"
    assert any(
        "required difference" in reason
        for reason in compatibility.incompatibility_reasons
    )
    new_stream = {
        BASELINE_STUDY_CELL.cell_id: same_stream[BASELINE_STUDY_CELL.cell_id],
        gated.cell_id: {
            "frozen_model_fit_id": "1" * 64,
            "gross_trade_stream_hash": "3" * 64,
        },
    }
    _, compatibility = build_comparison(
        baseline=BASELINE_STUDY_CELL,
        challengers=(gated,),
        comparison_class=ComparisonClass.MODEL_GATE_EXECUTION,
        computation_path=_replay_path(),
        interpretation="strategy_counterfactual",
        equality_evidence=new_stream,
    )
    assert compatibility.compatibility_status == "compatible"


def test_config_diff_only_carries_the_full_field_diff() -> None:
    from alpha_lab.agents.data_infra.ifvg.study.comparison_contracts import (
        build_config_diff,
    )

    challenger = _challenger_cell()
    diff = build_config_diff(BASELINE_STUDY_CELL, (challenger,))
    assert "strategy_profile" in diff
    assert "strategy_profile_id" in diff["strategy_profile"]
    evidence = {
        BASELINE_STUDY_CELL.cell_id: _evidence(BASELINE_STUDY_CELL.cell_id),
        challenger.cell_id: _evidence(
            challenger.cell_id, cost_policy_id="verified_nq_v1"
        ),
    }
    _, compatibility = build_comparison(
        baseline=BASELINE_STUDY_CELL,
        challengers=(challenger,),
        comparison_class=ComparisonClass.STRATEGY_COUNTERFACTUAL,
        computation_path=_replay_path(),
        interpretation="strategy_counterfactual",
        equality_evidence=evidence,
    )
    assert any(
        field.startswith("strategy_profile.")
        for field in compatibility.differing_fields
    )


def test_feature_only_requires_the_resolved_model_identity() -> None:
    for cls in (ComparisonClass.FEATURE_ONLY, ComparisonClass.COHORT_MODEL):
        assert "resolved_model_protocol_id" in REQUIRED_EQUALITIES[cls]
        assert "model_protocol_key" not in REQUIRED_EQUALITIES[cls]


def test_prop_families_only_for_prop_paths() -> None:
    for cls, families in DELTA_FAMILY_SELECTION.items():
        values = {family.value for family in families}
        if "prop_delta" in values:
            assert cls in {
                "risk_policy",
                "prop_realization",
                "payout_policy",
                "stress_scenario",
                "composite_preregistered",
            }, cls


def test_comparison_result_subject_is_a_typed_discriminated_reference() -> None:
    """R5-FIX (gate finding 6): study-cell references and search-lane
    derivation references are separate TYPES — the identity domain is
    enforced by the contract, not by a docstring."""

    import pytest as _pytest
    from pydantic import ValidationError

    from alpha_lab.agents.data_infra.ifvg.study.comparison_contracts import (
        ComparisonCompatibility,
        ComparisonResult,
        SearchDerivationComparisonSubject,
        StudyCellComparisonSubject,
    )

    compatibility = ComparisonCompatibility(
        per_dimension_match={"strategy_profile": False},
        required_equalities_satisfied=True,
        registered_single_axis_delta_only=True,
        compatible_for_metric_delta=True,
        compatibility_status="compatible",
        incompatibility_reasons=(),
        differing_fields=(),
    )
    common = {
        "compatibility": compatibility,
        "delta_reports": {"population_delta": {"entity_kind": "setup"}},
        "config_diff": None,
        "evidence_links": ("b" * 64,),
    }
    study = ComparisonResult(
        subject=StudyCellComparisonSubject(comparison_id="a" * 64), **common
    )
    search = ComparisonResult(
        subject=SearchDerivationComparisonSubject(derivation_id="c" * 64), **common
    )
    assert study.subject.subject_kind == "study_cell_comparison_v1"
    assert search.subject.subject_kind == "search_cross_profile_derivation_v1"
    # the discriminator refuses an untyped/mislabeled reference outright
    with _pytest.raises(ValidationError):
        ComparisonResult(subject={"comparison_id": "a" * 64}, **common)
    with _pytest.raises(ValidationError):
        ComparisonResult(
            subject={
                "subject_kind": "study_cell_comparison_v1",
                "derivation_id": "c" * 64,
            },
            **common,
        )
    # a bare 64-hex string is no longer a lawful subject at all
    with _pytest.raises(ValidationError):
        ComparisonResult(subject="a" * 64, **common)
