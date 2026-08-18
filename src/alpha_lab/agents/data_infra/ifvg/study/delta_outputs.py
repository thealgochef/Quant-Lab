"""Delta output families + automatic per-class selection (DT §4; brief §7A.6–8).

Guardrails encoded structurally: cohort classes never select execution/prop
families; prop families appear only for classes whose computation path
includes prop resimulation.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType

__all__ = ["DeltaOutputFamily", "DELTA_FAMILY_SELECTION", "select_delta_families"]


class DeltaOutputFamily(StrEnum):
    IDENTITY_DELTA = "identity_delta"
    COMPATIBILITY_DELTA = "compatibility_delta"
    POPULATION_DELTA = "population_delta"
    FUNNEL_DELTA = "funnel_delta"
    SEQUENCE_DELTA = "sequence_delta"
    TIMING_DELTA = "timing_delta"
    FEATURE_COVERAGE_DELTA = "feature_coverage_delta"
    PREDICTIVE_DELTA = "predictive_delta"
    CALIBRATION_DELTA = "calibration_delta"
    LABEL_DELTA = "label_delta"
    ECONOMIC_DELTA = "economic_delta"
    EXECUTION_DELTA = "execution_delta"
    COST_DELTA = "cost_delta"
    RISK_DELTA = "risk_delta"
    TAIL_DELTA = "tail_delta"
    STABILITY_DELTA = "stability_delta"
    CONCENTRATION_DELTA = "concentration_delta"
    PROP_DELTA = "prop_delta"
    PAYOUT_DELTA = "payout_delta"
    PORTFOLIO_DELTA = "portfolio_delta"
    ROBUSTNESS_DELTA = "robustness_delta"
    EVIDENCE_QUALITY_DELTA = "evidence_quality_delta"
    ENGINEERING_DELTA = "engineering_delta"


_F = DeltaOutputFamily
_BASE = (_F.IDENTITY_DELTA, _F.COMPATIBILITY_DELTA)
_STRATEGY = (
    *_BASE,
    _F.POPULATION_DELTA,
    _F.FUNNEL_DELTA,
    _F.SEQUENCE_DELTA,
    _F.TIMING_DELTA,
    _F.ECONOMIC_DELTA,
    _F.TAIL_DELTA,
    _F.STABILITY_DELTA,
    _F.CONCENTRATION_DELTA,
    _F.EVIDENCE_QUALITY_DELTA,
)

#: Automatic family selection per comparison class (acceptance §7A.19.8).
DELTA_FAMILY_SELECTION: MappingProxyType[str, tuple[DeltaOutputFamily, ...]] = (
    MappingProxyType(
        {
            "feature_only": (
                *_BASE,
                _F.FEATURE_COVERAGE_DELTA,
                _F.PREDICTIVE_DELTA,
                _F.CALIBRATION_DELTA,
                _F.STABILITY_DELTA,
                _F.CONCENTRATION_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "cohort_descriptive": (
                *_BASE,
                _F.POPULATION_DELTA,
                _F.FUNNEL_DELTA,
                _F.CONCENTRATION_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "cohort_model": (
                *_BASE,
                _F.PREDICTIVE_DELTA,
                _F.CALIBRATION_DELTA,
                _F.STABILITY_DELTA,
                _F.CONCENTRATION_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "strategy_counterfactual": _STRATEGY,
            "label_counterfactual": (
                *_BASE,
                _F.LABEL_DELTA,
                _F.PREDICTIVE_DELTA,
                _F.CALIBRATION_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "model_protocol": (
                *_BASE,
                _F.PREDICTIVE_DELTA,
                _F.CALIBRATION_DELTA,
                _F.STABILITY_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "model_gate_execution": (
                *_STRATEGY,
                _F.EXECUTION_DELTA,
                _F.COST_DELTA,
            ),
            "execution_policy": (
                *_BASE,
                _F.SEQUENCE_DELTA,
                _F.TIMING_DELTA,
                _F.ECONOMIC_DELTA,
                _F.EXECUTION_DELTA,
                _F.COST_DELTA,
                _F.TAIL_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "cost_policy": (
                *_BASE,
                _F.COST_DELTA,
                _F.ECONOMIC_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "risk_policy": (
                *_BASE,
                _F.RISK_DELTA,
                _F.PROP_DELTA,
                _F.PAYOUT_DELTA,
                _F.TAIL_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "prop_realization": (
                *_BASE,
                _F.PROP_DELTA,
                _F.PAYOUT_DELTA,
                _F.TAIL_DELTA,
                _F.RISK_DELTA,
                _F.ROBUSTNESS_DELTA,
            ),
            "payout_policy": (
                *_BASE,
                _F.PAYOUT_DELTA,
                _F.PROP_DELTA,
                _F.TAIL_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "portfolio_policy": (
                *_BASE,
                _F.PORTFOLIO_DELTA,
                _F.PAYOUT_DELTA,
                _F.TAIL_DELTA,
                _F.ROBUSTNESS_DELTA,
            ),
            "data_lineage": (
                *_BASE,
                _F.POPULATION_DELTA,
                _F.FUNNEL_DELTA,
                _F.PREDICTIVE_DELTA,
                _F.ECONOMIC_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
                _F.ENGINEERING_DELTA,
            ),
            "validation_protocol": (
                *_BASE,
                _F.PREDICTIVE_DELTA,
                _F.CALIBRATION_DELTA,
                _F.STABILITY_DELTA,
                _F.EVIDENCE_QUALITY_DELTA,
            ),
            "stress_scenario": (
                *_BASE,
                _F.PROP_DELTA,
                _F.TAIL_DELTA,
                _F.ROBUSTNESS_DELTA,
            ),
            "composite_preregistered": (
                *_STRATEGY,
                _F.PROP_DELTA,
                _F.ROBUSTNESS_DELTA,
            ),
            "configuration_neighbor": (
                *_STRATEGY,
                _F.ROBUSTNESS_DELTA,
            ),
            "baseline_to_child": _STRATEGY,
            "child_to_child": _STRATEGY,
        }
    )
)


def select_delta_families(comparison_class: str) -> tuple[DeltaOutputFamily, ...]:
    families = DELTA_FAMILY_SELECTION.get(comparison_class)
    if families is None:
        raise ValueError(f"unknown comparison class {comparison_class!r}")
    return families
