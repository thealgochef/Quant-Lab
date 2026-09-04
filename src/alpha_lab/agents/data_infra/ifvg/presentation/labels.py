"""The human-label registry (UI-3; plan §6.5) — pure.

Human names for the technical keys the screens show: strategy profiles,
feature bundles and blocks, charter objectives (from the metric registry —
one source), model protocols, regime algorithms, stamps, statuses, roles,
comparison classes, feature tiers and the reviewer verdicts. A label never
replaces the technical key — ``label_for`` returns the key itself when no
label is registered (never an invention) and ``technical_key_for`` recovers
the key from a label. The availability chips make proposed / ratified /
offline / planned / implemented / blocked / experimental / superseded
states distinct by glyph AND word.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from ..search.charter import OBJECTIVE_DIRECTIONS
from .metric_registry import describe
from .review_vocabulary import VERDICT_LABELS
from .status_vocabulary import STATUS_SPECS, UiStatus

__all__ = [
    "ALGORITHM_LABELS",
    "AVAILABILITY_CHIPS",
    "BLOCK_LABELS",
    "BLOCK_STATUS_LABELS",
    "BUNDLE_LABELS",
    "COMPARISON_CLASS_LABELS",
    "FEATURE_TIER_LABELS",
    "LABEL_REGISTRIES",
    "MODEL_PROTOCOL_LABELS",
    "OBJECTIVE_LABELS",
    "PREPARATION_STATUS_LABELS",
    "PROFILE_CAPABILITY_LABELS",
    "PROFILE_LABELS",
    "REGIME_ROLE_LABELS",
    "REGIME_STATUS_LABELS",
    "STAMP_LABELS",
    "AvailabilityKind",
    "availability_chip",
    "availability_for_algorithm",
    "availability_for_block_status",
    "availability_for_stamp",
    "label_for",
    "technical_key_for",
]


def _frozen(mapping: dict[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(mapping))


PROFILE_LABELS: Mapping[str, str] = _frozen(
    {
        "ifvg_v2_doc_default_fresh_static_1r": "Doc-default (fresh entries, static 1R)",
        "ifvg_v2_ict_clean_fresh_static_1r": "ICT clean (fresh entries, static 1R)",
        "ifvg_v2_ict_clean_pure_retest_static_1r": "ICT clean (pure retest, static 1R)",
        "ifvg_v2_weak_counter_displacement_research": "Weak counter-displacement (research)",
        "ifvg_v1_legacy_candidate_stream": "Legacy v1 candidate stream",
    }
)

BUNDLE_LABELS: Mapping[str, str] = _frozen(
    {
        "B0_CORE": "Core",
        "B1_CORE_STRUCTURE": "Core + structure",
        "B2_CORE_ORDER_FLOW": "Core + order flow",
        "B3_CORE_STRUCTURE_ORDER_FLOW": "Core + structure + order flow",
        "B4_CORE_STRUCTURE_LIQUIDITY": "Core + structure + liquidity",
        "B5_CORE_STRUCTURE_ORDER_FLOW_REGIME": "Core + structure + order flow + regime",
        "B6_CORE_STRUCTURE_ORDER_FLOW_EXECUTION_LIQUIDITY": (
            "Core + structure + order flow + execution liquidity"
        ),
        "B7_CORE_REGIME": "Core + regime",
        "BP0_CONTEXT_BAR_PANEL": "Context bar panel (panel artifact)",
    }
)

BLOCK_LABELS: Mapping[str, str] = _frozen(
    {
        "IFVG_CORE_BASELINE_V1": "Core baseline",
        "IFVG_SESSION_CONTEXT_V1": "Session context",
        "IFVG_STRUCTURE_CONTEXT_V1": "Structure context",
        "IFVG_STRUCTURE_CONTEXT_240_V1": "Structure context (240m)",
        "IFVG_DISPLACEMENT_CONTEXT_V2": "Displacement context",
        "IFVG_LIQUIDITY_CONTEXT_V1": "Liquidity context",
        "IFVG_VOLATILITY_CONTEXT_V1": "Volatility context",
        "IFVG_ORDER_FLOW_MBP1_V1": "Order flow (MBP-1)",
        "IFVG_REGIME_CONTEXT_V1": "Regime context",
        "IFVG_KEY_LEVEL_CONTEXT_V1": "Key-level context",
        "IFVG_EXECUTION_LIQUIDITY_V1": "Execution liquidity",
        "IFVG_CONTEXT_BAR_PANEL_V1": "Context bar panel",
    }
)

#: one source: the metric registry's human names
OBJECTIVE_LABELS: Mapping[str, str] = _frozen(
    {key: describe(key).human_name for key in OBJECTIVE_DIRECTIONS}
)

MODEL_PROTOCOL_LABELS: Mapping[str, str] = _frozen(
    {
        "reference_prevalence_v1": "Prevalence reference",
        "ifvg_context_logistic_l2_v1": "Logistic regression (L2)",
        "ifvg_context_catboost_binary_v1": "CatBoost (frozen tier lane)",
        "ifvg_context_catboost_bundle_v1": "CatBoost (bundle-aware rung)",
        "ifvg_context_gam_v1": "GAM (preregistered basis penalty)",
    }
)

ALGORITHM_LABELS: Mapping[str, str] = _frozen(
    {
        "kmeans_v1": "K-means",
        "minibatch_kmeans_v1": "Mini-batch K-means",
        "gaussian_mixture_v1": "Gaussian mixture",
        "spectral_clustering_train_only_v1": "Spectral clustering (training-only)",
        "nystrom_kmeans_v1": "Nyström K-means",
        "surrogate_assignment_logistic_v1": "Surrogate logistic assignment",
    }
)

STAMP_LABELS: Mapping[str, str] = _frozen(
    {
        "proposed_protocol_default": "Proposed default — owner ratification required",
        "registered_storage_budget": "Registered storage budget — refuses before publication",
        "custom": "Custom — not a proposed default",
        "owner_ratified": "Owner-ratified",
    }
)

BLOCK_STATUS_LABELS: Mapping[str, str] = _frozen(
    {
        "available": "Available",
        "planned": "Planned",
        "blocked_missing_source": "Blocked — missing source",
        "blocked_owner_decision": "Blocked — owner decision",
        "experimental": "Experimental",
        "superseded": "Superseded",
    }
)

REGIME_STATUS_LABELS: Mapping[str, str] = _frozen(
    {
        "planned": "Planned",
        "descriptive_only": "Descriptive only",
        "stratification_ready": "Stratification-ready",
        "feature_eligible": "Feature-eligible",
        "model_feature": "Model feature",
        "experimental": "Experimental",
        "blocked_no_oos_assignment": "Blocked — no out-of-sample assignment",
        "blocked_insufficient_coverage": "Blocked — insufficient coverage",
        "superseded": "Superseded",
    }
)

REGIME_ROLE_LABELS: Mapping[str, str] = _frozen(
    {
        "descriptive_only": "Descriptive only",
        "stratification_only": "Stratification only",
        "feature_generator": "Feature generator",
        "predictive_model": "Predictive model",
        "decision_policy": "Decision policy",
        "execution_gate_candidate": "Execution-gate candidate",
        "frozen_execution_gate": "Frozen execution gate",
        "monitoring_only": "Monitoring only",
    }
)

PROFILE_CAPABILITY_LABELS: Mapping[str, str] = _frozen(
    {
        "runnable": "Runnable",
        "blocked": "Blocked",
        "analysis_only": "Analysis only",
        "legacy_read_only": "Legacy read-only",
    }
)

PREPARATION_STATUS_LABELS: Mapping[str, str] = _frozen(
    {
        "not_prepared": "Not prepared",
        "preparing": "Preparing",
        "failed": "Failed",
        "context_ready": "Context-ready",
        "superseded": "Superseded",
        "legacy_only": "Legacy only",
        "blocked": "Blocked",
    }
)

COMPARISON_CLASS_LABELS: Mapping[str, str] = _frozen(
    {
        "cohort_descriptive": "Cohort descriptive",
        "feature_only": "Feature-only model",
        "cohort_model": "Cohort model",
        "stratified_prop": "Stratified prop",
        "stratified_frontier": "Stratified frontier",
    }
)

FEATURE_TIER_LABELS: Mapping[str, str] = _frozen(
    {
        "M0": "M0 — baseline candidate features",
        "M1_PRIMARY": "M1 — M0 + structure context (primary timeframes)",
        "M1_PLUS_240_EXPERIMENTAL": "M1 + 240m structure (experimental)",
        "M2": "M2 — M1 + displacement context",
        "M3": "M3 — M2 + equal-level pool and sweep context",
    }
)

LABEL_REGISTRIES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "profile": PROFILE_LABELS,
        "bundle": BUNDLE_LABELS,
        "block": BLOCK_LABELS,
        "objective": OBJECTIVE_LABELS,
        "model_protocol": MODEL_PROTOCOL_LABELS,
        "algorithm": ALGORITHM_LABELS,
        "stamp": STAMP_LABELS,
        "block_status": BLOCK_STATUS_LABELS,
        "regime_status": REGIME_STATUS_LABELS,
        "regime_role": REGIME_ROLE_LABELS,
        "profile_capability": PROFILE_CAPABILITY_LABELS,
        "preparation_status": PREPARATION_STATUS_LABELS,
        "comparison_class": COMPARISON_CLASS_LABELS,
        "feature_tier": FEATURE_TIER_LABELS,
        "verdict": VERDICT_LABELS,
    }
)


def _registry(kind: str) -> Mapping[str, str]:
    try:
        return LABEL_REGISTRIES[str(kind)]
    except KeyError:
        raise ValueError(f"unregistered label kind {kind!r}") from None


def label_for(kind: str, key: str) -> str:
    """The human label, or the technical key itself when none is registered."""

    return _registry(kind).get(str(key), str(key))


def technical_key_for(kind: str, label: str) -> str | None:
    """The technical key behind a human label (``None`` when unknown)."""

    for key, value in _registry(kind).items():
        if value == label:
            return key
    return None


# ── availability chips (plan §5.2 / §7: distinct states by glyph + word) ────


class AvailabilityKind(StrEnum):
    IMPLEMENTED = "implemented"
    PLANNED = "planned"
    PROPOSED = "proposed"
    RATIFIED = "ratified"
    RESEARCH_ONLY_OFFLINE = "research_only_offline"
    BLOCKED = "blocked"
    EXPERIMENTAL = "experimental"
    SUPERSEDED = "superseded"


AVAILABILITY_CHIPS: Mapping[AvailabilityKind, tuple[UiStatus, str]] = MappingProxyType(
    {
        AvailabilityKind.IMPLEMENTED: (UiStatus.COMPLETE, "Implemented (V1)"),
        AvailabilityKind.PLANNED: (UiStatus.NOT_SELECTED, "Planned post-V1 — visible, disabled"),
        AvailabilityKind.PROPOSED: (
            UiStatus.WARNING,
            "Proposed default — owner ratification required",
        ),
        AvailabilityKind.RATIFIED: (UiStatus.PASS, "Owner-ratified"),
        AvailabilityKind.RESEARCH_ONLY_OFFLINE: (
            UiStatus.INFORMATIONAL,
            "Research-only offline — no serving or execution role",
        ),
        AvailabilityKind.BLOCKED: (
            UiStatus.BLOCKED,
            "Blocked — a missing dependency or owner decision",
        ),
        AvailabilityKind.EXPERIMENTAL: (
            UiStatus.WARNING,
            "Experimental — excluded from every primary tier",
        ),
        AvailabilityKind.SUPERSEDED: (UiStatus.SUPERSEDED, "Superseded — read-only for audit"),
    }
)


def availability_chip(kind: AvailabilityKind | str) -> str:
    status, text = AVAILABILITY_CHIPS[AvailabilityKind(kind)]
    return f"{STATUS_SPECS[status].glyph} {text}"


_BLOCK_STATUS_AVAILABILITY: Mapping[str, AvailabilityKind] = MappingProxyType(
    {
        "available": AvailabilityKind.IMPLEMENTED,
        "planned": AvailabilityKind.PLANNED,
        "blocked_missing_source": AvailabilityKind.BLOCKED,
        "blocked_owner_decision": AvailabilityKind.BLOCKED,
        "experimental": AvailabilityKind.EXPERIMENTAL,
        "superseded": AvailabilityKind.SUPERSEDED,
    }
)


def availability_for_block_status(status: object) -> AvailabilityKind:
    value = str(getattr(status, "value", status))
    try:
        return _BLOCK_STATUS_AVAILABILITY[value]
    except KeyError:
        raise ValueError(f"unregistered feature block status {value!r}") from None


def availability_for_algorithm(entry: object) -> AvailabilityKind:
    """From the registry entry's ``implementation_status`` — never from wording."""

    status = str(getattr(entry, "implementation_status", ""))
    if status == "implemented":
        return AvailabilityKind.IMPLEMENTED
    if status == "planned":
        return AvailabilityKind.PLANNED
    raise ValueError(f"unregistered algorithm implementation status {status!r}")


_STAMP_AVAILABILITY: Mapping[str, AvailabilityKind] = MappingProxyType(
    {
        "proposed_protocol_default": AvailabilityKind.PROPOSED,
        "owner_ratified": AvailabilityKind.RATIFIED,
        "registered_storage_budget": AvailabilityKind.IMPLEMENTED,
    }
)


def availability_for_stamp(stamp: str) -> AvailabilityKind:
    try:
        return _STAMP_AVAILABILITY[str(stamp)]
    except KeyError:
        raise ValueError(f"unregistered stamp {stamp!r}") from None
