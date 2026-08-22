"""Deterministic insight rendering (CS §12; brief §9.22; FUX §26).

Seven fixed categories, verbatim template renders, exact ``EvidenceRef``s,
causally-neutral wording. Insights cite ``match_basis`` and are suppressed
for ``not_comparable`` populations; no AI-written summary exists or may
replace this panel. Forbidden publishable wording is structurally refused.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import Field

from ..study.population_delta import PopulationDeltaReport
from .identities import SHA256_PATTERN, EnvelopeBase, FrozenContract, register_identity_pair
from .robustness import RobustnessReport
from .strategy_metrics import StrategyMetrics

__all__ = [
    "InsightCategory",
    "EvidenceRef",
    "Insight",
    "InsightPanel",
    "InsightPanelPayload",
    "InsightPanelEnvelope",
    "render_insight_panel",
    "FORBIDDEN_INSIGHT_WORDING",
]

#: Wording that would misrepresent development output as validated/publishable.
FORBIDDEN_INSIGHT_WORDING: tuple[str, ...] = (
    "best",
    "winner",
    "production ready",
    "production-ready",
    "validated",
    "live ready",
    "live-ready",
    "robust representative",
    "publishable",
    "causes",
    "caused by",
    "because of the change",
)


class InsightCategory(StrEnum):
    WHAT_CHANGED = "What Changed"
    EDGE_EFFECT = "Edge Effect"
    PROP_EFFECT = "Prop Effect"
    ROBUSTNESS = "Robustness"
    CONCENTRATION_WARNING = "Concentration Warning"
    EVIDENCE_QUALITY = "Evidence Quality"
    RECOMMENDED_INSPECTION = "Recommended Inspection"


class EvidenceRef(FrozenContract):
    # CS §12's registered kinds plus the two population entity kinds the delta
    # layer emits (candidate/decision — DEV-R2-7); no dead vocabulary.
    kind: Literal[
        "setup", "trade", "candidate", "decision", "account_event",
        "child", "simulation", "axis",
    ]
    identifier: str


class Insight(FrozenContract):
    category: InsightCategory
    text: str
    evidence: tuple[EvidenceRef, ...]
    match_basis: str | None = None
    suppressed: bool = False
    suppression_reason: str | None = None


class InsightPanel(FrozenContract):
    insights: tuple[Insight, ...]

    def by_category(self, category: InsightCategory) -> tuple[Insight, ...]:
        return tuple(i for i in self.insights if i.category is category)


class InsightPanelPayload(FrozenContract):
    """One persisted deterministic insight panel (S14; DEV-R4-7 closure).

    The subject binding (search × child) is the hashed content together with
    the panel itself, so re-running S14 for the same search reuses the same
    immutable artifact and a different child/search can never alias it.
    """

    search_id: str = Field(pattern=SHA256_PATTERN)
    subject_core_replay_id: str = Field(pattern=SHA256_PATTERN)
    panel: InsightPanel


class InsightPanelEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "insight_panel_id"

    insight_panel_id: str = Field(pattern=SHA256_PATTERN)
    payload: InsightPanelPayload


def _assert_wording_lawful(text: str) -> str:
    lowered = text.lower()
    for token in FORBIDDEN_INSIGHT_WORDING:
        if token in lowered:
            raise ValueError(f"forbidden insight wording {token!r} in: {text!r}")
    return text


def render_insight_panel(
    *,
    changed_axis_labels: tuple[str, ...],
    child_id: str,
    metrics: StrategyMetrics,
    baseline_metrics: StrategyMetrics | None,
    population_deltas: tuple[PopulationDeltaReport, ...] = (),
    robustness: RobustnessReport | None = None,
    sample_caveat: str | None = None,
) -> InsightPanel:
    """Render the seven fixed categories deterministically (no free text)."""

    insights: list[Insight] = []
    child_ref = EvidenceRef(kind="child", identifier=child_id)

    # 1. What Changed
    changed = ", ".join(changed_axis_labels) if changed_axis_labels else "nothing"
    insights.append(
        Insight(
            category=InsightCategory.WHAT_CHANGED,
            text=_assert_wording_lawful(
                f"This configuration differs from its baseline on: {changed}. "
                "All other dimensions are frozen."
            ),
            evidence=(child_ref,),
        )
    )

    # 2. Edge Effect (observational; paired with the baseline when available)
    if baseline_metrics is not None and metrics.net_expectancy_r is not None and (
        baseline_metrics.net_expectancy_r is not None
    ):
        delta = metrics.net_expectancy_r - baseline_metrics.net_expectancy_r
        text = (
            f"Observed net expectancy is {metrics.net_expectancy_r:+.3f}R over "
            f"{metrics.executed_trades} resolved trades vs the baseline's "
            f"{baseline_metrics.net_expectancy_r:+.3f}R over "
            f"{baseline_metrics.executed_trades}; the observed difference is "
            f"{delta:+.3f}R. Sample sizes limit interpretation."
        )
    elif metrics.net_expectancy_r is not None:
        text = (
            f"Observed net expectancy is {metrics.net_expectancy_r:+.3f}R over "
            f"{metrics.executed_trades} resolved trades. No compatible baseline "
            "delta is available."
        )
    else:
        text = "No resolved trades exist; no edge observation is available."
    insights.append(
        Insight(
            category=InsightCategory.EDGE_EFFECT,
            text=_assert_wording_lawful(text),
            evidence=(child_ref,),
        )
    )

    # 3. Prop Effect (R3 wires simulations; explicit absence until then)
    insights.append(
        Insight(
            category=InsightCategory.PROP_EFFECT,
            text=_assert_wording_lawful(
                "No prop simulation is attached to this result yet; prop-effect "
                "insights render after the prop lifecycle produces a simulation "
                "for this exact trade stream."
            ),
            evidence=(child_ref,),
        )
    )

    # 4. Robustness
    if robustness is not None:
        if robustness.knife_edge:
            text = (
                "Neighbor checks flag a knife-edge: "
                + "; ".join(robustness.knife_edge_warnings)
            )
        elif robustness.neighbor_checks:
            text = (
                f"±1-step neighbors were checked ({len(robustness.neighbor_checks)}); "
                f"worst degradation {robustness.worst_neighbor_degradation_r:+.3f}R; "
                f"minimum plateau width {robustness.plateau_width}."
            )
        else:
            text = "No evaluable ±1-step neighbors exist on the searched grid."
        evidence = (
            child_ref,
            *(
                EvidenceRef(kind="child", identifier=check.neighbor_id)
                for check in robustness.neighbor_checks
            ),
        )
        insights.append(
            Insight(
                category=InsightCategory.ROBUSTNESS,
                text=_assert_wording_lawful(text),
                evidence=evidence,
            )
        )
    else:
        insights.append(
            Insight(
                category=InsightCategory.ROBUSTNESS,
                text="No robustness evaluation is attached to this result.",
                evidence=(child_ref,),
            )
        )

    # 5. Concentration Warning
    concentration_parts = []
    if metrics.top_day_pnl_share is not None and metrics.top_day_pnl_share > 0.4:
        concentration_parts.append(
            f"one trading day carries {metrics.top_day_pnl_share:.0%} of absolute PnL"
        )
    if metrics.top_setup_pnl_share is not None and metrics.top_setup_pnl_share > 0.25:
        concentration_parts.append(
            f"one setup carries {metrics.top_setup_pnl_share:.0%} of absolute PnL"
        )
    insights.append(
        Insight(
            category=InsightCategory.CONCENTRATION_WARNING,
            text=_assert_wording_lawful(
                "; ".join(concentration_parts)
                if concentration_parts
                else "No day/setup concentration threshold is exceeded."
            ),
            evidence=(child_ref,),
        )
    )

    # 6. Evidence Quality — population deltas cite match_basis; not_comparable
    #    populations SUPPRESS commonality claims entirely.
    for delta in population_deltas:
        if delta.match_basis == "not_comparable":
            insights.append(
                Insight(
                    category=InsightCategory.EVIDENCE_QUALITY,
                    text=_assert_wording_lawful(
                        f"The {delta.entity_kind} population comparison is disabled: "
                        f"{delta.match_basis_reason}. No common/added/removed "
                        "membership is claimed."
                    ),
                    evidence=(child_ref,),
                    match_basis=delta.match_basis,
                    suppressed=True,
                    suppression_reason=delta.match_basis_reason,
                )
            )
        else:
            insights.append(
                Insight(
                    category=InsightCategory.EVIDENCE_QUALITY,
                    text=_assert_wording_lawful(
                        f"The {delta.entity_kind} population delta uses "
                        f"{delta.match_basis}: {len(delta.common_keys)} common, "
                        f"{len(delta.added_keys)} added, {len(delta.removed_keys)} "
                        "removed."
                    ),
                    evidence=(
                        child_ref,
                        *(
                            EvidenceRef(kind=delta.entity_kind, identifier=key)
                            for key in (*delta.added_keys[:5], *delta.removed_keys[:5])
                        ),
                    ),
                    match_basis=delta.match_basis,
                )
            )
    if sample_caveat:
        insights.append(
            Insight(
                category=InsightCategory.EVIDENCE_QUALITY,
                text=_assert_wording_lawful(sample_caveat),
                evidence=(child_ref,),
            )
        )
    if not population_deltas and not sample_caveat:
        # all seven fixed categories always render (CS §12; FUX-RES-008)
        insights.append(
            Insight(
                category=InsightCategory.EVIDENCE_QUALITY,
                text=(
                    "No population delta or sample caveat is attached to this "
                    "result."
                ),
                evidence=(child_ref,),
            )
        )

    # 7. Recommended Inspection
    inspect: list[str] = []
    first_divergences = [
        d.first_divergence for d in population_deltas if d.first_divergence is not None
    ]
    if first_divergences:
        first = first_divergences[0]
        inspect.append(
            f"open the first divergent {first.side} entity "
            f"{first.entity_key[:16]}… on {first.trading_day} in the verifier"
        )
        divergence_evidence = (
            EvidenceRef(kind="setup", identifier=first.entity_key),
        )
    else:
        divergence_evidence = ()
        inspect.append("open the configuration's executed trades in the verifier")
    insights.append(
        Insight(
            category=InsightCategory.RECOMMENDED_INSPECTION,
            text=_assert_wording_lawful("; ".join(inspect)),
            evidence=(child_ref, *divergence_evidence),
        )
    )
    return InsightPanel(insights=tuple(insights))


def _example_insight_panel_payload() -> InsightPanelPayload:
    return InsightPanelPayload(
        search_id="a" * 64,
        subject_core_replay_id="b" * 64,
        panel=InsightPanel(
            insights=(
                Insight(
                    category=InsightCategory.WHAT_CHANGED,
                    text="Compared 1 changed axis against the study baseline.",
                    evidence=(EvidenceRef(kind="child", identifier="b" * 64),),
                ),
            )
        ),
    )


register_identity_pair(
    name="InsightPanel",
    envelope_cls=InsightPanelEnvelope,
    payload_cls=InsightPanelPayload,
    id_field="insight_panel_id",
    example_factory=_example_insight_panel_payload,
)
