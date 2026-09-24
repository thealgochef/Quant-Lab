"""Study-side helpers for the configuration comparison (no screen code).

The study screen resolves the owner's configurator selections to configurations
of a verified study, freezes the exact plan and — only on an explicit owner
action — stores the owner approval for that exact plan. Tests use the same
functions.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    save_envelope_immutable,
)
from alpha_lab.propsim.funded.clock import ELAPSED_48_HOURS, TWO_BUSINESS_DAYS_FED_1600
from alpha_lab.propsim.funded.comparison_plan import (
    APPROVAL_STORE,
    COMPARISON_DECISIONS,
    PLAN_STORE,
    ComparisonConfigurationRef,
    ComparisonSourceRef,
    FundedComparisonApprovalEnvelope,
    FundedComparisonApprovalPayload,
    FundedComparisonPlanEnvelope,
    FundedComparisonPlanPayload,
)
from alpha_lab.propsim.funded.comparison_source import ComparisonSource
from alpha_lab.propsim.funded.plan import OwnerDecisionRef
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES, INSTRUMENTS

__all__ = [
    "COMPARISON_AXES",
    "axis_choices",
    "resolve_selection",
    "size_problem",
    "build_comparison_plan",
    "save_plan",
    "record_owner_approval",
    "VARIATION_AXES",
    "build_variation_plan",
    "core_source_description",
    "variation_axis_values",
    "variation_variants",
]

CLOCKS = {"two_business_days": TWO_BUSINESS_DAYS_FED_1600, "elapsed_48_hours": ELAPSED_48_HOURS}

#: the configurator settings this study varies, in display order
COMPARISON_AXES = ("enabled_entry_sessions", "opposing_parent_distance_ticks_max",
                   "opposing_min_gap_ticks", "parent_timeframes")


def axis_choices(source: ComparisonSource) -> dict[str, list[str]]:
    """Registry value ids offered per axis (those the verified study approved)."""

    out: dict[str, list[str]] = {axis: [] for axis in COMPARISON_AXES}
    for config in source.configurations:
        for axis in COMPARISON_AXES:
            value = config.axis_value_ids.get(axis)
            if value and value not in out[axis]:
                out[axis].append(value)
    return out


def resolve_selection(source: ComparisonSource, selections: dict[str, list[str]]
                      ) -> tuple[list[Any], list[dict[str, str]]]:
    """Every combination of the selected values -> an approved configuration.

    Returns ``(configurations, unavailable)``: combinations that the verified
    study did not approve are listed as unavailable (never silently dropped).
    """

    index = {tuple(c.axis_value_ids.get(a) for a in COMPARISON_AXES): c
             for c in source.configurations}
    chosen, unavailable = [], []
    lists = [list(selections.get(axis) or []) for axis in COMPARISON_AXES]
    for combo in product(*lists):
        config = index.get(tuple(combo))
        if config is None:
            unavailable.append(dict(zip(COMPARISON_AXES, combo, strict=True)))
        else:
            chosen.append(config)
    return chosen, unavailable


def size_problem(instrument: str, quantity: int, firm_keys: list[str]) -> str | None:
    spec = INSTRUMENTS[instrument]
    exposure = spec.mini_equivalent_tenths * int(quantity)
    for key in firm_keys:
        profile = FIRM_PROFILES[key]
        if exposure > profile.max_mini_equivalent_tenths:
            return (f"{quantity} x {spec.label} is above the {profile.firm_name} limit of "
                    f"{profile.max_minis_label}. Choose a supported size; a size is never "
                    "reduced silently.")
    return None


def build_comparison_plan(source: ComparisonSource, configurations: list[Any], *,
                          firm_keys: list[str], instrument: str, quantity: int,
                          cost_per_side_cents: int, processing: str = "two_business_days",
                          purpose: str = "historical_comparison"
                          ) -> FundedComparisonPlanEnvelope:
    problem = size_problem(instrument, quantity, firm_keys)
    if problem:
        raise ValueError(problem)
    refs = tuple(
        ComparisonConfigurationRef(
            name=c.name, display_name=c.display_name,
            axis_value_ids=tuple(sorted(c.axis_value_ids.items())),
            resolved_section_config_hash=c.resolved_section_config_hash,
            approval_id=c.approval_id)
        for c in sorted(configurations, key=lambda c: c.name))
    owner_clock = processing == "two_business_days"
    owner_cost = int(cost_per_side_cents) == 514
    decisions = [d for d in COMPARISON_DECISIONS
                 if not (d.subject == "Payout processing clock" and not owner_clock)
                 and not (d.subject == "Costs" and not owner_cost)]
    if processing != "two_business_days":
        decisions.append(OwnerDecisionRef(
            decided_on="plan setting", subject="Payout processing clock in this plan",
            decision="48 elapsed hours: an engineering comparison, not the owner's choice.",
            status="assumption"))
    if int(cost_per_side_cents) != 514:
        decisions.append(OwnerDecisionRef(
            decided_on="plan setting", subject="Costs in this plan",
            decision=f"${int(cost_per_side_cents) / 100:.2f} per contract per fill (differs "
                     "from the owner's $5.14).", status="assumption"))
    payload = FundedComparisonPlanPayload(
        purpose=purpose,
        source=ComparisonSourceRef(
            package_run_id=source.package.run_id,
            package_manifest_sha256=source.package.manifest_sha256,
            package_root_name=source.package.root.name, title=source.package.title,
            warmup_dates=source.warmup_dates, evaluation_dates=source.evaluation_dates,
            cutoff_utc=source.cutoff_utc),
        configurations=refs,
        firm_profiles=tuple(FIRM_PROFILES[k] for k in FIRM_PROFILES if k in firm_keys),
        instrument=instrument, quantity=int(quantity),
        cost_per_side_cents=int(cost_per_side_cents), processing=CLOCKS[processing],
        owner_decisions=tuple(decisions),
    )
    return FundedComparisonPlanEnvelope.from_payload(payload)


def save_plan(store_root: Path, envelope: FundedComparisonPlanEnvelope) -> str:
    plan_id = envelope.funded_comparison_plan_id
    if not has_envelope(store_root, PLAN_STORE, plan_id):
        save_envelope_immutable(store_root, PLAN_STORE, envelope)
    return plan_id


def record_owner_approval(store_root: Path, plan_id: str, *, approved_on: str, channel: str,
                          statement: str, scope: str) -> str:
    """Store the owner's approval of this exact plan (called only on an owner action)."""

    if not has_envelope(store_root, PLAN_STORE, plan_id):
        raise ValueError("approve a saved plan only")
    envelope = FundedComparisonApprovalEnvelope.from_payload(FundedComparisonApprovalPayload(
        funded_comparison_plan_id=plan_id, approved_on=approved_on, channel=channel,
        statement=statement, scope=scope))
    if not has_envelope(store_root, APPROVAL_STORE, envelope.funded_comparison_approval_id):
        save_envelope_immutable(store_root, APPROVAL_STORE, envelope)
    return envelope.funded_comparison_approval_id


# ── v2 plans: variations around a base configuration, each with its own size ──

def build_variation_plan(source: ComparisonSource, base_name: str,
                         variants: list[dict[str, Any]], *, firm_keys: list[str],
                         core_source: dict[str, str], processing: str = "two_business_days",
                         purpose: str = "historical_comparison", description: str = ""):
    """``variants``: dicts with ``changes`` (axis -> value id), ``instrument``,
    ``quantity`` and ``cost_per_contract_mills``."""

    from alpha_lab.propsim.funded.comparison_plan import (
        ComparisonVariantRef,
        CoreSourceRef,
        FundedComparisonPlanEnvelopeV2,
        FundedComparisonPlanPayloadV2,
    )
    from alpha_lab.propsim.funded.comparison_source import variation_configurations

    configurations = variation_configurations(source, base_name,
                                              [v["changes"] for v in variants])
    refs = []
    for variant, config in zip(variants, configurations, strict=True):
        problem = size_problem(variant["instrument"], variant["quantity"], firm_keys)
        if problem:
            raise ValueError(problem)
        exit_policy = config.axis_value_ids.get(
            "exit_policy", "exit_policy.fixed_target_v1").split(".", 1)[1]
        refs.append(ComparisonVariantRef(
            name=config.name, display_name=config.display_name,
            axis_value_ids=tuple(sorted(config.axis_value_ids.items())),
            resolved_section_config_hash=config.resolved_section_config_hash,
            exit_policy=exit_policy, instrument=variant["instrument"],
            quantity=int(variant["quantity"]),
            cost_per_contract_mills=int(variant["cost_per_contract_mills"]),
            in_verified_study=config.name in source.by_name,
            cache_configuration=config.name if config.name in source.by_name else base_name))
    payload = FundedComparisonPlanPayloadV2(
        purpose=purpose,
        source=ComparisonSourceRef(
            package_run_id=source.package.run_id,
            package_manifest_sha256=source.package.manifest_sha256,
            package_root_name=source.package.root.name, title=source.package.title,
            warmup_dates=source.warmup_dates, evaluation_dates=source.evaluation_dates,
            cutoff_utc=source.cutoff_utc),
        base_configuration=base_name, variants=tuple(sorted(refs, key=lambda r: r.name)),
        firm_profiles=tuple(FIRM_PROFILES[k] for k in FIRM_PROFILES if k in firm_keys),
        processing=CLOCKS[processing],
        owner_decisions=variation_decisions(refs),
        limitations=variation_limitations(refs),
        core_source=CoreSourceRef(base_commit=core_source["base_commit"],
                                  branch=core_source["branch"],
                                  patch_sha256=core_source["patch_sha256"],
                                  description=description or core_source["branch"]),
    )
    return FundedComparisonPlanEnvelopeV2.from_payload(payload)


def variation_decisions(refs) -> tuple:
    """The owner decisions a v2 variation plan freezes (September 23, 2026)."""

    from alpha_lab.propsim.funded.comparison_plan import COMPARISON_DECISIONS
    from alpha_lab.propsim.funded.plan import OwnerDecisionRef

    kept = [d for d in COMPARISON_DECISIONS if d.subject != "Costs"]
    has_micro = any(r.instrument == "micro" for r in refs)
    kept += [
        OwnerDecisionRef(
            decided_on="2026-09-23", subject="Variations tested",
            decision="The variations recommended in the September 23 research notes around "
            "S0_D80_W1_P1: target 1R/2R/3R, one-hour+four-hour or one-hour-only gaps, parent "
            "charts with or without three-minute, long only or long and short, original "
            "windows or all open-market hours, plus the scale-out exit. Variations outside "
            "the verified study are authorized only through this plan's approval.",
            status="owner_confirmed"),
        OwnerDecisionRef(
            decided_on="2026-09-23", subject="Scale-out exit",
            decision="Half the position exits at 1R; the stop of the rest moves to the "
            "entry price and it is held to that stop or the daily close. Built in a "
            "separate Strategy-Core branch; the pinned Core is unchanged.",
            status="owner_confirmed"),
        OwnerDecisionRef(
            decided_on="2026-09-22", subject="Costs (mini)",
            decision="$5.14 per mini at entry and $5.14 at exit, deducted inside the "
            "account only.", status="owner_confirmed"),
    ]
    if has_micro:
        kept.append(OwnerDecisionRef(
            decided_on="2026-09-23", subject="Scale-out size and micro costs",
            decision="Scale-out configurations trade 10 Micro Nasdaq-100 contracts (same "
            "exposure as one mini; 5 exit at 1R, 5 are held) at $0.514 per micro per fill; "
            "fixed-target configurations trade one mini.", status="owner_confirmed"))
    return tuple(kept)


def variation_limitations(refs) -> tuple:
    from alpha_lab.propsim.funded.comparison_plan import COMPARISON_LIMITATIONS

    extra = [
        "Strategy-Core runs from a separate branch that adds the scale-out exit; "
        "fixed-target configurations behave exactly as in the pinned Core (checked "
        "against the saved study's trades).",
        "Inside one candle the strategy checks its break-even stop only from the next "
        "candle; the funded accounts use the recorded trades' exact order, so the two views "
        "can differ within a minute.",
        "Strategy measures come from each configuration's replay with no account limits "
        "and the strategy's own candle rules (stop first inside a candle); funded results "
        "come from the accounts' recorded-trade paths.",
    ]
    if any(r.instrument == "micro" for r in refs):
        extra.append(
            "Micro positions are priced on the E-mini Nasdaq-100 (NQ) recorded trades: the "
            "micro contract tracks the same index, but its own trades were not used, so "
            "micro fills (including stop gaps) are an approximation of the micro market.")
    return (*COMPARISON_LIMITATIONS, *extra)


# ── version-2 variation plans from the normal configurator ───────────────────

#: the settings a variation study may vary around one study configuration
VARIATION_AXES = ("enabled_entry_sessions", "tp_r_multiple", "htf_timeframes",
                  "parent_timeframes", "enable_shorts", "exit_policy")
SCALE_OUT_VALUE = "exit_policy.scale_out_half_breakeven_hold_to_close_v1"
_ONE_R = "tp_r_multiple.1.0"


def variation_axis_values(axis: str) -> list[str]:
    """Registry values offered for one variation setting (empty when unavailable).

    The exit rule is registered only when the imported Strategy-Core supports it
    (the research Core); the application's pinned Core offers no exit choice.
    """

    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1

    spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
    return [] if spec is None else [str(v) for v in spec.registered_values]


def variation_variants(source: ComparisonSource, base_name: str,
                       selections: dict[str, list[str]], *, whole: dict[str, Any],
                       scale_out: dict[str, Any]
                       ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, str], str]]]:
    """Every combination of the selected values around ``base_name``.

    ``whole`` / ``scale_out``: ``instrument``, ``quantity`` and
    ``cost_per_contract_mills`` for whole-position and half-exit configurations.
    A variant records only the settings that differ from the base configuration.
    Combinations that cannot run are returned with the reason, never dropped
    silently: a half exit away from a 1R target, or charts the base
    configuration's verified cache does not hold.
    """

    from alpha_lab.propsim.funded.comparison_source import resolve_configuration

    base = source.by_name[base_name].axis_value_ids
    _base_section, base_cfg = resolve_configuration(base)
    cached = set(base_cfg.timeframes_seconds())
    axes = [a for a in VARIATION_AXES if selections.get(a)]
    variants: list[dict[str, Any]] = []
    skipped: list[tuple[dict[str, str], str]] = []
    for combo in product(*(selections[a] for a in axes)):
        chosen = dict(zip(axes, combo, strict=True))
        half = chosen.get("exit_policy") == SCALE_OUT_VALUE
        target = chosen.get("tp_r_multiple") or base.get("tp_r_multiple") or _ONE_R
        if half and target != _ONE_R:
            skipped.append((chosen, "the half exit is taken at 1R, so it applies only to "
                                    "the 1R target"))
            continue
        changes = {a: v for a, v in chosen.items() if base.get(a) != v}
        _section, cfg = resolve_configuration({**base, **changes})
        if not set(cfg.timeframes_seconds()) <= cached:
            skipped.append((chosen, "it needs charts the base configuration's verified "
                                    "study data does not hold"))
            continue
        variants.append({"changes": changes, **(scale_out if half else whole)})
    return variants, skipped


def core_source_description(identity: dict[str, str], *, half_exit_available: bool) -> str:
    """Plain name of the imported Strategy-Core source a variation plan freezes."""

    commit = identity["base_commit"][:7]
    if identity.get("branch") in (None, "", "HEAD"):
        return f"Strategy-Core {commit}"
    extra = " + scale-out exit" if half_exit_available else ""
    return f"Strategy-Core branch {identity['branch']} ({commit}{extra})"
