"""Saved funded-comparison drafts: faithful reading, compatibility and plan rebuild.

A saved draft is the owner's experiment. This module reads it exactly as saved
and never fills, drops or substitutes a value. It says in plain English when the
running application cannot represent the saved settings (for example a
half-exit selection opened with the pinned Strategy-Core, whose registry has no
exit-rule setting), counts the configurations the saved settings describe
without needing that engine, finds saved plans that match the draft exactly,
and rebuilds the plan from the saved settings so a launch can prove that the
plan it starts is the one the draft describes. No screen code lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from alpha_lab.propsim.funded.profiles import FIRM_PROFILES, INSTRUMENTS

__all__ = [
    "DEFAULT_SETTINGS",
    "DEFAULT_VARIATION",
    "HALF_EXIT_REQUIREMENT",
    "SavedComparisonCheck",
    "check_saved_comparison",
    "matching_saved_plans",
    "rebuild_saved_plan",
    "saved_settings",
    "saved_study_selections",
    "value_label",
    "variation_selections",
]

DEFAULT_SETTINGS: dict[str, Any] = {
    "source_run_id": None,
    "firm_keys": ["takeprofittrader", "myfundedfutures"],
    "instrument": "mini",
    "quantity": 1,
    "cost_per_side_cents": 514,
    "processing": "two_business_days",
}
DEFAULT_VARIATION: dict[str, Any] = {
    "base": None, "selections": {}, "whole_quantity": 1, "whole_cost_mills": 5140,
    "half_quantity": 10, "half_cost_mills": 514,
}
_KNOWN_SETTINGS = frozenset((*DEFAULT_SETTINGS, "plan_kind", "variation"))
_KNOWN_VARIATION = frozenset(DEFAULT_VARIATION)
_PLAN_KINDS = ("study", "variations")
_CHART_AXES = ("htf_timeframes", "parent_timeframes")
_FIXED = "exit_policy.fixed_target_v1"
HALF_EXIT_REQUIREMENT = "the Strategy-Core version that supports partial exits (the half exit)"
_AXIS_LABELS = {
    "enabled_entry_sessions": "Entry hours",
    "opposing_parent_distance_ticks_max": "Opposing-to-parent distance cap",
    "opposing_min_gap_ticks": "One-minute opposing-pattern minimum gap",
    "parent_timeframes": "Supporting (parent) charts",
    "tp_r_multiple": "Target",
    "htf_timeframes": "Higher-timeframe gap charts",
    "enable_shorts": "Direction",
    "exit_policy": "Exit rule",
}
_PROCESSING = {
    "two_business_days": "Two business days, paid 4:00 PM Chicago",
    "elapsed_48_hours": "48 elapsed hours (engineering comparison only)",
}


def saved_settings(draft: Any) -> dict[str, Any]:
    """The funded choices exactly as saved (no defaults merged in)."""

    return dict(draft.steps.get("review", {}).get("funded_comparison", {}) or {})


def saved_study_selections(draft: Any) -> dict[str, list[str]]:
    """The saved configurator selections exactly as saved, every axis kept."""

    stored = draft.steps.get("search_space", {}).get("axis_selections", {}) or {}
    return {axis: list(values or []) for axis, values in stored.items()}


def value_label(value_id: str) -> str:
    """The complete plain name of one saved value, known to this engine or not."""

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import axis_value_text
    from alpha_lab.propsim.funded.comparison_source import variation_value_label

    plain, full = variation_value_label(value_id), axis_value_text(value_id)
    if plain and full:
        if full.lower().startswith(plain.lower()):
            return full  # e.g. the entry-hours name followed by its Chicago hours
        if full.startswith("[") and full.endswith("]"):
            return f"{plain} ({full[1:-1]})"  # e.g. the parent charts, listed in full
        return plain  # e.g. "Long only" rather than the registry's "Disabled"
    text = full or plain
    return text if text else f"{value_id} (not available in this application)"


def _merged(settings: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, "plan_kind": "study", **settings}


def _variation(settings: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_VARIATION, **(settings.get("variation") or {})}


def variation_selections(source: Any, variation: dict[str, Any]) -> dict[str, list[str]]:
    """Saved variation selections as the configurator shows them.

    An axis with no saved value shows the base configuration's value (the
    configurator's own starting value); every saved value is kept as saved.
    """

    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1
    from alpha_lab.propsim.funded.comparison_study import VARIATION_AXES, variation_axis_values

    saved = dict(variation.get("selections") or {})
    base_ids = source.by_name[variation["base"]].axis_value_ids
    out: dict[str, list[str]] = {}
    for axis in VARIATION_AXES:
        if saved.get(axis):
            out[axis] = list(saved[axis])
            continue
        if not variation_axis_values(axis):
            continue
        spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
        baseline = base_ids.get(axis) or (spec.baseline_value_id if spec else None)
        out[axis] = [baseline] if baseline else []
    for axis, values in saved.items():  # keep unknown saved axes visible
        out.setdefault(axis, list(values or []))
    return out


@dataclass(frozen=True)
class SavedComparisonCheck:
    """What the running application can do with one saved comparison draft."""

    plan_kind: str
    #: plain-English reasons this application cannot edit, approve or run it
    problems: tuple[str, ...]
    needs_half_exit_engine: bool
    #: configurations the saved settings describe (None when not countable here)
    configuration_count: int | None
    #: False when this engine cannot apply every rule used in the count
    count_is_exact: bool
    #: (setting, complete saved value text) for a faithful read-only display
    saved_rows: tuple[tuple[str, str], ...]
    #: saved plans in the store that match these saved settings exactly
    matching_plan_ids: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return not self.problems


def check_saved_comparison(settings: dict[str, Any], study_selections: dict[str, list[str]],
                           sources: dict[str, Any], *, store_root: Path | None = None
                           ) -> SavedComparisonCheck:
    """Compare a saved draft with what this application's engine and data offer."""

    from alpha_lab.propsim.funded.comparison_study import (
        CLOCKS,
        COMPARISON_AXES,
        SCALE_OUT_VALUE,
        VARIATION_AXES,
        axis_choices,
        variation_axis_values,
    )

    merged = _merged(settings)
    kind = merged["plan_kind"]
    problems: list[str] = []
    needs_half = False
    unknown = sorted(set(settings) - _KNOWN_SETTINGS)
    if unknown:
        problems.append("It contains settings this application does not recognize: "
                        + ", ".join(unknown) + ".")
    if kind not in _PLAN_KINDS:
        problems.append(f"Its study type ({kind}) is not available in this application.")
    source_id = merged.get("source_run_id")
    source = sources.get(source_id) if source_id else None
    if source_id and source is None:
        problems.append(f"The completed strategy study it uses (run {str(source_id)[:12]}…) "
                        "is not available in this application.")
    firms = list(merged.get("firm_keys") or [])
    bad_firms = [f for f in firms if f not in FIRM_PROFILES]
    if bad_firms:
        problems.append("It names firms this application does not know: "
                        + ", ".join(map(str, bad_firms)) + ".")
    if merged.get("processing") not in CLOCKS:
        problems.append(f"Its payout processing setting ({merged.get('processing')}) is not "
                        "available in this application.")
    rows: list[tuple[str, str]] = []
    if source is not None:
        rows.append(("Completed strategy study", source.package.title))
    elif source_id:
        rows.append(("Completed strategy study", f"run {source_id} (not available here)"))
    count: int | None = None
    exact = True
    variants_key: set | None = None
    if kind == "variations":
        variation = _variation(settings)
        unknown_v = sorted(set(settings.get("variation") or {}) - _KNOWN_VARIATION)
        if unknown_v:
            problems.append("Its variation settings include names this application does not "
                            "recognize: " + ", ".join(unknown_v) + ".")
        base = variation.get("base")
        base_ok = source is not None and base in source.by_name
        if source is not None and not base_ok:
            problems.append(f"Its base configuration ({base}) is not in the completed strategy "
                            "study available here.")
        rows.append(("Base configuration", source.by_name[base].display_name.replace(" | ", "; ")
                     if base_ok else str(base)))
        if base_ok:
            selections = variation_selections(source, variation)
        else:
            selections = {a: list(v or []) for a, v in (variation.get("selections") or {}).items()}
        for axis, values in selections.items():
            offered = variation_axis_values(axis) if axis in VARIATION_AXES else []
            missing = [v for v in values if v not in offered]
            if axis not in VARIATION_AXES:
                problems.append(f"It varies a setting this application does not offer ({axis}).")
            elif missing:
                if axis == "exit_policy" and not offered and set(values) <= {_FIXED,
                                                                            SCALE_OUT_VALUE}:
                    needs_half = True
                    problems.append(f"It requires {HALF_EXIT_REQUIREMENT}, which this "
                                    "application did not start with.")
                else:
                    problems.append("Saved values are not available in this application: "
                                    + "; ".join(value_label(v) for v in missing) + ".")
            rows.append((_AXIS_LABELS.get(axis, axis),
                         "; ".join(value_label(v) for v in values) or "none selected"))
        sizes = _variation_sizes(variation)
        if sizes is None:
            problems.append("Its saved sizes or costs are not whole numbers.")
        else:
            (whole_q, whole_c), (half_q, half_c) = sizes
            rows.append(("Whole-position exits", f"{whole_q} x E-mini Nasdaq-100 (NQ) at "
                         f"${whole_c / 1000:,.3f} per contract per fill"))
            if SCALE_OUT_VALUE in selections.get("exit_policy", []):
                rows.append(("Half exits", f"{half_q} x Micro E-mini Nasdaq-100 (MNQ) at "
                             f"${half_c / 1000:,.3f} per contract per fill"))
        if base_ok and sizes is not None:
            count, exact, variants_key = _count_variations(source, base, selections, sizes)
    elif kind == "study":
        chosen_axes = dict(study_selections)
        offered = axis_choices(source) if source is not None else {}
        for axis, values in chosen_axes.items():
            if axis not in COMPARISON_AXES:
                problems.append(f"It selects a setting this application does not offer ({axis}).")
            elif source is not None:
                missing = [v for v in values if v not in offered.get(axis, [])]
                if missing:
                    problems.append("Saved values are not approved values of this study: "
                                    + "; ".join(value_label(v) for v in missing) + ".")
            rows.append((_AXIS_LABELS.get(axis, axis),
                         "; ".join(value_label(v) for v in values) or "none selected"))
        instrument = merged.get("instrument")
        if instrument not in INSTRUMENTS:
            problems.append(f"Its contract ({instrument}) is not available in this application.")
        else:
            rows.append(("Contract", f"{merged.get('quantity')} x {INSTRUMENTS[instrument].label} "
                         f"at ${int(merged.get('cost_per_side_cents') or 0) / 100:,.2f} per "
                         "contract per fill"))
        if source is not None:
            from alpha_lab.propsim.funded.comparison_study import resolve_selection

            configurations, _unavailable = resolve_selection(
                source, {a: v for a, v in chosen_axes.items() if a in COMPARISON_AXES})
            count = len(configurations)
            exact = not any("not approved" in p or "does not offer" in p for p in problems)
            variants_key = {c.name for c in configurations}
    rows.append(("Firms (each a separate result)", "; ".join(
        FIRM_PROFILES[f].firm_name if f in FIRM_PROFILES else f"{f} (unknown)" for f in firms)
        or "none selected"))
    rows.append(("Payout processing pause",
                 _PROCESSING.get(merged.get("processing"), str(merged.get("processing")))))
    matches: tuple[str, ...] = ()
    if store_root is not None and source is not None and variants_key is not None:
        matches = matching_saved_plans(store_root, settings, variants_key)
    return SavedComparisonCheck(
        plan_kind=kind, problems=tuple(problems), needs_half_exit_engine=needs_half,
        configuration_count=count, count_is_exact=exact, saved_rows=tuple(rows),
        matching_plan_ids=matches)


def _variation_sizes(variation: dict[str, Any]):
    try:
        values = [variation[k] for k in ("whole_quantity", "whole_cost_mills",
                                         "half_quantity", "half_cost_mills")]
    except KeyError:
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return None
    return (values[0], values[1]), (values[2], values[3])


def _count_variations(source: Any, base: str, selections: dict[str, list[str]], sizes):
    """Configurations the saved selections describe, independent of the exit-rule engine.

    Applies the two rules that remove combinations (a half exit is taken only at
    the 1R target; a variation cannot use charts the base configuration's
    verified data lacks). The chart rule needs the registry to know the chart
    values; if it does not, the count is marked as not exact.
    """

    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import AXIS_VALUE_REGISTRY_V1
    from alpha_lab.propsim.funded.comparison_source import _named_ids, resolve_configuration
    from alpha_lab.propsim.funded.comparison_study import SCALE_OUT_VALUE, VARIATION_AXES

    base_ids = source.by_name[base].axis_value_ids
    try:
        _section, base_cfg = resolve_configuration(base_ids)
        cached = set(base_cfg.timeframes_seconds())
    except Exception:
        cached = None
    axes = [a for a in VARIATION_AXES if selections.get(a)]
    (whole_q, whole_c), (half_q, half_c) = sizes
    chart_ok: dict[tuple, bool | None] = {}
    count, exact = 0, True
    keys: set = set()
    for combo in product(*(selections[a] for a in axes)):
        chosen = dict(zip(axes, combo, strict=True))
        half = chosen.get("exit_policy") == SCALE_OUT_VALUE
        target = chosen.get("tp_r_multiple") or base_ids.get("tp_r_multiple") or "tp_r_multiple.1.0"
        if half and target != "tp_r_multiple.1.0":
            continue
        charts = tuple(chosen.get(a) or base_ids.get(a) for a in _CHART_AXES)
        if charts not in chart_ok:
            known = [v for v in charts if v is not None]
            if cached is None or any(v not in AXIS_VALUE_REGISTRY_V1 for v in known):
                chart_ok[charts] = None
            else:
                changes = {a: v for a, v in zip(_CHART_AXES, charts, strict=True) if v}
                try:
                    _s, cfg = resolve_configuration({**base_ids, **changes})
                    chart_ok[charts] = set(cfg.timeframes_seconds()) <= cached
                except Exception:
                    chart_ok[charts] = None
        verdict = chart_ok[charts]
        if verdict is False:
            continue
        if verdict is None:
            exact = False
        count += 1
        named = _named_ids({**base_ids, **chosen})
        keys.add((tuple(sorted(named.items())), "micro" if half else "mini",
                  half_q if half else whole_q, half_c if half else whole_c))
    return count, exact, keys


def matching_saved_plans(store_root: Path, settings: dict[str, Any], key: set) -> tuple[str, ...]:
    """Saved plans in ``store_root`` whose frozen contents equal these saved settings.

    Every recorded element is compared: source study, firms, clock, and the
    complete set of configurations with their sizes and costs. A similar name or
    a nearby run is never treated as a match.
    """

    from alpha_lab.propsim.funded.comparison_plan import PLAN_STORE
    from alpha_lab.propsim.funded.comparison_runner import is_v2, load_plan
    from alpha_lab.propsim.funded.comparison_source import _named_ids
    from alpha_lab.propsim.funded.comparison_study import CLOCKS

    merged = _merged(settings)
    folder = Path(store_root) / PLAN_STORE
    if not folder.is_dir() or merged.get("processing") not in CLOCKS:
        return ()
    firms = sorted(merged.get("firm_keys") or [])
    found = []
    for entry in sorted(folder.iterdir()):
        try:
            plan = load_plan(store_root, entry.name)
        except Exception:
            continue
        if (plan.source.package_run_id != merged.get("source_run_id")
                or sorted(p.firm_key for p in plan.firm_profiles) != firms
                or plan.processing != CLOCKS[merged["processing"]]):
            continue
        if merged["plan_kind"] == "variations" and is_v2(plan):
            if plan.base_configuration != _variation(settings).get("base"):
                continue
            recorded = {(tuple(sorted(_named_ids(dict(v.axis_value_ids)).items())),
                         v.instrument, v.quantity, v.cost_per_contract_mills)
                        for v in plan.variants}
            if recorded == key and len(plan.variants) == len(key):
                found.append(entry.name)
        elif merged["plan_kind"] == "study" and not is_v2(plan):
            if ({c.name for c in plan.configurations} == key
                    and plan.instrument == merged.get("instrument")
                    and plan.quantity == merged.get("quantity")
                    and plan.cost_per_side_cents == merged.get("cost_per_side_cents")):
                found.append(entry.name)
    return tuple(found)


def rebuild_saved_plan(source: Any, settings: dict[str, Any],
                       study_selections: dict[str, list[str]]):
    """The plan envelope the saved settings describe, built exactly as the screen does.

    Returns ``(envelope or None, skipped combinations, unavailable combinations)``.
    Used by the screen for display and by the launch to prove that the plan it
    starts is the plan rebuilt from the draft saved on disk right now.
    """

    from alpha_lab.propsim.funded.comparison_study import (
        COMPARISON_AXES,
        build_comparison_plan,
        build_variation_plan,
        core_source_description,
        resolve_selection,
        size_problem,
        variation_axis_values,
        variation_variants,
    )

    merged = _merged(settings)
    firm_keys = list(merged.get("firm_keys") or [])
    if not firm_keys:
        return None, [], []
    if merged["plan_kind"] == "variations":
        from alpha_lab.propsim.funded.core_identity import core_source_identity

        variation = _variation(settings)
        selections = variation_selections(source, variation)
        if any(not values for values in selections.values()) or variation["half_quantity"] % 2:
            return None, [], []
        variants, skipped = variation_variants(
            source, variation["base"], selections,
            whole={"instrument": "mini", "quantity": variation["whole_quantity"],
                   "cost_per_contract_mills": variation["whole_cost_mills"]},
            scale_out={"instrument": "micro", "quantity": variation["half_quantity"],
                       "cost_per_contract_mills": variation["half_cost_mills"]})
        if not variants:
            return None, skipped, []
        identity = core_source_identity()
        engine = core_source_description(identity, half_exit_available=bool(
            variation_axis_values("exit_policy")))
        envelope = build_variation_plan(source, variation["base"], variants,
                                        firm_keys=firm_keys, core_source=identity,
                                        processing=merged["processing"], description=engine)
        return envelope, skipped, []
    chosen = {axis: list(study_selections.get(axis) or []) for axis in COMPARISON_AXES}
    configurations, unavailable = resolve_selection(source, chosen)
    if not configurations or size_problem(merged["instrument"], int(merged["quantity"]),
                                          firm_keys):
        return None, [], unavailable
    envelope = build_comparison_plan(
        source, configurations, firm_keys=firm_keys, instrument=merged["instrument"],
        quantity=int(merged["quantity"]), cost_per_side_cents=int(merged["cost_per_side_cents"]),
        processing=merged["processing"])
    return envelope, [], unavailable
