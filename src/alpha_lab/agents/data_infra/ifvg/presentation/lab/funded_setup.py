"""New funded comparison (mocks 10, 10b) and Review and approve (mocks 11, 11b), pure part.

A funded comparison is saved as a normal study draft (mode
``funded_configuration_comparison``). The plan the existing gated path builds,
approves and launches lives, unchanged, in ``steps.review.funded_comparison``
(a version-2 "variations" plan around one configuration of the verified
strategy study: ``source_run_id``, ``firm_keys``, ``plan_kind`` and
``variation`` {base, selections, sizes}). The settings this redesign adds and no
plan can carry yet — the starting-configuration choice, the chosen dates, the
gap rules, the withdrawal triggers and the pass/fail checks — are saved under
the SEPARATE key ``steps.review.funded_comparison_redesign`` so the saved-draft
check (repair R1) never reads them as unknown plan settings.

Nothing here saves, approves or launches: it reads saved drafts, turns them into
readable choices, applies one owner edit at a time, and says in plain English
what still blocks approval. Streamlit code lives in
``scripts/ifvg_lab_new_funded.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt

__all__ = [
    "BASELINES",
    "CHIP_AXES",
    "COMPARISON_MODE",
    "DROP_CHOICES",
    "FIRM_AXIS",
    "FIXED_AXES",
    "GAP_AXIS",
    "GATES",
    "HALF_EXIT",
    "LEGACY",
    "LEGACY_LABEL",
    "NAMED",
    "REDESIGN_KEY",
    "TRIGGER_AXIS",
    "TRIGGER_CHOICES_USD",
    "WHOLE_EXIT",
    "Blocker",
    "GateRow",
    "GateSpec",
    "SetupChoices",
    "apply_action",
    "HALF_EXIT_ENGINE_WORDS",
    "PlanCount",
    "approval_blocked_note",
    "approval_sentence",
    "axis_title",
    "baseline_caption",
    "baseline_description",
    "blocked_engine_lead",
    "choices_from_draft",
    "date_blockers",
    "earliest_start_sentence",
    "firm_chip_label",
    "firm_terms_sentence",
    "format_gate",
    "gate_blockers",
    "gate_defaults",
    "gate_rows",
    "has_redesign_settings",
    "legacy_warning_parts",
    "named_baseline",
    "new_study_name",
    "new_variation_settings",
    "offered_options",
    "plan_count",
    "parse_gate",
    "plan_lines",
    "retained_cushion_usd",
    "review_rows",
    "rows_from_plan",
    "same_for_every",
    "saved_draft_blockers",
    "saved_strategy_rows",
    "setup_blockers",
    "short_label",
    "size_lines",
    "skipped_sentence",
    "today_trigger_usd",
    "trigger_blocker_text",
    "value_label",
    "varying_axes",
    "write_choices",
]

COMPARISON_MODE = "funded_configuration_comparison"
FUNDED_KEY = "funded_comparison"
REDESIGN_KEY = "funded_comparison_redesign"
NAMED, LEGACY = "named", "legacy"
BASELINES = (NAMED, LEGACY)
LEGACY_LABEL = "Legacy baseline — fresh entries, static 1R"

GAP_AXIS = "htf_gap_invalidation_policy"
GAP_CLOSE = f"{GAP_AXIS}.own_timeframe_close_v1"
GAP_WICK = f"{GAP_AXIS}.execution_wick_full_fill_v1"
TRIGGER_AXIS = "withdrawal_trigger_usd"
FIRM_AXIS = "firm"
TRIGGER_CHOICES_USD = (500, 1000, 2000)
WHOLE_EXIT = "exit_policy.fixed_target_v1"
HALF_EXIT = "exit_policy.scale_out_half_breakeven_hold_to_close_v1"
ONE_R = "tp_r_multiple.1.0"

#: the plan's variation settings (``comparison_study.VARIATION_AXES``), repeated here
#: only to order the screens; the plan builder's own tuple is always used to build
_VARIATION_AXES = ("enabled_entry_sessions", "tp_r_multiple", "htf_timeframes",
                   "parent_timeframes", "enable_shorts", "exit_policy")
#: setup order: chip rows first (mock 10), then the settings on the fixed line
CHIP_AXES = ("enabled_entry_sessions", "tp_r_multiple", GAP_AXIS, "exit_policy", TRIGGER_AXIS)
FIXED_AXES = ("enable_shorts", "htf_timeframes", "parent_timeframes")
#: review table column order (mock 11): settings that differ, nested in this order
ROW_ORDER = ("enabled_entry_sessions", "tp_r_multiple", "enable_shorts", "htf_timeframes",
             "parent_timeframes", GAP_AXIS, "exit_policy", TRIGGER_AXIS)
#: the settings added by the redesign that no plan can carry yet
NEW_AXES = (GAP_AXIS, TRIGGER_AXIS)

_TITLES = {
    "enabled_entry_sessions": "Entry hours",
    "tp_r_multiple": "Profit target",
    GAP_AXIS: "When a big gap stops counting",
    "exit_policy": "Exit",
    "enable_shorts": "Direction",
    "htf_timeframes": "Big gaps",
    "parent_timeframes": "Supporting charts",
    FIRM_AXIS: "Firms",
}
_COLUMNS = {
    "enabled_entry_sessions": "Entry hours", "tp_r_multiple": "Target",
    GAP_AXIS: "Gap stops counting when", "exit_policy": "Exit", TRIGGER_AXIS: "Withdraw at",
    "enable_shorts": "Direction", "htf_timeframes": "Big gaps",
    "parent_timeframes": "Supporting charts",
}
_NOUNS = {
    "enabled_entry_sessions": ("entry hours", "entry hours"),
    "tp_r_multiple": ("target", "targets"),
    GAP_AXIS: ("gap rule", "gap rules"),
    "exit_policy": ("exit", "exits"),
    TRIGGER_AXIS: ("withdrawal trigger", "withdrawal triggers"),
    "enable_shorts": ("direction", "directions"),
    "htf_timeframes": ("big-gap choice", "big-gap choices"),
    "parent_timeframes": ("supporting-chart choice", "supporting-chart choices"),
}
_ENTRY_HOURS = {
    "enabled_entry_sessions.asia-london-ny": (
        "Original three windows: 3:00 PM–12:45 AM, 1:00–6:00 AM, 7:00 AM–1:00 PM",
        "Original three windows"),
    "enabled_entry_sessions.all_open_market_v1": ("All open-market hours",
                                                  "All open-market hours"),
    "enabled_entry_sessions.daytime_chicago_0700_1555_v1": ("Daytime: 7:00 AM–3:55 PM",
                                                            "Daytime"),
    "enabled_entry_sessions.morning_chicago_0700_1030_v1": ("Morning: 7:00–10:30 AM",
                                                            "Morning"),
    "enabled_entry_sessions.asia": ("Asia only: 3:00 PM–12:45 AM", "Asia only"),
    "enabled_entry_sessions.london": ("London only: 1:00–6:00 AM", "London only"),
    "enabled_entry_sessions.ny": ("New York only: 7:00 AM–1:00 PM", "New York only"),
    "enabled_entry_sessions.ny_0700_1030": ("Legacy morning: 6:00–9:30 AM (historical)",
                                            "Legacy morning"),
}
_GAP_RULES = {
    GAP_CLOSE: ("A candle on its own chart closes through it", "Candle closes through it"),
    GAP_WICK: ("A one-minute wick reaches the far edge", "One-minute wick reaches far edge"),
}
_EXITS = {
    WHOLE_EXIT: ("Whole position at the target", "Whole position"),
    HALF_EXIT: ("Half at the target, rest held to break-even or 3:55 PM",
                "Half at target, rest to 3:55 PM"),
}
_DIRECTIONS = {"enable_shorts.false": "Long only", "enable_shorts.true": "Long and short"}
_WORD_HOURS = {"1H": "1-hour", "2H": "2-hour", "4H": "4-hour"}


# ── readable values ───────────────────────────────────────────────────────


def _fallback(value_id: str) -> str:
    try:
        from alpha_lab.propsim.funded.comparison_draft import value_label as saved_label

        return saved_label(value_id)
    except Exception:  # pragma: no cover - the registry is part of the application
        return value_id


def _minutes(value_id: str) -> str | None:
    """``parent_timeframes.1m-3m-5m-10m-15m-30m`` → ``1, 3, 5, 10, 15 and 30 minutes``."""

    tokens = value_id.split(".", 1)[-1].split("-")
    if not tokens or not all(re.fullmatch(r"\d+m", t) for t in tokens):
        return None
    numbers = [t[:-1] for t in tokens]
    listed = numbers[0] if len(numbers) == 1 else ", ".join(numbers[:-1]) + " and " + numbers[-1]
    return f"{listed} minute" + ("" if numbers == ["1"] else "s")


def _big_gaps(value_id: str) -> str | None:
    tokens = value_id.split(".", 1)[-1].split("-")
    if not all(t in _WORD_HOURS for t in tokens):
        return None
    words = [_WORD_HOURS[t] for t in tokens]
    return words[0] + " only" if len(words) == 1 else " and ".join(words)


def _target_multiple(value_id: str) -> str | None:
    found = re.fullmatch(r"tp_r_multiple\.(\d+(?:\.\d+)?)", value_id)
    return None if found is None else f"{float(found.group(1)):g}"


def value_label(axis: str, value: str) -> str:
    """The complete readable chip text of one value (rule 17: never an internal key)."""

    if axis == TRIGGER_AXIS:
        return fmt.money_whole(int(value))
    if axis == FIRM_AXIS:
        return firm_chip_label(value)
    if value in _ENTRY_HOURS:
        return _ENTRY_HOURS[value][0]
    if value in _GAP_RULES:
        return _GAP_RULES[value][0]
    if value in _EXITS:
        return _EXITS[value][0]
    if value in _DIRECTIONS:
        return _DIRECTIONS[value]
    if axis == "tp_r_multiple" and _target_multiple(value):
        return f"{_target_multiple(value)}× the stop distance"
    if axis == "htf_timeframes" and _big_gaps(value):
        return str(_big_gaps(value))
    if axis == "parent_timeframes" and _minutes(value):
        return str(_minutes(value))
    return _fallback(value)


def short_label(axis: str, value: str) -> str:
    """The shorter text for the review table."""

    if axis == TRIGGER_AXIS:
        return fmt.money_whole(int(value))
    if value in _ENTRY_HOURS:
        return _ENTRY_HOURS[value][1]
    if value in _GAP_RULES:
        return _GAP_RULES[value][1]
    if value in _EXITS:
        return _EXITS[value][1]
    if axis == "tp_r_multiple" and _target_multiple(value):
        return f"{_target_multiple(value)}R"
    return value_label(axis, value)


def axis_title(axis: str, *, cushion_usd: int | None = None) -> str:
    if axis == TRIGGER_AXIS:
        cushion = retained_cushion_usd() if cushion_usd is None else cushion_usd
        return f"Withdraw when the surplus above {fmt.money_whole(cushion)} reaches"
    return _TITLES.get(axis, axis)


def column_title(axis: str) -> str:
    return _COLUMNS.get(axis, axis)


def firm_terms_sentence(profile: Any) -> str:
    """The earlier configurator's firm terms caption, in full, from the firm's profile.

    ``TakeProfitTrader: $102 per account, 80% trader share, up to 6 minis or equivalent,
    $2,000 loss allowance, keeps $2,100 after each payout, $500 minimum gross request.
    Owner-defined simulation terms.``
    """

    return (f"{profile.firm_name}: ${profile.acquisition_cost_cents / 100:,.0f} per account, "
            f"{profile.trader_share_pct}% trader share, up to {profile.max_minis_label}, "
            f"${profile.loss_allowance_cents / 100:,.0f} loss allowance, keeps "
            f"${profile.retained_cushion_cents / 100:,.0f} after each payout, "
            f"${profile.minimum_gross_request_cents / 100:,.0f} minimum gross request. "
            "Owner-defined simulation terms.")


def firm_chip_label(firm_key: str) -> str:
    """``TakeProfitTrader · $102 per account · 80% share`` from the saved firm profile."""

    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    profile = FIRM_PROFILES.get(firm_key)
    if profile is None:
        return f"{firm_key} (not available in this application)"
    return (f"{profile.firm_name} · {fmt.money_whole(profile.acquisition_cost_cents / 100)} "
            f"per account · {profile.trader_share_pct}% share")


def today_trigger_usd() -> int | None:
    """The withdrawal trigger the funded simulator uses today: each firm's minimum request.

    ``FundedFirmProfile.minimum_gross_request_cents`` (the surplus above the retained
    cushion that starts a request). ``None`` if the firms disagree.
    """

    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    values = {p.minimum_gross_request_cents for p in FIRM_PROFILES.values()}
    return values.pop() // 100 if len(values) == 1 else None


def retained_cushion_usd() -> int:
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    values = {p.retained_cushion_cents for p in FIRM_PROFILES.values()}
    return max(values) // 100


def trigger_blocker_text() -> str:
    today = today_trigger_usd()
    today_text = fmt.money_whole(today) if today is not None else "each firm's own minimum"
    return ("Comparing withdrawal triggers needs the funded simulator to accept a trigger per "
            f"plan; today it uses one ({today_text} above the retained "
            f"{fmt.money_whole(retained_cushion_usd())}). Your saved settings haven't been "
            "changed.")


# ── the starting configuration ────────────────────────────────────────────


def _ticks_value(value_id: str | None) -> int | None:
    if not value_id:
        return None
    found = re.fullmatch(r"[a-z_0-9]+\.(\d+)", value_id)
    return int(found.group(1)) if found else None


def baseline_description(name: str, base_ids: Mapping[str, str]) -> str:
    """``S0_D80_W1_P1 — original three windows, opposing distance 80 ticks, …``.

    Built from the configuration's saved value ids (never retyped).
    """

    parts: list[str] = []
    hours = base_ids.get("enabled_entry_sessions")
    if hours:
        parts.append(short_label("enabled_entry_sessions", hours).lower()
                     if hours in _ENTRY_HOURS else value_label("enabled_entry_sessions", hours))
    distance = _ticks_value(base_ids.get("opposing_parent_distance_ticks_max"))
    if distance is not None:
        parts.append(f"opposing distance {distance} ticks")
    minimum = _ticks_value(base_ids.get("opposing_min_gap_ticks"))
    if minimum is not None:
        parts.append(f"{minimum}-tick minimum")
    parents = base_ids.get("parent_timeframes") or ""
    if parents:
        parts.append("one-minute parents" if "1m" in parents.split(".", 1)[-1].split("-")
                     else "no one-minute parents")
    caption = baseline_caption(base_ids)
    parts.append(caption[:1].lower() + caption[1:])
    return f"{name} — " + ", ".join(p for p in parts if p)


def baseline_caption(base_ids: Mapping[str, str]) -> str:
    holding = base_ids.get("holding_policy") or ""
    if holding.endswith("scheduled_daily_close_v1"):
        return "Flat by 3:55 PM"
    return "Holds across the daily close"


@lru_cache(maxsize=1)
def _legacy_section() -> Any:
    from alpha_lab.agents.data_infra.ifvg.named_baselines import LEGACY_PROFILE_NAME
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config

    return resolve_profile_config({"profile_name": LEGACY_PROFILE_NAME}).section


def legacy_warning_parts(section: Any = None) -> tuple[str, str] | None:
    """The legacy-baseline warning (mock 10b) from the repair-R7 wording.

    ``named_baselines.legacy_baseline_warning`` decides whether the section holds
    across the daily close and words it; ``None`` when it does not warn.
    """

    from alpha_lab.agents.data_infra.ifvg.named_baselines import legacy_baseline_warning

    text = legacy_baseline_warning(_legacy_section() if section is None else section)
    if not text:
        return None
    body = text.replace(" It does not follow the mandatory 3:55 PM Chicago close.", "")
    return ("This baseline doesn't follow the 3:55 PM close.",
            body + " It's here so older studies still open exactly as saved.")


# ── choices ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SetupChoices:
    """Everything the setup page shows, read from one draft (saved or not)."""

    source_run_id: str | None
    base: str | None
    #: chip axes and the fixed-line axes → selected value ids (triggers as dollar text)
    selections: dict[str, tuple[str, ...]]
    firm_keys: tuple[str, ...]
    baseline: str = NAMED
    whole_quantity: int = 1
    whole_cost_mills: int = 5140
    half_quantity: int = 10
    half_cost_mills: int = 514
    start: str | None = None
    end: str | None = None
    #: pass/fail checks the owner changed (saved); missing keys use the source's values
    gates: dict[str, Any] = field(default_factory=dict)

    def values(self, axis: str) -> tuple[str, ...]:
        if axis == FIRM_AXIS:
            return self.firm_keys
        return tuple(self.selections.get(axis) or ())

    @property
    def triggers(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.values(TRIGGER_AXIS))

    @property
    def half_exit(self) -> bool:
        return HALF_EXIT in self.values("exit_policy")


def _registry_values(axis: str) -> tuple[str, ...]:
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1

    spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
    return () if spec is None else tuple(str(v) for v in spec.registered_values)


def _registry_baseline(axis: str) -> str | None:
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1

    spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
    return None if spec is None else spec.baseline_value_id


def base_value(axis: str, base_ids: Mapping[str, str]) -> str | None:
    """The starting configuration's own value of one setting (today's behavior)."""

    if base_ids.get(axis):
        return base_ids[axis]
    if axis == "exit_policy":
        return WHOLE_EXIT
    return _registry_baseline(axis)


def offered_options(base_ids: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """Values the running application can offer per setting (empty when not offered).

    The plan's variation settings come from the registry exactly as the plan
    builder offers them (the exit rule only with the half-exit engine); the gap
    rule lists the registry's values; the withdrawal triggers are the mock's three.
    """

    from alpha_lab.propsim.funded.comparison_study import VARIATION_AXES, variation_axis_values
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    out: dict[str, tuple[str, ...]] = {}
    for axis in VARIATION_AXES:
        values = tuple(variation_axis_values(axis))
        if values:
            out[axis] = values
    gap_values = _registry_values(GAP_AXIS)
    own_gap = base_value(GAP_AXIS, base_ids)
    # the starting configuration's own rule (today's behavior) is listed first
    ordered = ((own_gap,) if own_gap else ()) + tuple(v for v in gap_values if v != own_gap)
    out[GAP_AXIS] = ordered
    out[TRIGGER_AXIS] = tuple(str(v) for v in TRIGGER_CHOICES_USD)
    out[FIRM_AXIS] = tuple(FIRM_PROFILES)
    return out


def new_variation_settings(source_run_id: str | None, base: str | None) -> dict[str, Any]:
    """The funded settings a new setup page starts from (a variations plan around ``base``)."""

    from alpha_lab.propsim.funded.comparison_draft import DEFAULT_SETTINGS, DEFAULT_VARIATION

    return {**DEFAULT_SETTINGS, "source_run_id": source_run_id, "plan_kind": "variations",
            "variation": {**DEFAULT_VARIATION, "base": base, "selections": {}}}


def _redesign(draft: Any) -> dict[str, Any]:
    return dict((draft.steps.get("review") or {}).get(REDESIGN_KEY) or {})


def choices_from_draft(draft: Any, *, base_ids: Mapping[str, str],
                       offered: Mapping[str, Sequence[str]]) -> SetupChoices:
    """The draft's choices exactly as saved; unsaved settings show today's behavior.

    A setting with no saved value shows the starting configuration's own value
    (as the earlier configurator does). Nothing is written.
    """

    from alpha_lab.propsim.funded.comparison_draft import (
        DEFAULT_SETTINGS,
        DEFAULT_VARIATION,
        saved_settings,
    )

    saved = {**DEFAULT_SETTINGS, **saved_settings(draft)}
    variation = {**DEFAULT_VARIATION, **(saved.get("variation") or {})}
    stored = dict(variation.get("selections") or {})
    selections: dict[str, tuple[str, ...]] = {}
    for axis in _VARIATION_AXES:
        if stored.get(axis):
            selections[axis] = tuple(stored[axis])
        elif offered.get(axis):
            own = base_value(axis, base_ids)
            selections[axis] = (own,) if own else ()
    for axis, values in stored.items():  # a saved setting is never dropped
        selections.setdefault(axis, tuple(values or ()))
    extra = _redesign(draft)
    own_gap = base_value(GAP_AXIS, base_ids)
    selections[GAP_AXIS] = tuple(extra.get("gap_rules") or ((own_gap,) if own_gap else ()))
    today = today_trigger_usd()
    triggers = extra.get("withdrawal_triggers_usd") or ([today] if today else [])
    selections[TRIGGER_AXIS] = tuple(str(int(v)) for v in triggers)
    dates = dict(extra.get("dates") or {})
    return SetupChoices(
        source_run_id=saved.get("source_run_id"), base=variation.get("base"),
        selections=selections, firm_keys=tuple(saved.get("firm_keys") or ()),
        baseline=extra.get("baseline", NAMED) if extra.get("baseline") in BASELINES else NAMED,
        whole_quantity=variation["whole_quantity"], whole_cost_mills=variation["whole_cost_mills"],
        half_quantity=variation["half_quantity"], half_cost_mills=variation["half_cost_mills"],
        start=dates.get("start"), end=dates.get("end"), gates=dict(extra.get("gates") or {}))


def write_choices(draft: Any, choices: SetupChoices, *,
                  offered: Mapping[str, Sequence[str]]) -> None:
    """Put the choices into the draft (in memory) after an owner edit.

    The plan settings keep the existing schema and every other saved value;
    the redesign's own settings go to their separate key. A plan setting the
    running engine does not offer is never written (the pinned engine never
    receives a half-exit selection).
    """

    from alpha_lab.propsim.funded.comparison_draft import (
        DEFAULT_SETTINGS,
        DEFAULT_VARIATION,
        saved_settings,
    )
    from alpha_lab.propsim.funded.comparison_study import VARIATION_AXES

    saved = saved_settings(draft)
    variation = {**DEFAULT_VARIATION, **(saved.get("variation") or {})}
    variation.update({
        "base": choices.base,
        # A whole-position-only exit is the builder's own default: it is left out so the
        # draft also opens (editable) with the engine that has no exit-rule setting.
        "selections": {axis: list(choices.values(axis)) for axis in VARIATION_AXES
                       if offered.get(axis) and choices.values(axis)
                       and not (axis == "exit_policy"
                                and choices.values(axis) == (WHOLE_EXIT,))},
        "whole_quantity": int(choices.whole_quantity),
        "whole_cost_mills": int(choices.whole_cost_mills),
        "half_quantity": int(choices.half_quantity),
        "half_cost_mills": int(choices.half_cost_mills),
    })
    review = draft.steps.setdefault("review", {})
    review[FUNDED_KEY] = {**DEFAULT_SETTINGS, **saved, "source_run_id": choices.source_run_id,
                          "firm_keys": list(choices.firm_keys), "plan_kind": "variations",
                          "variation": variation}
    extra: dict[str, Any] = {
        "baseline": choices.baseline,
        "gap_rules": list(choices.values(GAP_AXIS)),
        "withdrawal_triggers_usd": list(choices.triggers),
    }
    if choices.start or choices.end:
        extra["dates"] = {"start": choices.start, "end": choices.end}
    if choices.gates:
        extra["gates"] = dict(choices.gates)
    review[REDESIGN_KEY] = extra
    space = draft.steps.setdefault("search_space", {})
    space.setdefault("mode_id", COMPARISON_MODE)
    space.setdefault("axis_selections", {})


def _ordered(values: Iterable[str], options: Sequence[str]) -> tuple[str, ...]:
    order = {v: i for i, v in enumerate(options)}
    unique = list(dict.fromkeys(values))
    return tuple(sorted(unique, key=lambda v: (order.get(v, len(order)), unique.index(v))))


def apply_action(choices: SetupChoices, action: str | None, *,
                 options: Mapping[str, Sequence[str]], adding: str | None = None
                 ) -> tuple[SetupChoices, str | None, bool]:
    """One chip click: ``remove|axis|value``, ``add|axis``, ``vary|axis``, ``pick|axis|value``
    or ``cancel``.

    Returns ``(choices, axis whose options are open, whether the choices changed)``.
    The last value of a setting cannot be removed (the chip has no remove button);
    a value the application does not offer cannot be picked.
    """

    if not action:
        return choices, adding, False
    kind, _, rest = action.partition("|")
    axis, _, value = rest.partition("|")
    if kind == "cancel":
        return choices, None, False
    if kind in ("add", "vary"):
        return choices, (None if adding == axis else axis), False
    current = choices.values(axis)
    if kind == "remove":
        if value not in current or len(current) <= 1:
            return choices, adding, False
        updated = tuple(v for v in current if v != value)
    elif kind == "pick":
        if value in current or value not in (options.get(axis) or ()):
            return choices, adding, False
        updated = _ordered((*current, value), options.get(axis) or ())
    else:
        return choices, adding, False
    remaining = [v for v in (options.get(axis) or ()) if v not in updated]
    still_adding = None if (adding == axis and not remaining) else adding
    if axis == FIRM_AXIS:
        return replace(choices, firm_keys=updated), still_adding, True
    return (replace(choices, selections={**choices.selections, axis: updated}), still_adding,
            True)


# ── the plan ──────────────────────────────────────────────────────────────


def strategy_selections(choices: SetupChoices) -> dict[str, list[str]]:
    """The plan's variation selections (only the settings a plan can vary)."""

    from alpha_lab.propsim.funded.comparison_study import VARIATION_AXES

    return {axis: list(choices.values(axis)) for axis in VARIATION_AXES if choices.values(axis)}


def plan_lines(choices: SetupChoices) -> list[str]:
    """``2 entry hours``, ``× 2 targets`` … (the fixed settings only when varied)."""

    lines = []
    for axis in (*CHIP_AXES, *FIXED_AXES):
        count = len(choices.values(axis)) or (1 if axis == "exit_policy" else 0)
        if axis in FIXED_AXES and count <= 1:
            continue
        singular, plural = _NOUNS[axis]
        lines.append(f"{'× ' if lines else ''}{count} {singular if count == 1 else plural}")
    return lines


def skipped_sentence(skipped: Sequence[tuple[Mapping[str, str], str]], multiplier: int) -> str:
    """Why some combinations are not in the plan (never dropped silently)."""

    if not skipped:
        return ""
    reasons = sorted({reason for _combo, reason in skipped})
    count = len(skipped) * max(1, multiplier)
    return (f"{fmt.count(count, 'combination')} left out: " + "; ".join(reasons) + ".")


def size_lines(choices: SetupChoices) -> tuple[str, str]:
    """The two size boxes: whole-position and half exits."""

    whole = (f"{choices.whole_quantity} E-mini · "
             f"${choices.whole_cost_mills / 1000:,.2f} per fill")
    half = (f"{choices.half_quantity} micros · "
            f"${choices.half_cost_mills / 1000:,.3f} per micro per fill")
    return whole, half


# ── approval blockers ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Blocker:
    key: str
    text: str


def _values_text(axis: str, values: Iterable[str]) -> str:
    return " and ".join(value_label(axis, v).lower() if axis == GAP_AXIS
                        else value_label(axis, v) for v in values)


def setup_blockers(choices: SetupChoices, *, base_ids: Mapping[str, str],
                   named_problem: str | None = None) -> list[Blocker]:
    """What stops approval in the settings themselves (dates and checks are separate)."""

    from alpha_lab.propsim.funded.comparison_study import size_problem

    out: list[Blocker] = []
    if choices.baseline == LEGACY:
        out.append(Blocker("baseline", (
            "The legacy baseline isn't a configuration of the verified strategy study, so a "
            "funded comparison can't replay it. Choose S0_D80_W1_P1 to approve and run; your "
            "saved settings haven't been changed.")))
    elif named_problem:
        out.append(Blocker("baseline", named_problem))
    for axis in (*CHIP_AXES, *FIXED_AXES):
        if axis in choices.selections and not choices.values(axis):
            out.append(Blocker(f"empty_{axis}",
                               f"Choose at least one value for {axis_title(axis).lower()}."))
    if not choices.firm_keys:
        out.append(Blocker("firms", "Choose at least one firm."))
    own_gap = base_value(GAP_AXIS, base_ids)
    if choices.values(GAP_AXIS) and own_gap and choices.values(GAP_AXIS) != (own_gap,):
        out.append(Blocker("gap_rule", (
            "Comparing when a big gap stops counting needs the funded plan to accept a gap "
            "rule per configuration; today every configuration keeps its starting "
            f"configuration's own rule ({_values_text(GAP_AXIS, [own_gap])}). Your saved "
            "settings haven't been changed.")))
    today = today_trigger_usd()
    if choices.triggers and (today is None or choices.triggers != (today,)):
        out.append(Blocker("withdrawal_trigger", trigger_blocker_text()))
    firms = [f for f in choices.firm_keys]
    try:
        problem = size_problem("mini", int(choices.whole_quantity), firms) if firms else None
        if not problem and choices.half_exit and firms:
            problem = size_problem("micro", int(choices.half_quantity), firms)
    except KeyError:
        problem = None  # an unknown firm is reported by the saved-draft check
    if problem:
        out.append(Blocker("size", problem))
    if choices.half_exit and int(choices.half_quantity) % 2:
        out.append(Blocker("half_size", "The half exit needs an even number of micro contracts."))
    return out


def earliest_start_sentence(earliest_day: str, first_stored_day: str, warmup_days: int) -> str:
    """The earliest start the date picker allows, and why (the mock's "[earliest stored date]")."""

    return (f"Earliest start: {fmt.date_long(earliest_day)}. Stored market data begins "
            f"{fmt.date_long(first_stored_day)}, and every study first replays "
            f"{fmt.count(warmup_days, 'warmup day')} before its first evaluated day.")


def new_study_name(today: Any) -> str:
    """A new comparison's name, dated in words (rule 17: no technical identifiers)."""

    return f"Funded configuration comparison — {fmt.date_long(today)}"


def date_blockers(resolved: Any, source_dates: Sequence[str]) -> list[Blocker]:
    """A funded comparison replays its source study's saved dates; say so for any others."""

    out = [Blocker("dates", p) for p in getattr(resolved, "problems", ())]
    if out or resolved is None:
        return out
    days = tuple(resolved.trading_days)
    if source_dates and days != tuple(source_dates):
        out.append(Blocker("dates", (
            "A funded comparison replays its source study's saved dates "
            f"({fmt.date_range(source_dates[0], source_dates[-1])}, "
            f"{fmt.count(len(source_dates), 'trading day')}). The dates chosen here "
            f"({fmt.date_range(days[0], days[-1]) if days else 'none'}, "
            f"{fmt.count(len(days), 'trading day')}) are kept in this draft, but running them "
            "needs a completed strategy study over those dates first. Your saved settings "
            "haven't been changed.")))
    return out


# ── pass/fail checks ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateSpec:
    key: str
    label: str
    #: the key in the strategy study's saved ``feasibility_gates``
    package_key: str | None
    #: count · number · r · share · choice
    kind: str
    example: str


GATES = (
    GateSpec("min_trades", "Minimum trades", "min_executed_trades", "count", "60"),
    GateSpec("min_days", "Minimum days with a trade", "min_independent_days", "count", "50"),
    GateSpec("min_profit_factor", "Minimum profit factor", "min_profit_factor", "number", "1.1"),
    GateSpec("max_drawdown_r", "Largest drawdown", "max_drawdown_r", "r", "15R"),
    GateSpec("max_days_under_water", "Most trading days under water",
             "max_time_under_water_days", "count", "3"),
    GateSpec("max_best_day_share", "Best day's share of profit", "max_top_day_pnl_share",
             "share", "40%"),
    GateSpec("drop_largest_payout", "Drop the largest payout", None, "choice", "Pass/fail"),
)
DROP_CHOICES = ("Pass/fail", "Information only")
_GATE_BY_KEY = {g.key: g for g in GATES}


def gate_defaults(package_gates: Mapping[str, Any] | None) -> dict[str, Any]:
    """The source study's saved gate thresholds (``None`` where it saved none)."""

    gates = dict(package_gates or {})
    out: dict[str, Any] = {}
    for spec in GATES:
        if spec.kind == "choice":
            out[spec.key] = DROP_CHOICES[0]
        else:
            out[spec.key] = gates.get(spec.package_key) if spec.package_key else None
    return out


def format_gate(spec: GateSpec | str, value: Any) -> str:
    spec = _GATE_BY_KEY[spec] if isinstance(spec, str) else spec
    if value is None:
        return ""
    if spec.kind == "choice":
        return str(value)
    number = float(value)
    if spec.kind == "count":
        return f"{int(number):,}" if number.is_integer() else f"{number:,g}"
    if spec.kind == "r":
        return f"{number:g}R"
    if spec.kind == "share":
        return f"{number * 100:g}%"
    return f"{number:g}"


def parse_gate(spec: GateSpec | str, text: Any) -> tuple[Any, str | None]:
    """``(value, None)`` or ``(None, plain error)``; a value is never rounded or trimmed."""

    spec = _GATE_BY_KEY[spec] if isinstance(spec, str) else spec
    raw = str(text if text is not None else "").strip()
    if spec.kind == "choice":
        return (raw, None) if raw in DROP_CHOICES else (None, "Choose Pass/fail or "
                                                               "Information only.")
    cleaned = raw.replace(",", "").replace(" ", "")
    if spec.kind == "r":
        cleaned = re.sub(r"[rR]$", "", cleaned)
    if spec.kind == "share":
        cleaned = cleaned.rstrip("%")
    try:
        number = float(cleaned)
    except ValueError:
        number = None
    if number is None or number != number:  # not a number (or NaN)
        return None, f"Enter a number, like {spec.example}."
    if spec.kind == "count":
        if number < 0 or not number.is_integer():
            return None, f"Enter a whole number of 0 or more, like {spec.example}."
        return int(number), None
    if spec.kind == "share":
        if not 0 <= number <= 100:
            return None, f"Enter a percentage from 0 to 100, like {spec.example}."
        return number / 100, None
    if number < 0:
        return None, f"Enter a number of 0 or more, like {spec.example}."
    return number, None


@dataclass(frozen=True)
class GateRow:
    spec: GateSpec
    text: str
    #: ok · flag (kept as saved, cannot pass) · decision (owner's open decision) · invalid
    status: str
    message: str
    changed: bool


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return abs(float(a) - float(b)) < 1e-12


def gate_rows(saved: Mapping[str, Any], defaults: Mapping[str, Any], *,
              trading_days: Sequence[str], warmup_days: Sequence[str] = (),
              typed: Mapping[str, str] | None = None) -> list[GateRow]:
    """One validated row per check: the owner's value (saved or typed) or the source's.

    Repair R5: an impossible "days with a trade" threshold is kept exactly as
    saved and flagged — never clamped. The under-water limit is the owner's open
    decision and is only marked, never chosen.
    """

    from alpha_lab.agents.data_infra.ifvg.search.charter_day_threshold import (
        DEVELOPMENT_ACCESS_POLICY_ID,
        check_day_threshold,
    )

    typed = typed or {}
    n_days = len(trading_days)
    rows: list[GateRow] = []
    for spec in GATES:
        default = defaults.get(spec.key)
        value = saved.get(spec.key, default)
        text = format_gate(spec, value)
        if spec.key in typed and str(typed[spec.key]).strip() != text:
            parsed, error = parse_gate(spec, typed[spec.key])
            if error:
                rows.append(GateRow(spec, str(typed[spec.key]), "invalid",
                                    error + " Nothing was saved.", False))
                continue
            value, text = parsed, str(typed[spec.key])
        changed = not _same(value, default)
        if value is None:
            rows.append(GateRow(spec, "", "ok", "Not saved in the source study.", False))
            continue
        if spec.key == "min_days":
            check = check_day_threshold(
                min_independent_days=value, replay_dates=(*warmup_days, *trading_days),
                warmup_dates=tuple(warmup_days), access_policy_id=DEVELOPMENT_ACCESS_POLICY_ID)
            if not check.passed:
                rows.append(GateRow(spec, text, "flag", (
                    f"Can't be met: only {n_days:,} trading days are selected. Kept as "
                    "saved, not trimmed — change it or the dates."), changed))
                continue
            rows.append(GateRow(spec, text, "ok", f"OK against {n_days:,} trading days",
                                changed))
            continue
        if spec.key == "max_days_under_water":
            # The owner's pending decision, stated neutrally: the saved limit stays as saved
            # and this page never proposes another value (correction A11).
            if changed:
                limit_words = (f"the saved limit is {format_gate(spec, default)}"
                               if default is not None else "the source study saved no limit")
                message = (f"Needs your decision. You entered {format_gate(spec, value)}; "
                           f"{limit_words}. This page doesn't propose a value.")
            else:
                days = "trading day" if _same(value, 1) else "trading days"
                message = (f"Needs your decision. The saved limit is {format_gate(spec, value)} "
                           f"{days} under water; it stays as saved, and this page doesn't "
                           "propose another value.")
            rows.append(GateRow(spec, text, "decision", message, changed))
            continue
        rows.append(GateRow(spec, text, "ok",
                            "OK" if not changed else
                            f"OK · changed from the saved {format_gate(spec, default)}",
                            changed))
    return rows


def gate_blockers(rows: Sequence[GateRow]) -> list[Blocker]:
    out: list[Blocker] = []
    for row in rows:
        if row.status == "invalid":
            out.append(Blocker(f"gate_{row.spec.key}", f"{row.spec.label}: {row.message}"))
        elif row.status == "flag":
            out.append(Blocker(f"gate_{row.spec.key}", (
                f"{row.spec.label} can't be met with the selected dates. Change it or the "
                "dates.")))
    if any(row.changed for row in rows):
        out.append(Blocker("gates_changed", (
            "Changed pass/fail checks need the funded plan to carry its own checks; today the "
            "results screens use the source study's saved checks. Your saved settings haven't "
            "been changed.")))
    return out


# ── a saved draft's blockers, for every approval and launch path ─────────


UNREADABLE_TEXT = ("This draft's saved dates, gap rules, withdrawal triggers or pass/fail checks "
                   "can't be read here, so approval and launch are off. Your saved settings "
                   "haven't been changed.")


def has_redesign_settings(draft: Any) -> bool:
    """Whether the draft holds this redesign's own settings (``steps.review`` + REDESIGN_KEY)."""

    review = (getattr(draft, "steps", None) or {}).get("review") or {}
    return isinstance(review, Mapping) and REDESIGN_KEY in review


def named_baseline() -> tuple[Any, str | None]:
    """``(S0_D80_W1_P1 as saved, None)`` or ``(None, why it isn't offered)``."""

    from alpha_lab.agents.data_infra.ifvg.named_baselines import (
        NamedBaselineUnavailableError,
        owner_selected_baseline,
    )

    try:
        return owner_selected_baseline(), None
    except NamedBaselineUnavailableError as error:
        return None, str(error)
    except Exception:
        return None, ("The named baseline S0_D80_W1_P1 could not be checked in this "
                      "application, so it is not offered.")


def saved_draft_blockers(draft: Any, source: Any, *, repo_root: Any = None,
                         day_has_data: Callable[[str], bool] | None = None,
                         named_problem: str | None = None) -> list[Blocker]:
    """What stops approval of a SAVED draft: the list Review and approve shows for it.

    For the other approval and launch paths (the earlier configurator's "Record my
    approval" and "Run funded comparison", and the shared launch check), so the
    settings no plan can carry yet — other dates, other withdrawal triggers, another
    gap rule, changed pass/fail checks, the legacy baseline — can't be approved or
    run there either. The inputs are Review's: the starting configuration's own
    values, the chosen dates resolved with the same calendar and local day files
    against the source study's saved dates, and the source's saved pass/fail
    thresholds (never a typed, unsaved value). A plan that can't be built is each
    path's own check and is not repeated here.

    A draft without the redesign's key returns ``[]``: it behaves exactly as before.
    Nothing is written.
    """

    if not has_redesign_settings(draft):
        return []
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        package_gate_thresholds,
    )
    from alpha_lab.agents.data_infra.ifvg.research_period import (
        local_day_has_data,
        resolve_research_range,
    )
    from alpha_lab.propsim.funded.comparison_draft import DEFAULT_VARIATION, saved_settings

    try:
        settings = saved_settings(draft)
        variation = {**DEFAULT_VARIATION, **(settings.get("variation") or {})}
        base = variation.get("base")
        by_name = source.by_name if source is not None else {}
        base_ids = dict(by_name[base].axis_value_ids) if base in by_name else {}
        choices = choices_from_draft(draft, base_ids=base_ids,
                                     offered=offered_options(base_ids))
    except (AttributeError, KeyError, TypeError, ValueError):
        return [Blocker("redesign", UNREADABLE_TEXT)]
    out = setup_blockers(choices, base_ids=base_ids, named_problem=named_problem)
    resolved = None
    if source is not None:
        days = tuple(source.evaluation_dates)
        check = day_has_data or local_day_has_data(Path(repo_root or Path.cwd()))
        resolved = resolve_research_range(choices.start or days[0], choices.end or days[-1],
                                          day_has_data=check)
        out += date_blockers(resolved, days)
    defaults = gate_defaults(package_gate_thresholds(
        source.package.root if source is not None else None))
    rows = gate_rows(choices.gates, defaults,
                     trading_days=resolved.trading_days if resolved is not None else (),
                     warmup_days=resolved.warmup_dates if resolved is not None else ())
    return out + gate_blockers(rows)


# ── the review table ──────────────────────────────────────────────────────


def _row_value(ids: Mapping[str, str], axis: str, base_ids: Mapping[str, str]) -> str:
    return ids.get(axis) or base_value(axis, base_ids) or ""


def rows_from_plan(variants: Sequence[Any], base_ids: Mapping[str, str]) -> list[dict[str, str]]:
    """The plan's configurations as (setting → value id) rows (plan built by the builder)."""

    rows = []
    for variant in variants:
        ids = dict(variant.axis_value_ids)
        row = {axis: _row_value(ids, axis, base_ids) for axis in _VARIATION_AXES
               if axis != "exit_policy"}
        row["exit_policy"] = f"exit_policy.{variant.exit_policy}"
        rows.append(row)
    return rows


def saved_strategy_rows(source: Any, variation: Mapping[str, Any]) -> list[dict[str, str]] | None:
    """The saved configurations without this engine's help (the R1 engine-independent count).

    Used when the running engine cannot build the plan (for example the
    half-exit plan opened with the pinned engine): the same combinations the
    saved-draft check counts, shown exactly as saved. ``None`` when uncountable.
    """

    from alpha_lab.propsim.funded.comparison_draft import (
        DEFAULT_VARIATION,
        _count_variations,
        _variation_sizes,
        variation_selections,
    )

    merged = {**DEFAULT_VARIATION, **dict(variation)}
    base = merged.get("base")
    sizes = _variation_sizes(merged)
    if source is None or base not in source.by_name or sizes is None:
        return None
    selections = variation_selections(source, merged)
    try:
        _count, _exact, keys = _count_variations(source, base, selections, sizes)
    except Exception:
        return None
    rows = []
    for named, _instrument, _quantity, _cost in keys:
        values = dict(named)
        rows.append({axis: values.get(axis, "") for axis in _VARIATION_AXES})
    return rows


def review_rows(strategy_rows: Sequence[Mapping[str, str]], choices: SetupChoices,
                options: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    """Every configuration: strategy rows × gap rules × withdrawal triggers, in mock order."""

    gaps = choices.values(GAP_AXIS) or ("",)
    triggers = choices.values(TRIGGER_AXIS) or ("",)
    rows = [{**row, GAP_AXIS: gap, TRIGGER_AXIS: trigger}
            for row in strategy_rows for gap in gaps for trigger in triggers]

    def order(row: Mapping[str, str]) -> tuple:
        key = []
        for axis in ROW_ORDER:
            listed = list(options.get(axis) or choices.values(axis) or ())
            value = row.get(axis, "")
            key.append((listed.index(value) if value in listed else len(listed), value))
        return tuple(key)

    return sorted(rows, key=order)


def varying_axes(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Only the settings that differ between configurations, in column order."""

    return [axis for axis in ROW_ORDER if len({row.get(axis) for row in rows}) > 1]


def same_for_every(rows: Sequence[Mapping[str, str]], base_ids: Mapping[str, str],
                   choices: SetupChoices) -> list[str]:
    """Plain sentences for the settings every configuration shares."""

    varying = set(varying_axes(rows))
    first = rows[0] if rows else {}
    out: list[str] = []

    def shared(axis: str) -> str | None:
        if axis in varying:
            return None
        value = first.get(axis) or (choices.values(axis) or ("",))[0]
        return value or None

    phrases = {
        "enable_shorts": lambda v: value_label("enable_shorts", v),
        "htf_timeframes": lambda v: f"{value_label('htf_timeframes', v)} gaps",
        "parent_timeframes": lambda v: (f"{_minutes(v)} supporting charts".replace(
            " minutes supporting", "-minute supporting").replace(" minute supporting",
                                                                 "-minute supporting")
            if _minutes(v) else value_label("parent_timeframes", v)),
        "enabled_entry_sessions": lambda v: value_label("enabled_entry_sessions", v),
        "tp_r_multiple": lambda v: f"Target {short_label('tp_r_multiple', v)}",
        GAP_AXIS: lambda v: f"Gap stops counting when {value_label(GAP_AXIS, v).lower()}",
        "exit_policy": lambda v: value_label("exit_policy", v),
        TRIGGER_AXIS: lambda v: f"Withdraw at {fmt.money_whole(int(v))}",
    }
    for axis in ("enable_shorts", "htf_timeframes", "parent_timeframes",
                 "enabled_entry_sessions", "tp_r_multiple", GAP_AXIS, "exit_policy",
                 TRIGGER_AXIS):
        value = shared(axis)
        if value:
            out.append(phrases[axis](value))
    distance = _ticks_value(base_ids.get("opposing_parent_distance_ticks_max"))
    if distance is not None:
        out.append(f"Opposing distance {distance} ticks")
    minimum = _ticks_value(base_ids.get("opposing_min_gap_ticks"))
    if minimum is not None:
        out.append(f"Smallest opposing gap {minimum} tick" + ("" if minimum == 1 else "s"))
    out.append(baseline_caption(base_ids))
    retest = _ticks_value(base_ids.get("parent_retest_timeout_1m_bars"))
    if retest is not None:
        out.append(f"Parent retest within {retest} one-minute candles")
    opposing_wait = _ticks_value(base_ids.get("opposing_timeout_1m_bars"))
    if opposing_wait is not None:
        out.append(f"Opposing pattern within {opposing_wait} one-minute candles")
    parent_distance = _ticks_value(base_ids.get("parent_htf_distance_ticks_max"))
    if parent_distance is not None:
        out.append(f"Parent within {parent_distance} ticks of the big gap")
    per_chart = _ticks_value(base_ids.get("htf_selection_max_per_timeframe"))
    if per_chart is not None:
        out.append(f"Up to {per_chart} newest big gaps per chart")
    whole, half = size_lines(choices)
    out.append(f"Whole-position exits: {whole}")
    if any(row.get("exit_policy") == HALF_EXIT for row in rows):
        out.append(f"Half exits: {half}")
    out.append("Payouts processed in two business days, paid 4:00 PM")
    return out


# ── one configuration count for a draft (My studies, setup, review) ──────


HALF_EXIT_ENGINE_WORDS = "on the half-exit engine"


@dataclass(frozen=True)
class PlanCount:
    """A draft's configuration count, from its saved selections.

    ``strategy`` counts the strategy variations (the half exit is taken only at the
    1R target), independent of the engine running now; the saved gap rules and
    withdrawal triggers multiply it. A plan with the half exit only runs on the
    half-exit engine, so its count is always given for that engine.
    """

    strategy: int | None
    gap_rules: int
    triggers: int
    exact: bool
    half_exit: bool

    @property
    def total(self) -> int | None:
        return None if self.strategy is None else self.strategy * self.gap_rules * self.triggers

    @property
    def engine_words(self) -> str:
        return HALF_EXIT_ENGINE_WORDS if self.half_exit else ""

    def text(self) -> str:
        """``36 configurations on the half-exit engine``, ``Up to 12 configurations``."""

        if self.total is None:
            return "Configurations not counted yet"
        words = fmt.count(self.total, "configuration")
        if not self.exact:
            words = f"Up to {words}"
        return f"{words} {self.engine_words}" if self.engine_words else words


def plan_count(check: Any, draft: Any) -> PlanCount:
    """The ONE count a saved (or in-memory) draft shows on every screen, in both apps.

    ``check`` is ``comparison_draft.check_saved_comparison`` of the same draft.
    """

    from alpha_lab.propsim.funded.comparison_draft import DEFAULT_VARIATION, saved_settings

    extra = _redesign(draft) if draft is not None else {}
    gaps = len(extra.get("gap_rules") or ()) or 1
    triggers = len(extra.get("withdrawal_triggers_usd") or ()) or 1
    half = False
    if draft is not None:
        variation = {**DEFAULT_VARIATION, **(saved_settings(draft).get("variation") or {})}
        half = HALF_EXIT in ((variation.get("selections") or {}).get("exit_policy") or ())
    return PlanCount(strategy=getattr(check, "configuration_count", None), gap_rules=gaps,
                     triggers=triggers, exact=bool(getattr(check, "count_is_exact", True)),
                     half_exit=half or bool(getattr(check, "needs_half_exit_engine", False)))


#: blockers that wait on engine or plan support, not on an owner edit
ENGINE_SUPPORT_KEYS = frozenset({"gap_rule", "withdrawal_trigger", "gates_changed"})


def approval_blocked_note(blockers: Sequence[Blocker]) -> str | None:
    """``Approval blocked: 2 settings need engine support.`` (None when nothing blocks)."""

    if not blockers:
        return None
    engine = sum(1 for b in blockers if b.key in ENGINE_SUPPORT_KEYS)
    other = len(blockers) - engine
    parts = []
    if engine:
        parts.append(f"{fmt.count(engine, 'setting')} need{'s' if engine == 1 else ''} "
                     "engine support")
    if other:
        parts.append(f"{fmt.count(other, 'setting')} need{'s' if other == 1 else ''} a "
                     "change")
    return "Approval blocked: " + " and ".join(parts) + "."


def approval_sentence(count: int, firm_keys: Sequence[str]) -> str:
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    if len(firm_keys) == 2:
        firms = "both firms"
    elif len(firm_keys) == 1:
        profile = FIRM_PROFILES.get(firm_keys[0])
        firms = f"{profile.firm_name if profile else firm_keys[0]} only"
    else:
        firms = "no firm"
    return (f"I approve running this exact plan: {fmt.count(count, 'configuration')}, {firms}, "
            "the dates above. No live trading.")


def blocked_engine_lead(count: PlanCount | int | None, exact: bool = True,
                        needs_half_exit: bool = False) -> str:
    """Mock 11b's first sentence for a saved plan this engine cannot represent.

    With a :class:`PlanCount` the sentence uses the draft's one shared count and its
    engine wording ("This study contains 36 configurations on the half-exit engine").
    """

    if isinstance(count, PlanCount):
        exact, count_words = count.exact, count.text()
        count = count.total
        count_words = count_words.removeprefix("Up to ")
    else:
        count_words = fmt.count(count, "configuration") if count is not None else ""
    if count is None:
        lead = "This study's configurations can't be counted in this application"
    elif exact:
        lead = f"This study contains {count_words}"
    else:
        lead = f"This study describes up to {count_words}"
    need = ("needs the version that supports half exits" if needs_half_exit
            else "can't be edited in this application")
    return f"{lead} and {need}."
