"""Deterministic, read-only explanations of resolved IFVG trading rules.

The vocabulary here follows the executable v2 reducer, not the research-axis
labels. A complete effective section is required: this module never resolves a
profile name, applies today's defaults, reads data, or changes a saved study.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import ValidationError
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

__all__ = ["RuleDescription", "describe_strategy", "describe_strategy_variations"]


@dataclass(frozen=True, slots=True)
class RuleDescription:
    status: Literal["available", "partial", "unavailable", "unconfigured"]
    preview_bullets: tuple[str, ...] = ()
    detail_bullets: tuple[str, ...] = ()
    variations: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    is_preview: bool = False


SectionInput = Mapping[str, Any] | IfvgSmcSection

# Explicit coverage makes an added engine field a reviewable change, rather
# than silently giving the old explanation to a new strategy.
_RULE_FIELDS = {
    "enable_longs", "enable_shorts", "htf_timeframes", "parent_timeframes",
    "min_gap_ticks_capture", "parent_reaction_window_parent_bars",
    "parent_reaction_window_1m_bars_max", "parent_retest_timeout_1m_bars",
    "opposing_timeout_1m_bars", "inversion_timeout_1m_bars",
    "post_inversion_expiry_1m_bars_max", "parent_htf_distance_ticks_max",
    "opposing_parent_distance_ticks_max", "entry_near_parent",
    "entry_parent_distance_ticks_max", "htf_selection_max_per_timeframe",
    "sl_buffer_ticks", "tp_r_multiple", "causality_parent", "causality_opposing",
    "causality_entry", "enabled_entry_sessions", "outside_session_policy",
    "parent_full_fill_invalidation", "parent_structural_invalidation",
    "max_executed_trades_per_day", "doc_sessions", "session_scheme",
    "htf_registry_max_age_days", "ltf_registry_max_live",
}
_CAPABILITY_FIELDS = {
    "runnable", "execution_enabled", "non_runnable_reason", "entry_family",
    "resolver_policy", "anchor_policy",
}
_MEASUREMENT_FIELDS = {
    "profile_name", "qualification_mode", "entry_families", "retest_trigger",
    "label_family", "swing_strength_bars", "swing_pool_max",
}
_INERT_FIELDS = {"break_even_enabled", "legacy_candidate_row_limit"}
_CLASSIFIED_FIELDS = (
    _RULE_FIELDS | _CAPABILITY_FIELDS | _MEASUREMENT_FIELDS | _INERT_FIELDS
)


def _unavailable(message: str) -> RuleDescription:
    return RuleDescription("unavailable", issues=(message,))


def _load(section: SectionInput) -> tuple[IfvgSmcSection | None, tuple[str, ...]]:
    raw = (
        section.model_dump(mode="json")
        if isinstance(section, IfvgSmcSection)
        else dict(section)
    )
    required = _RULE_FIELDS | (_CAPABILITY_FIELDS - {"non_runnable_reason"})
    if required - raw.keys():
        return None, ("The saved strategy settings are incomplete; its rules cannot be verified.",)
    unknown = raw.keys() - _CLASSIFIED_FIELDS
    issues = (
        ("Some saved settings are not yet covered by this description.",)
        if unknown else ()
    )
    try:
        parsed = IfvgSmcSection.model_validate(
            {key: value for key, value in raw.items() if key in IfvgSmcSection.model_fields}
        )
    except (ValidationError, TypeError, ValueError):
        return None, ("The saved strategy settings cannot be read reliably.",)
    return parsed, issues


def _join(values: Sequence[str], conjunction: str = "and") -> str:
    if len(values) < 2:
        return "".join(values)
    if len(values) == 2:
        return f" {conjunction} ".join(values)
    return ", ".join(values[:-1]) + f" {conjunction} " + values[-1]


def _timeframes(values: Sequence[str], conjunction: str = "and") -> str:
    return _join([
        value[:-1] + ("-hour" if value.endswith("H") else "-minute")
        for value in values
    ], conjunction)


def _after(policy: str, event: str) -> str:
    if policy == "fully_formed_after":
        return f"starts forming at or after {event}"
    return f"is confirmed after {event}"


def _retest_rule(section: IfvgSmcSection) -> str:
    limit = section.parent_retest_timeout_1m_bars
    wait = (
        "There is no time limit for this retest"
        if limit is None else
        f"If more than {limit} one-minute candles pass after selecting the main zone "
        "without a retest, abandon the setup"
    )
    return (
        "If a later candle touches the main zone, continue. "
        f"{wait}; other invalidation rules still apply. "
        "A replacement main zone restarts this wait. Count recorded candles, not clock time."
    )


def _wait(limit: int | None, start: str) -> str:
    if limit is None:
        return ""
    return (
        f"If more than {limit} one-minute candles pass after {start} "
        "without the next step, abandon the setup."
    )


def _sessions(section: IfvgSmcSection) -> str:
    names = {"asia": "Asia", "london": "London", "ny": "New York"}
    windows = []
    for name in section.enabled_entry_sessions:
        if name in section.doc_sessions:
            start, end = section.doc_sessions[name]
            overnight = " the next day" if start > end else ""
            windows.append(f"{names.get(name, name)} {start}–{end}{overnight}")
    zone = section.session_scheme.timezone
    zone = "Eastern time" if zone == "America/New_York" else zone
    allowed = _join(windows) or "no entry windows"
    outside = (
        "If an entry signal appears outside these windows, abandon that setup."
        if section.outside_session_policy == "reset_setup_as_missed" else
        "Otherwise, skip that signal and keep waiting while the setup remains valid."
    )
    cap = section.max_executed_trades_per_day
    capped = (
        "No daily trade-count limit."
        if cap is None else
        f"After {cap} executed trade{'s' if cap != 1 else ''} in a trading day, "
        "skip further entries until the next trading day."
    )
    return (
        f"Enter during {allowed} ({zone}; end times excluded). "
        f"{outside} {capped} Each trading day starts at "
        f"{section.session_scheme.trading_day_boundary[:5]} {zone}."
    )


def _rules(section: IfvgSmcSection) -> dict[str, str]:
    both = section.enable_longs and section.enable_shorts
    long = section.enable_longs
    side = "buying and selling" if both else "buying" if long else "selling"
    matching_gap = (
        "gap in the trade direction" if both else "bullish gap" if long else "bearish gap"
    )
    matching_minute = (
        "one-minute gap in the trade direction" if both else
        "bullish one-minute gap" if long else "bearish one-minute gap"
    )
    opposing_gap = (
        "one-minute gap in the opposite direction" if both else
        "bearish one-minute gap" if long else "bullish one-minute gap"
    )
    inversion = (
        "above it for a buy or below it for a sell" if both else
        "above its upper edge" if long else "below its lower edge"
    )
    entry = "buy or sell in the setup's direction" if both else "buy" if long else "sell"
    rules = {}
    rules["setup"] = (
        f"Look for {side} opportunities. If price touches a selected gap on the "
        f"{_timeframes(section.htf_timeframes, 'or')} charts, "
        "start a setup in that gap's direction. "
        "A gap is the space between the first and third candles of a three-candle move. "
        f"Use gaps at least {section.min_gap_ticks_capture} ticks wide (a tick is one price step)."
    )
    rules["sessions"] = _sessions(section)
    parent_window = (
        f"within {section.parent_reaction_window_parent_bars} candles of its own timeframe"
    )
    legacy = section.parent_reaction_window_1m_bars_max
    if legacy is not None:
        parent_window += f" and within {legacy} recorded one-minute candles of the initial touch"
    rules["parent"] = (
        f"If a new {matching_gap} on the {_timeframes(section.parent_timeframes, 'or')} charts "
        f"{_after(section.causality_parent, 'that touch')}, {parent_window}, and lies within "
        f"{section.parent_htf_distance_ticks_max} ticks of the touched zone, use it as the "
        "main zone. Until its first retest, prefer replacements on larger timeframes, "
        "then newer gaps on the same timeframe."
    )
    rules["retest"] = _retest_rule(section)
    rules["inversion"] = (
        f"After the retest, wait for a {opposing_gap} that "
        f"{_after(section.causality_opposing, 'the retest')} and lies within "
        f"{section.opposing_parent_distance_ticks_max} ticks of the main zone. "
        f"{_wait(section.opposing_timeout_1m_bars, 'the retest')} "
        f"Then wait for a later candle to close {inversion}. "
        f"{_wait(section.inversion_timeout_1m_bars, 'selecting the opposing gap')} "
        "A newer qualifying opposing gap replaces the old one and restarts its wait."
    )
    near = (
        f" The entry price must also be within {section.entry_parent_distance_ticks_max} "
        "ticks of the main zone."
        if section.entry_near_parent else ""
    )
    rules["entry"] = (
        f"If a fresh {matching_minute} "
        f"{_after(section.causality_entry, 'that close-through')} within the next "
        f"{section.post_inversion_expiry_1m_bars_max} one-minute candles, {entry} at the "
        "confirming candle's close, if entries are allowed and the stop "
        f"is at least one tick away.{near} If more than "
        f"{section.post_inversion_expiry_1m_bars_max} candles have passed, abandon the setup."
    )
    buffer = section.sl_buffer_ticks
    anchor = (
        "below the lowest price since the main-zone retest for a buy, or above the highest "
        "price since that retest for a sell"
        if both else
        "below the lowest price since the main-zone retest" if long else
        "above the highest price since the main-zone retest"
    )
    target = (
        "Set a fixed profit target the same distance from entry as the stop. "
        if section.tp_r_multiple == 1 else
        f"Set a fixed profit target {section.tp_r_multiple:g} times "
        "the initial distance to the stop. "
    )
    rules["exit"] = (
        f"Place the stop {buffer} tick{'s' if buffer != 1 else ''} {anchor}, "
        f"including the entry candle. {target}"
        "Check exits from the next one-minute candle; if both prices are touched, count the "
        "stop first. The stop stays fixed. Session or trading-day changes do not close the trade."
    )
    causes = []
    if section.parent_full_fill_invalidation:
        causes.append("price completely fills the main zone")
    if section.parent_structural_invalidation:
        causes.append("a candle on its own timeframe closes beyond its far edge against the trade")
    invalidation = (
        f"If {_join(causes, 'or')} before its retest, seek another main zone within the "
        "original reaction window; after the retest, abandon the setup."
        if causes else
        "A main-zone fill or a close through its far edge does not cancel the setup."
    )
    rules["cancellation"] = (
        "Before entry, if price completely fills the original higher-timeframe gap, "
        f"abandon the setup. {invalidation} Work on one setup or open trade at a time."
    )
    selection = section.htf_selection_max_per_timeframe
    rules["selection"] = (
        f"Consider only the {selection} most recent active higher-timeframe "
        f"gap{'s' if selection != 1 else ''} per timeframe. For simultaneous touches, "
        "prefer larger timeframes, then newer gaps; skip opposing directions tied on "
        "the highest eligible timeframe."
    )
    age = section.htf_registry_max_age_days
    live = section.ltf_registry_max_live
    retention = (
        f"Remove higher-timeframe gaps older than {age} days, measured by trading-day dates."
        if age is not None else "Higher-timeframe gaps have no age limit."
    )
    lower = (
        f"Keep at most {live} active gaps per lower timeframe, dropping the oldest first."
        if live is not None else "There is no count limit for active lower-timeframe gaps."
    )
    rules["retention"] = f"{retention} {lower} Fully filled gaps leave the active list."
    return rules


def _visible_rules(
    rules: Mapping[str, str], section: IfvgSmcSection, omit: set[str] | None = None,
) -> tuple[str, ...]:
    """Keep everyday rules short; retention/selection details appear when varied."""
    visible = []
    for key in ("sessions", "setup", "parent", "retest", "inversion", "entry", "exit",
                "cancellation"):
        if key in (omit or set()):
            if key == "retest":
                visible.append(
                    "If a later candle returns to the main zone (a retest), continue."
                )
            continue
        text = rules[key]
        if key == "sessions":
            if section.session_scheme.trading_day_boundary[:5] == "18:00":
                text = text.split(" Each trading day starts at ", 1)[0]
            text = text.replace(" No daily trade-count limit.", "")
        visible.append(text)
    # Reviewed document-profile values are only brevity choices here, never
    # missing configuration defaults. Exact non-default profiles expose them.
    if section.htf_selection_max_per_timeframe != 1 and "selection" not in (omit or set()):
        visible.append(rules["selection"])
    if (
        (section.htf_registry_max_age_days, section.ltf_registry_max_live) != (15, 512)
        and "retention" not in (omit or set())
    ):
        visible.append(rules["retention"])
    return tuple(visible)


def describe_strategy(
    section: SectionInput,
    semantic_metadata: Mapping[str, Any] | None = None,
) -> RuleDescription:
    """Explain a complete effective v2 fresh-entry section without applying defaults."""
    metadata = semantic_metadata or {}
    for key, supported in (
        ("strategy_id", "ifvg_smc"), ("strategy_version", "2"),
        ("resolver_policy", "next_1m_bar_stop_first_v1"),
    ):
        if key in metadata and str(metadata[key]) != supported:
            return _unavailable("The saved strategy version is not supported by this description.")
    parsed, issues = _load(section)
    if parsed is None:
        return RuleDescription("unavailable", issues=issues)
    if not parsed.runnable or not parsed.execution_enabled:
        return _unavailable("This strategy configuration is not enabled to execute trades.")
    if parsed.entry_family != "fresh_fvg_continuation":
        return _unavailable("This entry style is not supported for executable trading rules.")
    if not parsed.enable_longs and not parsed.enable_shorts:
        return RuleDescription(
            "available", ("Both buying and selling are disabled; this strategy takes no trades.",),
            ("Both buying and selling are disabled; this strategy takes no trades.",),
        )
    if parsed.resolver_policy != "next_1m_bar_stop_first_v1":
        return _unavailable("The saved exit rules are not supported by this description.")
    if parsed.anchor_policy != "trading_day_18et_elapsed_v1":
        issues += ("The configured chart timing policy is not covered by this description.",)
    if set(parsed.enabled_entry_sessions) - parsed.doc_sessions.keys():
        issues += ("Some enabled entry sessions have no saved time window.",)
    if parsed.break_even_enabled:
        issues += (
            "Moving the stop to break-even is configured, "
            "but the current engine does not apply it.",
        )
    if parsed.legacy_candidate_row_limit is not None:
        issues += ("The saved legacy candidate limit does not limit executed trades.",)
    rules = _rules(parsed)
    side = (
        "buy or sell" if parsed.enable_longs and parsed.enable_shorts else
        "buy" if parsed.enable_longs else "sell"
    )
    target_preview = (
        "a profit target equal to the initial risk" if parsed.tp_r_multiple == 1 else
        f"a profit target of {parsed.tp_r_multiple:g} times risk"
    )
    previews = (
        "If price touches a selected gap on a "
        f"{_timeframes(parsed.htf_timeframes, 'or')} chart, "
        "look for a nearby main zone and wait for price to return to it (a retest).",
        "If a one-minute opposing gap fails and a fresh gap confirms the setup, "
        f"{side} during the allowed trading windows.",
        "Set a fixed stop beyond the price extreme since the retest and "
        f"{target_preview}; abandon invalid or expired setups.",
    )
    return RuleDescription(
        "partial" if issues else "available", previews, _visible_rules(rules, parsed),
        issues=issues,
    )


def describe_strategy_variations(
    base_section: SectionInput,
    configurations: Sequence[tuple[str, SectionInput]],
    *,
    semantic_metadata: Mapping[str, Any] | None = None,
) -> RuleDescription:
    """Explain the baseline and every named, fully resolved alternative.

    Configuration names are data only. Changes are compared using semantic
    groups, so dependent controls (such as entry locality) are interpreted
    together and inactive values do not become fictional trade filters.
    """
    base = describe_strategy(base_section, semantic_metadata)
    base_parsed, _ = _load(base_section)
    if base.status == "unavailable" or base_parsed is None or not configurations:
        return base
    if not base_parsed.enable_longs and not base_parsed.enable_shorts:
        alternatives = [
            (name, describe_strategy(section, semantic_metadata))
            for name, section in configurations
        ]
        issues = tuple(dict.fromkeys(
            f"{name}: {issue}" for name, description in alternatives
            for issue in description.issues
        ))
        return replace(
            base, status="partial" if issues else base.status, issues=issues,
            variations=tuple(
                f"{name}: {text}" for name, description in alternatives
                if description.detail_bullets != base.detail_bullets
                for text in description.detail_bullets
            ),
        )
    base_rules = _rules(base_parsed)
    variations = []
    grouped_changes: dict[tuple[str, str], list[str]] = {}
    issues = list(base.issues)
    retest_values = [base_parsed.parent_retest_timeout_1m_bars]
    changed_groups: set[str] = set()
    seen_rules = {tuple(base_rules.values())}
    version_count = 1
    for name, section in configurations:
        description = describe_strategy(section, semantic_metadata)
        parsed, _ = _load(section)
        if description.status == "unavailable" or parsed is None:
            variations.append(f"{name}: this version's trading rules are unavailable.")
            issues.extend(f"{name}: {issue}" for issue in description.issues)
            version_count += 1
            continue
        issues.extend(f"{name}: {issue}" for issue in description.issues)
        if not parsed.enable_longs and not parsed.enable_shorts:
            variations.append(f"{name}: buying and selling are disabled, so no trades are taken.")
            changed_groups.update(base_rules)
            version_count += 1
            continue
        candidate_rules = _rules(parsed)
        if (
            parsed.session_scheme != base_parsed.session_scheme
            and candidate_rules["sessions"] == base_rules["sessions"]
        ):
            issues.append(
                f"{name}: the chart session schedule differs and is not fully explained here."
            )
        fingerprint = tuple(candidate_rules.values())
        if fingerprint in seen_rules:
            continue
        seen_rules.add(fingerprint)
        version_count += 1
        changed_groups.update(
            key for key, text in candidate_rules.items() if base_rules[key] != text
        )
        retest_values.append(parsed.parent_retest_timeout_1m_bars)
        for key, text in candidate_rules.items():
            if base_rules[key] != text:
                grouped_changes.setdefault((key, text), []).append(name)
    variations.extend(
        f"{_join(names)}: {text}" for (_key, text), names in grouped_changes.items()
    )
    if variations:
        baseline_changes = [
            f"Baseline: {text}" for key, text in base_rules.items() if key in changed_groups
        ]
        variations = [*baseline_changes, *variations]
    preview = base.preview_bullets
    if variations and len(preview) == 3:
        if changed_groups == {"retest"}:
            values = sorted({value for value in retest_values if value is not None})
            limits = _join([str(value) for value in values])
            comparison = (
                f"Compare no main-zone retest timeout with limits of {limits} one-minute candles."
                if None in retest_values else
                f"Compare main-zone retest time limits of {limits} one-minute candles."
            )
            baseline_wait = (
                "no time limit for the main-zone retest" if retest_values[0] is None else
                f"abandon the main-zone retest wait after more than {retest_values[0]} "
                "recorded one-minute candles"
            )
            alternative_limits = list(dict.fromkeys(retest_values[1:]))
            alternatives = _join([
                "no timeout" if limit is None else f"more than {limit} candles"
                for limit in alternative_limits
            ], "or")
            variations = [
                f"Baseline: {baseline_wait}. Alternatives: {alternatives} "
                "before abandoning the wait.",
                "Count recorded one-minute candles from the current main zone's selection, "
                "not clock time. A replacement main zone restarts the count; "
                "other invalidation rules still apply.",
            ]
        else:
            comparison = (
                f"Compare {version_count} configured versions; "
                "the changed trading rules are listed below."
            )
        preview = (*preview[:2], comparison)
    return replace(
        base, status="partial" if issues else base.status, preview_bullets=preview,
        detail_bullets=_visible_rules(base_rules, base_parsed, changed_groups),
        variations=tuple(variations), issues=tuple(dict.fromkeys(issues)),
    )
