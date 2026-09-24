"""Named starting points for new studies (repair R7).

The owner's current selected configuration, S0_D80_W1_P1 from the completed
daily-close study, is offered as a named baseline: the registered legacy
profile plus that configuration's ten saved setting values. The values are
read from the verified study package (never retyped) and must re-resolve to
the configuration's saved section hash, or the named baseline is not offered.

It is deliberately NOT a newly registered profile: registering a profile with
the same content would rename the configuration's canonical identity and break
matching with its saved replays. The legacy baseline itself is not edited,
renamed or re-resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "LEGACY_PROFILE_NAME",
    "OWNER_BASELINE_ID",
    "NamedBaseline",
    "NamedBaselineUnavailableError",
    "holding_rule_statement",
    "legacy_baseline_warning",
    "owner_selected_baseline",
]

OWNER_BASELINE_ID = "S0_D80_W1_P1"
OWNER_BASELINE_RUN_ID = "a0f66422ef9f6b3393ad21d0b9740ebc0f23684cfe3463139a90743141512fd7"
OWNER_BASELINE_SECTION_HASH = "86261cc90c14c16db61f28e5cf77ee9079d1808b79af29160cec012b27e534d4"
LEGACY_PROFILE_NAME = "ifvg_v2_doc_default_fresh_static_1r"
DAILY_CLOSE = "scheduled_daily_close_v1"


class NamedBaselineUnavailableError(RuntimeError):
    """The named baseline cannot be offered exactly as saved."""


@dataclass(frozen=True)
class NamedBaseline:
    baseline_id: str
    label: str
    profile_name: str
    axis_value_ids: tuple[tuple[str, str], ...]
    section_hash: str
    source_run_id: str
    source_manifest_sha256: str


@lru_cache(maxsize=1)
def owner_selected_baseline() -> NamedBaseline:
    """S0_D80_W1_P1 exactly as the verified daily-close study saved it."""

    from alpha_lab.propsim.funded.comparison_source import (
        BASELINE_PROFILE,
        discover_comparison_sources,
        resolve_configuration,
    )

    source = next((s for s in discover_comparison_sources()
                   if s.package.run_id == OWNER_BASELINE_RUN_ID), None)
    if source is None:
        raise NamedBaselineUnavailableError(
            "The verified daily-close study that holds S0_D80_W1_P1 is not available on this "
            "computer, so the named baseline is not offered.")
    config = source.by_name.get(OWNER_BASELINE_ID)
    if config is None or config.resolved_section_config_hash != OWNER_BASELINE_SECTION_HASH:
        raise NamedBaselineUnavailableError(
            "S0_D80_W1_P1 in the verified study does not match its recorded identity; the "
            "named baseline is not offered.")
    _section, cfg = resolve_configuration(config.axis_value_ids)
    if cfg.profile_hash != OWNER_BASELINE_SECTION_HASH or BASELINE_PROFILE != LEGACY_PROFILE_NAME:
        raise NamedBaselineUnavailableError(
            "S0_D80_W1_P1 resolves differently with this application's strategy engine; the "
            "named baseline is not offered rather than guessed.")
    return NamedBaseline(
        baseline_id=OWNER_BASELINE_ID,
        label="Owner's selected configuration S0_D80_W1_P1 (mandatory 3:55 PM Chicago close)",
        profile_name=LEGACY_PROFILE_NAME,
        axis_value_ids=tuple(sorted(config.axis_value_ids.items())),
        section_hash=OWNER_BASELINE_SECTION_HASH,
        source_run_id=source.package.run_id,
        source_manifest_sha256=source.package.manifest_sha256,
    )


def legacy_baseline_warning(section: Any) -> str | None:
    """The plain warning for a configuration that does not follow the daily close."""

    if getattr(section, "holding_policy", DAILY_CLOSE) == DAILY_CLOSE:
        return None
    text = "This baseline holds positions across the daily close and weekends"
    if getattr(section, "parent_retest_timeout_1m_bars", 0) is None:
        text += " and has no retest time limit"
    return text + ". It does not follow the mandatory 3:55 PM Chicago close."


def holding_rule_statement(labels_without_daily_close: list[str], total: int) -> str | None:
    """The review-step statement before approval (None when every one closes daily)."""

    if not labels_without_daily_close:
        return None
    if total == 1:
        return ("This study's holding rule is not the mandatory 3:55 PM Chicago daily close: "
                "positions can be carried across the daily close and weekends. It will run "
                "exactly as shown; nothing is converted.")
    return (f"{len(labels_without_daily_close)} of {total} configurations in this study do not "
            "use the mandatory 3:55 PM Chicago daily close: positions can be carried across "
            "the daily close and weekends. They will run exactly as shown; nothing is "
            "converted.")
