"""Readable configuration names built from each configuration's SAVED settings.

The mocks show a configuration in two lines — ``All open-market hours · Long
only · Half at 1R`` over ``1R · 1-hour + 4-hour gaps · 1- and 3-minute
parents`` — and, in the detail header, a one-line settings summary. These are
derived from the saved plain-English setting values; a value this module does
not recognize is shown as saved, never guessed or dropped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["ConfigurationName", "configuration_name", "short_picker_name", "study_names"]

_WORD_MINUTES = {
    "one": 1, "three": 3, "five": 5, "ten": 10, "fifteen": 15, "thirty": 30,
    "sixty": 60, "two": 2, "four": 4,
}


@dataclass(frozen=True)
class ConfigurationName:
    line1: str
    line2: str
    summary: str

    @property
    def full(self) -> str:
        return f"{self.line1} · {self.line2}"


def _entry_hours(value: str) -> str:
    if value.startswith("Original three windows"):
        return "Original three windows"
    return value.split(":", 1)[0] if ":" in value and "open-market" not in value else value


def _target(value: str) -> str:
    if "1 to 1" in value or "equal to the initial risk" in value:
        return "1R"
    found = re.match(r"(\d+(?:\.\d+)?) times the initial risk", value)
    return f"{found.group(1)}R" if found else value


def _exit(value: str, target: str) -> str:
    if value.startswith("Half"):
        found = re.search(r"\((\d+(?:\.\d+)?R)\)", value)
        return f"Half at {found.group(1) if found else target}"
    if value.startswith("The whole position"):
        return "Whole position"
    return value


def _gaps(value: str) -> str:
    hours = [w for w in ("one-hour", "four-hour", "two-hour", "thirty-minute") if w in value]
    if not hours:
        return value
    short = [f"{_WORD_MINUTES[h.split('-')[0]]}-{h.split('-')[1]}" for h in hours]
    return (" + ".join(short) + " gaps") if len(short) > 1 else f"{short[0]} gaps only"


def _parents(value: str) -> str:
    has_one = "one-minute" in value
    has_three = "three-minute" in value
    if has_one and has_three:
        return "1- and 3-minute parents"
    if has_one:
        return "1-minute, no 3-minute parents"
    if has_three:
        return "3-minute, no 1-minute parents"
    return "no 1- or 3-minute parents"


def _ticks(value: str) -> str:
    found = re.match(r"(\d+) ticks?", value)
    return f"{found.group(1)} ticks" if found else value


def _daily_close(value: str) -> str:
    found = re.search(r"closed by (\d{1,2}:\d{2} [AP]M)", value)
    return f"flat by {found.group(1)}" if found else "no daily close"


def _saved_parents(timeframes: Iterable[str]) -> str:
    labels = tuple(timeframes)
    if labels and all(label.endswith("m") for label in labels):
        return ", ".join(label[:-1] for label in labels) + "-minute parents"
    return ", ".join(labels) + " parent charts" if labels else "no parent charts"


def configuration_name(settings: Iterable[tuple[str, str]],
                       fallback: str = "Unnamed configuration", *,
                       parent_timeframes: Iterable[str] | None = None) -> ConfigurationName:
    """Two readable lines and a summary from the saved (setting, value) pairs."""

    values = {name: value for name, value in settings}
    if not values:
        if parent_timeframes is not None:
            parents = _saved_parents(parent_timeframes)
            return ConfigurationName(fallback, parents, f"{fallback} · {parents}")
        return ConfigurationName(fallback, "", fallback)
    target = _target(values.get("Profit target", ""))
    hours = _entry_hours(values.get("Entry hours", ""))
    direction = values.get("Direction", "")
    exit_rule = _exit(values.get("Exit rule", ""), target)
    gaps = _gaps(values.get("Higher-timeframe gap charts", ""))
    parents = (_parents(values.get("Supporting (parent) charts", ""))
               if parent_timeframes is None else _saved_parents(parent_timeframes))
    line1 = " · ".join(p for p in (hours, direction, exit_rule) if p)
    line2 = " · ".join(p for p in (target, gaps, parents) if p)
    parts = [f"{target} target" if target else "", gaps, parents]
    if "Largest distance from the parent gap to the opposing gap" in values:
        parts.append("opposing distance "
                     + _ticks(values["Largest distance from the parent gap to the opposing gap"]))
    if "Smallest opposing gap" in values:
        minimum = _ticks(values["Smallest opposing gap"]).replace(" ticks", "-tick")
        minimum = re.sub(r"^1 tick.*", "1-tick", minimum)
        parts.append(f"{minimum} minimum")
    if "Daily close" in values:
        parts.append(_daily_close(values["Daily close"]))
    return ConfigurationName(line1 or fallback, line2, " · ".join(p for p in parts if p))


def _minimum_text(value: str) -> str:
    minimum = _ticks(value).replace(" ticks", "-tick")
    return f"{re.sub(r'^1 tick.*', '1-tick', minimum)} minimum"


#: settings the two lines leave out, added to line 2 only when they vary in the study
_EXTRA = (
    ("Largest distance from the parent gap to the opposing gap",
     lambda v: "opposing distance " + _ticks(v)),
    ("Smallest opposing gap", _minimum_text),
    ("Daily close", _daily_close),
)


def study_names(settings_by_configuration: dict[str, Iterable[tuple[str, str]]], *,
                parent_timeframes_by_configuration: dict[str, Iterable[str]] | None = None,
                ) -> dict[str, ConfigurationName]:
    """Readable names for every configuration of one study, never two alike.

    The two lines leave out the opposing-gap distance, its minimum and the daily
    close. When a study varies one of those, it is added to line 2 of every
    configuration; a name that is still shared ends with its configuration key.
    """

    settings = {key: dict(value) for key, value in settings_by_configuration.items()}
    parents = parent_timeframes_by_configuration or {}
    names = {key: configuration_name(values.items(), key, parent_timeframes=parents.get(key))
             for key, values in settings.items()}
    varying = [(setting, words) for setting, words in _EXTRA
               if len({values.get(setting) for values in settings.values()}) > 1]
    if varying:
        for key, values in settings.items():
            extra = [words(values[setting]) for setting, words in varying if values.get(setting)]
            name = names[key]
            names[key] = ConfigurationName(
                name.line1, " · ".join(p for p in (name.line2, *extra) if p), name.summary)
    seen: dict[str, list[str]] = {}
    for key, name in names.items():
        seen.setdefault(name.full, []).append(key)
    for keys in seen.values():
        if len(keys) > 1:
            for key in keys:
                name = names[key]
                names[key] = ConfigurationName(name.line1, f"{name.line2} · {key}", name.summary)
    return names


def short_picker_name(name: ConfigurationName) -> str:
    """Compact form for pickers: ``All hours · Long · Half at 1R``."""

    text = name.line1.replace("All open-market hours", "All hours")
    text = text.replace("Original three windows", "Three windows")
    return text.replace("Long only", "Long").replace("Long and short", "Long and short")
