"""Start/end research dates for a new study (repair R8), shared by the wizards.

The owner picks a start and an end date; the range resolves to logical
trading days with the existing calendar, leaving out days without stored
market data and known market closures, each with its reason. The count, the
first and last day, the exact list and the left-out days are shown before
anything is saved. June 11, 2026 onward cannot be picked.

A saved date list is never rewritten by opening the step: it is shown as
saved, and only a change to the start or end date replaces it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
from alpha_lab.agents.data_infra.ifvg.research_period import (
    EARLIEST_LOCAL_MARKET_DATE,
    LAST_PERMITTED_DAY,
    earliest_evidence_day,
    local_day_has_data,
    resolve_research_range,
    warmup_dates_for,
)

__all__ = ["render_research_dates"]


def _long(day: str) -> str:
    parsed = date.fromisoformat(day)
    return f"{parsed:%B} {parsed.day}, {parsed.year}"


def render_research_dates(st, payload, key: str, repo_root: Path
                          ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(evidence dates, warmup dates) chosen on this step."""

    saved = tuple(payload.get("real_dates") or ())
    saved_warmup = tuple(payload.get("warmup_dates") or ()) or (
        warmup_dates_for(saved[0]) if saved else FROZEN_WARMUP_DATES)
    first_allowed = earliest_evidence_day()
    st.write(f"Permitted research period: {_long(first_allowed)} to "
             f"{_long(LAST_PERMITTED_DAY)}. Stored market data starts "
             f"{_long(EARLIEST_LOCAL_MARKET_DATE)}; June 11, 2026 onward is protected and "
             "cannot be chosen.")
    bounds = {"min_value": date.fromisoformat(first_allowed),
              "max_value": date.fromisoformat(LAST_PERMITTED_DAY), "format": "YYYY-MM-DD"}
    ends = _pickable_ends(saved, bounds)
    columns = st.columns(2)
    start = columns[0].date_input(
        "Start date", value=ends[0], key=key + "start", **bounds)
    end = columns[1].date_input(
        "End date", value=ends[1], key=key + "end", **bounds)
    if start is None or end is None:
        if saved and ends == (None, None):
            # a saved list the pickers cannot show (outside the permitted period, or not
            # dates) is kept as saved; the step's own checks report its problems
            st.warning(f"This study's saved list of {len(saved)} days cannot be shown as a "
                       "start and end date here. It is kept exactly as saved; choose a start "
                       "and an end date to replace it.")
            _list(st, saved, saved_warmup, (), key)
        else:
            st.caption("Choose a start and an end date. The range becomes every trading day "
                       "between them with stored market data.")
        return saved, saved_warmup
    resolved = resolve_research_range(start.isoformat(), end.isoformat(),
                                      day_has_data=local_day_has_data(repo_root))
    for problem in resolved.problems:
        st.error(problem)
    unchanged = bool(saved) and (start.isoformat(), end.isoformat()) == (saved[0], saved[-1])
    if unchanged and resolved.trading_days != saved:
        # a list saved before date ranges existed (or picked day by day) stays as saved
        st.info(f"This study's saved list of {len(saved)} trading days "
                f"({_long(saved[0])} to {_long(saved[-1])}) is kept exactly as saved. "
                "Changing the start or end date replaces it with the resolved range.")
        _list(st, saved, saved_warmup, (), key)
        return saved, saved_warmup
    if not resolved.usable:
        # opening never replaces a saved list; a changed, unusable range never keeps the
        # old list silently (nothing to save until the range is usable)
        return (saved, saved_warmup) if unchanged else ((), ())
    days, warmup = resolved.trading_days, resolved.warmup_dates
    st.success(f"{len(days)} trading days: {_long(days[0])} to {_long(days[-1])} "
               f"· {len(warmup)} warmup days before them")
    for note in resolved.warnings:
        st.info(note)
    _list(st, days, warmup, resolved.excluded, key)
    return days, warmup


def _pickable_ends(saved, bounds) -> tuple[date | None, date | None]:
    """The saved first and last day for the pickers, or (None, None) when either is
    not a date or lies outside the permitted period (the picker would refuse it)."""

    if not saved:
        return None, None
    try:
        first, last = date.fromisoformat(saved[0]), date.fromisoformat(saved[-1])
    except (TypeError, ValueError):
        return None, None
    low, high = bounds["min_value"], bounds["max_value"]
    if not (low <= first <= high and low <= last <= high):
        return None, None
    return first, last


def _list(st, days, warmup, excluded, key) -> None:
    with st.expander(f"The exact {len(days)} trading days and {len(warmup)} warmup days"):
        st.caption("Warmup (replayed first, not evaluated): " + ", ".join(warmup))
        st.code("\n".join(days), language="text")
    if excluded:
        with st.expander(f"{len(excluded)} calendar trading days left out, with the reason"):
            import pandas as pd

            st.table(pd.DataFrame(excluded, columns=["Day", "Reason"]).set_index("Day"))
