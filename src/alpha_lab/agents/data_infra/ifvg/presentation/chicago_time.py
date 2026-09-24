"""Human-readable America/Chicago time for IFVG study and review screens.

One presentation path for every time a person reads: selectors, tables,
chart axes, hovers and review summaries. It converts instants (never relabels
them), uses a 12-hour clock with AM/PM, and always names the zone (CST or CDT,
which also tells apart the repeated hour when clocks fall back). Stored
timestamps, trading-date labels, session windows, hashes and exports are never
touched: callers convert only what they display and keep sorting by the
original instant.

An input without a time zone is refused unless the caller names the source's
documented convention (``naive="utc"`` for columns defined as UTC).
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

__all__ = [
    "AXIS_TITLE",
    "CHICAGO",
    "HOVERFORMAT",
    "TICKFORMAT",
    "chicago_label",
    "chicago_wall",
    "chicago_walls",
    "style_chicago_axis",
    "utc_instant",
]

CHICAGO = ZoneInfo("America/Chicago")
AXIS_TITLE = "Chicago time (CST/CDT)"
TICKFORMAT = "%-I:%M %p<br>%b %-d, %Y"
HOVERFORMAT = "%b %-d, %Y %-I:%M:%S %p"
Naive = Literal["reject", "utc"]


def utc_instant(value: Any, *, naive: Naive = "reject") -> pd.Timestamp | None:
    """The instant as a UTC timestamp (None for a missing value).

    Accepts an aware ``Timestamp``/``datetime``, integer nanoseconds since the
    epoch, or ISO text (``Z`` or an offset, nanosecond fractions kept).
    """

    if value is None or (not isinstance(value, (str, Integral)) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        raise TypeError("a boolean is not a time")
    if isinstance(value, Integral):  # Python and NumPy integers alike
        return pd.Timestamp(int(value), unit="ns", tz="UTC")
    if isinstance(value, str) and not value.strip():
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        if naive != "utc":
            raise ValueError(f"time without a zone: {value!r}; name its source convention")
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def chicago_label(value: Any, *, seconds: bool = False, short: bool = False,
                  naive: Naive = "reject", missing: str = "—") -> str:
    """``January 13, 2026 5:02 PM CST`` (``short``: ``Jan 13, 2026 5:02 PM CST``)."""

    instant = utc_instant(value, naive=naive)
    if instant is None:
        return missing
    local = instant.tz_convert(CHICAGO)
    hour = local.hour % 12 or 12
    clock = f"{hour}:{local.minute:02d}"
    if seconds:
        clock += f":{local.second:02d}"
        if local.nanosecond or local.microsecond:
            clock += f".{local.microsecond:06d}{local.nanosecond:03d}".rstrip("0")
    month = local.strftime("%b" if short else "%B")
    meridiem = "AM" if local.hour < 12 else "PM"
    return f"{month} {local.day}, {local.year} {clock} {meridiem} {local.tzname()}"


def chicago_wall(value: Any, *, naive: Naive = "reject") -> pd.Timestamp | None:
    """Naive Chicago wall-clock time for a chart axis (None for a missing value).

    A chart position is kept to the microsecond, the finest a chart can draw
    (finer digits would only be discarded, with a warning, when it is drawn);
    the exact recorded nanoseconds stay in the labels (:func:`chicago_label`).
    """

    instant = utc_instant(value, naive=naive)
    if instant is None:
        return None
    return instant.tz_convert(CHICAGO).tz_localize(None).floor("us")


def chicago_walls(values: Any, *, naive: Naive = "reject") -> pd.Series:
    """Vectorized :func:`chicago_wall` for a column of instants (order kept)."""

    series = pd.Series(values)
    stamps = pd.to_datetime(series, utc=False)
    if getattr(stamps.dt, "tz", None) is None:
        if naive != "utc":
            raise ValueError("times without a zone; name their source convention")
        stamps = stamps.dt.tz_localize("UTC")
    return stamps.dt.tz_convert(CHICAGO).dt.tz_localize(None).dt.floor("us")


def style_chicago_axis(fig: Any, *, title: str = AXIS_TITLE) -> Any:
    """Label a figure's time axis as Chicago time with 12-hour ticks and hovers."""

    fig.update_xaxes(title_text=title, tickformat=TICKFORMAT, hoverformat=HOVERFORMAT)
    return fig
