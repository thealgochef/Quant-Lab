"""Human-facing America/Chicago time (repair R4): convert instants, never relabel them."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import (
    chicago_label,
    chicago_wall,
    chicago_walls,
    utc_instant,
)


@pytest.mark.parametrize(("utc", "expected"), [
    ("2026-01-13T23:02:00Z", "January 13, 2026 5:02 PM CST"),  # winter; the 5 PM reopen
    ("2026-06-05T15:30:00Z", "June 5, 2026 10:30 AM CDT"),  # summer
    ("2026-03-08T07:59:00Z", "March 8, 2026 1:59 AM CST"),  # last minute before spring forward
    ("2026-03-08T08:00:00Z", "March 8, 2026 3:00 AM CDT"),  # first minute after it
    ("2026-11-01T06:30:00Z", "November 1, 2026 1:30 AM CDT"),  # repeated hour, first pass
    ("2026-11-01T07:30:00Z", "November 1, 2026 1:30 AM CST"),  # repeated hour, second pass
    ("2026-01-14T06:00:00Z", "January 14, 2026 12:00 AM CST"),  # midnight
    ("2026-01-14T18:00:00Z", "January 14, 2026 12:00 PM CST"),  # noon
])
def test_winter_summer_clock_changes_midnight_and_noon(utc, expected):
    assert chicago_label(utc) == expected


def test_every_input_form_names_the_same_instant():
    iso = "2026-01-27T21:40:53.641855957Z"
    ns = 1769550053641855957
    stamp = pd.Timestamp(iso)
    assert utc_instant(ns) == utc_instant(iso) == utc_instant(stamp)
    import numpy as np

    assert utc_instant(np.int64(ns)) == utc_instant(ns)  # NumPy integers are nanoseconds too
    assert chicago_label(ns, seconds=True) == "January 27, 2026 3:40:53.641855957 PM CST"
    assert chicago_label(datetime(2026, 1, 13, 23, 2, tzinfo=UTC), short=True) == (
        "Jan 13, 2026 5:02 PM CST")
    assert chicago_label(None) == "—" and chicago_label("") == "—"


def test_a_time_without_a_zone_is_refused_unless_its_convention_is_named():
    with pytest.raises(ValueError):
        chicago_label("2026-01-13 23:02:00")
    assert chicago_label("2026-01-13 23:02:00", naive="utc") == "January 13, 2026 5:02 PM CST"


def test_trading_date_labels_and_stored_instants_are_not_changed():
    # 5:02 PM Chicago on January 13 belongs to trading day January 14: the label shows
    # the Chicago calendar date only; the saved trading-day field is untouched.
    row = {"entry_utc": "2026-01-13T23:02:00Z", "trading_day": "2026-01-14"}
    before = dict(row)
    assert chicago_label(row["entry_utc"]).startswith("January 13, 2026")
    assert row == before


def test_sorting_uses_the_instant_not_the_text():
    utc = ["2026-01-13T16:00:00Z", "2026-01-13T15:00:00Z", "2026-01-13T20:00:00Z"]
    by_instant = [chicago_label(u) for u in sorted(utc, key=utc_instant)]
    assert by_instant == ["January 13, 2026 9:00 AM CST", "January 13, 2026 10:00 AM CST",
                          "January 13, 2026 2:00 PM CST"]
    assert sorted(by_instant) != by_instant  # sorting the text would put 10 AM first


def test_chart_positions_move_candles_and_markers_by_the_same_offset():
    candles = pd.Series(pd.to_datetime(
        ["2026-01-13T04:30:00Z", "2026-01-13T04:31:00Z"], utc=True))
    walls = chicago_walls(candles)
    entry_marker = chicago_wall("2026-01-13T04:31:00Z")
    assert walls.iloc[1] == pd.Timestamp(entry_marker)  # marker still on its candle
    assert str(walls.iloc[0]) == "2026-01-12 22:30:00"
    assert candles.iloc[0] == pd.Timestamp("2026-01-13T04:30:00Z")  # input unchanged
    summer = chicago_wall("2026-06-05T15:30:00Z")
    assert summer == datetime(2026, 6, 5, 10, 30)  # no fixed offset


def test_a_fill_at_a_nanosecond_instant_is_drawn_without_a_warning():
    """Chart positions stop at microseconds (a chart cannot draw finer); the label keeps
    the exact nanoseconds. Warnings are errors in this suite."""

    import plotly.graph_objects as go

    ns = 1769550053641855957
    at = chicago_wall(ns)
    assert at == pd.Timestamp("2026-01-27 15:40:53.641855")
    assert chicago_label(ns, seconds=True).endswith("53.641855957 PM CST")
    figure = go.Figure(go.Scatter(x=[at], y=[1.0]))
    figure.add_shape(type="line", x0=at, x1=at, y0=0, y1=1)
    figure.to_json()
    assert chicago_walls(pd.Series([pd.Timestamp(ns, tz="UTC")])).iloc[0] == at

