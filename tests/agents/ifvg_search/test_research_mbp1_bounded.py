"""Bounded MBP sources retain canonical identities without retaining all days."""

import weakref

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.features.mbp1_feature_materializer import (
    materialize_mbp1_features,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
    build_mbp1_source_artifact,
)
from alpha_lab.agents.data_infra.ifvg.search.research_mbp1 import BoundedDayEvents
from tests.agents.ifvg_search.mbp1_fixture import (
    MBP1_RESOLVED_BLOCK,
    anchor_row,
    build_fixture_source,
    default_day_events,
    synthetic_contract,
)


def events_for(day):
    frame = default_day_events()
    shift = (pd.Timestamp(day) - pd.Timestamp("2026-01-13")).value
    frame["ts_event"] += shift
    frame["ts_recv"] += shift
    return frame


def test_no_retained_event_bytes_preserves_source_identity_and_releases_frames():
    days = ("2026-01-13", "2026-01-14", "2026-01-15")
    prior = []

    def load(day):
        assert not prior or prior[-1]() is None
        frame = events_for(day)
        prior.append(weakref.ref(frame))
        return frame

    bounded = BoundedDayEvents(days, load)
    actual, discarded = build_mbp1_source_artifact(
        bounded,
        contract=synthetic_contract(),
        authorized_date_set_id="fixture",
        events_stored=False,
        retain_event_bytes=False,
    )
    assert discarded == {}
    assert bounded.max_resident_days == 1
    assert bounded.load_count == len(days)
    expected, retained = build_mbp1_source_artifact(
        {day: events_for(day) for day in days},
        contract=synthetic_contract(),
        authorized_date_set_id="fixture",
        events_stored=False,
    )
    assert actual == expected
    assert len(retained) == len(days)
    bounded.release()
    assert bounded.resident_days == ()
    assert prior[-1]() is None
    with pytest.raises(ValueError, match="stored"):
        build_mbp1_source_artifact(
            {},
            contract=synthetic_contract(),
            authorized_date_set_id="fixture",
            retain_event_bytes=False,
        )


def test_day_order_materialization_matches_legacy_candidate_order_hashes(monkeypatch):
    days = ("2026-01-13", "2026-01-14")
    eager = {day: events_for(day) for day in days}
    source, _bytes, _events = build_fixture_source(eager)
    rows = []
    for candidate_id, day in (("a", days[1]), ("b", days[0]), ("c", days[1]), ("d", days[0])):
        row = anchor_row(candidate_id, day=day)
        if day == days[1]:
            for key in (
                "tap_ts_utc",
                "lock_ts_utc",
                "armed_ts_utc",
                "inversion_ts_utc",
                "entry_ts_utc",
            ):
                row[key] = (pd.Timestamp(row[key]) + pd.Timedelta(days=1)).isoformat()
        rows.append(row)
    anchors = pd.DataFrame(rows)
    observed = []
    loaded_frames = []

    def load(day):
        assert not loaded_frames or loaded_frames[-1]() is None
        observed.append(day)
        frame = events_for(day)
        loaded_frames.append(weakref.ref(frame))
        return frame

    bounded = BoundedDayEvents(days, load)
    actual, features, evidence = materialize_mbp1_features(
        source,
        anchors,
        resolved_block=MBP1_RESOLVED_BLOCK,
        events_by_day=bounded,
    )
    assert observed == [*days, *days]  # verification pass, then materialization pass
    assert bounded.max_resident_days == 1
    assert features.candidate_id.tolist() == ["a", "b", "c", "d"]
    original_sort = pd.DataFrame.sort_values

    def legacy_sort(self, by, *args, **kwargs):
        if isinstance(by, list) and by == ["trading_day", "candidate_id"]:
            by = "candidate_id"
        return original_sort(self, by, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sort_values", legacy_sort)
    expected, old_features, old_evidence = materialize_mbp1_features(
        source,
        anchors,
        resolved_block=MBP1_RESOLVED_BLOCK,
        events_by_day=eager,
    )
    assert actual == expected
    pd.testing.assert_frame_equal(features, old_features)
    pd.testing.assert_frame_equal(evidence, old_evidence)
