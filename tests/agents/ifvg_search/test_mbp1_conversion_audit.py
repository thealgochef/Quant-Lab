"""The diagnostic audit must refuse scalar drift, reordering and extra rows."""

import pandas as pd
import pytest
from scripts.audit_mbp1_conversion_day import ParquetRows, assert_exact_rows

from tests.agents.ifvg_search.test_research_mbp1_adapter import raw, write_day
from tests.agents.ifvg_search.test_research_subject_data import subject_fixture


def test_chunk_comparison_is_ordinal_exact_and_detects_extra_or_missing_rows(tmp_path):
    frame = pd.DataFrame({"value": [1.0, None, 2.0], "sequence": [3, 1, 2]})
    path = tmp_path / "sample.parquet"
    frame.to_parquet(path, index=False)
    reader = ParquetRows(path)
    assert_exact_rows(reader.take(2), frame.iloc[:2])
    with pytest.raises(AssertionError, match="contains rows"):
        reader.assert_exhausted()
    assert_exact_rows(reader.take(1), frame.iloc[2:])
    reader.assert_exhausted()
    with pytest.raises(AssertionError, match="ended before"):
        reader.take(1)
    with pytest.raises(AssertionError):
        assert_exact_rows(frame.iloc[::-1], frame)
    changed = frame.copy()
    changed.loc[0, "value"] += 1e-12
    with pytest.raises(AssertionError):
        assert_exact_rows(changed, frame)


def test_reader_clips_midnight_snapshot_without_backdating_its_receive_time(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.development_access import DevelopmentReplayPolicy
    from alpha_lab.agents.data_infra.ifvg.search.research_mbp1 import read_research_mbp1_day

    subject = subject_fixture()
    prior = raw("2026-01-12T23:59:59Z")
    snapshot = raw("2026-01-12T23:59:59Z", action="A")
    snapshot.update(ts_recv=pd.Timestamp("2026-01-13T00:00:00Z"), flags=168)
    write_day(tmp_path, "2026-01-12", [prior])
    write_day(tmp_path, "2026-01-13", [snapshot, raw("2026-01-13T12:00:00Z")])
    normalized = read_research_mbp1_day(
        subject, tmp_path, "2026-01-13", DevelopmentReplayPolicy(subject.replay_dates)
    )
    assert len(normalized) == 2
    assert normalized["flags"].eq(0).all()
    assert normalized["ts_recv"].tolist() == [
        pd.Timestamp("2026-01-12T23:59:59Z").value,
        pd.Timestamp("2026-01-13T12:00:00Z").value,
    ]
