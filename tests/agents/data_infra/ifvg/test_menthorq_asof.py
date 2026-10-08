"""Point-in-time contracts for the separate MenthorQ v02 overnight policy."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, date, datetime

import pytest

from alpha_lab.agents.data_infra.ifvg.menthorq_asof import (
    EodAsOfIndex,
    level_set_eligible_from,
    load_v02_eod_asof_root,
    load_v02_eod_asof_zip,
    nominal_eligible_from,
)

_SHA = "a" * 64
_TABLE_HASHES = {"total_gamma": _SHA, "level_sets": _SHA, "gamma_long": _SHA}
_CUTOFF = datetime(2026, 6, 10, 21, tzinfo=UTC)


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _gamma(day: str, value: str, *, pointer: str | None = None) -> dict[str, str]:
    return {
        "vendor_report_date": day,
        "gex": value,
        "historical_publication_time": "",
        "source_path": "raw/gamma.json",
        "source_sha256": _SHA,
        "source_json_pointer": pointer or f"/{day}",
    }


def _set(requested: str, report: str, *, set_id: str | None = None) -> dict[str, str]:
    return {
        "level_set_id": set_id or requested,
        "requested_date": requested,
        "vendor_report_date": report,
        "ticker": "NQ1!",
        "level_type": "gamma_levels",
        "kind": "eod",
        "data_partition": "through_2026_06_10",
        "level_count": "2",
        "historical_publication_time": "",
        "source_path": f"raw/levels/{requested}.json",
        "source_sha256": _SHA,
        "source_json_pointer": f"/{requested}",
    }


def _levels(
    level_set: dict[str, str], *, price: str = "20000", gex: str = ""
) -> list[dict[str, str]]:
    def row(name: str, value: str, exposure: str) -> dict[str, str]:
        return {
            **level_set,
            "level_name": name,
            "value": value,
            "gex": exposure,
            "source_json_pointer": f"/{level_set['requested_date']}/{name}",
        }

    return [row("HVL", price, gex), row("1D Min", "19500", "")]


def _index(
    gamma: list[dict[str, str]],
    sets: list[dict[str, str]] | None = None,
    level_rows: list[dict[str, str]] | None = None,
    *,
    cutoff: datetime = _CUTOFF,
) -> EodAsOfIndex:
    sets = sets or []
    selected_level_rows = level_rows if level_rows is not None else [
        row for level_set in sets for row in _levels(level_set)
    ]
    return EodAsOfIndex(
        gamma_rows=gamma,
        level_set_rows=sets,
        gamma_long_rows=selected_level_rows,
        cutoff_utc=cutoff,
        bundle_sha256=_SHA,
        table_sha256=_TABLE_HASHES,
    )


def test_nominal_boundary_and_both_chicago_offsets() -> None:
    assert nominal_eligible_from(date(2025, 8, 26)) == _ts("2025-08-27T03:00:00Z")
    assert nominal_eligible_from(date(2026, 1, 15)) == _ts("2026-01-16T04:00:00Z")
    index = _index([_gamma("2025-08-25", "-1"), _gamma("2025-08-26", "2")])
    before = index.snapshot(_ts("2025-08-27T02:59:59Z"))["gamma"]
    at = index.snapshot(_ts("2025-08-27T03:00:00Z"))["gamma"]
    after_midnight = index.snapshot(_ts("2025-08-27T06:00:00Z"))["gamma"]
    assert (before["report_date"], before["sign"]) == ("2025-08-25", "negative")
    assert (at["report_date"], at["positive_run_age"]) == ("2025-08-26", 1)
    assert after_midnight["report_date"] == "2025-08-26"


def test_level_requested_date_convention_independent_of_gamma() -> None:
    old = _set("2025-08-26", "2025-08-25")
    new = _set("2025-08-27", "2025-08-26")
    index = _index([_gamma("2025-08-25", "-3")], [old, new])
    at_seven = index.snapshot(_ts("2025-08-27T00:00:00Z"))
    at_ten = index.snapshot(_ts("2025-08-27T03:00:00Z"))
    assert at_seven["levels"]["requested_date"] == "2025-08-26"
    assert at_ten["levels"]["requested_date"] == "2025-08-27"
    assert at_ten["gamma"]["report_date"] == "2025-08-25"
    assert at_ten["levels"]["report_date"] == "2025-08-26"
    assert at_ten["levels"]["items"]["1D Min"]["gex"] is None
    fields = index.policy_context_fields(_ts("2025-08-27T03:00:00Z"))
    assert fields["levels_status"] == fields["gamma_status"] == "available"
    assert fields["decision_ts_utc"] == _ts("2025-08-27T03:00:00Z")
    assert fields["gamma_eligible_from_utc"] == _ts("2025-08-26T03:00:00Z")
    assert fields["level_eligible_from_utc"] == _ts("2025-08-27T03:00:00Z")
    assert level_set_eligible_from(date(2025, 12, 26), date(2025, 12, 24)) == _ts(
        "2025-12-26T04:00:00Z"
    )


def test_distinct_report_age_left_censor_and_new_boundaries() -> None:
    days = ["2025-08-18", "2025-08-19", "2025-08-20", "2025-08-21", "2025-08-22", "2025-08-25"]
    index = _index([_gamma(day, "1") for day in days])
    fifth = index.snapshot(_ts("2025-08-23T03:00:00Z"))["gamma"]
    sixth = index.snapshot(_ts("2025-08-26T03:00:00Z"))["gamma"]
    assert fifth["positive_phase"] == "positive_age_unknown"
    assert fifth["positive_run_age"] is None
    assert sixth["positive_phase"] == "established_positive"
    assert sixth["positive_run_lower_bound"] == 6
    assert sixth["positive_run_age"] is None
    assert index.policy_context_fields(_ts("2025-08-26T03:00:00Z"))[
        "positive_run_age_is_lower_bound"
    ] is True

    bounded = _index([_gamma("2025-08-15", "0"), *[_gamma(day, "1") for day in days]])
    five = bounded.snapshot(_ts("2025-08-23T03:00:00Z"))["gamma"]
    repeat = bounded.snapshot(_ts("2025-08-25T12:00:00Z"))["gamma"]
    six = bounded.snapshot(_ts("2025-08-26T03:00:00Z"))["gamma"]
    assert five["positive_phase"] == "early_positive"
    assert five["positive_run_age"] == repeat["positive_run_age"] == 5
    assert six["positive_phase"] == "established_positive"
    assert six["positive_run_age"] == 6


def test_null_neutral_and_stale_keep_identity_without_backfill() -> None:
    level_set = _set("2025-08-27", "2025-08-26")
    index = _index(
        [_gamma("2025-08-25", "3"), _gamma("2025-08-26", "")], [level_set]
    )
    null = index.snapshot(_ts("2025-08-27T03:00:00Z"))["gamma"]
    assert (null["report_date"], null["status"], null["sign"]) == (
        "2025-08-26", "null", "unknown"
    )
    assert null["source_json_pointer"] == "/2025-08-26"
    assert null["positive_run_lower_bound"] == 0
    stale = index.snapshot(_ts("2025-09-04T17:00:00Z"))
    assert stale["gamma"]["status"] == stale["levels"]["status"] == "stale"
    assert stale["gamma"]["value"] is None
    assert stale["levels"]["items"] == {}
    assert stale["levels"]["level_set_id"] == "2025-08-27"

    neutral = _index([_gamma("2025-08-25", "1"), _gamma("2025-08-26", "0")])
    assert neutral.snapshot(_ts("2025-08-27T03:00:00Z"))["gamma"]["sign"] == "neutral"


def test_aged_exactly_seven_days_is_available() -> None:
    index = _index([_gamma("2025-08-18", "-1")])
    assert index.snapshot(_ts("2025-08-26T02:00:00Z"))["gamma"]["status"] == "selected"
    assert index.snapshot(_ts("2025-08-26T05:00:00Z"))["gamma"]["status"] == "stale"


def test_future_rows_cannot_change_past_and_measurements_are_not_parsed() -> None:
    base_set = _set("2025-08-27", "2025-08-26")
    future_set = _set("2026-06-11", "2026-06-10")
    future_set["level_count"] = "invalid future count"
    future_level = _levels(future_set, price="invalid future price")
    original = _index([_gamma("2025-08-26", "2")], [base_set])
    extended = _index(
        [_gamma("2025-08-26", "2"), _gamma("2026-06-10", "invalid future gex")],
        [base_set, future_set],
        [*_levels(base_set), *future_level],
    )
    decision = _ts("2025-08-27T12:00:00Z")
    assert original.snapshot(decision) == extended.snapshot(decision)
    assert extended.snapshot(_ts("2026-06-10T21:00:01Z"))["status"] == "outside_authorized_cutoff"
    with pytest.raises(ValueError, match="offset-aware"):
        extended.snapshot(datetime(2025, 8, 27, 12))
    json.dumps(extended.snapshot(decision))


def test_conflicting_same_requested_date_is_rejected() -> None:
    one = _set("2025-08-27", "2025-08-26")
    two = _set("2025-08-27", "2025-08-25", set_id="other")
    with pytest.raises(ValueError, match="duplicate/conflicting"):
        _index([], [one, two])
    with pytest.raises(ValueError, match="conflicting/duplicate gamma"):
        _index([_gamma("2025-08-26", "1"), _gamma("2025-08-26", "2")])
    with pytest.raises(ValueError, match="report date cannot follow"):
        _index([], [_set("2025-08-27", "2025-08-28")])


def test_sunday_reopen_and_holiday_carry_do_not_invent_reports() -> None:
    friday = _gamma("2025-08-29", "-1")
    before_sunday = _index([friday]).snapshot(_ts("2025-09-01T01:00:00Z"))["gamma"]
    after_sunday = _index([friday]).snapshot(_ts("2025-09-01T02:00:00Z"))["gamma"]
    assert before_sunday["report_date"] == after_sunday["report_date"] == "2025-08-29"
    assert before_sunday["age_calendar_days"] == after_sunday["age_calendar_days"] == 2
    old = _set("2025-12-24", "2025-12-23")
    holiday = _set("2025-12-26", "2025-12-24")
    index = _index([], [old, holiday])
    before = index.snapshot(_ts("2025-12-26T03:59:59Z"))["levels"]
    at = index.snapshot(_ts("2025-12-26T04:00:00Z"))["levels"]
    assert before["requested_date"] == "2025-12-24"
    assert (at["requested_date"], at["report_date"]) == ("2025-12-26", "2025-12-24")


def test_in_scope_publication_evidence_requires_review() -> None:
    row = _gamma("2025-08-26", "1")
    row["historical_publication_time"] = "2025-08-27T04:00:00Z"
    with pytest.raises(ValueError, match="publication evidence"):
        _index([row])


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def test_zip_loader_verifies_hashes_and_serializes_source_identity(tmp_path) -> None:
    level_set = _set("2025-08-27", "2025-08-26")
    prefix = "MenthorQ_Research_Data_v02/data/canonical/end_of_day/"
    members = {
        prefix + "metrics/total_gamma_by_report_date.csv": _csv_bytes([_gamma("2025-08-26", "1")]),
        prefix + "levels/level_sets.csv": _csv_bytes([level_set]),
        prefix + "levels/gamma_long.csv": _csv_bytes(_levels(level_set)),
    }
    archive = tmp_path / "v02.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for name, payload in members.items():
            package.writestr(name, payload)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    index = load_v02_eod_asof_zip(
        archive, cutoff_utc=_CUTOFF, expected_archive_sha256=archive_hash
    )
    result = index.snapshot(_ts("2025-08-27T12:00:00Z"))
    assert result["bundle_sha256"] == archive_hash
    assert result["table_sha256"]["total_gamma"] == hashlib.sha256(
        members[prefix + "metrics/total_gamma_by_report_date.csv"]
    ).hexdigest()
    assert result["gamma"]["source_sha256"] == _SHA
    assert result["levels"]["source_path"] == level_set["source_path"]
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        load_v02_eod_asof_zip(archive, cutoff_utc=_CUTOFF, expected_archive_sha256=_SHA)


def test_extracted_root_loader_rechecks_table_bytes(tmp_path) -> None:
    level_set = _set("2025-08-27", "2025-08-26")
    root = tmp_path / "MenthorQ_Research_Data_v02"
    payloads = {
        "total_gamma": (
            "metrics/total_gamma_by_report_date.csv",
            _csv_bytes([_gamma("2025-08-26", "1")]),
        ),
        "level_sets": ("levels/level_sets.csv", _csv_bytes([level_set])),
        "gamma_long": ("levels/gamma_long.csv", _csv_bytes(_levels(level_set))),
    }
    hashes = {}
    for key, (relative, payload) in payloads.items():
        target = root / "data" / "canonical" / "end_of_day" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[key] = hashlib.sha256(payload).hexdigest()
    index = load_v02_eod_asof_root(
        root, cutoff_utc=_CUTOFF, bundle_sha256=_SHA, expected_table_sha256=hashes
    )
    assert index.snapshot(_ts("2025-08-27T12:00:00Z"))["gamma"]["sign"] == "positive"
    with pytest.raises(ValueError, match="table SHA-256 mismatch"):
        load_v02_eod_asof_root(
            root, cutoff_utc=_CUTOFF, bundle_sha256=_SHA,
            expected_table_sha256={**hashes, "total_gamma": _SHA},
        )


def test_projection_constructs_task_core_context() -> None:
    core = pytest.importorskip("strategy_core.strategies.ifvg_smc.ifsm_policy_context")
    days = ["2025-08-18", "2025-08-19", "2025-08-20", "2025-08-21", "2025-08-22", "2025-08-25"]
    level_set = _set("2025-08-26", "2025-08-25")
    index = _index([_gamma(day, "1") for day in days], [level_set])
    fields = index.policy_context_fields(_ts("2025-08-26T12:00:00Z"))
    context = core.IfsmPolicyContext(**fields)
    assert context.gamma_sign == "positive"
    assert context.positive_run_age == 6
    assert context.positive_run_age_is_lower_bound is True
    assert context.early_positive is False
    assert context.level_prices["HVL"] == 20000.0
    assert context.gamma_eligible_from_utc <= context.decision_ts_utc
