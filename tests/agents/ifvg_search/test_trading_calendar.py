"""HARDENING-BACKEND Phase 3 §5.1 (F-22) — logical trading-day semantics.

A logical trading day is the Strategy-Core trading-day id ``td`` whose
stream is ``[td−1 18:00 ET, td 18:00 ET)``, composed from the physical UTC
partitions ``td−1`` and ``td``. A physical partition date alone (the Sunday
file that holds the Sunday 18:00 ET open) is NOT a trading-day id.
"""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.search.trading_calendar import (
    CANONICAL_CHAIN_START_DAY,
    CME_GLOBEX_18ET_WEEKDAY_V1,
    LEGACY_VERIFIED_REPLAY_SOURCE,
    PERMITTED_WINDOW_LAST_DAY,
    PUBLIC_SOURCE_KINDS,
    TRADING_CALENDAR_POLICY_ID,
    PhysicalSourceDescriptor,
    SourcePartitionRef,
    VerificationTradingDayRef,
    assert_consecutive_logical_days,
    assert_public_source_kind,
    consecutive_logical_windows,
    inventory_from_permitted_source_hashes,
    is_logical_trading_day,
    logical_trading_days,
    next_logical_trading_day,
    physical_descriptors_from_permitted_source_hashes,
    physical_partitions_for,
    physical_source_descriptor,
    previous_logical_trading_day,
    session_bounds_utc,
    source_kind_of_partition_file,
    store_day_chain,
    trading_day_ref_from_inventory,
)


def test_policy_is_weekday_minus_registered_full_closures() -> None:
    policy = CME_GLOBEX_18ET_WEEKDAY_V1
    assert policy.policy_id == TRADING_CALENDAR_POLICY_ID
    assert "2026-01-01" in policy.full_closure_days  # New Year's Day: no session (evidence)
    # Good Friday 2026-04-03 carried a full session in the accepted dataset
    # (9 lifecycle rows, 1 trade): the evidence contradicts a closure
    assert "2026-04-03" not in policy.full_closure_days
    assert is_logical_trading_day("2026-04-03")
    # partial-session holidays are trading days (MLK, Presidents, Memorial)
    for day in ("2026-01-19", "2026-02-16", "2026-05-25"):
        assert is_logical_trading_day(day)
    assert not is_logical_trading_day("2026-01-01")
    # Sunday and Saturday physical dates are never trading-day ids
    assert not is_logical_trading_day("2026-02-08")
    assert not is_logical_trading_day("2026-02-07")


def test_logical_days_are_consecutive_by_calendar_index() -> None:
    days = logical_trading_days("2026-02-05", "2026-02-13")
    assert days == (
        "2026-02-05",
        "2026-02-06",
        "2026-02-09",
        "2026-02-10",
        "2026-02-11",
        "2026-02-12",
        "2026-02-13",
    )
    assert next_logical_trading_day("2026-02-06") == "2026-02-09"
    assert previous_logical_trading_day("2026-02-09") == "2026-02-06"
    assert next_logical_trading_day("2025-12-31") == "2026-01-02"
    windows = consecutive_logical_windows(days, 5)
    assert windows[0] == ("2026-02-05", "2026-02-06", "2026-02-09", "2026-02-10", "2026-02-11")
    assert len(windows) == 3
    assert_consecutive_logical_days(("2026-02-06", "2026-02-09", "2026-02-10"))
    with pytest.raises(ValueError, match="not a logical trading day"):
        assert_consecutive_logical_days(("2026-02-06", "2026-02-08", "2026-02-09"))
    with pytest.raises(ValueError, match="consecutive"):
        assert_consecutive_logical_days(("2026-02-06", "2026-02-10"))
    with pytest.raises(ValueError, match="at most five"):
        consecutive_logical_windows(days, 6)


def test_physical_partitions_and_session_bounds() -> None:
    parts = physical_partitions_for("2026-02-09")  # Monday
    assert [p.physical_utc_date for p in parts] == ["2026-02-08", "2026-02-09"]
    assert [p.relative_logical_partition_key for p in parts] == ["prev_utc_date", "utc_date"]
    open_ts, close_ts = session_bounds_utc("2026-02-09")
    # 18:00 ET (EST, UTC−5) → 23:00 UTC the day before / on the day
    assert open_ts == "2026-02-08T23:00:00+00:00"
    assert close_ts == "2026-02-09T23:00:00+00:00"
    open_dst, close_dst = session_bounds_utc("2026-06-08")  # EDT, UTC−4
    assert open_dst == "2026-06-07T22:00:00+00:00"
    assert close_dst == "2026-06-08T22:00:00+00:00"


def test_store_day_chain_is_every_non_saturday_calendar_day() -> None:
    chain = store_day_chain("2026-02-05", "2026-02-10")
    assert chain == (
        "2026-02-05",
        "2026-02-06",
        "2026-02-08",  # the Sunday store day holds the Sunday 18:00 ET open
        "2026-02-09",
        "2026-02-10",
    )
    assert CANONICAL_CHAIN_START_DAY == "2026-01-01"
    assert PERMITTED_WINDOW_LAST_DAY == "2026-06-10"
    with pytest.raises(ValueError, match="Saturday"):
        store_day_chain("2026-02-05", "2026-02-07")


def test_trading_day_ref_binds_both_partitions_from_the_inventory() -> None:
    # a historical (pre-MBP-1) physical inventory: the PUBLIC kind is the
    # opaque legacy literal, never the physical file stem
    inventory = inventory_from_permitted_source_hashes(
        [
            ["2026-02-08/mbp10.parquet", "a" * 64],
            ["2026-02-09/mbp10.parquet", "b" * 64],
        ]
    )
    ref = trading_day_ref_from_inventory("2026-02-09", inventory)
    assert isinstance(ref, VerificationTradingDayRef)
    assert ref.logical_trading_day == "2026-02-09"
    assert [r.physical_utc_date for r in ref.ordered_source_partition_refs] == [
        "2026-02-08",
        "2026-02-09",
    ]
    assert ref.ordered_source_partition_refs[0].content_sha256 == "a" * 64
    assert ref.ordered_source_partition_refs[0].source_kind == LEGACY_VERIFIED_REPLAY_SOURCE
    # a missing partition means the day is NOT hash-addressable → None
    assert trading_day_ref_from_inventory("2026-02-10", inventory) is None
    # the contract refuses a ref whose partitions are not (td−1, td) in order
    with pytest.raises(ValueError):
        VerificationTradingDayRef(
            logical_trading_day="2026-02-09",
            session_open_ts_utc="2026-02-08T23:00:00+00:00",
            session_close_ts_utc="2026-02-09T23:00:00+00:00",
            ordered_source_partition_refs=(
                SourcePartitionRef(
                    physical_utc_date="2026-02-09",
                    relative_logical_partition_key="utc_date",
                    source_kind=LEGACY_VERIFIED_REPLAY_SOURCE,
                    content_sha256="b" * 64,
                ),
            ),
        )
    with pytest.raises(ValueError, match="not a logical trading day"):
        trading_day_ref_from_inventory("2026-02-08", inventory)


# ── HARDENING-BACKEND-FIX §5 — the public source-kind boundary ───────────────


def test_physical_pre_mbp1_file_resolves_to_the_public_legacy_source_kind() -> None:
    """HB-FIX-03: a historical physical file is mapped deterministically to
    the opaque public provenance literal; MBP-1 / trades are unchanged."""

    assert PUBLIC_SOURCE_KINDS == ("mbp1", "trades", LEGACY_VERIFIED_REPLAY_SOURCE)
    assert source_kind_of_partition_file("2026-02-08/mbp10.parquet") == (
        LEGACY_VERIFIED_REPLAY_SOURCE
    )
    assert source_kind_of_partition_file("mbp1.parquet") == "mbp1"
    assert source_kind_of_partition_file("trades.parquet") == "trades"
    with pytest.raises(ValueError, match="unregistered source partition file"):
        source_kind_of_partition_file("mbp5.parquet")
    inventory = inventory_from_permitted_source_hashes(
        [["2026-02-08/mbp10.parquet", "a" * 64], ["2026-02-09/mbp1.parquet", "b" * 64]]
    )
    assert inventory == {
        "2026-02-08": (LEGACY_VERIFIED_REPLAY_SOURCE, "a" * 64),
        "2026-02-09": ("mbp1", "b" * 64),
    }
    # deterministic and content-preserving: the same entries give the same view
    assert inventory == inventory_from_permitted_source_hashes(
        [["2026-02-09/mbp1.parquet", "b" * 64], ["2026-02-08/mbp10.parquet", "a" * 64]]
    )
    with pytest.raises(ValueError, match="two partition files"):
        inventory_from_permitted_source_hashes(
            [["2026-02-08/mbp10.parquet", "a" * 64], ["2026-02-08/mbp1.parquet", "b" * 64]]
        )
    with pytest.raises(ValueError, match="content sha256"):
        inventory_from_permitted_source_hashes([["2026-02-08/mbp10.parquet", "not-a-hash"]])


def test_public_source_kind_refuses_the_physical_stem() -> None:
    for lawful in PUBLIC_SOURCE_KINDS:
        assert assert_public_source_kind(lawful) == lawful
    for unlawful in ("mbp10", "MBP10", "mbp-10", "mbp_10", "", None, 10):
        with pytest.raises(ValueError, match="not a public source kind"):
            assert_public_source_kind(unlawful)
    # the frozen contract refuses it too (pydantic Literal)
    with pytest.raises(ValueError):
        SourcePartitionRef(
            physical_utc_date="2026-02-09",
            relative_logical_partition_key="utc_date",
            source_kind="mbp10",  # type: ignore[arg-type]
            content_sha256="b" * 64,
        )
    # and an inventory mapping that smuggles the stem never reaches a ref
    with pytest.raises(ValueError, match="not a public source kind"):
        trading_day_ref_from_inventory(
            "2026-02-09",
            {"2026-02-08": ("mbp10", "a" * 64), "2026-02-09": ("mbp10", "b" * 64)},
        )


def test_physical_descriptor_keeps_truthful_provenance_without_public_capability() -> None:
    descriptor = physical_source_descriptor("2026-02-08/mbp10.parquet", "a" * 64)
    assert isinstance(descriptor, PhysicalSourceDescriptor)
    assert descriptor.physical_filename == "mbp10.parquet"  # the truthful on-disk name
    assert descriptor.physical_partition_key == "2026-02-08/mbp10.parquet"
    assert descriptor.physical_content_sha256 == "a" * 64
    assert descriptor.physical_schema_era_id == "legacy_verified_replay_source_era_v1"
    assert descriptor.source_kind == LEGACY_VERIFIED_REPLAY_SOURCE
    assert descriptor.replay_bytes_only is True
    # the descriptor is an internal dataclass, not a registered contract
    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        FrozenContract,
        registered_identity_pairs,
    )

    assert not issubclass(PhysicalSourceDescriptor, FrozenContract)
    assert all(
        pair.payload_cls is not PhysicalSourceDescriptor
        and pair.envelope_cls is not PhysicalSourceDescriptor
        for pair in registered_identity_pairs()
    )
    descriptors = physical_descriptors_from_permitted_source_hashes(
        [["2026-02-08/mbp10.parquet", "a" * 64], ["2026-02-09/mbp1.parquet", "b" * 64]]
    )
    assert descriptors["2026-02-09"].physical_schema_era_id == "mbp1_era_v1"
    assert descriptors["2026-02-09"].source_kind == "mbp1"
    # the MBP-1 feature/source contract is untouched by the legacy mapping
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
        DEEP_BOOK_EXEMPT_LITERALS,
        DEEP_BOOK_IDENTIFIER_REGEX,
    )

    assert LEGACY_VERIFIED_REPLAY_SOURCE in DEEP_BOOK_EXEMPT_LITERALS
    assert DEEP_BOOK_IDENTIFIER_REGEX.search(LEGACY_VERIFIED_REPLAY_SOURCE) is None
    assert DEEP_BOOK_IDENTIFIER_REGEX.search(descriptor.physical_filename)  # truthful, internal
