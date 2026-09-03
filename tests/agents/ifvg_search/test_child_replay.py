"""Child replay worker + R0→R1 entry items, proven synthetically.

R0 item (a): capture-driver mid-chain start from a cached, profile-matching
seed — resolved via the documented ``start_after_artifact`` QL-only fallback;
the snapshot-restart chain is emission-identical to the continuous chain.
R0 item (b): native-ID replay determinism — the same synthetic day chain
replayed twice yields byte-identical record ids and tables.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest
from strategy_core.strategies.ifvg_smc.state import seed_hash

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable, partition_capture_tables
from alpha_lab.agents.data_infra.ifvg.data_access import ExplorationDataPolicy
from alpha_lab.agents.data_infra.ifvg.dataset import (
    ChainStart,
    build_ifvg_v2_capture,
    table_content_hash,
)
from alpha_lab.agents.data_infra.ifvg.day_artifacts import DaySeeds, write_day_artifacts
from alpha_lab.agents.data_infra.ifvg.development_access import VerificationReplayPolicy
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
    ArtifactProvenanceReadAdapter,
    DaySeedsRecord,
    SeedSnapshotError,
    SeedSnapshotPayload,
    build_neutrality_report,
    load_seed_snapshot,
    save_seed_snapshot,
)
from tests.agents.ifvg_search.conftest import (
    SYNTHETIC_DAYS,
    build_artifact_chain,
    run_synthetic_chain,
)


def _tables_of(chain) -> dict:
    frames = [r.rows for r in chain if len(r.rows)]
    trace = pd.concat(frames, ignore_index=True, sort=False)
    trace["trace_ordinal"] = range(len(trace))
    return partition_capture_tables(trace)


# ── R0 item (b): native-ID replay determinism ────────────────────────────────


def test_native_ids_are_deterministic_across_replays(doc_default_cfg) -> None:
    chain_a = run_synthetic_chain(doc_default_cfg, build_artifact_chain(doc_default_cfg))
    chain_b = run_synthetic_chain(doc_default_cfg, build_artifact_chain(doc_default_cfg))
    totals = {}
    for result in chain_a:
        for key, value in result.funnel.items():
            totals[key] = totals.get(key, 0) + value
    assert totals.get("setups_born", 0) >= 1  # the walk exercises the funnel
    tables_a = _tables_of(chain_a)
    tables_b = _tables_of(chain_b)
    for table in RecordTable:
        assert table_content_hash(table, tables_a[table]) == table_content_hash(
            table, tables_b[table]
        ), f"native determinism failed for {table.value}"
    lifecycle = tables_a[RecordTable.SETUP_LIFECYCLE]
    assert set(lifecycle["envelope_setup_id"]) == set(
        tables_b[RecordTable.SETUP_LIFECYCLE]["envelope_setup_id"]
    )


# ── R0 item (a): mid-chain start from a profile-matching seed ────────────────


def test_snapshot_restart_is_emission_identical_to_continuous_chain(
    doc_default_cfg, synthetic_artifacts, synthetic_chain
) -> None:
    continuous_last = synthetic_chain[-1]
    restart = run_synthetic_chain(
        doc_default_cfg,
        synthetic_artifacts[2:],
        start_seed=synthetic_chain[1].end_seed,
    )[0]
    assert restart.entering_seed_hash == seed_hash(synthetic_chain[1].end_seed)
    pd.testing.assert_frame_equal(
        restart.rows.reset_index(drop=True),
        continuous_last.rows.reset_index(drop=True),
    )
    assert restart.funnel == continuous_last.funnel
    assert seed_hash(restart.end_seed) == seed_hash(continuous_last.end_seed)


# ── the driver-level fallback on disk (start_after_artifact) ─────────────────


@pytest.fixture()
def disk_chain(tmp_path, doc_default_cfg, synthetic_artifacts):
    cfg = replace(doc_default_cfg, data_dir=tmp_path / "databento")
    policy = ExplorationDataPolicy()
    for artifacts in synthetic_artifacts:
        write_day_artifacts(artifacts, cfg, access_policy=policy)
    return cfg


def test_build_ifvg_v2_capture_supports_mid_chain_start(disk_chain, synthetic_artifacts) -> None:
    resolved = resolve_profile_config({})
    full = build_ifvg_v2_capture(
        list(SYNTHETIC_DAYS),
        disk_chain,
        resolved,
        access_policy=ExplorationDataPolicy(),
        cached_artifacts_only=True,
    )
    assert full.cached_artifact_days == list(SYNTHETIC_DAYS)
    # snapshot state entering day 3 = end of day 2 — the prefix must NOT mark
    # its final day dataset-exhausted (the chain continues into the restart)
    prefix = build_ifvg_v2_capture(
        list(SYNTHETIC_DAYS[:2]),
        disk_chain,
        resolved,
        access_policy=ExplorationDataPolicy(),
        cached_artifacts_only=True,
        final_day_exhausts_dataset=False,
    )
    day2 = synthetic_artifacts[1]
    restart = build_ifvg_v2_capture(
        [SYNTHETIC_DAYS[2]],
        disk_chain,
        resolved,
        access_policy=ExplorationDataPolicy(),
        cached_artifacts_only=True,
        start_after_artifact=ChainStart(
            seed=prefix.end_seed,
            day_seeds=DaySeeds(
                prev_day=date.fromisoformat(day2.date_str),
                prev_full_hl=day2.day_hl,
                prev_ny_day=None,
                prev_ny_hl=None,
            ),
        ),
    )
    # the restarted tail reproduces the full chain's day-3 funnel exactly
    assert restart.day_funnels[SYNTHETIC_DAYS[2]] == full.day_funnels[SYNTHETIC_DAYS[2]]
    assert seed_hash(restart.end_seed) == seed_hash(full.end_seed)


def test_profile_seed_mismatch_is_refused_before_any_source_read(
    disk_chain, synthetic_chain
) -> None:
    child = resolve_profile_config(
        {
            "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "section_overrides": {"parent_retest_timeout_1m_bars": 480},
        }
    )
    child_cfg = replace(disk_chain, section=child.section)
    policy = ExplorationDataPolicy()
    with pytest.raises(PermissionError, match="profile-bound|profile_hash"):
        build_ifvg_v2_capture(
            [SYNTHETIC_DAYS[2]],
            child_cfg,
            child,
            access_policy=policy,
            cached_artifacts_only=True,
            start_after_artifact=ChainStart(
                seed=synthetic_chain[1].end_seed,  # baseline-profile seed
                day_seeds=DaySeeds(None, None, None, None),
            ),
        )
    assert policy.audit.path_constructions == 0  # refused BEFORE any path


# ── dual-drive audit neutrality (§3.3) ───────────────────────────────────────


def test_dual_drive_neutrality_on_synthetic_chain(doc_default_cfg, synthetic_artifacts) -> None:
    disabled_chain = run_synthetic_chain(doc_default_cfg, synthetic_artifacts)
    enabled_chain = run_synthetic_chain(
        doc_default_cfg, synthetic_artifacts, audit_capture_mode="fsm_audit_v1"
    )

    class _Result:  # duck-typed V2CaptureResult surface for the report builder
        def __init__(self, chain, audit):
            self.tables = _tables_of(chain)
            self.audit_frames = (
                {r.date_str: r.audit_rows for r in chain if r.audit_rows is not None}
                if audit
                else None
            )

    report = build_neutrality_report(
        core_replay_id="3" * 64,
        disabled=_Result(disabled_chain, audit=False),
        enabled=_Result(enabled_chain, audit=True),
    )
    assert report.mechanism == "dual_drive_ab_v1"
    assert report.tables_equal is True
    assert report.audit_stamp_referential_integrity is True
    assert report.passed is True
    assert dict(report.audit_disabled_core_table_hashes) == dict(
        report.audit_enabled_core_table_hashes
    )
    # any audit rows produced must reference real core setups
    assert enabled_chain[1].audit_rows is not None


# ── seed snapshots (profile-bound, verified) ─────────────────────────────────


def _snapshot_payload(seed) -> SeedSnapshotPayload:
    from strategy_core.strategies.ifvg_smc.state import IFVG_SEED_SCHEMA_VERSION

    return SeedSnapshotPayload(
        profile_name="ifvg_v2_doc_default_fresh_static_1r",
        resolved_section_config_hash=seed.profile_hash,
        seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
        seed_hash=seed_hash(seed),
        snapshot_through_day=SYNTHETIC_DAYS[1],
        first_replay_day=SYNTHETIC_DAYS[2],
        entering_day_seeds=DaySeedsRecord(
            prev_day=SYNTHETIC_DAYS[1],
            prev_full_hl=(20400, 19900),
            prev_ny_day=None,
            prev_ny_hl=None,
        ),
        chain_policy_id="development_explicit_dates_before_path_v2",
        chain_date_count=2,
        strategy_core_commit="c" * 40,
    )


def test_seed_snapshot_roundtrip_and_profile_binding(tmp_path, synthetic_chain) -> None:
    seed = synthetic_chain[1].end_seed
    envelope = save_seed_snapshot(tmp_path, _snapshot_payload(seed), seed)
    loaded, chain_start = load_seed_snapshot(
        tmp_path,
        envelope.seed_snapshot_id,
        expected_section_config_hash=seed.profile_hash,
    )
    assert loaded.seed_snapshot_id == envelope.seed_snapshot_id
    assert seed_hash(chain_start.seed) == seed_hash(seed)
    assert chain_start.day_seeds.prev_full_hl == (20400, 19900)
    with pytest.raises(SeedSnapshotError, match="profile-bound"):
        load_seed_snapshot(
            tmp_path,
            envelope.seed_snapshot_id,
            expected_section_config_hash="9" * 64,
        )


def test_seed_snapshot_save_refuses_mismatched_seed(tmp_path, synthetic_chain) -> None:
    seed = synthetic_chain[1].end_seed
    payload = _snapshot_payload(seed).model_copy(update={"seed_hash": "9" * 64})
    with pytest.raises(SeedSnapshotError, match="seed does not hash"):
        save_seed_snapshot(tmp_path, payload, seed)


# ── provenance read adapter (verification reads of dev-chain caches) ─────────


def test_provenance_adapter_reads_dev_caches_under_strict_authorization(
    disk_chain, synthetic_artifacts
) -> None:
    from alpha_lab.agents.data_infra.ifvg.data_access import EXPLORATION_DATE_ALLOWLIST
    from alpha_lab.agents.data_infra.ifvg.day_artifacts import load_day_artifacts

    inner = VerificationReplayPolicy((SYNTHETIC_DAYS[2],))
    adapter = ArtifactProvenanceReadAdapter(
        inner, artifact_provenance_dates=tuple(sorted(EXPLORATION_DATE_ALLOWLIST))
    )
    day3 = synthetic_artifacts[2]
    loaded = load_day_artifacts(
        day3.date_str,
        disk_chain,
        expected_seeds=day3.seeds,
        access_policy=adapter,
    )
    assert loaded is not None
    assert loaded.day_hl == day3.day_hl
    # off-allowlist dates are refused before path construction by the INNER policy
    with pytest.raises(PermissionError):
        load_day_artifacts(
            SYNTHETIC_DAYS[0],
            disk_chain,
            expected_seeds=None,
            access_policy=adapter,
        )
    assert inner.audit.denied_dates.get(SYNTHETIC_DAYS[0]) == 1
    # provenance dates past the cutoff are unrepresentable
    with pytest.raises(PermissionError, match="development cutoff"):
        ArtifactProvenanceReadAdapter(inner, artifact_provenance_dates=("2026-06-12",))


def test_executor_seed_source_returns_the_loaded_artifacts_id(
    tmp_path, synthetic_chain
) -> None:
    """R5-FIX (gate finding 7): the pipeline's expected seed comes from the
    VERIFIED artifact load — the source refuses when no such artifact
    exists and returns the loaded envelope's content-derived id when it
    does."""

    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.search.executors import (
        loaded_seed_snapshot_id_source,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError

    seed = synthetic_chain[1].end_seed
    envelope = save_seed_snapshot(tmp_path, _snapshot_payload(seed), seed)
    run = SimpleNamespace(
        payload=SimpleNamespace(seed_snapshot_id=envelope.seed_snapshot_id)
    )
    resolved = SimpleNamespace(section_config_hash=seed.profile_hash)
    source = loaded_seed_snapshot_id_source(tmp_path, run, resolved)
    assert source() == envelope.seed_snapshot_id

    missing = SimpleNamespace(payload=SimpleNamespace(seed_snapshot_id="9" * 64))
    with pytest.raises(SearchStoreError, match="missing search-store entry"):
        loaded_seed_snapshot_id_source(tmp_path, missing, resolved)()

    wrong_profile = SimpleNamespace(section_config_hash="8" * 64)
    with pytest.raises(SeedSnapshotError, match="profile-bound"):
        loaded_seed_snapshot_id_source(tmp_path, run, wrong_profile)()


def test_real_scope_seed_requires_the_loaded_artifact_source() -> None:
    """R5-FIX (gate finding 7): the real verification scope REFUSES a
    caller-provided expected seed id without the loaded-artifact source;
    a wired-but-disagreeing caller expectation also refuses."""

    from dataclasses import replace as dc_replace

    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
        PipelineWiring,
        _loaded_seed_snapshot_id_for_real_scope,
    )

    bare = PipelineWiring(
        identity_resolver=lambda spec: None, child_runner=lambda **kwargs: None
    )
    with pytest.raises(PermissionError, match="not evidence"):
        _loaded_seed_snapshot_id_for_real_scope(bare)
    # a caller string ALONE is still refused — it is a cross-check only
    caller_only = dc_replace(bare, expected_seed_snapshot_id="a" * 64)
    with pytest.raises(PermissionError, match="not evidence"):
        _loaded_seed_snapshot_id_for_real_scope(caller_only)
    wired = dc_replace(bare, loaded_seed_snapshot_id_source=lambda: "a" * 64)
    assert _loaded_seed_snapshot_id_for_real_scope(wired) == "a" * 64
    agrees = dc_replace(wired, expected_seed_snapshot_id="a" * 64)
    assert _loaded_seed_snapshot_id_for_real_scope(agrees) == "a" * 64
    disagrees = dc_replace(wired, expected_seed_snapshot_id="b" * 64)
    with pytest.raises(PermissionError, match="disagrees"):
        _loaded_seed_snapshot_id_for_real_scope(disagrees)


# ── HARDENING-BACKEND-FIX §8 — canonical UTC seed datetimes at the ONE seam ──


def _zones() -> dict:
    from datetime import UTC, timedelta, timezone
    from zoneinfo import ZoneInfo

    import pytz

    return {
        "stdlib_utc": UTC,
        "pytz_utc": pytz.UTC,
        "pytz_new_york": pytz.timezone("America/New_York"),
        "zoneinfo_utc": ZoneInfo("UTC"),
        "zoneinfo_tokyo": ZoneInfo("Asia/Tokyo"),
        "fixed_minus_5": timezone(timedelta(hours=-5)),
        "fixed_plus_9_30": timezone(timedelta(hours=9, minutes=30)),
    }


@pytest.mark.parametrize("zone_label", sorted(_zones()))
def test_direct_seed_save_canonicalizes_every_aware_timezone_to_utc(
    tmp_path, synthetic_chain, zone_label
) -> None:
    """HB-FIX-10: the same instants under pytz / zoneinfo / fixed offsets
    produce the identical seed identity, the identical snapshot id and
    byte-identical canonical sidecar bytes through a DIRECT
    ``save_seed_snapshot``; the caller's object is never mutated; a seed
    already in stdlib UTC keeps its existing (golden) hash."""

    from datetime import UTC

    from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
        canonical_seed_hash,
        canonicalize_seed_datetimes,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import load_sidecar_bytes
    from tests.agents.ifvg_search.conftest import aware_seed_datetimes, rezone_seed_datetimes

    zone = _zones()[zone_label]
    seed = synthetic_chain[1].end_seed
    stamps = aware_seed_datetimes(seed)
    assert stamps, "the synthetic seed carries aware datetimes"
    assert all(stamp.tzinfo is UTC for stamp in stamps)
    golden = seed_hash(seed)
    assert canonical_seed_hash(seed) == golden  # an already-canonical seed is unchanged
    rezoned = rezone_seed_datetimes(seed, zone)
    rezoned_stamps = aware_seed_datetimes(rezoned)
    assert [s.timestamp() for s in rezoned_stamps] == [s.timestamp() for s in stamps]
    canonical = canonicalize_seed_datetimes(rezoned)
    assert seed_hash(canonical) == golden
    assert canonical_seed_hash(rezoned) == golden
    assert all(stamp.tzinfo is UTC for stamp in aware_seed_datetimes(canonical))
    # the instants are preserved exactly
    assert [s.timestamp() for s in aware_seed_datetimes(canonical)] == [
        s.timestamp() for s in stamps
    ]
    # the caller's object was not mutated by the canonicalization
    assert [s.tzinfo for s in aware_seed_datetimes(rezoned)] == [s.tzinfo for s in rezoned_stamps]
    # the DIRECT save accepts the rezoned seed against the canonical payload hash,
    # mints the same snapshot id and persists byte-identical canonical bytes
    envelope = save_seed_snapshot(tmp_path / "rezoned", _snapshot_payload(seed), rezoned)
    reference = save_seed_snapshot(tmp_path / "reference", _snapshot_payload(seed), seed)
    assert envelope.seed_snapshot_id == reference.seed_snapshot_id
    assert load_sidecar_bytes(
        tmp_path / "rezoned", "seed_snapshots", envelope.seed_snapshot_id, "seed.pickle"
    ) == load_sidecar_bytes(
        tmp_path / "reference", "seed_snapshots", reference.seed_snapshot_id, "seed.pickle"
    )
    _loaded, chain_start = load_seed_snapshot(
        tmp_path / "rezoned",
        envelope.seed_snapshot_id,
        expected_section_config_hash=seed.profile_hash,
    )
    assert seed_hash(chain_start.seed) == golden
    assert all(stamp.tzinfo is UTC for stamp in aware_seed_datetimes(chain_start.seed))


def test_seed_canonicalization_covers_nested_containers_and_keeps_naive_datetimes() -> None:
    from dataclasses import dataclass
    from datetime import UTC, datetime, timedelta, timezone
    from typing import NamedTuple

    import pytz
    from pydantic import BaseModel

    from alpha_lab.agents.data_infra.ifvg.search.child_replay import canonicalize_seed_datetimes

    instant = datetime(2026, 1, 13, 14, 30, tzinfo=UTC)
    minus_five = instant.astimezone(timezone(timedelta(hours=-5)))
    tokyo = instant.astimezone(pytz.timezone("Asia/Tokyo"))
    naive = datetime(2026, 1, 13, 14, 30)

    class Pair(NamedTuple):
        first: datetime
        second: datetime

    class Model(BaseModel):
        when: datetime
        items: tuple[datetime, ...]

    @dataclass(frozen=True)
    class Node:
        when: datetime
        naive: datetime
        children: tuple
        listed: list
        mapped: dict
        pair: Pair
        model: Model
        day: date

    node = Node(
        when=minus_five,
        naive=naive,
        children=(tokyo, (minus_five,)),
        listed=[tokyo, [minus_five]],
        mapped={"a": tokyo, "b": {"c": minus_five}},
        pair=Pair(minus_five, tokyo),
        model=Model(when=tokyo, items=(minus_five,)),
        day=date(2026, 1, 13),
    )
    canonical = canonicalize_seed_datetimes(node)
    # every aware datetime is the same instant under stdlib UTC …
    for stamp in (
        canonical.when,
        canonical.children[0],
        canonical.children[1][0],
        canonical.listed[0],
        canonical.listed[1][0],
        canonical.mapped["a"],
        canonical.mapped["b"]["c"],
        canonical.pair.first,
        canonical.pair.second,
        canonical.model.when,
        canonical.model.items[0],
    ):
        assert stamp.tzinfo is UTC and stamp == instant and stamp.isoformat() == instant.isoformat()
    assert isinstance(canonical.pair, Pair) and isinstance(canonical.model, Model)
    # … the naive datetime and the date pass through untouched (never localized)
    assert canonical.naive is naive and canonical.naive.tzinfo is None
    assert canonical.day == date(2026, 1, 13)
    # the caller's graph is untouched
    assert node.when.utcoffset() == timedelta(hours=-5)
    assert node.mapped["b"]["c"].utcoffset() == timedelta(hours=-5)
    assert node.model.when.utcoffset() == timedelta(hours=9)
    # a stdlib fixed offset is NOT UTC (the former ``isinstance(tzinfo, timezone)`` shortcut)
    assert canonicalize_seed_datetimes(minus_five).tzinfo is UTC
    assert canonicalize_seed_datetimes(minus_five).isoformat() == instant.isoformat()
    # an already-canonical stdlib UTC datetime keeps its value (a fresh, equal object)
    assert canonicalize_seed_datetimes(instant) == instant
    assert canonicalize_seed_datetimes(instant).isoformat() == instant.isoformat()
    # pytz UTC → stdlib UTC (the same isoformat)
    pytz_instant = instant.astimezone(pytz.UTC)
    assert canonicalize_seed_datetimes(pytz_instant).tzinfo is UTC
    assert canonicalize_seed_datetimes(pytz_instant).isoformat() == instant.isoformat()
    # review RA-03: datetime dict KEYS, set / frozenset members, pydantic
    # ``extra="allow"`` values and ``init=False`` dataclass fields are seed
    # content too — canonicalized, never dropped
    from dataclasses import field as dc_field

    import numpy as np
    from pydantic import ConfigDict

    class Extra(BaseModel):
        model_config = ConfigDict(extra="allow")
        when: datetime

    @dataclass
    class Derived:
        when: datetime
        derived: datetime = dc_field(init=False)

        def __post_init__(self) -> None:
            self.derived = self.when + timedelta(hours=1)

    keyed = canonicalize_seed_datetimes({minus_five: tokyo})
    assert list(keyed) == [instant] and list(keyed)[0].tzinfo is UTC
    assert keyed[instant].tzinfo is UTC and keyed[instant] == instant
    members = canonicalize_seed_datetimes(frozenset({tokyo}))
    assert isinstance(members, frozenset) and next(iter(members)).tzinfo is UTC
    plain = canonicalize_seed_datetimes({minus_five})
    assert isinstance(plain, set) and next(iter(plain)).tzinfo is UTC
    extra = canonicalize_seed_datetimes(Extra(when=tokyo, later=minus_five))
    assert extra.when.tzinfo is UTC and extra.model_extra["later"].tzinfo is UTC
    assert extra.model_extra["later"] == instant
    derived = Derived(when=minus_five)
    canonical_derived = canonicalize_seed_datetimes(derived)
    assert canonical_derived.when.tzinfo is UTC and canonical_derived.when == instant
    assert canonical_derived.derived == derived.derived  # the init=False value survives
    assert canonical_derived.derived.tzinfo is UTC
    assert derived.when.utcoffset() == timedelta(hours=-5)  # the caller's object untouched
    # a pandas Timestamp is rebuilt as the stdlib datetime it represents …
    stamp = canonicalize_seed_datetimes(pd.Timestamp(tokyo))
    assert type(stamp) is datetime and stamp == instant and stamp.tzinfo is UTC
    # … while sub-microsecond precision, numpy datetime64 and NaT are refused, never dropped
    with pytest.raises(SeedSnapshotError):
        canonicalize_seed_datetimes(pd.Timestamp(instant) + pd.Timedelta(1, "ns"))
    with pytest.raises(SeedSnapshotError):
        canonicalize_seed_datetimes(np.datetime64("2026-01-13T14:30"))
    with pytest.raises(SeedSnapshotError):
        canonicalize_seed_datetimes(pd.NaT)
