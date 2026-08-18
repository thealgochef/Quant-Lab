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
