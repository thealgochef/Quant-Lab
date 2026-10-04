"""Task B store routing, original segment provenance and bounded preparation."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date, datetime, timedelta

import pyarrow.parquet as pq
import pytest
from strategy_core.types import Bar, BarKind

from alpha_lab.agents.data_infra.ifvg import prepared_store as stores
from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig, IfvgV3CaptureConfig
from alpha_lab.agents.data_infra.ifvg.data_access import require_fixed_exploration_allowlist
from alpha_lab.agents.data_infra.ifvg.day_artifacts import (
    DayArtifacts,
    DaySeeds,
    write_day_artifacts,
)


def _artifacts(day, seeds):
    ts = datetime.fromisoformat(f"{day}T15:00:00+00:00")
    bar = Bar(
        timeframe_ticks=60,
        trading_day=date.fromisoformat(day),
        bar_index=0,
        bar_id=f"{day}:1m",
        open_ts_utc=ts - timedelta(minutes=1),
        close_ts_utc=ts,
        open_ticks=100,
        high_ticks=101,
        low_ticks=99,
        close_ticks=100,
        volume=1,
        trade_count=1,
        is_complete=True,
        is_partial=False,
        kind=BarKind.TIME,
    )
    return DayArtifacts(day, [bar], {}, seeds, (101, 99), (101, 99), ())


def _make_segment(tmp_path, name, days, *, owned=None, source=None, high_low=(101, 99)):
    raw = tmp_path / "raw"
    cfg = IfvgCaptureConfig(data_dir=raw, artifact_cache_dir=tmp_path / name / "artifacts")
    author = tmp_path / "TASK_B.md"
    if not author.exists():
        author.write_text("Synthetic owner authorization fixture.", encoding="utf-8")
    definition = stores._definition(
        cfg,
        preparation_dates=days,
        source_dates=source or days,
        owned_dates=owned or days,
        artifact_cache_dir=cfg.artifact_cache_dir,
        catalog_path=tmp_path / name / "catalog.json",
        authorization_file=author,
        mode="preserved_existing",
    )
    policy = stores.PreparedStoreSourcePolicy(definition)
    seeds = DaySeeds(None, None, None, None)
    for day in days:
        artifacts = _artifacts(day, seeds)
        artifacts = replace(
            artifacts,
            bars=[replace(artifacts.bars[0], high_ticks=high_low[0], low_ticks=high_low[1])],
            day_hl=high_low,
            ny_hl=high_low,
        )
        write_day_artifacts(artifacts, cfg, access_policy=policy)
        seeds = DaySeeds(date.fromisoformat(day), high_low, date.fromisoformat(day), high_low)
    registration = stores.register_existing_prepared_store(
        cfg=cfg,
        store_root=tmp_path / name,
        preparation_dates=days,
        source_dates=source or days,
        owned_dates=owned or days,
        authorization_file=author,
        catalog_path=tmp_path / name / "catalog.json",
    )
    return cfg, registration


def test_artifact_routing_is_identity_neutral_and_raw_paths_stay_source_paths(tmp_path):
    original = IfvgCaptureConfig(data_dir=tmp_path / "raw")
    separate = replace(original, artifact_cache_dir=tmp_path / "prepared")
    assert separate.artifacts_tag() == original.artifacts_tag()
    assert separate.capture_tag() == original.capture_tag()
    assert separate.day_dir("2025-06-02") == original.day_dir("2025-06-02")
    assert separate.bars_path("2025-06-02").parent == tmp_path / "prepared/NQ/2025-06-02"
    assert separate.levels_path("2025-06-02").parent == separate.bars_path("2025-06-02").parent
    assert original.bars_path("2026-01-02").parent == original.day_dir("2026-01-02")


def test_two_stores_retain_distinct_writing_scopes_and_original_seed_lineage(tmp_path, monkeypatch):
    first_dates = (*stores.TASK_B_WARMUP_DATES, "2025-06-16", "2026-01-01")
    first_cfg, first = _make_segment(tmp_path, "prepared_2025", first_dates)
    second_cfg, second = _make_segment(
        tmp_path, "prepared_2026", ("2026-01-01", "2026-01-02"), owned=("2026-01-02",)
    )
    before = {
        p: p.read_bytes()
        for p in (
            second_cfg.bars_path("2026-01-01"),
            second_cfg.bars_path("2026-01-02"),
            second_cfg.levels_path("2026-01-02"),
        )
    }
    cfg = replace(
        first_cfg, artifact_cache_dir=None, prepared_store_registry_paths=(first.path, second.path)
    )
    dates = (*first_dates, "2026-01-02")
    policy = stores.PreparedStoreReplayPolicy(
        dates, registry_paths=cfg.prepared_store_registry_paths
    )
    require_fixed_exploration_allowlist(policy)
    assert cfg.bars_path("2026-01-01") == first_cfg.bars_path("2026-01-01")
    assert cfg.bars_path("2026-01-02") == second_cfg.bars_path("2026-01-02")
    assert policy.for_day("2026-01-02").allowlist == frozenset(second.source_dates)
    assert policy.for_day("2026-01-01").allowlist == frozenset(first.source_dates)
    # Original 2026 January 2 seed refers to the ORIGINAL January 1 artifact,
    # rather than the last preparation date in the newly registered 2025 store.
    assert stores.registered_day_seeds("2026-01-02", cfg).prev_day == date(2026, 1, 1)
    monkeypatch.setattr(stores, "build_day_artifacts", lambda *a, **k: pytest.fail("bar rebuild"))
    loaded = stores.load_registered_day_artifacts("2026-01-02", cfg, access_policy=policy)
    assert loaded.bars[0].trading_day == date(2026, 1, 2)
    provenance, stamps = policy.verify_registered_metadata(cfg)
    assert provenance == tuple(sorted(set(first.source_dates) | set(second.source_dates)))
    assert len(stamps) == len(dates)
    assert before == {p: p.read_bytes() for p in before}
    policy.assert_zero_forbidden_access()


def test_store_ownership_and_protected_dates_fail_before_artifact_paths(tmp_path):
    dates = (*stores.TASK_B_WARMUP_DATES, "2025-06-16")
    cfg, registration = _make_segment(tmp_path, "prepared", dates)
    conflicting = replace(cfg, prepared_store_registry_paths=(registration.path, registration.path))
    with pytest.raises(ValueError, match="exactly one registered store owner"):
        conflicting.bars_path("2025-06-16")
    missing = replace(cfg, prepared_store_registry_paths=(registration.path,))
    with pytest.raises(ValueError, match="exactly one registered store owner"):
        missing.bars_path("2025-07-08")
    with pytest.raises(PermissionError, match="permitted window"):
        missing.bars_path("2026-06-11")
    with pytest.raises(PermissionError, match="permitted window"):
        stores.PreparedStoreReplayPolicy(
            (*dates, "2026-06-11"), registry_paths=(registration.path,)
        )
    with pytest.raises(ValueError, match="exact June"):
        stores.PreparedStoreReplayPolicy(dates[1:], registry_paths=(registration.path,))


@pytest.mark.parametrize("lane", ["v2", "v3", "audit"])
def test_capture_builders_use_registered_segment_seeds_and_never_rebuild(
    tmp_path,
    monkeypatch,
    lane,
):
    from alpha_lab.agents.data_infra.ifvg import dataset
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config

    first_dates = (*stores.TASK_B_WARMUP_DATES, "2025-06-16", "2026-01-01")
    first_cfg, first = _make_segment(tmp_path, "prepared_2025", first_dates, high_low=(201, 98))
    second_cfg, second = _make_segment(
        tmp_path, "prepared_2026", ("2026-01-01", "2026-01-02"), owned=("2026-01-02",)
    )
    dates = (*first_dates, "2026-01-02")
    cfg = replace(
        first_cfg, artifact_cache_dir=None, prepared_store_registry_paths=(first.path, second.path)
    )
    assert stores.registered_day_seeds("2026-01-02", cfg).prev_full_hl == (101, 99)
    # A normal carried artifact seed would instead demand (201, 98), changing
    # the original 2026 cache. Only registered provenance admits its saved chain.
    before = {
        p: p.read_bytes()
        for p in (second_cfg.bars_path("2026-01-02"), second_cfg.levels_path("2026-01-02"))
    }
    monkeypatch.setattr(dataset, "build_day_artifacts", lambda *a, **k: pytest.fail("raw build"))
    monkeypatch.setattr(
        dataset, "write_day_artifacts", lambda *a, **k: pytest.fail("cache rewrite")
    )
    monkeypatch.setattr(
        dataset, "load_day_artifacts", lambda *a, **k: pytest.fail("ordinary loader")
    )
    resolved = resolve_profile_config({})
    def policy():
        return stores.PreparedStoreReplayPolicy(
            dates, registry_paths=cfg.prepared_store_registry_paths,
        )
    control = dataset.build_ifvg_v2_capture(
        dates,
        cfg,
        resolved,
        access_policy=policy(),
        cached_artifacts_only=True,
    )
    if lane == "v2":
        result = control
    elif lane == "v3":
        result = dataset.build_ifvg_v3_capture(
            dates,
            IfvgV3CaptureConfig(core=cfg),
            resolved,
            strategy_core_commit="a" * 40,
            strategy_core_source_tree_hash="b" * 64,
            access_policy=policy(),
            cached_artifacts_only=True,
            accepted_v2_tables=control.tables,
        )
        assert result.baseline_reconciliation["passed"]
    else:
        monkeypatch.setattr(dataset, "load_accepted_v2_tables", lambda *a, **k: control.tables)
        result = dataset.build_ifvg_fsm_audit_v1(
            dates,
            cfg,
            resolved,
            accepted_v2_exploration_dir=tmp_path / "synthetic_accepted",
            access_policy=policy(),
            cached_artifacts_only=True,
        )
        assert result.parity_report["passed"]
    assert result.cached_artifact_days == list(dates)
    assert result.rebuilt_days == []
    assert before == {p: p.read_bytes() for p in before}


def test_registered_bytes_and_footer_lineage_cannot_be_replaced(tmp_path):
    dates = (*stores.TASK_B_WARMUP_DATES, "2025-06-16")
    cfg, registration = _make_segment(tmp_path, "prepared", dates)
    cfg = replace(cfg, prepared_store_registry_paths=(registration.path,))
    policy = stores.PreparedStoreReplayPolicy(dates, registry_paths=(registration.path,))
    path = cfg.bars_path("2025-06-16")
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    stamp = json.loads(metadata[b"ifvg_artifacts_meta"])
    stamp["seeds"]["prev_full_hl"] = [999, 1]
    metadata[b"ifvg_artifacts_meta"] = json.dumps(stamp).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="bytes changed"):
        stores.load_registered_day_artifacts("2025-06-16", cfg, access_policy=policy)
    with pytest.raises(ValueError, match="different cache stamps"):
        policy.verify_registered_metadata(cfg)


def _coverage(tmp_path):
    coverage = tmp_path / "coverage.csv"
    start, end = date(2025, 6, 2), date(2026, 1, 1)
    rows = [
        {
            "date": (start + timedelta(days=n)).isoformat(),
            "symbol": "NQ",
            "raw_files": ""
            if (start + timedelta(days=n)).isoformat() in stores.TASK_B_MISSING_SOURCE_DATES
            else "mbp10.parquet",
        }
        for n in range((end - start).days + 1)
        if (start + timedelta(days=n)).weekday() < 5
    ]
    with coverage.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("date", "symbol", "raw_files"))
        writer.writeheader()
        writer.writerows(rows)
    return coverage, rows


def test_unattended_preparation_is_separate_exact_missing_and_resumable(tmp_path, monkeypatch):
    coverage, rows = _coverage(tmp_path)
    author = tmp_path / "TASK_B.md"
    author.write_text("Synthetic Task B authorization.")
    cfg = IfvgCaptureConfig(data_dir=tmp_path / "raw")
    preserved = cfg.bars_path("2026-01-02")
    preserved.parent.mkdir(parents=True)
    preserved.write_bytes(b"preserved original 2026 cache bytes")
    for row in rows:
        if row["raw_files"]:
            raw = cfg.day_dir(row["date"]) / "mbp10.parquet"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"synthetic raw source fingerprint")
    sunday = cfg.day_dir("2025-06-08") / "mbp10.parquet"
    sunday.parent.mkdir(parents=True)
    sunday.write_bytes(b"synthetic prior Sunday physical partition")
    built = []

    def build(day, actual_cfg, seeds, *, access_policy):
        assert date.fromisoformat(day).weekday() < 5
        assert day not in stores.TASK_B_MISSING_SOURCE_DATES
        built.append(day)
        catalog = actual_cfg.preparation_catalog_paths[0]
        entries = json.loads(catalog.read_text()) if catalog.exists() else {"days": {}}
        entries["days"][day] = {"selected_instrument_id": 123, "raw_symbol": "NQM5"}
        catalog.parent.mkdir(parents=True, exist_ok=True)
        catalog.write_text(json.dumps(entries))
        return _artifacts(day, seeds)

    monkeypatch.setattr(stores, "build_day_artifacts", build)
    registration = stores.prepare_task_b_store(
        coverage_csv=coverage,
        cfg=cfg,
        store_root=tmp_path / "separate_2025",
        authorization_file=author,
    )
    assert len(built) == 151
    assert registration.owned_dates[:10] == stores.TASK_B_WARMUP_DATES
    assert registration.owned_dates[10] == "2025-06-16"
    reports = json.loads((tmp_path / "separate_2025/preparation_days.json").read_text())
    assert tuple(day for day, row in reports.items() if row["status"] == "missing") == (
        stores.TASK_B_MISSING_SOURCE_DATES
    )
    assert all(
        reports[day]["missing_reason"] == "no_source_partition"
        for day in stores.TASK_B_MISSING_SOURCE_DATES
    )
    assert all(row["seconds"] >= 0 for row in reports.values())
    assert all(
        row["selected_instrument_id"] == 123
        for row in reports.values()
        if row["status"] == "prepared"
    )
    assert [item["physical_date"] for item in reports["2025-06-09"]["source_files"]] == [
        "2025-06-08",
        "2025-06-09",
    ]
    assert "2025-06-08" not in built
    assert preserved.read_bytes() == b"preserved original 2026 cache bytes"
    assert not cfg.bars_path("2025-06-02").exists()
    monkeypatch.setattr(
        stores, "build_day_artifacts", lambda *a, **k: pytest.fail("resume rebuild")
    )
    resumed = stores.prepare_task_b_store(
        coverage_csv=coverage,
        cfg=cfg,
        store_root=tmp_path / "separate_2025",
        authorization_file=author,
    )
    assert resumed.registration_id == registration.registration_id
    run = json.loads((tmp_path / "separate_2025/preparation_run.json").read_text())
    assert run["prepared_days"] == 151 and run["missing_days"] == 3


def test_separate_preparation_refuses_raw_root_or_changed_inventory(tmp_path):
    coverage, rows = _coverage(tmp_path)
    author = tmp_path / "TASK_B.md"
    author.write_text("Synthetic Task B authorization.")
    cfg = IfvgCaptureConfig(data_dir=tmp_path / "raw")
    with pytest.raises(PermissionError, match="separate from"):
        stores.prepare_task_b_store(
            coverage_csv=coverage, cfg=cfg, store_root=tmp_path / "raw", authorization_file=author
        )
    rows[0]["raw_files"] = ""
    with coverage.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("date", "symbol", "raw_files"))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="three recorded missing"):
        stores.task_b_preparation_coverage(coverage)
