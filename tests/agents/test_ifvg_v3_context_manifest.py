"""Content-addressed v3 manifests reference v2 and persist context only."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from strategy_core.strategies.ifvg_smc.context_config import (
    ContextFeatureConfig,
    context_config_hash,
    feature_schema_hash,
)

from alpha_lab.agents.data_infra.ifvg.config import (
    ACCEPTED_V2_DATASET_ID,
    ACCEPTED_V2_MANIFEST_SHA256,
)
from alpha_lab.agents.data_infra.ifvg.manifest import (
    RepositoryState,
    V3DatasetIdentity,
    save_v3_dataset_immutable,
    v3_dataset_id_for,
)

from .ifvg_v3_fixtures import context_fixture


def _identity() -> V3DatasetIdentity:
    cfg = ContextFeatureConfig()
    repository = RepositoryState(
        name="Strategy-Core",
        path="C:/checkout",
        head="1" * 40,
        dirty_status_sha256="2" * 64,
        source_tree_hash="3" * 64,
    )
    return V3DatasetIdentity(
        repositories=(repository,),
        authoritative_source_blob="9b5f6f163ae060030c5695dbc0aede94e0ebebcd",
        accepted_v2_dataset_id=ACCEPTED_V2_DATASET_ID,
        accepted_v2_manifest_payload_sha256=ACCEPTED_V2_MANIFEST_SHA256,
        resolved_profile_hash="4" * 64,
        feature_set_version=cfg.feature_set_version,
        feature_formula_version=cfg.feature_formula_version,
        feature_schema_hash=feature_schema_hash(cfg),
        context_config_hash=context_config_hash(cfg),
        normalized_timeframes=cfg.normalized_timeframes,
        anchor_status_240m="experimental_q40_open",
        date_allowlist=("2026-01-01", "2026-01-02", "2026-01-04"),
        warmup_dates=("2026-01-01", "2026-01-02"),
        evidence_dates=("2026-01-04",),
        permitted_source_hashes=(("2026-01-01/trades.parquet", "5" * 64),),
    )


def _reports() -> dict:
    return {
        "validity_report": {"passed": True, "violations": {}},
        "coverage_report": {"capture_count": 9},
        "reconciliation_report": {"passed": True, "violations": []},
        "capacity_report": {"passed": True, "violations": {}},
        "identity_report": {"passed": True, "violations": {}},
        "performance_report": {"passed": True, "violations": {}},
        "data_access_audit": {
            "protected_file_opens": 0,
            "counts_by_source_class": {
                "prior_research": {"label_bar": 0},
            },
        },
    }


def test_v3_identity_is_path_independent_and_formula_sensitive() -> None:
    identity = _identity()
    moved = replace(identity.repositories[0], path="D:/different")
    assert v3_dataset_id_for(identity) == v3_dataset_id_for(
        replace(identity, repositories=(moved,))
    )
    assert v3_dataset_id_for(identity) != v3_dataset_id_for(
        replace(identity, feature_formula_version="ifvg_context_formula_v1")
    )
    assert v3_dataset_id_for(identity) != v3_dataset_id_for(
        replace(identity, accepted_v2_dataset_id="9" * 64)
    )


def test_v3_save_is_immutable_and_never_duplicates_core_tables(tmp_path: Path) -> None:
    _day, tables, _core, _emissions = context_fixture()
    identity = _identity()
    diagnostics: dict[str, object] = {}
    output = save_v3_dataset_immutable(
        base_dir=tmp_path,
        identity=identity,
        raw_config={"mode": "context_measurement_only"},
        effective_config={"feature_set_version": "ifvg_context_v1"},
        context_tables=tables,
        diagnostics_out=diagnostics,
        **_reports(),
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == v3_dataset_id_for(identity)
    assert manifest["accepted_v2_reference"] == {
        "dataset_id": ACCEPTED_V2_DATASET_ID,
        "manifest_payload_sha256": ACCEPTED_V2_MANIFEST_SHA256,
        "core_tables_duplicated": False,
    }
    assert not (output / "entry_candidate.parquet").exists()
    assert not (output / "executed_trade.parquet").exists()
    assert (output / "context_capture.parquet").exists()
    assert float(diagnostics["persistence_seconds"]) > 0
    assert "persistence_seconds" not in manifest
    with pytest.raises(FileExistsError, match="already exists"):
        save_v3_dataset_immutable(
            base_dir=tmp_path,
            identity=identity,
            raw_config={},
            effective_config={},
            context_tables=tables,
            **_reports(),
        )


def test_v3_save_rejects_performance_or_outcome_payloads(tmp_path: Path) -> None:
    _day, tables, _core, _emissions = context_fixture()
    reports = _reports()
    reports["coverage_report"] = {"profit_factor": 1.2}
    with pytest.raises(ValueError, match="forbidden key"):
        save_v3_dataset_immutable(
            base_dir=tmp_path,
            identity=_identity(),
            raw_config={},
            effective_config={},
            context_tables=tables,
            **reports,
        )

    reports = _reports()
    reports["data_access_audit"] = {"label": 0}
    with pytest.raises(ValueError, match="forbidden key"):
        save_v3_dataset_immutable(
            base_dir=tmp_path,
            identity=_identity(),
            raw_config={},
            effective_config={},
            context_tables=tables,
            **reports,
        )
