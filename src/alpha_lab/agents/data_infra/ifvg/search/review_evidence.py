"""Exact stored search executions and their original input bars for review.

Search replays do not need a fabricated v3/context pair to be inspectable.
Every table and source file is checked against the original replay's hashes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..artifact_io import VerifiedIfvgArtifact, load_verified_v2_artifact
from ..config import IfvgCaptureConfig
from ..contracts import RecordTable
from ..dataset import table_content_hash
from ..development_access import DevelopmentReplayPolicy
from ..manifest import file_sha256
from .identities import CoreStrategyReplayIdentity, ReplayInputBundleEnvelope
from .store import load_sidecar_bytes, load_verified_envelope


@dataclass(frozen=True)
class SearchReviewEvidence:
    core: CoreStrategyReplayIdentity
    dataset: VerifiedIfvgArtifact
    inputs: ReplayInputBundleEnvelope

    @property
    def reference(self) -> dict[str, str]:
        return {
            "core_replay_id": self.core.core_replay_id,
            "v2_dataset_id": self.dataset.reference.artifact_id,
            "v2_manifest_hash": self.dataset.reference.manifest_payload_sha256,
            "replay_input_bundle_id": self.inputs.replay_input_bundle_id,
        }


def load_search_review_evidence(store_root: Path, core_replay_id: str) -> SearchReviewEvidence:
    root = Path(store_root)
    core = load_verified_envelope(root, "core_replays", core_replay_id, CoreStrategyReplayIdentity)
    ref = json.loads(
        load_sidecar_bytes(root, "core_replays", core_replay_id, "artifact_reference.json")
    )
    if ref["core_replay_id"] != core_replay_id:
        raise ValueError("search replay reference belongs to another child")
    dataset = load_verified_v2_artifact(root / "v2_datasets", ref["v2_dataset_artifact_id"])
    if dataset.reference.manifest_payload_sha256 != ref["manifest_payload_sha256"]:
        raise ValueError("search replay dataset manifest differs from its reference")
    if dataset.reference.profile_hash != core.payload.resolved_section_config_hash:
        raise ValueError("search replay profile differs from its dataset")
    trades = dataset.tables[RecordTable.EXECUTED_TRADE]
    if table_content_hash(RecordTable.EXECUTED_TRADE, trades) != ref["gross_trade_stream_hash"]:
        raise ValueError("search execution stream differs from its replay reference")
    inputs = load_verified_envelope(
        root, "replay_input_bundles", core.payload.replay_input_bundle_id, ReplayInputBundleEnvelope
    )
    return SearchReviewEvidence(core, dataset, inputs)


def load_search_day_bars(repo_root: Path, evidence: SearchReviewEvidence, day: str) -> pd.DataFrame:
    """Authorize the exact day before resolving a source path; verify bytes."""
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

    dates = tuple(
        sorted({ref.trading_day for ref in evidence.inputs.payload.ordered_day_artifacts})
    )
    policy = DevelopmentReplayPolicy(dates)
    refs = [
        ref
        for ref in evidence.inputs.payload.ordered_day_artifacts
        if ref.trading_day == day and ref.artifact_kind == "bars"
    ]
    if len(refs) != 1:
        raise ValueError("review requires exactly one original bars reference for this day")
    ref = refs[0]
    if Path(ref.artifact_id).name != ref.artifact_id:
        raise ValueError("invalid bars reference filename")
    section = IfvgSmcSection.model_validate(
        evidence.dataset.reports["effective_config.json"]["section"]
    )
    cfg = IfvgCaptureConfig(section=section, data_dir=Path(repo_root) / "data/databento")
    path = policy.resolve_source_path(day, lambda value: cfg.day_dir(value) / ref.artifact_id)
    policy.record_file_open(day)
    if file_sha256(path) != ref.content_sha256:
        raise ValueError("original replay bars changed since the study ran")
    bars = pd.read_parquet(path)
    policy.record_rows_read(day, rows=len(bars))
    policy.assert_zero_forbidden_access()
    return bars
