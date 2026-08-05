"""Setup-verifier provider tests over real immutable artifacts.

These are the §12.3/§12.4 golden gates: every one of the 215 setups must be
selectable and yield contract-ordered evidence; stage gating must hide future
evidence point-in-time; bundle identity isolation must refuse mismatched
artifacts. Skipped cleanly when the local store lacks the artifacts (CI
without data).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ID = "143b510f8a73896072f44e08f331ef5156e85eb8e5124d25bdf441c4fb6b2ac7"
V3_ID = "09ef35d0c37ca4c2b688ec88c57b2881f02d52f11cc99c4daf0f74b5c46245c3"
_FSM_STORE = REPO_ROOT / "data" / "ifvg_datasets" / "fsm_audit" / "v1"
_CHART_STORE = REPO_ROOT / "data" / "ifvg_datasets" / "replay_chart" / "v1"


def _discover_bundle():
    """Newest on-disk fsm-audit artifact + a v2-kind replay chart pinning it."""
    if not _FSM_STORE.is_dir():
        return None
    audits = sorted(
        (p for p in _FSM_STORE.iterdir() if (p / "exploration" / "manifest.json").is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    for audit_dir in reversed(audits):
        audit_id = audit_dir.name
        if not _CHART_STORE.is_dir():
            continue
        for chart_dir in _CHART_STORE.iterdir():
            manifest_path = chart_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("artifact_kind") != "ifvg_replay_chart_v2":
                continue
            config = manifest.get("effective_config", {})
            if config.get("fsm_audit_artifact_id") != audit_id:
                continue
            if manifest.get("source_pair", {}).get("v2_dataset_id") != V2_ID:
                continue
            return audit_id, chart_dir.name, manifest
    return None


_BUNDLE = _discover_bundle()

pytestmark = pytest.mark.skipif(
    _BUNDLE is None,
    reason="no local ifvg_fsm_audit_v1 + replay-chart v2 artifacts",
)


@pytest.fixture(scope="module")
def ctx():
    from alpha_lab.agents.data_infra.ifvg.artifact_io import load_verified_ifvg_pair
    from alpha_lab.agents.data_infra.ifvg.config import V2_DATASET_DIR, V3_DATASET_DIR
    from alpha_lab.agents.data_infra.ifvg.fsm_audit_io import (
        load_verified_fsm_audit_artifact,
    )
    from alpha_lab.agents.data_infra.ifvg.replay_chart_store import VerifierBundleRef
    from alpha_lab.agents.data_infra.ifvg.setup_verifier_provider import (
        open_setup_replay_context,
    )

    audit_id, chart_id, chart_manifest = _BUNDLE
    pair = load_verified_ifvg_pair(
        v2_root=REPO_ROOT / V2_DATASET_DIR,
        v2_artifact_id=V2_ID,
        v3_root=REPO_ROOT / V3_DATASET_DIR,
        v3_artifact_id=V3_ID,
    )
    audit = load_verified_fsm_audit_artifact(_FSM_STORE, audit_id)
    bundle = VerifierBundleRef(
        profile_name="ifvg_v2_doc_default_fresh_static_1r",
        v2_dataset_id=V2_ID,
        v2_manifest_hash=pair.reference.v2.manifest_payload_sha256,
        v3_dataset_id=V3_ID,
        v3_manifest_hash=pair.reference.v3.manifest_payload_sha256,
        fsm_audit_artifact_id=audit_id,
        fsm_audit_manifest_hash=audit.manifest_payload_sha256,
        replay_chart_artifact_id=chart_id,
        replay_chart_manifest_hash=str(chart_manifest["manifest_payload_sha256"]),
    )
    return open_setup_replay_context(REPO_ROOT, bundle)


def test_all_215_setups_selectable_with_evidence(ctx) -> None:
    from alpha_lab.agents.data_infra.ifvg.setup_verifier_provider import (
        list_setups,
        setup_evidence,
    )

    setups = list_setups(ctx)
    assert len(setups) == 215
    assert setups["setup_id"].is_unique
    assert int(setups["candidate_less"].sum()) == 170
    for setup_id in setups["setup_id"].astype(str):
        evidence = setup_evidence(ctx, setup_id)
        assert not evidence.events.empty, setup_id
        assert evidence.terminal is not None, setup_id


def test_known_death_cohorts_match_the_audit(ctx) -> None:
    from alpha_lab.agents.data_infra.ifvg.setup_verifier_provider import list_setups

    setups = list_setups(ctx)
    reasons = setups["terminal_reason"].value_counts().to_dict()
    assert reasons["invalidated_parent_filled"] == 83
    assert reasons["slot_freed"] == 33
    phases = setups["phase_at_death"].value_counts().to_dict()
    assert phases["S4"] == 13 and phases["S3"] == 15
    # D-1 present-but-empty surfaces: zero conflicts, zero structural closes.
    assert int(setups["conflict_flag"].sum()) == 0
    assert int(setups["structural_suppression_flag"].sum()) == 0


def test_stage_gate_is_point_in_time(ctx) -> None:
    from alpha_lab.agents.data_infra.ifvg.setup_verifier_provider import (
        list_setups,
        setup_evidence,
    )

    setups = list_setups(ctx)
    target = str(
        setups.loc[
            setups["candidate_less"]
            & (setups["terminal_reason"] == "invalidated_htf_filled"),
            "setup_id",
        ].iloc[0]
    )
    full = setup_evidence(ctx, target)
    gated = setup_evidence(ctx, target, stage="activation")
    assert gated.gating_report["hidden_events"] > 0
    assert not (gated.events["stage"] == "terminal").any()
    assert len(gated.events) + gated.gating_report["hidden_events"] == len(full.events)
    # no fill event that killed the setup leaks through the activation gate.
    killing = full.events.loc[
        (full.events["event_kind"] == "fvg_fill_event")
        & (full.events["fill_kind"] == "filled")
    ]
    if len(killing):
        assert not gated.events["event_kind"].eq("fvg_fill_event").where(
            gated.events.get("fill_kind", pd.Series(dtype=object)).eq("filled"),
            False,
        ).any()


def test_bundle_identity_isolation(ctx) -> None:
    from dataclasses import replace

    from alpha_lab.agents.data_infra.ifvg.setup_verifier_provider import (
        open_setup_replay_context,
    )

    bad = replace(ctx.bundle, fsm_audit_manifest_hash="0" * 64)
    with pytest.raises(Exception, match="manifest hash"):
        open_setup_replay_context(REPO_ROOT, bad)


def test_parentless_interval_reconciliation_visible(ctx) -> None:
    from alpha_lab.agents.data_infra.ifvg.audit_contracts import AuditTable
    from alpha_lab.agents.data_infra.ifvg.setup_verifier_provider import (
        list_setups,
        setup_evidence,
    )

    setups = list_setups(ctx)
    target = str(
        setups.loc[setups["parentless_interval_count"] > 0, "setup_id"].iloc[0]
    )
    evidence = setup_evidence(ctx, target)
    intervals = evidence.parentless_intervals
    assert not intervals.empty
    steps = ctx.fsm_audit.tables[AuditTable.PARENTLESS_STEP]
    mine = steps.loc[steps["setup_id"].astype(str) == target]
    assert int(intervals["bars_count"].sum()) == int(len(mine))
