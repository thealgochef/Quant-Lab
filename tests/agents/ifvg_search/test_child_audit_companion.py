"""Per-child neutrality-gated FSM-audit companion + slice companion wiring (R2).

Everything here runs the REAL machinery over the synthetic three-day chain:
``build_ifvg_v2_capture`` dual-drives on cached day artifacts, the child audit
companion assembles from the retained trace/audit channels, publication uses
the immutable search store, and the slice companion builder closes the
DEV-R1-6 gates synthetically. The doc-default accepted-parity gate is never
invoked — child audits are gated by the ``ChildAuditNeutralityReport``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from alpha_lab.agents.data_infra.ifvg.audit_contracts import AuditTable
from alpha_lab.agents.data_infra.ifvg.data_access import (
    ExplorationDataPolicy,
    allowlist_sha256,
)
from alpha_lab.agents.data_infra.ifvg.day_artifacts import write_day_artifacts
from alpha_lab.agents.data_infra.ifvg.fsm_audit_preparation import (
    ChildFsmAuditEnvelope,
    build_child_fsm_audit,
    publish_child_fsm_audit,
)
from alpha_lab.agents.data_infra.ifvg.manifest import RepositoryState
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
    build_slice_companions,
    run_child_replay,
    verify_exact_drill_targets,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    CoreStrategyReplayPayload,
    ReplayAccessAuthorizationRef,
    ReplayDayArtifactRef,
    build_replay_input_bundle,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    load_verified_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    VERIFICATION_POLICY_ID,
    VerificationRunEnvelope,
    VerificationRunPayload,
)
from tests.agents.ifvg_search.conftest import SYNTHETIC_DAYS

_CORE_ID = "3" * 64


@pytest.fixture(scope="module")
def audit_disk_cfg(tmp_path_factory, doc_default_cfg, synthetic_artifacts):
    # warmup_days=0: every synthetic day is evidence, and the audit-channel
    # warmup stamps agree with the companion's DAY_FUNNEL stamps (B-M1).
    cfg = replace(
        doc_default_cfg,
        data_dir=tmp_path_factory.mktemp("databento-audit"),
        warmup_days=0,
    )
    policy = ExplorationDataPolicy()
    for artifacts in synthetic_artifacts:
        write_day_artifacts(artifacts, cfg, access_policy=policy)
    return cfg


@pytest.fixture(scope="module")
def dual_drive_result(audit_disk_cfg):
    resolved = resolve_profile_config({})
    return run_child_replay(
        dates=SYNTHETIC_DAYS,
        cfg=audit_disk_cfg,
        resolved_profile=resolved,
        access_policy_factory=ExplorationDataPolicy,
        core_replay_id=_CORE_ID,
        cached_artifacts_only=True,
        dual_drive=True,
    )


def test_dual_drive_retains_audit_channels_and_neutrality(dual_drive_result) -> None:
    result = dual_drive_result
    assert result.neutrality is not None and result.neutrality.passed
    assert result.audit_capture is not None
    assert result.audit_capture.audit_frames is not None
    # DEV-R2-1: the audit-enabled drive retains the trace-cut audit rows; the
    # canonical (audit-disabled) drive retains nothing extra.
    assert result.audit_capture.trace_audit_rows is not None
    assert result.capture.trace_audit_rows is None
    kinds = set(result.audit_capture.trace_audit_rows.get("kind", ()))
    assert kinds <= {
        "htf_tap",
        "parent_candidate",
        "opposing",
        "parent_lock",
        "inversion",
        "setup_resolution",
    }
    assert len(result.audit_capture.trace_audit_rows) > 0


def test_child_audit_build_assembles_and_reconciles(dual_drive_result) -> None:
    build = build_child_fsm_audit(
        core_replay_id=_CORE_ID,
        audit_capture=dual_drive_result.audit_capture,
        neutrality=dual_drive_result.neutrality,
        chain_dates=SYNTHETIC_DAYS,
        warmup_days=0,
    )
    assert build.identity.core_replay_id == _CORE_ID
    assert build.identity.neutrality_mechanism_id == "dual_drive_ab_v1"
    # typed audit tables exist for every contract table (possibly empty)
    assert set(build.audit_tables) == set(AuditTable)
    taps = build.audit_tables[AuditTable.HTF_TAP]
    assert len(taps) > 0  # the synthetic walk produces tap evidence
    assert build.reconciliation_report  # exact reconciliation returned (raises on inexact)
    counts = build.coverage_report["drop_reason_counts"]
    assert set(counts)  # coverage over the full drop-reason enumeration


def test_child_audit_refuses_without_or_with_failed_neutrality(dual_drive_result) -> None:
    with pytest.raises(PermissionError, match="requires a ChildAuditNeutralityReport"):
        build_child_fsm_audit(
            core_replay_id=_CORE_ID,
            audit_capture=dual_drive_result.audit_capture,
            neutrality=None,
            chain_dates=SYNTHETIC_DAYS,
        )
    failed = dual_drive_result.neutrality.model_copy(
        update={"passed": False, "tables_equal": False}
    )
    with pytest.raises(PermissionError, match="neutrality FAILED"):
        build_child_fsm_audit(
            core_replay_id=_CORE_ID,
            audit_capture=dual_drive_result.audit_capture,
            neutrality=failed,
            chain_dates=SYNTHETIC_DAYS,
        )
    mismatched = dual_drive_result.neutrality.model_copy(
        update={"core_replay_id": "4" * 64}
    )
    with pytest.raises(PermissionError, match="different core replay id"):
        build_child_fsm_audit(
            core_replay_id=_CORE_ID,
            audit_capture=dual_drive_result.audit_capture,
            neutrality=mismatched,
            chain_dates=SYNTHETIC_DAYS,
        )
    # a capture without the retained trace refuses (no silent empty tables)
    bare = dual_drive_result.capture  # audit-disabled: trace_audit_rows is None
    with pytest.raises(PermissionError, match="audit-enabled capture"):
        build_child_fsm_audit(
            core_replay_id=_CORE_ID,
            audit_capture=bare,
            neutrality=dual_drive_result.neutrality,
            chain_dates=SYNTHETIC_DAYS,
        )


def test_warmup_stamp_disagreement_refuses(dual_drive_result) -> None:
    """B-M1: one artifact may never carry two warmup truths."""

    with pytest.raises(PermissionError, match="warmup stamps are inconsistent"):
        build_child_fsm_audit(
            core_replay_id=_CORE_ID,
            audit_capture=dual_drive_result.audit_capture,
            neutrality=dual_drive_result.neutrality,
            chain_dates=SYNTHETIC_DAYS,
            warmup_days=2,  # capture stamped 0 warmup days
        )


def test_run_child_replay_raises_on_failed_neutrality(
    audit_disk_cfg, monkeypatch
) -> None:
    """CS §3.3: the worker refuses loudly — publication is unreachable."""

    import alpha_lab.agents.data_infra.ifvg.search.child_replay as child_replay_module
    from alpha_lab.agents.data_infra.ifvg.search.failure import ChildNeutralityError

    real_builder = child_replay_module.build_neutrality_report

    def _failing(**kwargs):
        report = real_builder(**kwargs)
        return report.model_copy(update={"passed": False, "tables_equal": False})

    monkeypatch.setattr(child_replay_module, "build_neutrality_report", _failing)
    resolved = resolve_profile_config({})
    with pytest.raises(ChildNeutralityError, match="audit-neutrality FAILED"):
        run_child_replay(
            dates=SYNTHETIC_DAYS,
            cfg=audit_disk_cfg,
            resolved_profile=resolved,
            access_policy_factory=ExplorationDataPolicy,
            core_replay_id=_CORE_ID,
            cached_artifacts_only=True,
            dual_drive=True,
        )


def test_child_audit_publishes_immutably_and_reuses(tmp_path, dual_drive_result) -> None:
    build = build_child_fsm_audit(
        core_replay_id=_CORE_ID,
        audit_capture=dual_drive_result.audit_capture,
        neutrality=dual_drive_result.neutrality,
        chain_dates=SYNTHETIC_DAYS,
    )
    envelope, reused = publish_child_fsm_audit(tmp_path, build)
    assert reused is False
    assert has_envelope(tmp_path, "fsm_audit_companions", envelope.child_fsm_audit_id)
    again, reused_again = publish_child_fsm_audit(tmp_path, build)
    assert reused_again is True
    assert again.child_fsm_audit_id == envelope.child_fsm_audit_id
    reloaded = load_verified_envelope(
        tmp_path,
        "fsm_audit_companions",
        envelope.child_fsm_audit_id,
        ChildFsmAuditEnvelope,
    )
    assert reloaded.payload == build.identity


def _verification_run(allowlist: tuple[str, ...], section_hash: str, store_root=None):
    from tests.agents.ifvg_search.namespace_fixture import verification_authorization_ref

    allowlist_hash = allowlist_sha256(allowlist)
    authorization = verification_authorization_ref(
        store_root,
        approved_allowlist_hash=allowlist_hash,
        coverage_matrix_artifact_id="b" * 64,
        seed_snapshot_id="c" * 64,
    )
    return VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id="a" * 64,
            verification_authorization=authorization,
            allowlist=allowlist,
            allowlist_hash=allowlist_hash,
            seed_snapshot_id="c" * 64,
            baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            baseline_section_config_hash=section_hash,
            coverage_matrix_artifact_id="b" * 64,
        )
    )


def test_slice_companions_close_the_dev_r1_6_gates_synthetically(
    tmp_path, dual_drive_result
) -> None:
    """The full companion path: audit + neutrality + v2 publication + links."""

    resolved = resolve_profile_config({})
    run = _verification_run(SYNTHETIC_DAYS, resolved.section_config_hash, tmp_path)
    day_refs = tuple(
        ReplayDayArtifactRef(
            trading_day=day,
            artifact_kind=kind,
            artifact_id=f"{kind}-{day}",
            manifest_payload_sha256=f"{index:064x}",
            content_sha256=None,
        )
        for index, (day, kind) in enumerate(
            (day, kind) for day in SYNTHETIC_DAYS for kind in ("bars", "levels")
        )
    )
    bundle = build_replay_input_bundle(
        authorized_date_set_id=run.payload.allowlist_hash,
        source_partitions=(),
        day_artifacts=day_refs,
        source_contract_id="databento_nq_v1",
        source_schema_era_id="mbp1_era_v1",
        access_authorization=ReplayAccessAuthorizationRef(
            access_policy_id=VERIFICATION_POLICY_ID,
            authorized_date_set_id=run.payload.allowlist_hash,
            expected_source_inventory_hash="6" * 64,
        ),
    )
    core = CoreStrategyReplayIdentity.from_payload(
        CoreStrategyReplayPayload(
            replay_input_bundle_id=bundle.replay_input_bundle_id,
            quant_lab_replay_source_identity="e" * 64,
            strategy_core_commit="f" * 40,
            strategy_core_source_identity="f" * 64,
            resolved_section_config_hash=resolved.section_config_hash,
            canonical_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            warmup_seed_identity="cold_start_v1",
            anchor_policy=resolved.section.anchor_policy,
            capture_schema_version=2,
            record_schema_version=2,
        )
    )
    # key the neutrality/audit evidence to the REAL minted core id
    result = dual_drive_result
    rekeyed_neutrality = result.neutrality.model_copy(
        update={"core_replay_id": core.core_replay_id}
    )

    class _Result:
        resolved_profile = result.resolved_profile
        capture = result.capture
        audit_capture = result.audit_capture
        neutrality = rekeyed_neutrality
        gross_trade_stream_hash = result.gross_trade_stream_hash

    states = (
        RepositoryState(
            name="quant-lab",
            path="synthetic",
            head="1" * 40,
            dirty_status_sha256="7" * 64,
            source_tree_hash="8" * 64,
        ),
        RepositoryState(
            name="strategy-core",
            path="synthetic",
            head="2" * 40,
            dirty_status_sha256="9" * 64,
            source_tree_hash="a" * 64,
        ),
    )
    report = build_slice_companions(
        result=_Result(),
        run=run,
        core=core,
        bundle=bundle,
        store_root=tmp_path,
        repo_root=tmp_path,
        repository_states=states,
    )
    assert report["invariants_passed"] is True
    assert report["published"] is True
    assert report["verifier_link_resolves"] is True
    assert report["verifier_link_evidence"]["target_count"] > 0
    assert has_envelope(
        tmp_path, "fsm_audit_companions", report["child_fsm_audit_id"]
    )
    assert has_envelope(
        tmp_path, "neutrality_reports", report["neutrality_report_id"]
    )
    v2_manifest = (
        tmp_path
        / "v2_datasets"
        / report["v2_dataset_artifact_id"]
        / "exploration"
        / "manifest.json"
    )
    assert v2_manifest.exists()
    reference = report["core_replay_artifact_reference"]
    assert reference["core_replay_id"] == core.core_replay_id
    assert reference["gross_trade_stream_hash"] == result.gross_trade_stream_hash
    # idempotent re-run: every publication is reused, nothing overwritten
    rerun = build_slice_companions(
        result=_Result(),
        run=run,
        core=core,
        bundle=bundle,
        store_root=tmp_path,
        repo_root=tmp_path,
        repository_states=states,
    )
    assert rerun["child_fsm_audit_reused"] is True
    assert rerun["neutrality_report_reused"] is True
    assert rerun["v2_dataset_reused"] is True


def test_slice_companions_require_repository_states(tmp_path, dual_drive_result) -> None:
    resolved = resolve_profile_config({})
    run = _verification_run(SYNTHETIC_DAYS, resolved.section_config_hash, tmp_path)
    with pytest.raises(PermissionError, match="repository_states"):
        build_slice_companions(
            result=dual_drive_result,
            run=run,
            core=None,
            bundle=None,
            store_root=tmp_path,
            repo_root=tmp_path,
        )


def test_exact_drill_targets_zero_entities_is_vacuous_not_resolved_claim() -> None:
    import pandas as pd

    link = verify_exact_drill_targets({}, {})
    assert link["vacuous_zero_targets"] is True
    assert link["target_count"] == 0
    assert link["resolves"] is True  # machinery ran; explicitly marked vacuous
    # an audit setup id with no lifecycle row is a hard failure, never dropped
    audit_tables = {
        "ifvg_audit_htf_tap": pd.DataFrame({"setup_id": ["orphan-setup"]})
    }
    link = verify_exact_drill_targets({}, audit_tables)
    assert link["resolves"] is False
    assert any("no lifecycle setup" in failure for failure in link["failures"])
