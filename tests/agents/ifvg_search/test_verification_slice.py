"""Synthetic composition test of the Path-A vertical slice runner.

Proves the R1-owned slice sequence end-to-end with an injected replay stub:
fail-before-path validation → canonical-namespace binding → program-allowlist
marker → snapshot continuity → REAL input-bundle + core-identity assembly →
dual-drive handoff keyed by the real core id → immutable publication
(save→reload→reuse) → honest open gates → exact stamps. No real market data
is touched; every artifact lives under tmp paths.
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest
from strategy_core.strategies.ifvg_smc.state import IFVG_SEED_SCHEMA_VERSION, seed_hash

from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.development_access import VerificationReplayPolicy
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
    ArtifactProvenanceReadAdapter,
    ChildAuditNeutralityReport,
    DaySeedsRecord,
    SeedSnapshotError,
    SeedSnapshotPayload,
    _load_seed_bytes,
    run_baseline_verification_slice,
    run_child_replay,
    save_seed_snapshot,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SEARCH_TEST_STORE_ROOT,
    has_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    VerificationRunEnvelope,
    VerificationRunPayload,
    VerificationRunValidationError,
)

_ALLOWLIST = ("2026-06-04", "2026-06-05")


@pytest.fixture()
def slice_env(tmp_path, synthetic_chain):
    repo_root = tmp_path / "repo"
    store_root = repo_root / SEARCH_TEST_STORE_ROOT
    store_root.mkdir(parents=True)
    data_dir = repo_root / "data" / "databento"
    resolved = resolve_profile_config({})
    cfg = replace(IfvgCaptureConfig(), section=resolved.section, data_dir=data_dir)
    for day in _ALLOWLIST:
        for factory in (cfg.bars_path, cfg.levels_path):
            path = factory(day)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"cached-artifact:{path.name}:{day}".encode())
    seed = synthetic_chain[1].end_seed
    snapshot = save_seed_snapshot(
        store_root,
        SeedSnapshotPayload(
            profile_name="ifvg_v2_doc_default_fresh_static_1r",
            resolved_section_config_hash=seed.profile_hash,
            seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
            seed_hash=seed_hash(seed),
            snapshot_through_day="2026-06-03",
            first_replay_day=_ALLOWLIST[0],
            entering_day_seeds=DaySeedsRecord(
                prev_day="2026-06-03",
                prev_full_hl=(100, 50),
                prev_ny_day=None,
                prev_ny_hl=None,
            ),
            chain_policy_id="development_explicit_dates_before_path_v2",
            chain_date_count=2,
            strategy_core_commit="c" * 40,
        ),
        seed,
    )
    allowlist_hash = allowlist_sha256(_ALLOWLIST)
    from tests.agents.ifvg_search.namespace_fixture import verification_authorization_ref

    # HARDENING-BACKEND §4.1: the verification store is an explicit test
    # namespace and the authorization binds it + the current head
    authorization = verification_authorization_ref(
        store_root,
        approved_allowlist_hash=allowlist_hash,
        coverage_matrix_artifact_id="b" * 64,
        seed_snapshot_id=snapshot.seed_snapshot_id,
    )
    run = VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id="a" * 64,
            verification_authorization=authorization,
            allowlist=_ALLOWLIST,
            allowlist_hash=allowlist_hash,
            seed_snapshot_id=snapshot.seed_snapshot_id,
            baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            baseline_section_config_hash=resolved.section_config_hash,
            coverage_matrix_artifact_id="b" * 64,
        )
    )
    return SimpleNamespace(
        repo_root=repo_root,
        store_root=store_root,
        data_dir=data_dir,
        run=run,
        authorization=authorization,
        snapshot=snapshot,
        seed=seed,
    )


def _stub_replay_runner(observed: dict):
    def _runner(**kwargs):
        observed.update(kwargs)

        class _Policy:
            def assert_zero_forbidden_access(self):
                return None

        capture = SimpleNamespace(tables={}, audit_frames=None, access_policy=_Policy())
        neutrality = ChildAuditNeutralityReport(
            core_replay_id=kwargs["core_replay_id"],
            mechanism="dual_drive_ab_v1",
            audit_disabled_core_table_hashes={},
            audit_enabled_core_table_hashes={},
            tables_equal=True,
            mechanism_evidence_refs=(),
            core_trace_content_hash="9" * 64,
            audit_stamp_referential_integrity=True,
            passed=True,
        )
        return SimpleNamespace(
            capture=capture,
            audit_capture=capture,
            neutrality=neutrality,
            gross_trade_stream_hash="8" * 64,
        )

    return _runner


def test_slice_composes_identity_publication_and_honest_gates(slice_env) -> None:
    observed: dict = {}
    summary = run_baseline_verification_slice(
        run=slice_env.run,
        authorization=slice_env.authorization,
        pipeline_semantic_id="a" * 64,
        store_root=slice_env.store_root,
        repo_root=slice_env.repo_root,
        data_dir=slice_env.data_dir,
        ql_source_identity="e" * 64,
        sc_identity=("f" * 40, "f" * 64),
        replay_runner=_stub_replay_runner(observed),
    )
    # the REAL minted core id keyed the replay + neutrality (no zero sentinel)
    assert observed["core_replay_id"] == summary["core_replay_id"]
    assert summary["core_replay_id"] != "0" * 64
    assert observed["cached_artifacts_only"] is True
    # bundle + core envelope published immutably into the canonical namespace
    assert has_envelope(
        slice_env.store_root, "replay_input_bundles", summary["replay_input_bundle_id"]
    )
    assert has_envelope(slice_env.store_root, "core_replays", summary["core_replay_id"])
    assert summary["core_replay_reused"] is False
    # program marker written once, canonical
    assert (slice_env.store_root / "VERIFICATION_ALLOWLIST_MARKER.json").exists()
    # honest gates: neutrality passed; companion-owned gates stay OPEN
    gates = summary["gate_policy"]["results"]
    assert dict(gates)["neutrality_passed"] is True
    assert dict(gates)["artifacts_published_and_reloaded"] is False
    assert dict(gates)["verifier_link_resolves"] is False
    # exact stamps
    assert summary["verification_only"] is True
    assert summary["not_for_research_interpretation"] is True
    assert summary["full_pipeline_not_run"] is True
    assert summary["real_date_count"] == len(_ALLOWLIST)
    # idempotent re-run reuses the SAME identities (save→reload→reuse)
    rerun = run_baseline_verification_slice(
        run=slice_env.run,
        authorization=slice_env.authorization,
        pipeline_semantic_id="a" * 64,
        store_root=slice_env.store_root,
        repo_root=slice_env.repo_root,
        data_dir=slice_env.data_dir,
        ql_source_identity="e" * 64,
        sc_identity=("f" * 40, "f" * 64),
        replay_runner=_stub_replay_runner({}),
    )
    assert rerun["core_replay_id"] == summary["core_replay_id"]
    assert rerun["core_replay_reused"] is True


def test_slice_refuses_failed_neutrality_before_any_publication(slice_env) -> None:
    """F1: CS §3.3 — neutrality gates PUBLICATION, not just the gate booleans."""

    from alpha_lab.agents.data_infra.ifvg.search.failure import ChildNeutralityError

    def _failed_runner(**kwargs):
        class _Policy:
            def assert_zero_forbidden_access(self):
                return None

        capture = SimpleNamespace(tables={}, audit_frames=None, access_policy=_Policy())
        neutrality = ChildAuditNeutralityReport(
            core_replay_id=kwargs["core_replay_id"],
            mechanism="dual_drive_ab_v1",
            audit_disabled_core_table_hashes={},
            audit_enabled_core_table_hashes={},
            tables_equal=False,
            mechanism_evidence_refs=(),
            core_trace_content_hash="9" * 64,
            audit_stamp_referential_integrity=True,
            passed=False,
        )
        return SimpleNamespace(
            capture=capture,
            audit_capture=capture,
            neutrality=neutrality,
            gross_trade_stream_hash="8" * 64,
        )

    with pytest.raises(ChildNeutralityError, match="PASSING dual-drive"):
        run_baseline_verification_slice(
            run=slice_env.run,
            authorization=slice_env.authorization,
            pipeline_semantic_id="a" * 64,
            store_root=slice_env.store_root,
            repo_root=slice_env.repo_root,
            data_dir=slice_env.data_dir,
            ql_source_identity="e" * 64,
            sc_identity=("f" * 40, "f" * 64),
            replay_runner=_failed_runner,
        )
    # NOTHING was published for the refused slice
    from alpha_lab.agents.data_infra.ifvg.search.store import SEARCH_STORE_NAMES

    for store_name in ("core_replays", "replay_input_bundles"):
        store_dir = slice_env.store_root / store_name
        published = list(store_dir.iterdir()) if store_dir.exists() else []
        assert published == [], f"{store_name} must stay empty on refusal"
    del SEARCH_STORE_NAMES


def test_slice_refuses_non_canonical_store_root(slice_env, tmp_path) -> None:
    with pytest.raises(VerificationRunValidationError, match="search_test/v1"):
        run_baseline_verification_slice(
            run=slice_env.run,
            authorization=slice_env.authorization,
            pipeline_semantic_id="a" * 64,
            store_root=tmp_path / "elsewhere",
            repo_root=slice_env.repo_root,
            data_dir=slice_env.data_dir,
            ql_source_identity="e" * 64,
            sc_identity=("f" * 40, "f" * 64),
            replay_runner=_stub_replay_runner({}),
        )


def test_slice_refuses_snapshot_discontinuous_with_allowlist(slice_env) -> None:
    seed = slice_env.seed
    wrong_window = save_seed_snapshot(
        slice_env.store_root,
        SeedSnapshotPayload(
            profile_name="ifvg_v2_doc_default_fresh_static_1r",
            resolved_section_config_hash=seed.profile_hash,
            seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
            seed_hash=seed_hash(seed),
            snapshot_through_day="2026-02-05",
            first_replay_day="2026-02-06",
            entering_day_seeds=DaySeedsRecord(
                prev_day="2026-02-05",
                prev_full_hl=(100, 50),
                prev_ny_day=None,
                prev_ny_hl=None,
            ),
            chain_policy_id="development_explicit_dates_before_path_v2",
            chain_date_count=2,
            strategy_core_commit="c" * 40,
        ),
        seed,
    )
    authorization = slice_env.authorization.model_copy(
        update={"seed_snapshot_id": wrong_window.seed_snapshot_id}
    )
    run = VerificationRunEnvelope.from_payload(
        slice_env.run.payload.model_copy(
            update={
                "verification_authorization": authorization,
                "seed_snapshot_id": wrong_window.seed_snapshot_id,
            }
        )
    )
    with pytest.raises(SeedSnapshotError, match="not continuous"):
        run_baseline_verification_slice(
            run=run,
            authorization=authorization,
            pipeline_semantic_id="a" * 64,
            store_root=slice_env.store_root,
            repo_root=slice_env.repo_root,
            data_dir=slice_env.data_dir,
            ql_source_identity="e" * 64,
            sc_identity=("f" * 40, "f" * 64),
            replay_runner=_stub_replay_runner({}),
        )


def test_provenance_adapter_is_read_only_in_the_worker(doc_default_cfg) -> None:
    resolved = resolve_profile_config({})

    def _factory():
        return ArtifactProvenanceReadAdapter(
            VerificationReplayPolicy(("2026-06-04",)),
            artifact_provenance_dates=("2026-01-13", "2026-06-04"),
        )

    with pytest.raises(PermissionError, match="read-only"):
        run_child_replay(
            dates=("2026-06-04",),
            cfg=doc_default_cfg,
            resolved_profile=resolved,
            access_policy_factory=_factory,
            core_replay_id="0" * 64,
            cached_artifacts_only=False,
        )


def test_output_namespace_literal_rejects_research_store(slice_env) -> None:
    with pytest.raises(ValueError):
        VerificationRunPayload.model_validate(
            {
                **slice_env.run.payload.model_dump(mode="json"),
                "output_namespace": "search/v1",
            }
        )


def test_seed_sidecar_unpickler_refuses_gadgets(tmp_path) -> None:
    class Evil:
        def __reduce__(self):
            import os  # noqa: PLC0415

            return (os.system, ("echo pwned",))

    with pytest.raises(SeedSnapshotError, match="disallowed type"):
        _load_seed_bytes(pickle.dumps(Evil()))
