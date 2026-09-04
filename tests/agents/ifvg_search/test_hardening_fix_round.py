"""HARDENING-BACKEND adversarial fix round — the main agent's findings.

RA-01 the MBP-1 diagnostic seam binds the namespace + current head before any
path; RA-02 the lock's release / refresh / reclaim survive a transient reader
and type a persistent failure; RA-03 research-catalog activation re-verifies
the charter bundle against the CURRENT head; RA-06 the synthetic fixture never
marks a non-temp root; RA-08 the executor selects the run by pipeline id and
refuses ambiguity; RA-09 the deployment check runs inside the bound check;
RA-10 the head / namespace atomic replace retries and types a persistent
failure; B-06 worker counts are strict ints; B-03 S08 persists a typed
``fold_summary.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.search import owner_decision_lock as lock_module
from alpha_lab.agents.data_infra.ifvg.search import store_namespace as namespace_module
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    AuthorizationError,
    assert_authorization_bound_to_store,
)
from alpha_lab.agents.data_infra.ifvg.search.catalog import append_catalog_event
from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope, save_charter
from alpha_lab.agents.data_infra.ifvg.search.executors import _verification_run_envelope
from alpha_lab.agents.data_infra.ifvg.search.owner_decision_lock import (
    OWNER_DECISION_LOCK_FILE,
    OwnerDecisionLock,
    OwnerDecisionLockError,
)
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
    OwnerDecisionRefusalError,
    synthetic_owner_decision_fixture,
)
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    PUBLICATION_GATE_IDS,
    PublicationError,
    QuantLabPipelineStage,
    WorkerPolicy,
    activate_pipeline_result,
    read_pipeline_state,
    run_pipeline,
    worker_parallelism_refusal,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SEARCH_TEST_STORE_ROOT,
    load_json_sidecar,
    save_or_reuse_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    OWNER_DECISION_STORE,
    StoreNamespaceError,
    initialize_store_namespace,
    initialize_test_namespace,
    supersession_head_path,
    write_supersession_head_atomic,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    current_supersession_head_witness,
    publish_supersession,
)
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    CoverageMatrixEnvelope,
    CoverageMatrixPayload,
    DayCoverageRow,
    VerificationRunEnvelope,
    VerificationRunPayload,
)
from tests.agents.ifvg_search import pipeline_fixture as pipeline_fixture_module
from tests.agents.ifvg_search.namespace_fixture import (
    namespace_and_witness,
    owner_authorization_bundle,
    verification_authorization_ref,
)
from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture

_AT = "2026-09-01T00:00:00+00:00"


# ── RA-02 / RA-10 ────────────────────────────────────────────────────────────


def _lock_root(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    initialize_test_namespace(root)
    return root


def test_release_survives_a_transient_reader_and_never_orphans_silently(tmp_path) -> None:
    root = _lock_root(tmp_path)
    lock = OwnerDecisionLock(root, wait_seconds=1.0)
    lock.acquire()
    path = root / OWNER_DECISION_STORE / OWNER_DECISION_LOCK_FILE
    # a polling waiter holds a read handle open across the release (the RA-02
    # Windows sharing violation); the release retries until the handle closes
    handle = open(path, encoding="utf-8")  # noqa: SIM115 — held deliberately

    def _close_later() -> None:
        time.sleep(0.15)
        handle.close()

    thread = threading.Thread(target=_close_later)
    thread.start()
    lock.release()
    thread.join()
    assert not path.exists()
    # a PERSISTENT failure is typed, never silent, and the file keeps our token
    lock = OwnerDecisionLock(root, wait_seconds=1.0)
    lock.acquire()

    def _always_busy(_path):
        raise PermissionError("sharing violation (simulated, persistent)")

    original_unlink = lock_module.os.unlink
    lock_module.os.unlink = _always_busy  # type: ignore[assignment]
    try:
        with pytest.raises(OwnerDecisionLockError) as failed:
            lock.release()
    finally:
        lock_module.os.unlink = original_unlink  # type: ignore[assignment]
    assert failed.value.reason == "lock_release_failed"
    assert json.loads(path.read_text(encoding="utf-8"))["lock_token"] == lock.token
    lock.release()
    assert not path.exists()


def test_refresh_retries_transient_failures_and_types_persistent_ones(tmp_path, monkeypatch):
    root = _lock_root(tmp_path)
    lock = OwnerDecisionLock(root, wait_seconds=1.0)
    lock.acquire()
    real_replace = lock_module.os.replace
    calls = {"n": 0}

    def _flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("sharing violation (simulated, transient)")
        return real_replace(src, dst)

    monkeypatch.setattr(lock_module.os, "replace", _flaky)
    monkeypatch.setattr(lock_module, "_FS_RETRY_DELAY_SECONDS", 0.001)
    lock.refresh()
    assert calls["n"] == 3 and lock.verify_held().lock_token == lock.token
    monkeypatch.setattr(lock_module.os, "replace", lambda *_: (_ for _ in ()).throw(
        PermissionError("sharing violation (simulated, persistent)")
    ))
    with pytest.raises(OwnerDecisionLockError) as failed:
        lock.refresh()
    assert failed.value.reason == "lock_refresh_failed"
    # no heartbeat temp file survives a failed refresh; the lock body is intact
    assert not list((root / OWNER_DECISION_STORE).glob(".SUPERSESSIONS.lock.hb-*"))
    assert lock.verify_held().lock_token == lock.token
    monkeypatch.setattr(lock_module.os, "replace", real_replace)
    lock.release()


def test_atomic_head_write_retries_and_types_a_persistent_failure(tmp_path, monkeypatch):
    root = tmp_path / "store"
    namespace = initialize_test_namespace(root)
    head = supersession_head_path(root)
    real_replace = namespace_module.os.replace
    calls = {"n": 0}

    def _flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("sharing violation (simulated, transient)")
        return real_replace(src, dst)

    monkeypatch.setattr(namespace_module.os, "replace", _flaky)
    monkeypatch.setattr(namespace_module, "_REPLACE_RETRY_DELAY_SECONDS", 0.001)
    write_supersession_head_atomic(
        root,
        store_namespace_id=namespace.store_namespace_id,
        record_id=None,
        line_count=0,
        head_sha256=namespace.payload.authority_genesis_id,
    )
    assert calls["n"] == 3 and current_supersession_head_witness(root).line_count == 0
    monkeypatch.setattr(namespace_module, "_REPLACE_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(namespace_module.os, "replace", lambda *_: (_ for _ in ()).throw(
        PermissionError("sharing violation (simulated, persistent)")
    ))
    with pytest.raises(StoreNamespaceError) as failed:
        write_supersession_head_atomic(
            root,
            store_namespace_id=namespace.store_namespace_id,
            record_id=None,
            line_count=0,
            head_sha256=namespace.payload.authority_genesis_id,
        )
    assert failed.value.reason == "atomic_write_failed"
    assert not list(head.parent.glob(".SUPERSESSIONS.head.tmp-*"))
    monkeypatch.setattr(namespace_module.os, "replace", real_replace)
    assert current_supersession_head_witness(root).line_count == 0  # the old head survived


# ── RA-09 / B-06 / RA-06 / RA-08 ─────────────────────────────────────────────


def test_bound_check_refuses_an_incoherently_deployed_test_namespace(tmp_path) -> None:
    root = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"  # research-looking
    namespace_id, witness = namespace_and_witness(root)  # marked test
    with pytest.raises(AuthorizationError) as refused:
        assert_authorization_bound_to_store(
            root, store_namespace_id=namespace_id, supersession_head_witness=witness
        )
    assert refused.value.reason == "store_namespace_deployment_incoherent"


@pytest.mark.parametrize("value", [0, 2, True, "1", 1.0])
def test_worker_policy_accepts_only_the_int_one_with_the_typed_reason(value) -> None:
    with pytest.raises(ValidationError) as refused:
        WorkerPolicy(max_workers=value, max_tasks_per_child=1, memory_budget_bytes=0)
    typed = worker_parallelism_refusal(refused.value)
    assert typed is not None and typed.reason == "unsupported_worker_parallelism_v1"
    accepted = WorkerPolicy(max_workers=1, max_tasks_per_child=1, memory_budget_bytes=0)
    assert accepted.max_workers == 1


def test_synthetic_fixture_never_marks_a_root_outside_the_temp_directory() -> None:
    probe = Path.cwd() / ".hardening_fixture_probe_never_marked"
    probe.mkdir(exist_ok=True)
    try:
        with pytest.raises(OwnerDecisionRefusalError) as refused:
            synthetic_owner_decision_fixture(probe, protocol=None, assessment=None)
        assert getattr(refused.value, "reason", None) == "store_namespace_missing"
        assert not (probe / "STORE_NAMESPACE.json").exists()
        assert not (probe / OWNER_DECISION_STORE).exists()
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def _bound_run(root: Path, *, pipeline_semantic_id: str, days: tuple[str, ...]):
    authorization = verification_authorization_ref(
        root,
        approved_allowlist_hash=allowlist_sha256(days),
        coverage_matrix_artifact_id="b" * 64,
        seed_snapshot_id="c" * 64,
    )
    return VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id=pipeline_semantic_id,
            verification_authorization=authorization,
            allowlist=days,
            allowlist_hash=allowlist_sha256(days),
            seed_snapshot_id="c" * 64,
            baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            baseline_section_config_hash="e" * 64,
            coverage_matrix_artifact_id="b" * 64,
        )
    )


def test_verification_run_is_selected_by_pipeline_id_and_ambiguity_refuses(tmp_path) -> None:
    root = tmp_path / "repo" / SEARCH_TEST_STORE_ROOT
    root.mkdir(parents=True)
    initialize_test_namespace(root)
    first = _bound_run(root, pipeline_semantic_id="1" * 64, days=("2026-06-04",))
    second = _bound_run(root, pipeline_semantic_id="2" * 64, days=("2026-06-05",))
    for run in (first, second):
        save_or_reuse_envelope(root, "verification_runs", run)
        append_catalog_event(
            root, kind="display_name", artifact_id=run.verification_run_id,
            payload={"display_name": run.payload.pipeline_semantic_id[:4]},
        )
    with pytest.raises(PermissionError, match="never positional"):
        _verification_run_envelope(root)
    chosen = _verification_run_envelope(root, pipeline_semantic_id="2" * 64)
    assert chosen.verification_run_id == second.verification_run_id
    with pytest.raises(PermissionError, match="binds the pipeline semantic id"):
        _verification_run_envelope(root, pipeline_semantic_id="3" * 64)


# ── RA-01: the MBP-1 diagnostic seam ─────────────────────────────────────────


def _diagnostic_store(root: Path, days: tuple[str, ...]):
    matrix = CoverageMatrixEnvelope.from_payload(
        CoverageMatrixPayload(
            evidence_source_dataset_id="1" * 64,
            evidence_source_manifest_sha256="2" * 64,
            candidate_allowlist=days,
            candidate_allowlist_hash=allowlist_sha256(days),
            rows=tuple(
                DayCoverageRow(
                    trading_day=day,
                    source_partition_recorded=True,
                    setup_lifecycle_rows=1,
                    entry_candidate_rows=1,
                    eligible_decision_rows=0,
                    executed_trade_rows=0,
                    candidate_label_rows=1,
                    audit_event_rows=None,
                    replay_chart_available=None,
                )
                for day in days
            ),
            lifecycle_paths_covered={"entry_candidate": True},
            uncovered_paths_note="synthetic",
        )
    )
    save_or_reuse_envelope(root, "coverage_matrices", matrix)
    authorization = verification_authorization_ref(
        root,
        approved_allowlist_hash=allowlist_sha256(days),
        coverage_matrix_artifact_id=matrix.coverage_matrix_id,
        seed_snapshot_id="b" * 64,
        approved_by="synthetic",
    )
    run = VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id="a" * 64,
            verification_authorization=authorization,
            allowlist=days,
            allowlist_hash=allowlist_sha256(days),
            seed_snapshot_id="b" * 64,
            baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            baseline_section_config_hash="e" * 64,
            coverage_matrix_artifact_id=matrix.coverage_matrix_id,
        )
    )
    return run


def test_mbp1_diagnostic_seam_binds_namespace_and_current_head_before_any_path(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.development_access import VerificationReplayPolicy
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_diagnostic import (
        assert_diagnostic_authorized,
    )

    days = ("2026-01-13",)
    root = tmp_path / "bound" / "search_test" / "v1"
    root.mkdir(parents=True)
    run = _diagnostic_store(root, days)  # marks the root as a test namespace
    marker = root / "VERIFICATION_ALLOWLIST_MARKER.json"
    assert assert_diagnostic_authorized(
        store_root=root, run_envelope=run, access_policy=VerificationReplayPolicy(days),
        allowlist=days, canonical_root=root,
    ) == run.payload.verification_authorization.content_hash
    assert marker.exists()
    # (a) an UNMARKED directory whose path ends in search_test/v1 is refused by
    # the namespace binding — before the coverage matrix loads (none exists
    # there) and before the program allowlist is registered
    unmarked = tmp_path / "unmarked" / "search_test" / "v1"
    unmarked.mkdir(parents=True)
    with pytest.raises(PermissionError, match="store_namespace_missing"):
        assert_diagnostic_authorized(
            store_root=unmarked, run_envelope=run, access_policy=VerificationReplayPolicy(days),
            allowlist=days, canonical_root=unmarked,
        )
    assert not (unmarked / "VERIFICATION_ALLOWLIST_MARKER.json").exists()
    # (b) a research-class namespace at that path
    research = tmp_path / "research" / "search_test" / "v1"
    research.mkdir(parents=True)
    initialize_store_namespace(research, namespace_class="research", store_instance_id="a2" * 16)
    with pytest.raises(PermissionError, match="store_namespace_class_mismatch"):
        assert_diagnostic_authorized(
            store_root=research, run_envelope=run, access_policy=VerificationReplayPolicy(days),
            allowlist=days, canonical_root=research,
        )
    # (c) the head moved after the owner signed
    publish_supersession(
        root,
        superseded_decision_id="1" * 64,
        replacement_decision_id="2" * 64,
        reason="moved after signing",
        effective_at=_AT,
        owner_evidence_ref="2" * 64,
    )
    with pytest.raises(PermissionError, match="supersession_head_witness_mismatch"):
        assert_diagnostic_authorized(
            store_root=root, run_envelope=run, access_policy=VerificationReplayPolicy(days),
            allowlist=days, canonical_root=root,
        )


# ── RA-03 + B-03 support: a completed synthetic pipeline ─────────────────────


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fix_round_pipeline")
    fixture = build_pipeline_fixture(tmp_path)
    save_charter(fixture["store_root"], fixture["charter"])
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    return {"fixture": fixture, "result": result, "tmp_path": tmp_path}


def test_s08_persists_a_typed_fold_summary_sidecar(completed) -> None:
    fixture = completed["fixture"]
    state = read_pipeline_state(fixture["state_root"], completed["result"].pipeline_semantic_id)
    entry = state["stages"][QuantLabPipelineStage.S08_BUILD_FOLDS.value]
    summary = load_json_sidecar(
        fixture["store_root"], "pipeline_stage_results", entry["stage_result_id"],
        "fold_summary.json",
    )
    assert summary is not None
    assert set(summary) == {
        "fold_count", "valid_fold_count", "invalid_reasons", "trading_day_count"
    }
    assert summary["valid_fold_count"] == 0 and summary["fold_count"] == 0
    assert summary["trading_day_count"] == len(pipeline_fixture_module.SYNTHETIC_DAYS)


def test_activation_re_verifies_the_charter_bundle_against_the_current_head(completed, tmp_path):
    fixture = completed["fixture"]
    store_root = fixture["store_root"]
    semantic_id = completed["result"].pipeline_semantic_id
    state_root = tmp_path / "state"
    shutil.copytree(fixture["state_root"], state_root)
    state_path = state_root / semantic_id / "pipeline_state.json"

    def _craft(**overrides) -> None:
        crafted = read_pipeline_state(state_root, semantic_id)
        crafted["run_scope"] = "full_authorized_development"
        crafted["publication"] = {
            "state": "gates_passed",
            "gates": {name: True for name in PUBLICATION_GATE_IDS},
            "activated": False,
            "pipeline_result_id": crafted["publication"]["pipeline_result_id"],
            "control_flow_gates_passed": True,
            "reload_failures": {},
        }
        crafted.update(overrides)
        state_path.write_text(json.dumps(crafted), encoding="utf-8")

    # a synthetic-marker charter can never activate a research entry, even
    # under a crafted development scope with every gate re-deriving true
    _craft()
    with pytest.raises(PublicationError, match="synthetic-authorization charter"):
        activate_pipeline_result(state_root, semantic_id, store_root=store_root)
    # a REAL bundle charter bound to this store's namespace + current head
    bundle = owner_authorization_bundle(
        store_root, requirement_set_id="1" * 64, decision_refs={}
    )
    real_charter = SearchCharterEnvelope.from_payload(
        fixture["charter"].payload.model_copy(update={"owner_authorization": bundle})
    )
    save_charter(store_root, real_charter)
    _craft(search_charter_id=real_charter.search_id)
    result_id = activate_pipeline_result(state_root, semantic_id, store_root=store_root)
    assert read_pipeline_state(state_root, semantic_id)["publication"]["activated"] is True
    assert result_id
    # a supersession that lands after the launch moves the head: the bundle
    # is no longer bound and the activation is refused
    publish_supersession(
        store_root,
        superseded_decision_id="1" * 64,
        replacement_decision_id="2" * 64,
        reason="landed after the launch",
        effective_at=_AT,
        owner_evidence_ref="2" * 64,
    )
    _craft(search_charter_id=real_charter.search_id)
    with pytest.raises(PublicationError, match="no longer bound"):
        activate_pipeline_result(state_root, semantic_id, store_root=store_root)
    assert read_pipeline_state(state_root, semantic_id)["publication"]["activated"] is False
    os.utime(state_path)  # touch only; nothing else persisted
