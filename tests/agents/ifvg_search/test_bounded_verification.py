"""Phase 4 (plan §6) — bounded-verification preflight, the R1 baseline gate
report, and the release-specific bounded control-flow report.

Everything here is synthetic: a store under ``tmp_path`` marked as a test
namespace, the conftest synthetic chain's seed, a bound authorization ref.
No real source path is ever constructed; the real run stays blocked on the
owner's signatures.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from strategy_core.strategies.ifvg_smc.state import IFVG_SEED_SCHEMA_VERSION, seed_hash

from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.authorization import SyntheticAuthorizationMarker
from alpha_lab.agents.data_infra.ifvg.search.bounded_verification import (
    BOUNDED_COMPONENTS,
    R1_BASELINE_PROOF_IDS,
    BoundedComponentResult,
    BoundedReleaseControlFlowPayload,
    BoundedVerificationRefusalError,
    R1BaselineGatePayload,
    build_bounded_release_control_flow_report,
    build_r1_baseline_gate_report,
    preflight_bounded_verification,
    save_bounded_release_control_flow_report,
    save_r1_baseline_gate_report,
)
from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
    DaySeedsRecord,
    SeedSnapshotPayload,
    save_seed_snapshot,
)
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    S11_BLOCKED_REASON,
    read_pipeline_state,
    run_pipeline,
)
from alpha_lab.agents.data_infra.ifvg.search.store import SEARCH_TEST_STORE_ROOT
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    initialize_store_namespace,
    initialize_test_namespace,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    current_supersession_head_witness,
    publish_supersession,
)
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    VerificationRunEnvelope,
    VerificationRunPayload,
    evaluate_control_flow_gates,
)
from tests.agents.ifvg_search.namespace_fixture import (
    namespace_and_witness,
    verification_authorization_ref,
)
from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture

_PROFILE = "ifvg_v2_doc_default_fresh_static_1r"
_WINDOW = ("2026-06-04", "2026-06-05")  # Thu, Fri — consecutive logical days


@pytest.fixture()
def bounded_env(tmp_path, synthetic_chain):
    repo_root = tmp_path / "repo"
    store_root = repo_root / SEARCH_TEST_STORE_ROOT
    store_root.mkdir(parents=True)
    initialize_test_namespace(store_root)
    resolved = resolve_profile_config({})
    cfg = replace(IfvgCaptureConfig(), section=resolved.section)
    seed = synthetic_chain[1].end_seed
    snapshot = save_seed_snapshot(
        store_root,
        SeedSnapshotPayload(
            profile_name=_PROFILE,
            resolved_section_config_hash=seed.profile_hash,
            seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
            seed_hash=seed_hash(seed),
            snapshot_through_day="2026-06-03",
            first_replay_day=_WINDOW[0],
            entering_day_seeds=DaySeedsRecord(
                prev_day="2026-06-03", prev_full_hl=(100, 50), prev_ny_day=None, prev_ny_hl=None
            ),
            chain_policy_id="development_explicit_dates_before_path_v2",
            chain_date_count=2,
            strategy_core_commit="c" * 40,
        ),
        seed,
    )

    def make_run(days=_WINDOW, *, root=store_root):
        authorization = verification_authorization_ref(
            root,
            approved_allowlist_hash=allowlist_sha256(days),
            coverage_matrix_artifact_id="b" * 64,
            seed_snapshot_id=snapshot.seed_snapshot_id,
        )
        run = VerificationRunEnvelope.from_payload(
            VerificationRunPayload(
                pipeline_semantic_id="a" * 64,
                verification_authorization=authorization,
                allowlist=tuple(days),
                allowlist_hash=allowlist_sha256(days),
                seed_snapshot_id=snapshot.seed_snapshot_id,
                baseline_profile_id=_PROFILE,
                baseline_section_config_hash=resolved.section_config_hash,
                coverage_matrix_artifact_id="b" * 64,
            )
        )
        return run, authorization

    return {
        "repo_root": repo_root,
        "store_root": store_root,
        "cfg": cfg,
        "resolved": resolved,
        "snapshot": snapshot,
        "make_run": make_run,
    }


def _preflight(env, run, authorization, **overrides):
    kwargs = dict(
        store_root=env["store_root"],
        repo_root=env["repo_root"],
        verification_run=run,
        authorization=authorization,
        pipeline_semantic_id="a" * 64,
        baseline_profile_id=_PROFILE,
        baseline_section_config_hash=env["resolved"].section_config_hash,
    )
    kwargs.update(overrides)
    return preflight_bounded_verification(**kwargs)


def _reason(env, run, authorization, **overrides) -> str:
    with pytest.raises(BoundedVerificationRefusalError) as refused:
        _preflight(env, run, authorization, **overrides)
    return refused.value.reason


# ── §6.1 preflight ──────────────────────────────────────────────────────────


def test_preflight_passes_on_a_bound_store_and_a_logical_window(bounded_env) -> None:
    run, authorization = bounded_env["make_run"]()
    record = _preflight(bounded_env, run, authorization)
    assert record.passed and all(record.checks.values())
    assert record.allowlist == _WINDOW
    assert record.logical_trading_days == _WINDOW
    # each logical day maps to its two physical UTC partitions (td−1, td)
    assert record.physical_partition_dates == ("2026-06-03", "2026-06-04", "2026-06-05")
    assert record.store_namespace_id == current_supersession_head_witness(
        bounded_env["store_root"]
    ).store_namespace_id
    assert record.seed_snapshot_id == bounded_env["snapshot"].seed_snapshot_id


def test_preflight_refuses_before_any_path_in_the_registered_order(bounded_env, tmp_path) -> None:
    env = bounded_env
    run, authorization = env["make_run"]()
    # the output root must BE the canonical search_test/v1 under repo_root
    assert _reason(env, run, authorization, store_root=tmp_path / "elsewhere") == (
        "output_namespace_not_locked"
    )
    # an unmarked canonical store has no authority
    other_repo = tmp_path / "other_repo"
    unmarked = other_repo / SEARCH_TEST_STORE_ROOT
    unmarked.mkdir(parents=True)
    assert _reason(env, run, authorization, store_root=unmarked, repo_root=other_repo) == (
        "store_namespace_refused"
    )
    # a research-class namespace at the canonical path is refused as well
    research_repo = tmp_path / "research_repo"
    research_store = research_repo / SEARCH_TEST_STORE_ROOT
    research_store.mkdir(parents=True)
    initialize_store_namespace(research_store, namespace_class="research")
    assert _reason(
        env, run, authorization, store_root=research_store, repo_root=research_repo
    ) == "store_namespace_refused"
    # the synthetic marker is never a verification authorization
    assert _reason(env, run, SyntheticAuthorizationMarker()) == "synthetic_marker_refused"
    # a head that moved after signing: the witness is stale
    publish_supersession(
        env["store_root"],
        superseded_decision_id="1" * 64,
        replacement_decision_id="2" * 64,
        reason="moved after signing",
        effective_at="2026-08-18T01:00:00+00:00",
        owner_evidence_ref="2" * 64,
    )
    assert _reason(env, run, authorization) == "supersession_head_witness_refused"


def test_preflight_types_every_window_defect(bounded_env) -> None:
    env = bounded_env
    six = ("2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08")
    run, authorization = env["make_run"](six)
    assert _reason(env, run, authorization) == "sixth_day_refused"
    run, authorization = env["make_run"](("2026-06-05", "2026-06-04"))
    assert _reason(env, run, authorization) == "allowlist_not_chronological"
    run, authorization = env["make_run"](("2026-06-10", "2026-06-11"))
    assert _reason(env, run, authorization) == "protected_or_sealed_date"
    # a physical Sunday partition date is NOT a logical trading day (F-22)
    run, authorization = env["make_run"](("2026-06-04", "2026-06-05", "2026-06-07"))
    assert _reason(env, run, authorization) == "date_domain_mismatch"
    # skipping a logical day breaks consecutiveness
    run, authorization = env["make_run"](("2026-06-04", "2026-06-08"))
    assert _reason(env, run, authorization) == "logical_days_not_consecutive"
    # the ONE canonical program allowlist: a registered different window refuses
    marker = env["store_root"] / "VERIFICATION_ALLOWLIST_MARKER.json"
    marker.write_text(
        json.dumps({"allowlist": ["2026-06-08"], "allowlist_hash": "9" * 64}), encoding="utf-8"
    )
    run, authorization = env["make_run"]()
    assert _reason(env, run, authorization) == "rotated_window_refused"
    marker.unlink()
    # a mapping that does not cover the allowlist is a source-inventory mismatch
    from alpha_lab.agents.data_infra.ifvg.search.trading_calendar import (
        SourcePartitionRef,
        VerificationTradingDayRef,
        physical_partitions_for,
        session_bounds_utc,
    )

    open_ts, close_ts = session_bounds_utc("2026-06-04")
    partial = (
        VerificationTradingDayRef(
            logical_trading_day="2026-06-04",
            session_open_ts_utc=open_ts,
            session_close_ts_utc=close_ts,
            ordered_source_partition_refs=tuple(
                SourcePartitionRef(
                    physical_utc_date=partition.physical_utc_date,
                    relative_logical_partition_key=partition.relative_logical_partition_key,
                    source_kind="legacy_verified_replay_source",
                    content_sha256="7" * 64,
                )
                for partition in physical_partitions_for("2026-06-04")
            ),
        ),
    )
    assert _reason(env, run, authorization, logical_day_refs=partial) == (
        "source_inventory_mismatch"
    )


def test_preflight_binds_the_seed_the_runner_will_load(bounded_env) -> None:
    env = bounded_env
    run, authorization = env["make_run"]()
    # a profile-mismatched seed is refused inside the verified load
    assert _reason(env, run, authorization, baseline_section_config_hash="f" * 64) in (
        "verification_authorization_invalid",  # the run payload's section hash disagrees first
    )
    # a seed that does not continue into the window (first_replay_day ≠ allowlist[0])
    discontinuous = ("2026-06-05", "2026-06-08")
    run, authorization = env["make_run"](discontinuous)
    assert _reason(env, run, authorization) == "seed_discontinuous_with_window"


# ── §6.2 R1 baseline gate report ────────────────────────────────────────────


def _gates(**overrides):
    results = {
        "replay_completed": True,
        "invariants_passed": True,
        "artifacts_published_and_reloaded": True,
        "neutrality_passed": True,
        "verifier_link_resolves": True,
        "zero_forbidden_counters": True,
    }
    results.update(overrides)
    return evaluate_control_flow_gates(results)


def test_r1_baseline_gate_report_derives_passed_and_refuses_inconsistency(tmp_path) -> None:
    root = tmp_path / "store"
    namespace_id, witness = namespace_and_witness(root)
    proofs = {proof: True for proof in R1_BASELINE_PROOF_IDS}
    envelope = build_r1_baseline_gate_report(
        verification_run_id="a" * 64,
        pipeline_semantic_id="b" * 64,
        store_namespace_id=namespace_id,
        supersession_head_witness=witness,
        first_attempt_gates=_gates(),
        second_attempt_gates=_gates(),
        audit_disabled_core_table_hashes_sha256="e" * 64,
        audit_enabled_core_table_hashes_sha256="e" * 64,
        proofs=proofs,
        evidence_refs={"core_replay_id": "f" * 64},
    )
    assert envelope.payload.passed and envelope.payload.audit_modes_equal
    stored, reused = save_r1_baseline_gate_report(root, envelope)
    assert not reused and stored.r1_baseline_gate_report_id == envelope.r1_baseline_gate_report_id
    # any failed proof, gate, or audit-mode divergence fails the report
    failing = build_r1_baseline_gate_report(
        verification_run_id="a" * 64,
        pipeline_semantic_id="b" * 64,
        store_namespace_id=namespace_id,
        supersession_head_witness=witness,
        first_attempt_gates=_gates(),
        second_attempt_gates=_gates(neutrality_passed=False),
        audit_disabled_core_table_hashes_sha256="e" * 64,
        audit_enabled_core_table_hashes_sha256="d" * 64,
        proofs={**proofs, "second_invocation_reused_with_zero_replay": False},
        evidence_refs={},
    )
    assert not failing.payload.passed and not failing.payload.audit_modes_equal
    # a missing proof is refused by the contract; a lying `passed` too
    with pytest.raises(ValueError, match="exactly the registered proof ids"):
        R1BaselineGatePayload.model_validate(
            {**envelope.payload.model_dump(mode="json"), "proofs": (("native_ids_repeat", True),)}
        )
    with pytest.raises(ValueError, match="passed disagrees"):
        R1BaselineGatePayload.model_validate(
            {**failing.payload.model_dump(mode="json"), "passed": True}
        )
    assert envelope.payload.verification_only and envelope.payload.full_pipeline_not_run


# ── §6.3 bounded release control-flow report ────────────────────────────────


@pytest.fixture(scope="module")
def completed_pipeline(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bounded_pipeline")
    fixture = build_pipeline_fixture(tmp_path)
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter

    save_charter(fixture["store_root"], fixture["charter"])
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    state = read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)
    return {"fixture": fixture, "result": result, "state": state}


def test_bounded_report_types_every_component_of_the_synthetic_fixture(completed_pipeline) -> None:
    fixture = completed_pipeline["fixture"]
    state = completed_pipeline["state"]
    root = fixture["store_root"]
    namespace_id, witness = namespace_and_witness(root)
    envelope = build_bounded_release_control_flow_report(
        store_root=root,
        state=state,
        store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
    payload = envelope.payload
    outcomes = {result.component: result.outcome for result in payload.components}
    assert tuple(outcomes) == BOUNDED_COMPONENTS
    assert outcomes["mbp1_diagnostic"] == "not_planned"
    assert outcomes["context_bar_panel"] == "not_planned"
    assert outcomes["fold_construction"] == "typed_no_valid_fold"
    assert outcomes["regime_fit"] == "not_planned"
    assert outcomes["supervised_models"] == "typed_non_fit_zero_predictions"
    assert outcomes["s14_reports"] in (
        "verification_report_zero_fitting",
        "typed_skip_with_evidence",
    )
    assert outcomes["s15_publication"] == "verification_only_result"
    assert outcomes["s11_model_gated_replays"] == "blocked_with_registered_reason"
    assert payload.passed and payload.not_research_evidence and payload.full_pipeline_not_run
    stored, reused = save_bounded_release_control_flow_report(root, envelope)
    assert not reused
    again, reused_again = save_bounded_release_control_flow_report(root, envelope)
    assert reused_again and again.bounded_release_control_flow_report_id == (
        stored.bounded_release_control_flow_report_id
    )
    # the supervised component records zero fabricated predictions
    supervised = next(r for r in payload.components if r.component == "supervised_models")
    assert supervised.evidence["oos_row_count"] == 0
    s11 = next(r for r in payload.components if r.component == "s11_model_gated_replays")
    assert s11.evidence["reason"] == S11_BLOCKED_REASON


def test_bounded_report_fails_closed_on_untyped_states(completed_pipeline) -> None:
    fixture = completed_pipeline["fixture"]
    root = fixture["store_root"]
    namespace_id, witness = namespace_and_witness(root)
    state = json.loads(json.dumps(completed_pipeline["state"]))
    # S11 unblocked with an unregistered explanation is never a passing outcome
    state["stages"]["11_run_frozen_model_gated_replays"]["explanation"] = "ran anyway"
    # an activated publication is not a verification-only result
    state["publication"]["activated"] = True
    envelope = build_bounded_release_control_flow_report(
        store_root=root, state=state, store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
    outcomes = {r.component: r.outcome for r in envelope.payload.components}
    assert outcomes["s11_model_gated_replays"] == "unexpected_state"
    assert outcomes["s15_publication"] == "unexpected_state"
    assert not envelope.payload.passed
    # the contract refuses an unregistered outcome and a lying `passed`
    with pytest.raises(ValueError, match="unregistered outcome"):
        BoundedComponentResult(
            component="regime_fit", outcome="promoted", passed=True, detail="x", evidence={}
        )
    with pytest.raises(ValueError, match="passed disagrees"):
        BoundedComponentResult(
            component="regime_fit", outcome="unexpected_state", passed=True, detail="x",
            evidence={},
        )
    with pytest.raises(ValueError, match="exactly the registered components"):
        BoundedReleaseControlFlowPayload(
            pipeline_semantic_id="b" * 64,
            store_namespace_id=namespace_id,
            supersession_head_witness=witness,
            components=envelope.payload.components[:-1],
            passed=False,
        )


def test_bounded_report_types_a_regime_non_fit_from_the_persisted_record(
    completed_pipeline, monkeypatch
) -> None:
    """A regime study with zero lawful fits on the bounded fixture is the
    typed non-fit capability result — never a fabricated fit."""

    from alpha_lab.agents.data_infra.ifvg.search import bounded_verification as module

    fixture = completed_pipeline["fixture"]
    root = fixture["store_root"]
    namespace_id, witness = namespace_and_witness(root)
    state = json.loads(json.dumps(completed_pipeline["state"]))
    real = module._stage_sidecar

    def _fake(root_, entry, name):
        if name == "regime_run.json":
            return {
                "S09a": {
                    "regime_fit_ids": [],
                    "gates_passed": False,
                    "gate_failures": ["no_valid_folds", "no_oos_assignment_coverage"],
                    "regime_capability_assessment_id": "c" * 64,
                }
            }
        return real(root_, entry, name)

    monkeypatch.setattr(module, "_stage_sidecar", _fake)
    envelope = build_bounded_release_control_flow_report(
        store_root=root, state=state, store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
    regime = next(r for r in envelope.payload.components if r.component == "regime_fit")
    assert regime.outcome == "typed_non_fit_recorded" and regime.passed
    assert regime.evidence["gate_failures"] == ["no_valid_folds", "no_oos_assignment_coverage"]


# ── adversarial fix round (Fix-E): RA-04 / RA-05 / B-01 / B-03 / B-04 ────────


def _make_run_with_seed(env, seed_snapshot_id: str, days=_WINDOW):
    authorization = verification_authorization_ref(
        env["store_root"],
        approved_allowlist_hash=allowlist_sha256(days),
        coverage_matrix_artifact_id="b" * 64,
        seed_snapshot_id=seed_snapshot_id,
    )
    run = VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id="a" * 64,
            verification_authorization=authorization,
            allowlist=tuple(days),
            allowlist_hash=allowlist_sha256(days),
            seed_snapshot_id=seed_snapshot_id,
            baseline_profile_id=_PROFILE,
            baseline_section_config_hash=env["resolved"].section_config_hash,
            coverage_matrix_artifact_id="b" * 64,
        )
    )
    return run, authorization


def test_preflight_types_seed_failures_by_their_cause(bounded_env) -> None:
    """RA-05: a missing or corrupt seed entry is ``seed_snapshot_unverifiable``, a
    profile refusal of the verified loader is ``seed_profile_mismatch``, and any
    other exception propagates untouched (never a wrong typed reason)."""

    from alpha_lab.agents.data_infra.ifvg.search.child_replay import SeedSnapshotError
    from alpha_lab.agents.data_infra.ifvg.search.store import envelope_destination

    env = bounded_env
    # (a) a missing seed entry
    run, authorization = _make_run_with_seed(env, "9" * 64)
    assert _reason(env, run, authorization) == "seed_snapshot_unverifiable"
    # (b) a corrupt seed sidecar (a flipped byte; the manifest hash disagrees)
    run, authorization = env["make_run"]()
    sidecar = (
        envelope_destination(
            env["store_root"], "seed_snapshots", env["snapshot"].seed_snapshot_id
        )
        / "seed.pickle"
    )
    original = sidecar.read_bytes()
    sidecar.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
    try:
        assert _reason(env, run, authorization) == "seed_snapshot_unverifiable"
    finally:
        sidecar.write_bytes(original)
    _preflight(env, run, authorization)  # restored → passes again

    # (c) a profile refusal raised by the verified loader itself
    def _profile_refusal(*_args, **_kwargs):
        raise SeedSnapshotError("seed snapshot is bound to a different profile section hash")

    assert _reason(env, run, authorization, seed_loader=_profile_refusal) == (
        "seed_profile_mismatch"
    )

    # (d) anything else propagates
    def _bug(*_args, **_kwargs):
        raise RuntimeError("programming error")

    with pytest.raises(RuntimeError, match="programming error"):
        _preflight(env, run, authorization, seed_loader=_bug)


def test_fold_outcome_comes_from_the_typed_fold_summary_not_the_explanation(
    completed_pipeline, monkeypatch
) -> None:
    """B-03: the fold outcome is read from the typed ``fold_summary.json``
    stage sidecar; the sanitized explanation is never parsed; neither sidecar
    present is an untyped state."""

    from alpha_lab.agents.data_infra.ifvg.search import bounded_verification as module

    fixture = completed_pipeline["fixture"]
    root = fixture["store_root"]
    namespace_id, witness = namespace_and_witness(root)
    state = json.loads(json.dumps(completed_pipeline["state"]))
    # the explanation LIES; the typed summary decides
    state["stages"]["08_build_folds"]["explanation"] = "7 folds under x; 7 valid"
    envelope = build_bounded_release_control_flow_report(
        store_root=root, state=state, store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
    folds = next(r for r in envelope.payload.components if r.component == "fold_construction")
    assert folds.outcome == "typed_no_valid_fold"
    assert folds.evidence["source"] == "fold_summary.json"
    assert folds.evidence["fold_count"] == 0 and folds.evidence["valid_fold_count"] == 0
    # no typed sidecar at all → unexpected_state (never inferred from text)
    real = module._stage_sidecar

    def _none(root_, entry, name):
        if name in ("fold_summary.json", "fold_sample_adequacy.json"):
            return None
        return real(root_, entry, name)

    monkeypatch.setattr(module, "_stage_sidecar", _none)
    envelope = build_bounded_release_control_flow_report(
        store_root=root, state=state, store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
    folds = next(r for r in envelope.payload.components if r.component == "fold_construction")
    assert folds.outcome == "unexpected_state" and not folds.passed


def test_bounded_report_carries_typed_child_skips_from_the_s14_record(
    completed_pipeline, monkeypatch
) -> None:
    """B-01: the S14 record's ``children_skipped`` reaches the component evidence
    and a record that claims fitting is never a passing outcome."""

    from alpha_lab.agents.data_infra.ifvg.search import bounded_verification as module

    fixture = completed_pipeline["fixture"]
    root = fixture["store_root"]
    namespace_id, witness = namespace_and_witness(root)
    state = completed_pipeline["state"]
    real = module._stage_sidecar
    skipped = {"c" * 64: "executed_trade_table_unavailable"}

    def _with_skip(root_, entry, name):
        if name == "regime_stratified_reports.json":
            return {"children_skipped": skipped, "fitting_performed": False, "report_ids": []}
        return real(root_, entry, name)

    monkeypatch.setattr(module, "_stage_sidecar", _with_skip)
    envelope = build_bounded_release_control_flow_report(
        store_root=root, state=state, store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
    s14 = next(r for r in envelope.payload.components if r.component == "s14_reports")
    assert s14.passed and s14.evidence["children_skipped"] == skipped
    assert s14.evidence["fitting_performed"] is False

    def _fitted(root_, entry, name):
        if name == "regime_stratified_reports.json":
            return {"children_skipped": {}, "fitting_performed": True, "report_ids": []}
        return real(root_, entry, name)

    monkeypatch.setattr(module, "_stage_sidecar", _fitted)
    envelope = build_bounded_release_control_flow_report(
        store_root=root, state=state, store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
    s14 = next(r for r in envelope.payload.components if r.component == "s14_reports")
    assert s14.outcome == "unexpected_state" and not s14.passed


@pytest.fixture(scope="module")
def completed_panel_regime_pipeline(tmp_path_factory):
    """A panel-grain regime study (stratified reporting requested) on the
    synthetic fixture: S05 materializes the persisted panel artifact, S08
    writes the typed fold summary, S14 persists ``regime_stratified_reports.json``."""

    tmp_path = tmp_path_factory.mktemp("bounded_panel_regime")
    fixture = build_pipeline_fixture(tmp_path, regime_study="panel")
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    state = read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)
    return {"fixture": fixture, "result": result, "state": state}


def test_bounded_report_reads_the_real_s14_record_fold_summary_and_panel_artifact(
    completed_panel_regime_pipeline,
) -> None:
    """B-01 / B-03 / B-04 over a REAL regime run: the S14 sidecar under its
    persisted name, the typed fold summary, and the persisted panel artifact's
    typed validity counts (never a frame from memory)."""

    from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
        PANEL_MISSING_REASONS,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import load_json_sidecar

    fixture = completed_panel_regime_pipeline["fixture"]
    state = completed_panel_regime_pipeline["state"]
    root = fixture["store_root"]
    s14 = state["stages"]["14_build_frontier_and_insights"]
    assert s14["status"] in ("completed", "reused"), s14
    record = load_json_sidecar(
        root, "pipeline_stage_results", s14["stage_result_id"], "regime_stratified_reports.json"
    )
    assert isinstance(record, dict) and "children_skipped" in record
    namespace_id, witness = namespace_and_witness(root)
    envelope = build_bounded_release_control_flow_report(
        store_root=root, state=state, store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
    by = {r.component: r for r in envelope.payload.components}
    assert by["s14_reports"].evidence["children_skipped"] == record["children_skipped"]
    assert by["s14_reports"].evidence["fitting_performed"] is False
    assert by["s14_reports"].evidence["report_ids"] == record["report_ids"]
    # B-04: the panel component reads the PERSISTED artifact's typed validity
    panel = by["context_bar_panel"]
    assert panel.outcome in ("panel_materialized", "typed_insufficiency"), panel
    assert panel.evidence["panel_artifact_id"] and panel.evidence["row_count"] > 0
    reasons = panel.evidence["typed_null_reason_counts"]
    assert set(reasons) <= set(PANEL_MISSING_REASONS)
    assert panel.evidence["valid_row_count"] + sum(reasons.values()) == panel.evidence["row_count"]
    # B-03: the fold outcome is the typed summary of the labeled folds
    folds = by["fold_construction"]
    assert folds.evidence["source"] == "fold_summary.json"
    assert folds.evidence["fold_count"] >= folds.evidence["valid_fold_count"]
    assert folds.outcome == (
        "valid_folds_present" if folds.evidence["valid_fold_count"] else "typed_no_valid_fold"
    )
    assert "expected_gate_outcome" in folds.evidence["regime_sample_adequacy"]
    assert by["regime_fit"].outcome == "diagnostic_fit_verification_only"
    assert by["s11_model_gated_replays"].outcome == "blocked_with_registered_reason"


def test_store_behavior_proofs_probe_the_fixture_table_and_detect_a_tampered_source(
    completed_pipeline, tmp_path
) -> None:
    """RA-04: the four immutable-store proofs are GATHERED on the fixture's own
    executed-trade table through scratch copies (identical bytes reuse;
    different bytes, a missing / corrupt manifest and a corrupt sidecar fail
    closed), the scratch root never survives, and an already-tampered source
    is reported as a failed proof."""

    import shutil

    from alpha_lab.agents.data_infra.ifvg.search.bounded_verification import (
        store_behavior_proofs,
    )
    from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
        EXECUTED_TRADE_TABLE_SIDECAR,
        EXECUTED_TRADE_TABLE_STORE,
    )

    fixture = completed_pipeline["fixture"]
    root = fixture["store_root"]
    table_ids = [
        row["executed_trade_table_id"]
        for row in completed_pipeline["state"]["children"]
        if row.get("executed_trade_table_id")
    ]
    assert table_ids
    scratch = tmp_path / "scratch"
    proofs, evidence = store_behavior_proofs(root, table_ids[0], scratch)
    assert proofs == {
        "identical_bytes_reuse": True,
        "different_bytes_fail_closed": True,
        "missing_or_corrupt_manifest_fails_closed": True,
        "corrupt_sidecar_fails_closed": True,
    }
    assert not scratch.exists()
    assert evidence["observations"]["different_bytes"] == "sidecar_hash_mismatch"
    assert evidence["observations"]["missing_manifest"] == "manifest_missing_for_existing_entry"
    assert evidence["observations"]["corrupt_manifest"] == "malformed_manifest"
    assert evidence["observations"]["corrupt_sidecar"] == "sidecar_hash_mismatch"
    # the source itself tampered: the identical-bytes proof must FAIL
    tampered = tmp_path / "tampered_store"
    source_entry = root / EXECUTED_TRADE_TABLE_STORE / table_ids[0]
    target_entry = tampered / EXECUTED_TRADE_TABLE_STORE / table_ids[0]
    shutil.copytree(source_entry, target_entry)
    sidecar = target_entry / EXECUTED_TRADE_TABLE_SIDECAR
    data = sidecar.read_bytes()
    sidecar.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
    proofs2, evidence2 = store_behavior_proofs(tampered, table_ids[0], tmp_path / "scratch2")
    assert proofs2["identical_bytes_reuse"] is False
    assert evidence2["observations"]["identical_bytes"] == "sidecar_hash_mismatch"
    assert not (tmp_path / "scratch2").exists()
