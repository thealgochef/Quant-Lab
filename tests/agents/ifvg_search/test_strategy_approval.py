"""Real-scope approval tests use disposable stores and never replay real data."""

import json
from datetime import UTC, datetime

import pytest

from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
from alpha_lab.agents.data_infra.ifvg.search.authorization import derive_authorization_requirements
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    registry_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.catalog import append_catalog_event
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    DatePolicy,
    SearchCharterEnvelope,
    _example_charter_payload,
    save_charter,
    validate_charter,
)
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import enumerate_children
from alpha_lab.agents.data_infra.ifvg.search.runner_registry import runner_entry_key_for_charter
from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_store_namespace
from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import (
    STORE,
    StrategySearchApprovalEnvelope,
    StrategySearchApprovalPayload,
    charter_intent,
    charter_intent_hash,
    load_strategy_approval,
    persist_strategy_approval,
)
from alpha_lab.agents.data_infra.ifvg.search.strategy_executor import (
    search_strategy_development_entry,
)
from alpha_lab.agents.data_infra.ifvg.study.computation_path import ComputationPath
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    owner_authorization_bundle_from_store,
    owner_authorization_readiness,
)

NOW = "2026-09-08T00:00:00+00:00"
AXIS = "parent_retest_timeout_1m_bars"


@pytest.fixture
def approved(tmp_path):
    root = tmp_path / "research"
    namespace = initialize_store_namespace(
        root, namespace_class="research", store_instance_id="a" * 32
    )
    example = _example_charter_payload()
    payload = example.model_copy(
        update={
            "objective_policy": example.objective_policy.model_copy(
                update={"pareto_objectives": ("net_expectancy_r",)}
            ),
            "axes": {AXIS: tuple(f"{AXIS}.{v}" for v in ("none", "240", "360", "480"))},
            "locked_invariants_registry_sha256": registry_sha256(),
            "max_child_count": 4,
            "date_policy": DatePolicy(
                replay_dates=(*FROZEN_WARMUP_DATES, "2026-01-13"),
                warmup_dates=FROZEN_WARMUP_DATES,
                access_policy_id="development_explicit_dates_before_path_v2",
            ),
        }
    )
    requirements = derive_authorization_requirements(
        "full_authorized_development",
        (f"strategy_profile.{AXIS}",),
        ComputationPath(
            full_strategy_replay=True,
            feature_materialization=False,
            label_recomputation=False,
            model_refit=False,
            model_gated_sequential_replay=False,
            cost_recomputation=True,
            prop_resimulation=False,
            bootstrap_resimulation=False,
            reuse_trade_stream_hash=False,
        ),
        (),
        (),
    )
    approval = StrategySearchApprovalEnvelope.from_payload(
        StrategySearchApprovalPayload(
            store_namespace_id=namespace.store_namespace_id,
            requirement_set_id=requirements.requirement_set_id,
            charter_intent_sha256=charter_intent_hash(payload),
            approved_charter_json=json.dumps(charter_intent(payload), sort_keys=True),
            artifact_provenance_dates=tuple(payload.date_policy.replay_dates),
            author="test owner",
            approved_at=NOW,
            effective_from=NOW,
            reviewed_evidence_refs=("test approval request",),
            approval_statement="Approve this exact disposable test configuration; no real replay.",
        )
    )
    persist_strategy_approval(root, approval)
    append_catalog_event(
        root,
        kind="display_name",
        artifact_id=approval.strategy_search_approval_id,
        payload={"display_name": "test approval"},
    )
    bundle = owner_authorization_bundle_from_store(
        root, requirements, charter_intent_sha256=charter_intent_hash(payload)
    )
    assert bundle is not None
    payload = payload.model_copy(update={"owner_authorization": bundle})
    payload = type(payload).model_validate(payload.model_dump(mode="json"))
    return root, payload, approval, requirements


def test_exact_approval_enables_four_children_without_global_ratification(approved):
    root, payload, approval, requirements = approved
    ready = owner_authorization_readiness(
        root, requirements, charter_intent_sha256=charter_intent_hash(payload)
    )
    assert ready.status == "ready"
    validate_charter(payload, as_of_utc=NOW, store_root=root)
    charter = SearchCharterEnvelope.from_payload(payload)
    children = enumerate_children(
        charter, store_root=root, identity_resolver=lambda spec: spec.resolved_section_config_hash
    )
    assert len(children) == 4
    assert [child.comparison_role for child in children] == ["baseline"] + ["challenger"] * 3
    assert all(child.capability.status == "generated_runnable" for child in children[1:])
    assert AXIS_VALUE_REGISTRY_V1[f"{AXIS}.240"].owner_ratification_status == "pending"
    assert load_strategy_approval(root, approval.strategy_search_approval_id) == approval
    save_charter(root, charter)
    assert (
        load_verified_envelope(root, "charters", charter.search_id, SearchCharterEnvelope)
        == charter
    )
    assert runner_entry_key_for_charter(charter) == "search_strategy_development_v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 19),
        ("max_child_count", 12),
        ("axes", {AXIS: (f"{AXIS}.none", f"{AXIS}.240")}),
        ("baseline_profile_name", "ifvg_v2_doc_default_retest_static_1r"),
    ],
)
def test_changed_request_cannot_use_previous_approval(approved, field, value):
    root, payload, _, _ = approved
    changed = payload.model_copy(update={field: value})
    with pytest.raises((ValueError, PermissionError)):
        validate_charter(changed, as_of_utc=NOW, store_root=root)
    with pytest.raises((ValueError, PermissionError)):
        search_strategy_development_entry(
            SearchCharterEnvelope.from_payload(changed), store_root=root
        )


def test_wrong_scope_lookup_and_no_scope_lookup_do_not_unlock(approved):
    root, _, _, requirements = approved
    for intent in (None, "f" * 64):
        assert (
            owner_authorization_readiness(root, requirements, charter_intent_sha256=intent).status
            == "missing"
        )


def test_factory_and_selection_do_not_read_data_or_run(approved, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.search import strategy_executor as executor

    def forbidden(*args, **kwargs):
        pytest.fail("opening/preparing the approved study must never read data or run a replay")

    for name in ("load_day_artifacts", "run_child_replay", "file_sha256"):
        monkeypatch.setattr(executor, name, forbidden)
    root, payload, _, _ = approved
    wiring = search_strategy_development_entry(
        SearchCharterEnvelope.from_payload(payload), store_root=root
    )
    assert set(wiring) == {"identity_resolver", "child_runner"}
    assert not (root / "core_replays").exists()
    assert not (root / "charters").exists()


def test_corrupt_or_missing_approval_refuses_before_data_access(approved, monkeypatch):
    root, payload, approval, requirements = approved
    path = root / STORE / approval.strategy_search_approval_id / "envelope.json"
    path.write_text("{}")
    assert (
        owner_authorization_readiness(
            root, requirements, charter_intent_sha256=charter_intent_hash(payload)
        ).status
        != "ready"
    )
    with pytest.raises((ValueError, PermissionError, RuntimeError)):
        search_strategy_development_entry(
            SearchCharterEnvelope.from_payload(payload), store_root=root
        )


def test_copying_approval_to_other_namespace_is_refused(approved, tmp_path):
    root, _, approval, _ = approved
    other = tmp_path / "other"
    initialize_store_namespace(other, namespace_class="research", store_instance_id="b" * 32)
    with pytest.raises(PermissionError, match="another namespace"):
        persist_strategy_approval(other, approval)
    assert load_strategy_approval(root, approval.strategy_search_approval_id)


def test_approval_is_idempotent_and_time_bound(approved):
    root, _, approval, _ = approved
    saved, reused = persist_strategy_approval(root, approval)
    assert reused and saved == approval
    with pytest.raises(PermissionError, match="not yet effective"):
        load_strategy_approval(
            root, approval.strategy_search_approval_id, as_of_utc="2026-09-07T00:00:00+00:00"
        )
    assert datetime.fromisoformat(NOW) <= datetime.now(UTC)


def test_registered_worker_wiring_completes_and_reuses_four_fixture_children(approved, monkeypatch):
    """Exercise dispatch, identities, metrics and resume with fixture data only."""
    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
    from alpha_lab.agents.data_infra.ifvg.search import strategy_executor as executor
    from alpha_lab.agents.data_infra.ifvg.search.child_replay import ChildReplayResult
    from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import run_search
    from tests.agents.ifvg_search.conftest import make_resolved_trades_frame

    root, payload, _, _ = approved
    calls = []

    def artifacts(day, cfg, *, expected_seeds, access_policy):
        access_policy.authorize_date(day)
        return SimpleNamespace(date_str=day, day_hl=None, seeds=expected_seeds)

    def replay(**kwargs):
        calls.append(kwargs)
        assert kwargs["cached_artifacts_only"] and kwargs["dual_drive"]
        assert kwargs["cfg"].warmup_days == 10
        assert kwargs["cfg"].profile_hash == kwargs["resolved_profile"].section_config_hash
        policy = kwargs["access_policy_factory"]()
        assert policy.inner.allowlist == frozenset(payload.date_policy.replay_dates)
        tables = {RecordTable.EXECUTED_TRADE: make_resolved_trades_frame(("2026-01-13",))}
        return ChildReplayResult(
            resolved_profile=kwargs["resolved_profile"],
            capture=SimpleNamespace(tables=tables),
            audit_capture=None,
            neutrality=SimpleNamespace(passed=True),
            gross_trade_stream_hash=canonical_contract_sha256(
                {"fixture": kwargs["core_replay_id"]}
            ),
        )

    def publish(**kwargs):
        assert kwargs["warmup_days"] == 10
        assert kwargs["replay_dates"] == tuple(payload.date_policy.replay_dates)
        assert kwargs["cost_points"] == payload.cost_policy.cost_points_round_turn
        return {"invariants_passed": True, "core_replay_artifact_reference": {"fixture": True}}

    monkeypatch.setattr(executor, "load_day_artifacts", artifacts)
    monkeypatch.setattr(executor, "file_sha256", lambda path: "d" * 64)
    monkeypatch.setattr(executor, "run_child_replay", replay)
    monkeypatch.setattr(executor, "build_child_companions", publish)
    monkeypatch.setattr(executor, "read_repository_state", lambda *a, **kw: None)
    monkeypatch.setattr(
        executor, "strategy_core_source_identity", lambda **kw: ("a" * 40, "a" * 64)
    )
    monkeypatch.setattr(executor, "quant_lab_replay_source_identity", lambda **kw: "b" * 64)
    charter = SearchCharterEnvelope.from_payload(payload)
    wiring = search_strategy_development_entry(charter, store_root=root)
    first = run_search(charter, store_root=root, state_root=root / "state", **wiring)
    assert first.phase == "search_complete"
    assert len(calls) == 4 and all(child.state == "completed" for child in first.children)
    assert {call["cfg"].section.parent_retest_timeout_1m_bars for call in calls} == {
        None,
        240,
        360,
        480,
    }
    resumed = run_search(charter, store_root=root, state_root=root / "state", **wiring)
    assert resumed.phase == "search_complete"
    assert len(calls) == 4 and all(child.state == "reused" for child in resumed.children)


def test_input_verification_saves_startup_then_failure_without_replay(approved):
    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import read_search_state, run_search

    root, payload, _, _ = approved
    charter = SearchCharterEnvelope.from_payload(payload)
    state_root = root / "state"

    def unavailable(spec):
        state = read_search_state(state_root, charter.search_id)
        assert state["phase"] == "charter_frozen" and state["children"] == []
        raise PermissionError("required day artifact is unavailable")

    def never_run(**kwargs):
        pytest.fail("input failure must never reach a child replay")

    with pytest.raises(PermissionError, match="unavailable"):
        run_search(
            charter,
            store_root=root,
            state_root=state_root,
            identity_resolver=unavailable,
            child_runner=never_run,
        )
    failed = read_search_state(state_root, charter.search_id)
    assert failed["phase"] == "failed" and failed["children"] == []
    assert "unavailable" in failed["phase_notes"]["input_verification"]
