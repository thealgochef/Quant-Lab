"""Approval reviews inspect disposable cache metadata and never launch a replay."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from strategy_core.candles._buckets import HTF_ANCHOR_POLICY

from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.day_artifacts import _META_KEY, DaySeeds
from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search import strategy_approval_review as service
from alpha_lab.agents.data_infra.ifvg.search.authorization import derive_authorization_requirements
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    registry_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import DatePolicy, _example_charter_payload
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_store_namespace
from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import STORE, charter_intent
from alpha_lab.agents.data_infra.ifvg.study.computation_path import ComputationPath
from alpha_lab.agents.data_infra.ifvg.study_providers import owner_authorization_readiness

AXIS = "parent_retest_timeout_1m_bars"


@pytest.fixture
def request_fixture(tmp_path, monkeypatch):
    root = tmp_path / "research"
    initialize_store_namespace(root, namespace_class="research", store_instance_id="a" * 32)
    baseline = resolve_profile_config()
    payload = _example_charter_payload()
    payload = payload.model_copy(
        update={
            "baseline_section_config_hash": baseline.section_config_hash,
            "locked_invariants_registry_sha256": registry_sha256(),
            "axes": {AXIS: (f"{AXIS}.none", f"{AXIS}.60")},
            "max_child_count": 2,
            "date_policy": DatePolicy(
                replay_dates=(*FROZEN_WARMUP_DATES, "2026-01-13"),
                warmup_dates=FROZEN_WARMUP_DATES,
                access_policy_id="development_explicit_dates_before_path_v2",
            ),
            "objective_policy": payload.objective_policy.model_copy(
                update={
                    "pareto_objectives": ("net_expectancy_r",),
                    # one evaluated date: the threshold must be reachable (R5)
                    "feasibility_gates": payload.objective_policy.feasibility_gates.model_copy(
                        update={"min_independent_days": 1}
                    ),
                }
            ),
        }
    )
    fields = charter_intent(payload)
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
    cfg = IfvgCaptureConfig(data_dir=tmp_path / "cache")
    monkeypatch.setattr(service, "IfvgCaptureConfig", lambda **kw: replace(cfg, **kw))
    monkeypatch.setattr(service, "_check_source_commits", lambda _intent: None)
    dates = payload.date_policy.replay_dates
    for day in dates:
        _write_metadata_pair(cfg, day, dates)
    return root, fields, requirements, cfg


def _write_metadata_pair(cfg, day, dates, **changes):
    metadata = {
        "artifact_schema_version": 2,
        "artifacts_tag": cfg.artifacts_tag(),
        "anchor_policy": HTF_ANCHOR_POLICY,
        "source_access_policy": "explicit_allowlist_before_path_v1",
        "source_allowlist_sha256": allowlist_sha256(dates),
        "seeds": DaySeeds(None, None, None, None).meta(),
        "day_hl": None,
        "ny_hl": None,
        "warnings": [],
        **changes,
    }
    table = pa.table({"fixture": [1]}).replace_schema_metadata(
        {_META_KEY: json.dumps(metadata).encode()}
    )
    for factory in (cfg.bars_path, cfg.levels_path):
        path = factory(day)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


def _prepare(fixture):
    root, fields, requirements, _cfg = fixture
    return service.prepare_strategy_approval_review(fields, requirements, root)


def _record(fixture, review, **changes):
    root, fields, requirements, _cfg = fixture
    return service.record_strategy_approval(
        review,
        current_fields=changes.get("current_fields", fields),
        requirement_set=requirements,
        store_root=root,
        author=changes.get("author", "Fixture owner"),
        approval_statement="Approve this exact disposable fixture only.",
        approved_at=datetime.now(UTC).isoformat(),
    )


def test_review_checks_metadata_without_rows_or_publication(request_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("review must never read rows, publish approval, or launch work")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(service, "persist_strategy_approval", forbidden)
    review = _prepare(request_fixture)
    assert review.blockers == ()
    assert review.cache_checked_dates == 11
    assert len(review.configuration_rows) == 2
    assert not (request_fixture[0] / STORE).exists()
    assert not (request_fixture[0] / "charters").exists()
    assert AXIS_VALUE_REGISTRY_V1[f"{AXIS}.60"].owner_ratification_status == "pending"


def test_ny_morning_configuration_can_be_reviewed_without_changing_day_artifacts(request_fixture):
    root, fields, _requirements, cfg = request_fixture
    axis = "enabled_entry_sessions"
    fields = {
        **fields,
        "axes": {axis: (f"{axis}.asia-london-ny", f"{axis}.ny_0700_1030")},
    }
    requirements = derive_authorization_requirements(
        "full_authorized_development",
        (f"strategy_profile.{axis}",),
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
    review = service.prepare_strategy_approval_review(fields, requirements, root)
    assert review.blockers == ()
    assert len(review.configuration_rows) == 2
    child = next(row for row in review.configuration_rows if row["comparison_role"] == "challenger")
    assert child["section_overrides"] == {
        "enabled_entry_sessions": ["ny_0700_1030"],
        "doc_sessions": {"ny_0700_1030": ["07:00", "10:30"]},
    }
    resolved = resolve_profile_config({"section_overrides": child["section_overrides"]})
    assert IfvgCaptureConfig(section=resolved.section).artifacts_tag() == cfg.artifacts_tag()
    assert not (root / STORE).exists()


def test_explicit_record_enables_exact_request_without_launch(request_fixture, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.search import strategy_executor

    def forbidden(*args, **kwargs):
        pytest.fail("recording approval must never replay")

    monkeypatch.setattr(strategy_executor, "run_child_replay", forbidden)
    root, fields, requirements, _cfg = request_fixture
    review = _prepare(request_fixture)
    saved = _record(request_fixture, review)
    assert saved.payload.charter_intent_sha256 == review.intent_sha256
    assert (
        owner_authorization_readiness(
            root, requirements, charter_intent_sha256=review.intent_sha256
        ).status
        == "ready"
    )
    assert (
        owner_authorization_readiness(
            root,
            requirements,
            charter_intent_sha256=service.charter_intent_hash({**fields, "seed": 9}),
        ).status
        == "missing"
    )
    assert not (root / "charters").exists()
    assert not (root / "core_replays").exists()


def test_single_configuration_exact_approval(request_fixture):
    root, fields, requirements, cfg = request_fixture
    fields = {
        **fields,
        "search_mode": "single_configuration",
        "axes": {AXIS: [f"{AXIS}.240"]},
        "max_child_count": 1,
    }
    fixture = (root, fields, requirements, cfg)
    review = _prepare(fixture)
    assert review.blockers == ()
    assert len(review.configuration_rows) == 1
    saved = _record(fixture, review)
    assert json.loads(saved.payload.approved_charter_json)["axes"] == fields["axes"]


def test_single_configuration_refuses_multiple_values(request_fixture):
    root, fields, requirements, cfg = request_fixture
    fields = {**fields, "search_mode": "single_configuration"}
    review = _prepare((root, fields, requirements, cfg))
    assert any("exactly one configuration" in value for value in review.blockers)


def test_single_configuration_payload_also_refuses_multiple_children(request_fixture):
    from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import (
        StrategySearchApprovalPayload,
    )

    saved = _record(request_fixture, _prepare(request_fixture))
    content = json.loads(saved.payload.approved_charter_json)
    content["search_mode"] = "single_configuration"
    payload = saved.payload.model_dump(mode="json")
    payload["approved_charter_json"] = json.dumps(content)
    payload["charter_intent_sha256"] = service.charter_intent_hash(content)
    with pytest.raises(ValueError, match="exactly one configuration"):
        StrategySearchApprovalPayload.model_validate(payload)


def test_cross_store_preparation_evidence_does_not_reuse_authority(request_fixture, tmp_path):
    source_root, fields, requirements, _cfg = request_fixture
    saved = _record(request_fixture, _prepare(request_fixture))
    destination = tmp_path / "separate_research"
    initialize_store_namespace(destination, namespace_class="research", store_instance_id="b" * 32)
    review = service.prepare_strategy_approval_review(
        fields,
        requirements,
        destination,
        preparation_approval=(source_root, saved.strategy_search_approval_id),
    )
    assert review.blockers == ()
    assert review.artifact_provenance_dates == saved.payload.artifact_provenance_dates
    assert not (destination / STORE).exists()
    assert owner_authorization_readiness(
        destination, requirements, charter_intent_sha256=review.intent_sha256
    ).status != "ready"


def test_cross_store_preparation_refuses_unverified_approval(request_fixture):
    root, fields, requirements, _cfg = request_fixture
    review = service.prepare_strategy_approval_review(
        fields, requirements, root, preparation_approval=(root, "f" * 64)
    )
    assert review.blockers


def test_dedicated_ui_preparation_reference_is_verified_before_use(request_fixture, monkeypatch):
    root, _fields, _requirements, _cfg = request_fixture
    monkeypatch.setattr(service, "research_preparation_approval", lambda _repo: (root, "f" * 64))
    assert _prepare(request_fixture).blockers


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_section_config_hash", "f" * 64),
        ("locked_invariants_registry_sha256", "f" * 64),
        ("axes", {AXIS: (f"{AXIS}.none", "opposing_timeout_1m_bars.60")}),
        ("authorized_firm_contract_ids", ["unexpected-firm"]),
        ("max_child_count", 1),
    ],
)
def test_invalid_plan_refuses_before_cache_access(request_fixture, monkeypatch, field, value):
    root, fields, requirements, _cfg = request_fixture

    def forbidden(*args, **kwargs):
        pytest.fail("invalid scope must fail before cache access")

    monkeypatch.setattr(service, "_cache_review", forbidden)
    review = service.prepare_strategy_approval_review({**fields, field: value}, requirements, root)
    assert review.blockers
    assert not (root / STORE).exists()


def test_changed_review_refuses_before_saving(request_fixture):
    review = _prepare(request_fixture)
    with pytest.raises(ValueError, match="changed"):
        _record(request_fixture, review, current_fields={**request_fixture[1], "seed": 17})
    assert not (request_fixture[0] / STORE).exists()


def test_cache_change_invalidates_confirmation(request_fixture):
    root, fields, _requirements, cfg = request_fixture
    review = _prepare(request_fixture)
    dates = fields["date_policy"]["replay_dates"]
    _write_metadata_pair(cfg, dates[-1], dates, warnings=["Changed since review"])
    with pytest.raises(ValueError, match="changed"):
        _record(request_fixture, review)
    assert not (root / STORE).exists()


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"artifact_schema_version": 1}, "incompatible"),
        ({"source_allowlist_sha256": "f" * 64}, "unrecognized"),
        ({"seeds": {}}, "day chain"),
    ],
)
def test_bad_cache_stamps_block(request_fixture, changes, expected):
    _root, fields, _requirements, cfg = request_fixture
    dates = fields["date_policy"]["replay_dates"]
    _write_metadata_pair(cfg, dates[0], dates, **changes)
    review = _prepare(request_fixture)
    assert any(expected in message for message in review.blockers)


def test_missing_cache_and_blank_owner_are_actionable(request_fixture):
    review = _prepare(request_fixture)
    with pytest.raises(ValueError, match="owner"):
        _record(request_fixture, review, author="   ")
    request_fixture[3].bars_path(FROZEN_WARMUP_DATES[0]).unlink()
    assert "missing" in _prepare(request_fixture).blockers[0]


def test_wider_verified_cache_provenance_is_reused_for_narrower_scope(request_fixture):
    root, fields, requirements, cfg = request_fixture
    wide_dates = (*fields["date_policy"]["replay_dates"], "2026-01-14")
    wider = {**fields, "date_policy": {**fields["date_policy"], "replay_dates": wide_dates}}
    for day in wide_dates:
        _write_metadata_pair(cfg, day, wide_dates)
    wide_review = service.prepare_strategy_approval_review(wider, requirements, root)
    assert not wide_review.blockers
    _record(request_fixture, wide_review, current_fields=wider)
    narrow_review = _prepare(request_fixture)
    assert not narrow_review.blockers
    assert narrow_review.artifact_provenance_dates == wide_dates
    assert narrow_review.cache_checked_dates == 11


def test_nonpermitted_date_refuses_before_cache_path(request_fixture, monkeypatch):
    root, fields, requirements, _cfg = request_fixture
    changed = {
        **fields,
        "date_policy": {
            **fields["date_policy"],
            "replay_dates": (*fields["date_policy"]["replay_dates"], "2026-06-12"),
        },
    }
    monkeypatch.setattr(service, "_cache_review", lambda *a: pytest.fail("no cache path permitted"))
    assert service.prepare_strategy_approval_review(changed, requirements, root).blockers


def test_actual_source_commit_query_is_read_only_and_rejects_unknown():
    with pytest.raises(ValueError, match="revision is unavailable"):
        service._check_source_commits({"quant_lab_commit": "unknown", "strategy_core_commit": ""})


def test_parent_and_opposing_comparison_resolves_all_eight_configs(request_fixture):
    root, fields, _requirements, _cfg = request_fixture
    opposing = "opposing_timeout_1m_bars"
    fields = {
        **fields,
        "axes": {
            AXIS: tuple(f"{AXIS}.{value}" for value in ("none", "240", "360", "480")),
            opposing: (f"{opposing}.none", f"{opposing}.90"),
        },
        "max_child_count": 8,
    }
    requirements = derive_authorization_requirements(
        "full_authorized_development",
        tuple(f"strategy_profile.{axis}" for axis in sorted(fields["axes"])),
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
    review = service.prepare_strategy_approval_review(fields, requirements, root)
    assert not review.blockers
    assert len(review.configuration_rows) == 8
    assert len({row["resolved_section_config_hash"] for row in review.configuration_rows}) == 8
    assert sum(row["comparison_role"] == "baseline" for row in review.configuration_rows) == 1


def test_metadata_seed_chain_carries_day_and_ny_extrema(request_fixture):
    _root, fields, _requirements, cfg = request_fixture
    dates = fields["date_policy"]["replay_dates"]
    _write_metadata_pair(cfg, dates[0], dates, day_hl=[200, 100], ny_hl=[190, 110])
    seeds = {
        "prev_day": dates[0],
        "prev_full_hl": [200, 100],
        "prev_ny_day": dates[0],
        "prev_ny_hl": [190, 110],
    }
    for day in dates[1:]:
        _write_metadata_pair(cfg, day, dates, seeds=seeds)
    assert not _prepare(request_fixture).blockers
    _write_metadata_pair(cfg, dates[-1], dates, seeds={**seeds, "prev_day": dates[-2]})
    assert "day chain" in _prepare(request_fixture).blockers[0]


def test_extended_research_dates_cannot_be_approved_before_their_inputs_exist(request_fixture):
    """Repair R8: the window is extended, but pre-2026 inputs are not prepared yet."""

    from alpha_lab.agents.data_infra.ifvg.research_period import warmup_dates_for

    root, fields, requirements, _cfg = request_fixture
    warmup = warmup_dates_for("2025-06-13")
    extended = {**fields, "date_policy": {**fields["date_policy"],
                                          "replay_dates": (*warmup, "2025-06-13"),
                                          "warmup_dates": warmup}}
    review = service.prepare_strategy_approval_review(extended, requirements, root)
    assert any("have not been prepared" in blocker for blocker in review.blockers)
    assert not (root / STORE).exists() or not any((root / STORE).iterdir())
