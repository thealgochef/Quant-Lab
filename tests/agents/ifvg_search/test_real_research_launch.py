"""Real launch authorization/service tests using temporary source metadata only."""

import json
from dataclasses import asdict

import pytest

from alpha_lab.agents.data_infra.ifvg.search import (
    research_executor,
    research_runs,
    research_subject,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import ChildSpec
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    PipelineSemanticIdentity,
    _ensure_specs,
    _initial_state,
    _RunContext,
    _stage_s00_validate,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    load_verified_envelope,
    save_or_reuse_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_store_namespace
from tests.agents.ifvg_search.test_orchestrator import _charter
from tests.agents.ifvg_search.test_research_subject_data import subject_fixture


@pytest.fixture
def sources(tmp_path, monkeypatch):
    initialize_store_namespace(tmp_path, namespace_class="research", store_instance_id="b" * 32)
    subject = subject_fixture()
    original = _charter()
    original = SearchCharterEnvelope.from_payload(
        original.payload.model_copy(
            update={
                "date_policy": original.payload.date_policy.model_copy(
                    update={
                        "replay_dates": subject.replay_dates,
                        "warmup_dates": subject.warmup_dates,
                        "access_policy_id": "development_explicit_dates_before_path_v2",
                    }
                ),
            }
        )
    )
    # Metadata-only fixture; production original subjects are independently verified by binder.
    save_or_reuse_envelope(tmp_path, "charters", original)
    child = ChildSpec(
        1, {}, "baseline", "fixture", subject.section_config_hash, None, subject.section_mapping
    )
    subject = subject.model_copy(
        update={
            "original_search_id": original.search_id,
            "child_spec_json": json.dumps(asdict(child)),
        }
    )

    def bind(root, core_id, *, evaluation_dates=None, cohort="all_candidates"):
        assert core_id == subject.core_replay_id
        return subject.model_copy(
            update={
                "evaluation_dates": evaluation_dates or subject.evaluation_dates,
                "cohort": cohort,
            }
        )

    monkeypatch.setattr(research_subject, "bind_research_subject", bind)
    monkeypatch.setattr(research_subject, "preflight_research_subject", lambda *a: {"passed": True})
    monkeypatch.setattr(
        research_executor, "preflight_research_subject", lambda *a: {"passed": True}
    )
    software = {
        "quant_lab": "a" * 40,
        "strategy_core": "b" * 40,
        "quant_lab_research_source": "c" * 64,
        "strategy_core_source": "d" * 64,
    }
    monkeypatch.setattr(research_runs, "_software_identity", lambda: software)
    monkeypatch.setattr(research_executor, "_software_identity", lambda: software)
    monkeypatch.setattr(
        research_executor.ResearchPreparation,
        "prepare",
        lambda *a: pytest.fail("preflight/freeze/factory must never prepare data"),
    )
    monkeypatch.setattr(
        research_runs.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    return subject


def frozen(tmp_path, subject):
    preflight = research_runs.build_research_preflight(
        tmp_path,
        [subject.core_replay_id],
        {
            "display_name": "Temporary metadata research",
            "evaluation_start": "2026-01-13",
            "evaluation_end": "2026-01-14",
            "base_bundle_key": "B4_CORE_STRUCTURE_LIQUIDITY",
        },
    )
    assert preflight["ready"], preflight["blockers"]
    result = research_runs.freeze_research_group(
        tmp_path,
        preflight,
        authorization_statement="Fixture owner approves this exact plan",
        author="Fixture owner",
    )
    semantic = load_verified_envelope(
        tmp_path,
        "pipeline_specs",
        result["cells"][0]["pipeline_semantic_id"],
        PipelineSemanticIdentity,
    )
    charter = load_verified_envelope(
        tmp_path, "charters", semantic.payload.search_charter_id, SearchCharterEnvelope
    )
    return preflight, result, semantic, charter


def test_freeze_exact_real_source_reaches_s00_without_replay_or_fitting(tmp_path, sources):
    preflight, group, semantic, charter = frozen(tmp_path, sources)
    wiring = research_executor.pipeline_real_research_entry(charter, semantic, store_root=tmp_path)
    context = _RunContext(
        semantic, charter, wiring, tmp_path, tmp_path / "state", _initial_state(semantic)
    )
    _stage_s00_validate(context)
    _ensure_specs(context)
    assert len(context.specs) == 1
    assert context.specs[0].section_overrides == sources.section_mapping
    assert context.state["full_pipeline_not_run"] is True
    assert semantic.payload.label_policy_id == research_runs.REAL_LABEL_POLICY_ID
    assert preflight["subjects"][0]["cutoff_ts_utc"] == sources.cutoff_ts_utc
    assert research_runs.read_research_group(tmp_path, group["group_id"], tmp_path / "state")[
        "cells"
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"label_policy_id": "synthetic_fixture_labels_v1"},
        {"fold_protocol_id": "different_fold_protocol"},
        {"warmup_policy_id": "warmup_drop_disabled"},
        {"source_artifact_ids": ()},
        {"cost_policy_sha256": "f" * 64},
    ],
)
def test_worker_rejects_modified_frozen_semantics_before_sources(tmp_path, sources, change):
    _, _, semantic, charter = frozen(tmp_path, sources)
    altered = PipelineSemanticIdentity.from_payload(semantic.payload.model_copy(update=change))
    with pytest.raises(PermissionError, match="approved subject|authorized research"):
        research_executor.pipeline_real_research_entry(charter, altered, store_root=tmp_path)


def test_changed_sources_require_new_review_without_publishing_approval(
    tmp_path, sources, monkeypatch
):
    preflight = research_runs.build_research_preflight(
        tmp_path,
        [sources.core_replay_id],
        {
            "display_name": "Changed code",
            "evaluation_start": "2026-01-13",
            "evaluation_end": "2026-01-14",
        },
    )
    monkeypatch.setattr(
        research_runs,
        "_software_identity",
        lambda: {"quant_lab": "f" * 40, "strategy_core": "b" * 40},
    )
    with pytest.raises(ValueError, match="changed after review"):
        research_runs.freeze_research_group(
            tmp_path, preflight, author="Reviewer", authorization_statement="Reviewed"
        )
    assert not (tmp_path / "research_approvals").exists()


def test_cancel_survives_worker_start_without_constructing_factory(tmp_path, sources, monkeypatch):
    _, group, _, _ = frozen(tmp_path, sources)
    state = tmp_path / "state"
    research_runs.cancel_research_group(tmp_path, group["group_id"], state)
    monkeypatch.setattr(
        research_executor,
        "pipeline_real_research_entry",
        lambda *a, **kw: pytest.fail("cancelled group reached a child"),
    )
    result = research_runs.run_research_group(tmp_path, group["group_id"], state)
    assert result["status"] == "cancelled_at_safe_boundary"


def test_worker_startup_authority_failure_is_visible_for_recovery(tmp_path, sources, monkeypatch):
    _, group, _, _ = frozen(tmp_path, sources)
    state = tmp_path / "state"
    def refused(*args):
        raise PermissionError("research authority changed")
    monkeypatch.setattr(research_runs, "load_research_authorization", refused)
    outcome = research_runs.run_research_group(tmp_path, group["group_id"], state)
    assert outcome["status"] == "failed"
    assert "authority changed" in outcome["reason"]
    restored = research_runs.read_research_group(tmp_path, group["group_id"], state)
    assert restored["status"] == "failed"
    assert restored["cells"][0]["status"] == "queued"
