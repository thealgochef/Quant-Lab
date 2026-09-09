"""Strategy descriptions are bound to saved JSON, without replay/data reads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.artifact_io import (
    ArtifactVerificationError,
    load_verified_v2_configuration,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    IfvgContextExperimentConfig,
)
from alpha_lab.agents.data_infra.ifvg.context_run_store import (
    ImmutableContextStoreError,
    load_context_experiment_configuration,
)
from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256
from alpha_lab.agents.data_infra.ifvg.presentation import study_rules
from alpha_lab.agents.data_infra.ifvg.presentation.workspace import StudySummary
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import registry_sha256
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    SearchCharterEnvelope,
    _example_charter_payload,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    SearchChildMembership,
    _example_core_replay_payload,
)
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import SearchChildMembershipEnvelope
from alpha_lab.agents.data_infra.ifvg.search.store import save_envelope_immutable
from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft


def _bytes(value):
    return json.dumps(value, sort_keys=True).encode()


def _study(**changes):
    return StudySummary(
        **{
            "key": "study",
            "kind": "draft",
            "name": "An intentionally misleading name",
            "question": "Research question",
            "dates": "Dates",
            "status": "Draft",
            "scope": "research",
            **changes,
        }
    )


def _draft(mode="single_configuration"):
    draft = new_draft(mode)
    if mode == "single_configuration":
        draft.steps["objective"] = {"question_id": "evaluate_one_configuration"}
    draft.steps["baseline"] = {"baseline_profile_name": "ifvg_v2_doc_default_fresh_static_1r"}
    draft.steps["validation"] = {"run_scope": "verification_5d"}
    draft.steps["search_space"] = {
        "axis_selections": {"parent_retest_timeout_1m_bars": ["parent_retest_timeout_1m_bars.480"]}
    }
    return draft


def _dataset(root: Path, section):
    identity = {
        "resolved_profile_hash": canonical_sha256(section),
        "repositories": [
            {"name": "strategy-core", "source_tree_hash": study_rules._REVIEWED_STRATEGY_TREE}
        ],
    }
    artifact_id = canonical_sha256(identity)
    directory = root / artifact_id / "exploration"
    directory.mkdir(parents=True)
    data = _bytes({"section": section})
    config_path = directory / "effective_config.json"
    config_path.write_bytes(data)
    manifest = {
        "manifest_schema_version": 2,
        "dataset_id": artifact_id,
        "identity": identity,
        "artifacts": [
            {
                "path": "exploration/effective_config.json",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            # This table deliberately does not exist. A rules summary must
            # not perform full result verification or access market data.
            {
                "path": "exploration/executed_trade.parquet",
                "bytes": 900,
                "sha256": "a" * 64,
                "rows": 1,
            },
        ],
    }
    manifest_hash = canonical_sha256(manifest)
    manifest["manifest_payload_sha256"] = manifest_hash
    (directory / "manifest.json").write_bytes(_bytes(manifest))
    return artifact_id, manifest_hash, config_path


def _charter(root, **updates):
    base = resolve_profile_config()
    payload = _example_charter_payload().model_copy(
        update={
            "baseline_section_config_hash": base.section_config_hash,
            "locked_invariants_registry_sha256": registry_sha256(),
            "strategy_core_commit": study_rules._REVIEWED_STRATEGY_COMMIT,
            **updates,
        }
    )
    charter = SearchCharterEnvelope.from_payload(payload)
    save_envelope_immutable(root, "charters", charter)
    return charter


def _child(root, charter, *, timeout=None, ordinal=0, role="baseline"):
    resolved = resolve_profile_config(
        {"section_overrides": {"parent_retest_timeout_1m_bars": timeout}}
    )
    section = resolved.effective_config
    dataset_id, manifest_hash, config_path = _dataset(root / "v2_datasets", section)
    core = CoreStrategyReplayIdentity.from_payload(
        _example_core_replay_payload().model_copy(
            update={
                "resolved_section_config_hash": canonical_sha256(section),
                "canonical_profile_id": section["profile_name"],
                "strategy_core_source_identity": study_rules._REVIEWED_STRATEGY_SOURCE_ID,
                "strategy_core_commit": study_rules._REVIEWED_STRATEGY_COMMIT,
            }
        )
    )
    reference = {
        "core_replay_id": core.core_replay_id,
        "v2_dataset_artifact_id": dataset_id,
        "manifest_payload_sha256": manifest_hash,
        "gross_trade_stream_hash": "a" * 64,
    }
    save_envelope_immutable(
        root, "core_replays", core, extra_files={"artifact_reference.json": _bytes(reference)}
    )
    selection = {
        "parent_retest_timeout_1m_bars": f"parent_retest_timeout_1m_bars.{timeout or 'none'}"
    }
    member = SearchChildMembershipEnvelope.from_payload(
        SearchChildMembership(
            parent_search_id=charter.search_id,
            child_ordinal=ordinal,
            axis_value_ids=selection,
            core_replay_id=core.core_replay_id,
            comparison_role=role,
        )
    )
    save_envelope_immutable(root, "memberships", member)
    row = {
        "ordinal": ordinal,
        "axis_value_ids": selection,
        "core_replay_id": core.core_replay_id,
        "comparison_role": role,
        "state": "completed",
    }
    return row, config_path


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "store"
    charter = _charter(root)
    baseline, _ = _child(root, charter)
    challenger, path = _child(root, charter, timeout=480, ordinal=1, role="challenger")
    state = {"search_id": charter.search_id, "children": [baseline, challenger]}
    study = _study(
        key=charter.search_id,
        kind="search",
        status="Completed",
        store_root=root,
        charter_id=charter.search_id,
        state=state,
    )
    return study, challenger, path


def test_empty_draft_is_unconfigured():
    result = study_rules.load_study_rules(_study(draft=new_draft("single_configuration")), {})
    assert result.status == "unconfigured"
    assert result.issues == ("Choose a strategy configuration to see its rules.",)
    assert result.preview_bullets == ()


def test_skipped_draft_search_space_is_inactive_and_single_has_no_variations():
    result = study_rules.load_study_rules(_study(draft=_draft()), {})
    assert result.status == "available"
    assert result.is_preview
    assert result.variations == ()
    assert "Compare" not in " ".join(result.preview_bullets)


def test_draft_search_variations_are_resolved_and_baseline_not_double_counted():
    result = study_rules.load_study_rules(_study(draft=_draft("fsm_config_search")), {})
    assert result.status == "available"
    assert result.variations
    assert "limits of 480" in " ".join(result.preview_bullets)
    assert len(result.variations) == 2


def test_frozen_rules_ignore_mutable_draft_and_fail_on_hash_drift(tmp_path):
    charter = _charter(tmp_path, axes={})
    draft = _draft()
    draft.steps["baseline"]["baseline_profile_name"] = "unknown profile"
    study = _study(draft=draft, charter_id=charter.search_id, store_root=tmp_path)
    result = study_rules.load_study_rules(study, {})
    assert result.status == "available"
    assert not result.is_preview
    drifted = _charter(tmp_path, axes={}, baseline_section_config_hash="0" * 64)
    assert (
        study_rules.load_study_rules(replace(study, charter_id=drifted.search_id), {}).status
        == "unavailable"
    )


def test_generated_baseline_does_not_fall_back_to_a_named_profile(tmp_path):
    charter = _charter(
        tmp_path, axes={}, baseline_profile_name="ifvg_search_profile_0123456789abcdef"
    )
    result = study_rules.load_study_rules(
        _study(charter_id=charter.search_id, store_root=tmp_path), {}
    )
    assert result.status == "unavailable"


def test_completed_exact_settings_survive_current_registry_drift(completed, monkeypatch):
    study, _, _ = completed
    monkeypatch.setattr(study_rules, "registry_sha256", lambda: "0" * 64)

    def no_default(*args, **kwargs):
        raise AssertionError("completed descriptions must use their saved sections")

    monkeypatch.setattr(study_rules, "resolve_profile_config", no_default)
    result = study_rules.load_study_rules(study, {})
    assert result.status == "available"
    assert "limits of 480" in " ".join(result.preview_bullets)
    assert len(result.variations) == 2
    assert result.variations


def test_selected_result_reads_only_json_and_uses_exact_child(completed, monkeypatch):
    study, challenger, _ = completed

    def no_parquet(*args, **kwargs):
        raise AssertionError("strategy descriptions must never read Parquet")

    monkeypatch.setattr(pd, "read_parquet", no_parquet)
    original = Path.read_bytes

    def json_only(path):
        assert path.suffix == ".json"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", json_only)
    result = study_rules.load_study_rules(study, {}, challenger["core_replay_id"])
    assert result.status == "available"
    assert "480" in " ".join(result.detail_bullets)
    assert not result.variations
    assert study_rules.load_study_rules(study, {}, "a" * 64).status == "unavailable"


def test_pipeline_uses_its_own_children_and_checks_its_identity(completed):
    study, challenger, _ = completed
    pipeline = replace(
        study,
        kind="pipeline",
        key="p" * 64,
        state={
            "pipeline_semantic_id": "p" * 64,
            "search_charter_id": study.charter_id,
            "children": study.state["children"],
        },
    )
    assert (
        study_rules.load_study_rules(pipeline, {}, challenger["core_replay_id"]).status
        == "available"
    )
    wrong = replace(pipeline, state={**pipeline.state, "pipeline_semantic_id": "q" * 64})
    assert (
        study_rules.load_study_rules(wrong, {}, challenger["core_replay_id"]).status
        == "unavailable"
    )


def test_modified_configuration_is_isolated_to_affected_result(completed):
    study, challenger, path = completed
    path.write_text("{}")
    assert (
        study_rules.load_study_rules(study, {}, challenger["core_replay_id"]).status
        == "unavailable"
    )
    baseline_id = study.state["children"][0]["core_replay_id"]
    assert study_rules.load_study_rules(study, {}, baseline_id).status == "available"
    assert study_rules.load_study_rules(study, {}).status == "partial"


def test_unreviewed_implementation_has_no_rules(completed, monkeypatch):
    study, challenger, _ = completed
    monkeypatch.setattr(study_rules, "_REVIEWED_STRATEGY_SOURCE_ID", "0" * 64)
    result = study_rules.load_study_rules(study, {}, challenger["core_replay_id"])
    assert result.status == "unavailable"
    assert not result.detail_bullets
    assert "implementation" in result.issues[0]


@pytest.mark.parametrize("invalid_manifest", [[], None])
def test_non_object_manifests_are_isolated(completed, invalid_manifest):
    study, challenger, path = completed
    (path.parent / "manifest.json").write_bytes(_bytes(invalid_manifest))
    result = study_rules.load_study_rules(study, {}, challenger["core_replay_id"])
    assert result.status == "unavailable"


def _context_run(tmp_path):
    section = resolve_profile_config().effective_config
    dataset_id, manifest_hash, _ = _dataset(tmp_path / "data/ifvg_datasets/v2", section)
    config = IfvgContextExperimentConfig.model_validate(
        {
            "dataset": {
                "artifact_pair": {
                    "v2": {
                        "artifact_id": dataset_id,
                        "manifest_payload_sha256": manifest_hash,
                        "artifact_kind": "v2",
                        "dataset_schema_version": 2,
                        "profile_hash": canonical_sha256(section),
                    },
                    "v3": {
                        "artifact_id": "b" * 64,
                        "manifest_payload_sha256": "c" * 64,
                        "artifact_kind": "v3",
                        "dataset_schema_version": 4,
                    },
                }
            },
            "feature_tier": "M0",
        }
    )
    run_id = "d" * 64
    root = tmp_path / "runs"
    directory = root / run_id
    directory.mkdir(parents=True)
    data = config.model_dump_json().encode()
    (directory / "config.json").write_bytes(data)
    manifest = {
        "run_id": run_id,
        "config_hash": config.identity,
        "artifacts": [
            {"path": "config.json", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        ],
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    (directory / "manifest.json").write_bytes(_bytes(manifest))
    return run_id, root


def test_context_rules_use_saved_underlying_strategy_without_predictions(tmp_path, monkeypatch):
    run_id, root = _context_run(tmp_path)

    def no_parquet(*args, **kwargs):
        raise AssertionError("context descriptions must not load predictions")

    monkeypatch.setattr(pd, "read_parquet", no_parquet)
    result = study_rules.load_study_rules(
        _study(key=run_id, kind="context"),
        {"context_run_root": root, "repo_root": tmp_path},
    )
    assert result.status == "available"
    (root / run_id / "config.json").write_text("{}")
    with pytest.raises(ImmutableContextStoreError):
        load_context_experiment_configuration(run_id, base_dir=root)


@pytest.mark.parametrize("invalid_manifest", [[], None])
def test_context_non_object_manifest_is_unavailable(tmp_path, invalid_manifest):
    run_id, root = _context_run(tmp_path)
    (root / run_id / "manifest.json").write_bytes(_bytes(invalid_manifest))
    result = study_rules.load_study_rules(
        _study(key=run_id, kind="context"),
        {"context_run_root": root, "repo_root": tmp_path},
    )
    assert result.status == "unavailable"


def test_missing_membership_cannot_borrow_another_studys_result(completed, tmp_path):
    study, challenger, _ = completed
    another = _charter(tmp_path / "another", axes={})
    impostor = replace(
        study,
        charter_id=another.search_id,
        state={
            "search_id": another.search_id,
            "children": study.state["children"],
        },
    )
    assert (
        study_rules.load_study_rules(impostor, {}, challenger["core_replay_id"]).status
        == "unavailable"
    )


@pytest.mark.parametrize("damage", ["manifest", "config", "profile", "path"])
def test_lightweight_configuration_reader_verifies_exact_source(tmp_path, damage):
    section = resolve_profile_config().effective_config
    artifact_id, manifest_hash, path = _dataset(tmp_path, section)
    expected = canonical_sha256(section)
    if damage == "manifest":
        manifest_hash = "0" * 64
    elif damage == "config":
        path.write_text("{}")
    elif damage == "profile":
        expected = "0" * 64
    else:
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][0]["path"] = "../../outside.json"
        manifest.pop("manifest_payload_sha256")
        manifest_hash = canonical_sha256(manifest)
        manifest["manifest_payload_sha256"] = manifest_hash
        manifest_path.write_bytes(_bytes(manifest))
    with pytest.raises(ArtifactVerificationError):
        load_verified_v2_configuration(
            tmp_path,
            artifact_id,
            expected_manifest_hash=manifest_hash,
            expected_profile_hash=expected,
        )
