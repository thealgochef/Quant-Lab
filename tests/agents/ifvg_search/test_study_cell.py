"""Study-cell identity suites (TEST_MATRIX §3.8 — P0-B, P1-G, V3 P0-6)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.study.dimension_contracts import (
    EXPERIMENT_DIMENSION_REGISTRY,
    DimensionBlockedError,
    DimensionType,
    assert_dimension_usable,
    resolve_dimension,
)
from alpha_lab.agents.data_infra.ifvg.study.study_cell import (
    BASELINE_STUDY_CELL,
    LEGACY_REPLAY_PROVENANCE,
    DataLineagePayload,
    EngineeringProtocolAnnotation,
    StudyCellAnnotation,
    StudyCellIdentity,
    StudyCellSemanticPayload,
    _baseline_cell_payload,
)


def test_semantic_payload_has_exactly_sixteen_dimensions() -> None:
    assert len(StudyCellSemanticPayload.model_fields) == 16
    assert "engineering_protocol" not in StudyCellSemanticPayload.model_fields


def test_annotation_changes_never_fork_cell_id() -> None:
    cell = BASELINE_STUDY_CELL
    annotation_a = StudyCellAnnotation(
        cell_id=cell.cell_id,
        engineering_protocol=EngineeringProtocolAnnotation(
            worker_count=4, storage_root="data/x", page_size=100, runtime_estimate_seconds=60
        ),
        display_metadata={"display_name": "My study"},
    )
    annotation_b = StudyCellAnnotation(
        cell_id=cell.cell_id,
        engineering_protocol=EngineeringProtocolAnnotation(
            worker_count=1, storage_root="data/y", page_size=9, runtime_estimate_seconds=1
        ),
        display_metadata={"display_name": "Renamed"},
    )
    # both annotations legitimately reference the SAME immutable cell
    assert annotation_a.cell_id == annotation_b.cell_id == cell.cell_id
    recomputed = StudyCellIdentity.from_payload(cell.payload)
    assert recomputed.cell_id == cell.cell_id


def test_model_package_version_moves_the_correct_semantic_identity() -> None:
    base = _baseline_cell_payload()
    changed_model = base.model_copy(
        update={
            "model_protocol": base.model_protocol.model_copy(
                update={"resolved_model_protocol_id": "a" * 64}
            )
        }
    )
    with pytest.raises(ValueError):
        # a real model protocol without label/fold evidence is refused
        StudyCellSemanticPayload.model_validate(changed_model.model_dump(mode="json"))
    proper = base.model_copy(
        update={
            "model_protocol": base.model_protocol.model_copy(
                update={
                    "model_protocol_key": "ifvg_context_catboost_binary_v1",
                    "resolved_model_protocol_id": "a" * 64,
                }
            ),
            "label_policy": base.label_policy.model_copy(
                update={"label_derivation_id": "b" * 64}
            ),
            "validation_protocol": base.validation_protocol.model_copy(
                update={"fold_set_hash": "c" * 64}
            ),
            "data_lineage": base.data_lineage.model_copy(
                update={"label_view_id": "b" * 64, "fold_set_id": "c" * 64}
            ),
        }
    )
    validated = StudyCellSemanticPayload.model_validate(proper.model_dump(mode="json"))
    assert canonical_contract_sha256(validated) != canonical_contract_sha256(base)


def test_strategy_only_cell_carries_no_context_formula() -> None:
    lineage = BASELINE_STUDY_CELL.payload.data_lineage
    assert lineage.context_artifact_id is None
    assert lineage.context_formula_id is None
    assert lineage.feature_view_id is None
    assert lineage.model_fit_id is None
    assert lineage.artifact_pair_hash is None
    with pytest.raises(ValueError, match="strategy-only"):
        DataLineagePayload(
            core_replay_id=LEGACY_REPLAY_PROVENANCE,
            v2_artifact_id="1" * 64,
            v2_manifest_hash="2" * 64,
            context_formula_id="ifvg_context_formula_v2",
        )


def test_strategy_only_cell_is_never_blocked_by_missing_v3(tmp_path) -> None:
    payload = _baseline_cell_payload()
    cell = StudyCellIdentity.from_payload(payload)
    assert cell.cell_id == BASELINE_STUDY_CELL.cell_id
    # a required-evidence gap DOES fail (model without labels), asserted above;
    # an irrelevant gap (no v3 pair) does not — this cell validated fine.


def test_profile_names_are_gated_registered_or_generated() -> None:
    from alpha_lab.agents.data_infra.ifvg.study.study_cell import StrategyProfileIdentity

    base = _baseline_cell_payload().strategy_profile.model_dump(mode="json")
    with pytest.raises(ValueError, match="P0-D|neither"):
        StrategyProfileIdentity.model_validate(
            {**base, "strategy_profile_id": "rogue_profile"}
        )
    StrategyProfileIdentity.model_validate(  # generated names are lawful
        {**base, "strategy_profile_id": "ifvg_search_profile_0123456789abcdef"}
    )


def test_cell_payload_refuses_unregistered_and_blocked_dimensions() -> None:
    base = _baseline_cell_payload()
    # unknown registry-bound value at the PAYLOAD level (not just helpers)
    with pytest.raises(ValueError, match="not a registered value"):
        StudyCellSemanticPayload.model_validate(
            {
                **base.model_dump(mode="json"),
                "stress_scenario": {"stress_scenario_id": "nonexistent_scenario_v1"},
            }
        )
    with pytest.raises(ValueError, match="not a registered value"):
        StudyCellSemanticPayload.model_validate(
            {
                **base.model_dump(mode="json"),
                "feature_bundle": {
                    **base.feature_bundle.model_dump(mode="json"),
                    "feature_bundle_key": "IFVG_CORE_BASELINE_V1",  # a BLOCK key
                },
            }
        )
    # a BLOCKED active dimension (S11-gated decision policy) fails closed
    with pytest.raises(DimensionBlockedError, match="RejectedCandidatePolicy"):
        StudyCellSemanticPayload.model_validate(
            {
                **base.model_dump(mode="json"),
                "decision_policy": {
                    "decision_policy_key": "fixed_probability_threshold_v1",
                    "resolved_decision_policy_id": "9" * 64,
                },
                "label_policy": {
                    **base.label_policy.model_dump(mode="json"),
                    "label_derivation_id": "b" * 64,
                },
                "validation_protocol": {
                    **base.validation_protocol.model_dump(mode="json"),
                    "fold_set_hash": "c" * 64,
                },
            }
        )


def test_dimension_registry_fails_closed() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        resolve_dimension("no.such_dimension")
    blocked = resolve_dimension("decision_policy.execution_gate")
    with pytest.raises(DimensionBlockedError, match="RejectedCandidatePolicy"):
        assert_dimension_usable(blocked)
    usable = resolve_dimension("cost_policy.round_turn_cost")
    assert_dimension_usable(usable)
    assert len({spec.dimension_type for spec in EXPERIMENT_DIMENSION_REGISTRY.values()}) == len(
        DimensionType
    )


def test_annotation_only_dimension_is_absent_from_the_semantic_payload() -> None:
    engineering = resolve_dimension("engineering_protocol.runtime")
    assert engineering.identity_effect == "annotation_only"
    assert "engineering_protocol" not in StudyCellSemanticPayload.model_fields
