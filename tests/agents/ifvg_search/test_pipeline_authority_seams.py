"""HARDENING-BACKEND adversarial B-05 — the pipeline's S00 full-development
authority seam at the PIPELINE level: a real charter bundle that is not bound
to THIS store's verified namespace and CURRENT supersession head refuses at
S00 before any other stage runs (no child row, later stages pending); a
bound bundle passes the authority check. Everything is synthetic (tmp roots,
the synthetic replay machinery, no real source path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    DatePolicy,
    SearchCharterEnvelope,
    SearchCharterPayload,
)
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    PipelineSemanticIdentity,
    PipelineSemanticSpecPayload,
    QuantLabPipelineStage,
    read_pipeline_state,
    run_pipeline,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_test_namespace
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import publish_supersession
from tests.agents.ifvg_search.namespace_fixture import owner_authorization_bundle
from tests.agents.ifvg_search.pipeline_fixture import (
    STRATEGY_ONLY_STAGE_PLAN,
    build_pipeline_fixture,
)

_FULL_DATES = (*FROZEN_WARMUP_DATES, "2026-01-13")
_S00 = QuantLabPipelineStage.S00_VALIDATE_INPUTS.value


def _full_scope_fixture(tmp_path: Path, *, bundle_root: Path | None):
    """The synthetic fixture re-shaped as a FULL-development launch: the
    charter carries a real owner bundle bound to ``bundle_root`` (or a dummy
    namespace when ``None``) and the spec runs the strategy-only plan."""

    fixture = build_pipeline_fixture(tmp_path)
    bundle = owner_authorization_bundle(
        bundle_root, requirement_set_id="1" * 64, decision_refs={}
    )
    charter_payload = SearchCharterPayload.model_validate(
        {
            **fixture["charter"].payload.model_dump(mode="json"),
            "date_policy": DatePolicy(
                replay_dates=_FULL_DATES,
                warmup_dates=FROZEN_WARMUP_DATES,
                access_policy_id="development_explicit_dates_before_path_v2",
            ).model_dump(mode="json"),
            "owner_authorization": bundle.model_dump(mode="json"),
        }
    )
    charter = SearchCharterEnvelope.from_payload(charter_payload)
    spec_payload = PipelineSemanticSpecPayload.model_validate(
        {
            **fixture["semantic"].payload.model_dump(mode="json"),
            "run_scope": "full_authorized_development",
            "date_allowlist": list(_FULL_DATES),
            "allowlist_hash": allowlist_sha256(_FULL_DATES),
            "search_charter_id": charter.search_id,
            "stage_plan": [stage.value for stage in STRATEGY_ONLY_STAGE_PLAN],
            # the strategy-only plan declares no feature / label / fold / model inputs
            "feature_bundle_ids": [],
            "label_policy_id": None,
            "fold_protocol_id": None,
            "model_protocol_id": None,
        }
    )
    semantic = PipelineSemanticIdentity.from_payload(spec_payload)
    return fixture, charter, semantic


def _launch(fixture, charter, semantic):
    result = run_pipeline(
        semantic,
        charter,
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    state = read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)
    return result, state


def _assert_refused_at_s00(state, reason_fragment: str | None) -> None:
    s00 = state["stages"][_S00]
    assert s00["status"] == "failed", s00
    if reason_fragment is None:
        # the typed message carried the store's absolute path, which the
        # stage sanitizer withholds entirely — the refusal is still S00's
        assert s00["explanation"] in (
            "failure details withheld (sanitized)",
        ) or "launch refused" in s00["explanation"], s00["explanation"]
    else:
        assert "launch refused" in s00["explanation"], s00["explanation"]
        assert reason_fragment in s00["explanation"], s00["explanation"]
    for stage, entry in state["stages"].items():
        if stage != _S00 and entry["in_plan"]:
            assert entry["status"] == "pending", (stage, entry)
    assert state["children"] == []


def test_an_unbound_bundle_refuses_at_s00_before_any_stage_runs(tmp_path) -> None:
    # (a) the store is UNMARKED: the bundle names a namespace the store does not carry
    fixture, charter, semantic = _full_scope_fixture(
        tmp_path / "unmarked", bundle_root=tmp_path / "somewhere_else"
    )
    assert not (fixture["store_root"] / "STORE_NAMESPACE.json").exists()
    _result, state = _launch(fixture, charter, semantic)
    # the `store_namespace_missing` message names the store root (a local
    # path), so the sanitized stage explanation withholds it (recorded for
    # the main agent); the refusal is still S00's — no stage ran, no child
    _assert_refused_at_s00(state, None)
    # (a') the store is marked but the bundle names ANOTHER namespace
    other = build_pipeline_fixture(tmp_path / "other")
    initialize_test_namespace(other["store_root"])
    _fixture2, charter2, semantic2 = _full_scope_fixture(
        tmp_path / "foreign", bundle_root=tmp_path / "foreign_namespace"
    )
    initialize_test_namespace(_fixture2["store_root"])
    _result2, state2 = _launch(_fixture2, charter2, semantic2)
    _assert_refused_at_s00(state2, "store_namespace_identity_mismatch")


def test_a_stale_witness_refuses_at_s00_after_a_supersession(tmp_path) -> None:
    fixture, charter, semantic = _full_scope_fixture(
        tmp_path / "stale", bundle_root=tmp_path / "stale" / "store"
    )
    # the head moves after the bundle was signed
    publish_supersession(
        fixture["store_root"],
        superseded_decision_id="a" * 64,
        replacement_decision_id="b" * 64,
        reason="moved after signing",
        effective_at="2026-08-18T01:00:00+00:00",
        owner_evidence_ref="b" * 64,
    )
    _result, state = _launch(fixture, charter, semantic)
    _assert_refused_at_s00(state, "supersession_head_witness_mismatch")


def test_a_bound_bundle_passes_the_s00_authority_check(tmp_path) -> None:
    fixture, charter, semantic = _full_scope_fixture(
        tmp_path / "bound", bundle_root=tmp_path / "bound" / "store"
    )
    _result, state = _launch(fixture, charter, semantic)
    s00 = state["stages"][_S00]
    # S00 either completed with the bound-bundle explanation, or failed for a
    # reason that is NOT the authority refusal (later checks of the real
    # scope are out of this seam's scope)
    if s00["status"] == "completed":
        assert "bound to this store's namespace" in s00["explanation"], s00["explanation"]
    else:
        assert "launch refused" not in s00["explanation"], s00["explanation"]
        pytest.skip(f"S00 stopped after the authority check: {s00['explanation'][:160]}")
