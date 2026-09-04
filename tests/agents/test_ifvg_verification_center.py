"""UI-2 Verification Center AppTests (plan §5.3, §7, §9 Phase 2, §10):

* the center never renders research gates, prop / risk / model controls or a
  Publish route;
* logical trading days and physical partitions are displayed separately;
* seed production cannot launch without a validated
  ``SeedProductionAuthorizationRef`` (no run command until then);
* the final verification packet cannot exist before a verified
  profile-matching seed;
* real verification cannot launch without a validated final
  ``VerificationAuthorizationRef`` (no bounded-run command until the charter
  is frozen from the signed ref, the run registered and the preflight passed);
* the monitor lists only the stages of the resolved seed / verification plan.

Everything runs over the isolated synthetic fixture store under a tmp repo
root; nothing signs, produces a seed or launches — the center exposes exact
external commands and picks results up through receipts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

apptest = pytest.importorskip("streamlit.testing.v1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_pipeline_tab as pipeline_tab  # noqa: E402
import ifvg_study_tab as study_tab  # noqa: E402
import ifvg_verification_center as center  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.search.authorization import (  # noqa: E402
    OwnerAuthorizationBundle,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: E402
    has_envelope,
    load_verified_envelope,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: E402
    SEED_AUTHORIZATION_RECEIPT,
    SEED_RUN_RECEIPT,
    load_verification_center_record,
    verification_authorization_readiness,
)
from tests.agents.ifvg_search.conftest import (  # noqa: E402
    build_artifact_chain,
    run_synthetic_chain,
)
from tests.agents.ifvg_search.verification_center_fixture import (  # noqa: E402
    QL_IDENTITY,
    SC_COMMIT,
    SC_IDENTITY,
    WINDOW,
    build_verification_store,
    write_inventory_manifest,
    write_shortlist_document,
    write_signed_ref,
)

_WINDOW_LABEL = " · ".join(WINDOW)


@pytest.fixture(scope="module")
def synthetic_chain():
    """The conftest synthetic three-day chain (the seed through 2026-01-14)."""

    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig

    cfg = IfvgCaptureConfig()
    return run_synthetic_chain(cfg, build_artifact_chain(cfg))


def _app() -> None:
    import ifvg_study_tab as study_tab
    import streamlit as st

    study_tab.render_ifvg_study_tab(st)


def _wire(monkeypatch, tmp_path, fixture: dict) -> dict:
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", tmp_path / "research")
    monkeypatch.setattr(study_tab, "STORE_ROOT_VERIFICATION", fixture["store_root"])
    monkeypatch.setattr(study_tab, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", tmp_path / "drafts")
    monkeypatch.setattr(study_tab, "REPO_ROOT", fixture["repo_root"])
    monkeypatch.setattr(study_tab, "VERIFICATION_CENTER_ROOT", tmp_path / "center")
    monkeypatch.setattr(pipeline_tab, "PIPELINE_STATE_ROOT", tmp_path / "pipeline_jobs")
    shortlist_path = tmp_path / "evidence" / "LOGICAL_WINDOW_COVERAGE_SCAN.json"
    fixture["document"] = write_shortlist_document(shortlist_path)
    monkeypatch.setattr(center, "SHORTLIST_DOCUMENT_PATH", shortlist_path)
    manifest_path = write_inventory_manifest(tmp_path / "manifest.json", fixture["inventory"])
    monkeypatch.setattr(center, "INVENTORY_MANIFEST_PATH", manifest_path)
    fixture["center_root"] = tmp_path / "center"
    # the fixture authorization binds FAKE code identities; the center normally
    # computes the real repository identities (a read-only query) — pin them
    monkeypatch.setattr(
        center, "_code_identities", lambda repo_root: (QL_IDENTITY, (SC_COMMIT, SC_IDENTITY), None)
    )
    return fixture


def _run(**session):
    at = apptest.AppTest.from_function(_app, default_timeout=180)
    at.session_state[study_tab.ROUTE_KEY] = "Verify Implementation"
    for key, value in session.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return "\n".join(
        [str(b.value) for b in at.markdown]
        + [str(c.value) for c in at.caption]
        + [str(h.value) for h in at.subheader]
        + [str(w.value) for w in at.warning]
        + [str(e.value) for e in at.error]
        + [str(i.value) for i in at.info]
        + [str(s.value) for s in at.success]
    )


def _codes(at) -> str:
    return "\n".join(str(c.value) for c in at.code)


def _tables(at) -> str:
    return "\n".join(t.value.to_string() for t in at.table) + "\n".join(
        d.value.to_string() for d in at.dataframe
    )


def _button(at, label: str):
    return next(b for b in at.button if b.label == label)


def _labels(at) -> list[str]:
    return [b.label for b in at.button]


def _record_window(at):
    """Select the fixture window in the Fixture step and record it."""

    picker = at.selectbox(key=f"{center._VC}window")
    picker.set_value(_WINDOW_LABEL).run()
    assert not at.exception
    _button(at, "Record provisional window").click().run()
    assert not at.exception
    return at


# ── the surface ──────────────────────────────────────────────────────────────


def test_verification_center_never_renders_research_steps_or_publish(
    monkeypatch, tmp_path, synthetic_chain
) -> None:
    fixture = _wire(
        monkeypatch, tmp_path, build_verification_store(tmp_path / "repo", synthetic_chain)
    )
    at = _run()
    headings = " ".join(str(h.value) for h in at.subheader)
    for section in (
        "Purpose & readiness",
        "Fixture",
        "Seed",
        "Final authorization",
        "Review & run",
        "Monitor",
    ):
        assert section in headings, section
    assert "VERIFICATION ONLY" in _text(at)
    labels = [label.lower() for label in _labels(at)]
    for forbidden in ("publish", "activate", "sign", "launch", "delete", "promote"):
        assert not any(forbidden in label for label in labels), (forbidden, labels)
    widget_labels = [
        str(getattr(widget, "label", "") or "").lower()
        for kind in ("selectbox", "radio", "number_input", "multiselect", "checkbox", "slider")
        for widget in getattr(at, kind)
    ]
    for forbidden in ("prop", "risk", "model", "objective template", "gate", "worker", "run scope"):
        assert not any(forbidden in label for label in widget_labels), (forbidden, widget_labels)
    assert not at.slider
    text = _text(at)
    assert "sequential_children_v1" in text and "effective workers: 1" in text
    assert "Publish" not in headings
    # the sticky readiness card renders every typed state — never a Boolean
    for row in (
        "Semantic store namespace",
        "Logical-day fixture",
        "Physical partition coverage",
        "Seed-production authorization",
        "Seed job",
        "Verified seed",
        "Final verification authorization",
        "Bounded-run readiness",
    ):
        assert row in text, row
    assert fixture["namespace_id"][:16] in _codes(at)


def test_logical_days_and_physical_partitions_are_displayed_separately(
    monkeypatch, tmp_path, synthetic_chain
) -> None:
    fixture = _wire(
        monkeypatch, tmp_path, build_verification_store(tmp_path / "repo", synthetic_chain)
    )
    at = _run()
    text = _text(at)
    assert "No provisional verification window selected" in " ".join(
        str(h.value) for h in at.subheader
    )
    assert fixture["document"]["shortlist_id"][:16] in _codes(at)
    assert "NOT PERFORMED" in text  # the backend's owner_selection stays untouched
    _record_window(at)
    record = load_verification_center_record(tmp_path / "center")
    assert record.provisional_window["days"] == list(WINDOW)
    assert record.provisional_window["shortlist_id"] == fixture["document"]["shortlist_id"]
    tables = _tables(at)
    logical_start = tables.index("2026-01-15T") if "2026-01-15T" in tables else -1
    assert "Logical trading days" in _text(at) and "Physical partitions" in _text(at)
    # logical days with their session bounds …
    assert "2026-01-14T23:00:00+00:00" in tables  # the 18:00 ET roll of 2026-01-15
    # … and the physical partitions (td−1, td) as a SEPARATE table
    assert "prev_utc_date" in tables and "utc_date" in tables
    assert "2026-01-14" in tables  # a partition date that is NOT a logical day of the window
    assert tables.count("mbp1") >= 4
    assert logical_start != -1 or "2026-01-15" in tables
    exclusions = _text(at)
    assert "2026-06-11" in exclusions and "sealed" in exclusions.lower()
    # the shortlist document itself is never modified by the selection
    document = json.loads(
        (tmp_path / "evidence" / "LOGICAL_WINDOW_COVERAGE_SCAN.json").read_text(encoding="utf-8")
    )
    assert document["owner_selection"] == "NOT PERFORMED"
    assert document["register_program_allowlist_called"] is False
    # an ineligible window cannot be recorded
    options = at.selectbox(key=f"{center._VC}window").options
    assert all(
        "eligible" in option or option == _WINDOW_LABEL or "·" in option for option in options
    )


def test_seed_production_cannot_launch_without_a_validated_authorization(
    monkeypatch, tmp_path, synthetic_chain
) -> None:
    fixture = _wire(
        monkeypatch,
        tmp_path,
        build_verification_store(
            tmp_path / "repo", synthetic_chain, with_seed=False, with_authorization=False
        ),
    )
    at = _record_window(_run())
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "Seed production not authorized" in headings
    codes = _codes(at)
    assert "ifvg_seed_production.py register-authorization" in codes
    assert "ifvg_seed_production.py run" not in codes  # no job command without the ref
    assert "Prepare the seed-production packet" in _labels(at)
    _button(at, "Prepare the seed-production packet").click().run()
    assert not at.exception
    packet_path = tmp_path / "center" / "SEED_PRODUCTION_PACKET.json"
    assert packet_path.exists()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["payload"]["approved_by"] == "<OWNER_TO_FILL>"
    assert packet["payload"]["first_intended_verification_day"] == WINDOW[0]
    assert packet["payload"]["ordered_seed_chain_replay_days"][0] == "2026-01-01"
    assert packet["separately_authorized_preparation"] is True
    text = _text(at)
    assert "prohibited outputs" in text.lower()
    assert "research_catalog_publication" in text
    # a bogus recorded id is a typed not-found state — still no run command
    (tmp_path / "center" / SEED_AUTHORIZATION_RECEIPT).write_text(
        json.dumps({"seed_production_authorization_id": "f" * 64}), encoding="utf-8"
    )
    at = _run()
    assert "not_found" in _text(at)
    assert "ifvg_seed_production.py run" not in _codes(at)
    # a registered, VERIFIED authorization exposes the exact job command
    verified = build_verification_store(
        tmp_path / "repo2", synthetic_chain, with_seed=False, with_authorization=True
    )
    monkeypatch.setattr(study_tab, "STORE_ROOT_VERIFICATION", verified["store_root"])
    monkeypatch.setattr(study_tab, "REPO_ROOT", verified["repo_root"])
    authorization_id = verified["authorization"].seed_production_authorization_id
    (tmp_path / "center" / SEED_AUTHORIZATION_RECEIPT).write_text(
        json.dumps({"seed_production_authorization_id": authorization_id}), encoding="utf-8"
    )
    at = _run()
    text = _text(at)
    assert "verified" in text
    codes = _codes(at)
    assert "ifvg_seed_production.py run" in codes
    assert authorization_id in codes
    assert "--receipt-out" in codes and SEED_RUN_RECEIPT in codes
    assert fixture["repo_root"] != verified["repo_root"]


def test_final_verification_packet_cannot_exist_before_a_verified_seed(
    monkeypatch, tmp_path, synthetic_chain
) -> None:
    fixture = _wire(
        monkeypatch,
        tmp_path,
        build_verification_store(tmp_path / "repo", synthetic_chain, with_seed=False),
    )
    (tmp_path / "center").mkdir(parents=True, exist_ok=True)
    (tmp_path / "center" / SEED_AUTHORIZATION_RECEIPT).write_text(
        json.dumps(
            {
                "seed_production_authorization_id": fixture[
                    "authorization"
                ].seed_production_authorization_id
            }
        ),
        encoding="utf-8",
    )
    at = _record_window(_run())
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "No verified profile-matching seed" in headings
    assert "Prepare the final verification packet" not in _labels(at)
    assert not (tmp_path / "center" / "VERIFICATION_PACKET.json").exists()
    # the seed job's receipt brings the verified seed in
    seeded = build_verification_store(tmp_path / "repo", synthetic_chain)
    (tmp_path / "center" / SEED_RUN_RECEIPT).write_text(
        json.dumps(
            {
                "seed_snapshot_id": seeded["snapshot"].seed_snapshot_id,
                "seed_production_run_id": seeded["receipt"].seed_production_run_id,
            }
        ),
        encoding="utf-8",
    )
    at = _run()
    text = _text(at)
    assert "No verified profile-matching seed" not in " ".join(str(h.value) for h in at.subheader)
    assert seeded["snapshot"].payload.seed_hash[:16] in _codes(at)
    assert "synthetic_test_authorization_v1" in text + _tables(at)  # the seed's provenance
    _button(at, "Prepare the final verification packet").click().run()
    assert not at.exception
    packet = json.loads(
        (tmp_path / "center" / "VERIFICATION_PACKET.json").read_text(encoding="utf-8")
    )
    ref = packet["verification_authorization_ref"]
    assert ref["approved_by"] == "<OWNER_TO_FILL>" and ref["content_hash"] == "<OWNER_TO_FILL>"
    assert ref["seed_snapshot_id"] == seeded["snapshot"].seed_snapshot_id
    assert packet["corrected_logical_allowlist"] == list(WINDOW)
    assert len(packet["coverage_matrix_artifact_id"]) == 64
    assert packet["stamps"]["full_pipeline_not_run"] is True
    assert "Final verification authorization not signed" in " ".join(
        str(h.value) for h in at.subheader
    )


def test_real_verification_cannot_launch_without_a_validated_final_ref(
    monkeypatch, tmp_path, synthetic_chain
) -> None:
    fixture = _wire(
        monkeypatch, tmp_path, build_verification_store(tmp_path / "repo", synthetic_chain)
    )
    (tmp_path / "center").mkdir(parents=True, exist_ok=True)
    (tmp_path / "center" / SEED_AUTHORIZATION_RECEIPT).write_text(
        json.dumps(
            {
                "seed_production_authorization_id": fixture[
                    "authorization"
                ].seed_production_authorization_id
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "center" / SEED_RUN_RECEIPT).write_text(
        json.dumps(
            {
                "seed_snapshot_id": fixture["snapshot"].seed_snapshot_id,
                "seed_production_run_id": fixture["receipt"].seed_production_run_id,
            }
        ),
        encoding="utf-8",
    )
    at = _record_window(_run())
    _button(at, "Prepare the final verification packet").click().run()
    assert not at.exception
    codes = _codes(at)
    assert "ifvg_bounded_verification.py run" not in codes  # nothing to launch yet
    assert "Freeze the verification charter and register the run" not in _labels(at)
    assert verification_authorization_readiness(fixture["store_root"]).status == "missing"
    # the owner completes the reference outside the workspace (a fixture here)
    signed_path = tmp_path / "center" / "VERIFICATION_AUTHORIZATION_REF.signed.json"
    write_signed_ref(
        signed_path,
        fixture["store_root"],
        seed_snapshot_id=fixture["snapshot"].seed_snapshot_id,
        coverage_matrix_artifact_id=json.loads(
            (tmp_path / "center" / "VERIFICATION_PACKET.json").read_text(encoding="utf-8")
        )["coverage_matrix_artifact_id"],
    )
    at = _run()
    text = _text(at)
    assert "valid" in text and "test-owner" in text
    freeze = _button(at, "Freeze the verification charter and register the run")
    assert not freeze.disabled
    assert "ifvg_bounded_verification.py run" not in _codes(at)
    freeze.click().run()
    assert not at.exception, at.exception
    # the charter froze from the SIGNED ref (never the synthetic marker), the
    # pipeline spec exists, the run is registered and the readiness is ready
    readiness = verification_authorization_readiness(fixture["store_root"])
    assert readiness.status == "ready", readiness.detail
    record = load_verification_center_record(tmp_path / "center")
    assert record.provisional_window["days"] == list(WINDOW)
    charter_ids = [
        c
        for c in _codes(at).split("\n")
        if len(c) == 64 and has_envelope(fixture["store_root"], "charters", c)
    ]
    assert charter_ids
    charter = load_verified_envelope(
        fixture["store_root"], "charters", charter_ids[0], SearchCharterEnvelope
    )
    assert isinstance(charter.payload.owner_authorization, OwnerAuthorizationBundle)
    assert charter.payload.date_policy.replay_dates == WINDOW
    assert charter.payload.axes == {}  # the exact baseline only
    assert charter.payload.authorized_firm_contract_ids == ()
    # the preflight passed and the exact external run command is now shown
    text = _text(at)
    assert "preflight" in text.lower() and "passed" in text.lower()
    codes = _codes(at)
    assert "ifvg_bounded_verification.py run" in codes
    assert "--store-root" in codes and "--evidence-dir" in codes
    labels = [label.lower() for label in _labels(at)]
    assert not any("launch" in label or "publish" in label for label in labels)
    # once registered, the freeze control is gone (the readiness is ready) and a
    # re-render finds the same single registered run — never a second one
    assert "Freeze the verification charter and register the run" not in _labels(at)
    again = _run()
    assert "Freeze the verification charter and register the run" not in _labels(again)
    assert verification_authorization_readiness(fixture["store_root"]).evidence_ids == (
        readiness.evidence_ids
    )
    assert "ifvg_bounded_verification.py run" in _codes(again)


def test_monitor_lists_only_the_planned_stages(monkeypatch, tmp_path, synthetic_chain) -> None:
    fixture = _wire(
        monkeypatch, tmp_path, build_verification_store(tmp_path / "repo", synthetic_chain)
    )
    (tmp_path / "center").mkdir(parents=True, exist_ok=True)
    (tmp_path / "center" / SEED_RUN_RECEIPT).write_text(
        json.dumps(
            {
                "seed_snapshot_id": fixture["snapshot"].seed_snapshot_id,
                "seed_production_run_id": fixture["receipt"].seed_production_run_id,
            }
        ),
        encoding="utf-8",
    )
    at = _record_window(_run())
    _button(at, "Prepare the final verification packet").click().run()
    signed_path = tmp_path / "center" / "VERIFICATION_AUTHORIZATION_REF.signed.json"
    write_signed_ref(
        signed_path,
        fixture["store_root"],
        seed_snapshot_id=fixture["snapshot"].seed_snapshot_id,
        coverage_matrix_artifact_id=json.loads(
            (tmp_path / "center" / "VERIFICATION_PACKET.json").read_text(encoding="utf-8")
        )["coverage_matrix_artifact_id"],
    )
    at = _run()
    _button(at, "Freeze the verification charter and register the run").click().run()
    assert not at.exception
    readiness = verification_authorization_readiness(fixture["store_root"])
    from alpha_lab.agents.data_infra.ifvg.search.verification import VerificationRunEnvelope

    run = load_verified_envelope(
        fixture["store_root"],
        "verification_runs",
        readiness.evidence_ids[0],
        VerificationRunEnvelope,
    )
    pipeline_id = run.payload.pipeline_semantic_id
    state_dir = tmp_path / "pipeline_jobs" / pipeline_id
    state_dir.mkdir(parents=True)
    (state_dir / "pipeline_state.json").write_text(
        json.dumps(
            {
                "pipeline_semantic_id": pipeline_id,
                "run_scope": "verification_5d",
                "current_stage": "02_run_or_reuse_sequential_replays",
                "stages": {
                    "00_validate_inputs": {"in_plan": True, "status": "completed"},
                    "01_prepare_strategy_profiles": {"in_plan": True, "status": "completed"},
                    "02_run_or_reuse_sequential_replays": {"in_plan": True, "status": "running"},
                    "05_materialize_feature_views": {"in_plan": False},
                    "09_train_models": {"in_plan": False},
                    "11_run_frozen_model_gated_replays": {"in_plan": True, "status": "blocked"},
                    "15_verify_and_publish": {"in_plan": True, "status": "pending"},
                },
                "attempts": [
                    {
                        "started_at": "2026-09-04T00:00:00+00:00",
                        "execution_mode": "sequential_children_v1",
                        "effective_workers": 1,
                    }
                ],
                "children": [],
            }
        ),
        encoding="utf-8",
    )
    at = _run()
    tables = _tables(at)
    assert "Seed production" in tables  # the resolved seed stage first
    assert "Validate Inputs" in tables and "Frozen Model-Gated Replays" in tables
    assert "Materialize Feature Views" not in tables  # not in the plan → not listed
    assert "Train Models" not in tables
    text = _text(at)
    assert "sequential_children_v1" in text
    assert "Publish" not in " ".join(str(h.value) for h in at.subheader)
    assert not any(
        "publish" in label.lower() or "activate" in label.lower() for label in _labels(at)
    )
