"""External publications are additive, immutable and verified before navigation."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256, file_sha256
from alpha_lab.agents.data_infra.ifvg.presentation.lab import external_catalog as catalog

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


def _published(tmp_path):
    external = tmp_path / "external"
    bindings = []
    for family, key in (
        ("funded_comparison_results", "a" * 64),
        ("funded_comparison_plans", "b" * 64),
        ("funded_comparison_approvals", "c" * 64),
    ):
        folder = external / family / key
        folder.mkdir(parents=True)
        envelope = folder / "envelope.json"
        envelope.write_text(json.dumps({"payload": {"saved": family}}), encoding="utf-8")
        manifest = {
            "store_name": family,
            "envelope_id": key,
            "artifacts": [
                {
                    "path": "envelope.json",
                    "bytes": envelope.stat().st_size,
                    "sha256": file_sha256(envelope),
                }
            ],
        }
        manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        bindings.append(catalog._manifest_binding(external, family, key))
    view = {
        "schema": catalog.VIEW_VERSION,
        "result_id": "a" * 64,
        "plan_id": "b" * 64,
        "source_bindings": bindings,
        "result": {"cash": 12345},
    }
    view_path = tmp_path / "read_view.json"
    view_path.write_text(json.dumps(view), encoding="utf-8")
    pointer = {
        "result_id": "a" * 64,
        "plan_id": "b" * 64,
        "external_store_root": str(external),
        "source_bindings": bindings,
        "external_store_namespace": None,
        "view_path": str(view_path),
        "view_sha256": file_sha256(view_path),
        "view_bytes": view_path.stat().st_size,
        "reporting_definition_version": catalog.VIEW_VERSION,
        "qualification": "Original execution retained",
        "evaluation_dates": ["2026-01-02"],
    }
    pointer["binding_id"] = canonical_sha256(pointer)
    main = tmp_path / "main"
    catalog._write(
        catalog.catalog_path(main),
        {
            "schema": catalog.SCHEMA,
            "studies": {
                "study": {
                    "display_name": "Saved external study",
                    "versions": {"a" * 64: pointer},
                    "preferred_result_id": "a" * 64,
                    "updated_at_utc": "2026-10-07T20:00:00Z",
                }
            },
        },
    )
    return main, pointer


def test_catalog_and_view_bind_exact_sources_without_engine_promotion(tmp_path, monkeypatch):
    main, pointer = _published(tmp_path)
    before = copy.deepcopy(pointer)
    monkeypatch.delenv("IFSM_MFFU_REVIEW_STORE_ROOT", raising=False)
    monkeypatch.delenv("IFSM_MFFU_REVIEW_RESULT_ID", raising=False)
    groups, issues = catalog.registered_groups(main)
    assert not issues and groups[0]["pointer"] == pointer
    target = catalog.resolve_registered_result("a" * 64, store_root=main)
    assert target["app"] == "main" and target["external_review_only"]
    assert target["store_root"] == pointer["external_store_root"]
    assert catalog.load_registered_view(pointer) == {"cash": 12345}
    assert pointer == before
    assert catalog.resolve_registered_result("d" * 64, store_root=main) is None
    assert not (Path(pointer["external_store_root"]) / "STORE_NAMESPACE.json").exists()


@pytest.mark.parametrize("kind", ["manifest", "view", "pointer", "source_bytes"])
def test_corruption_fails_visibly_and_never_selects_another_result(tmp_path, kind):
    main, pointer = _published(tmp_path)
    if kind == "manifest":
        path = (
            Path(pointer["external_store_root"])
            / "funded_comparison_results"
            / pointer["result_id"]
            / "manifest.json"
        )
        path.write_text("{}", encoding="utf-8")
    elif kind == "view":
        Path(pointer["view_path"]).write_text("tampered", encoding="utf-8")
    elif kind == "source_bytes":
        path = (
            Path(pointer["external_store_root"])
            / "funded_comparison_results"
            / pointer["result_id"]
            / "envelope.json"
        )
        path.write_text(path.read_text().replace("saved", "other"), encoding="utf-8")
    else:
        pointer["external_store_root"] = str(tmp_path / "wrong")
    with pytest.raises((ValueError, OSError)):
        catalog.verify_pointer(pointer)
    if kind != "pointer":
        groups, issues = catalog.registered_groups(main)
        assert not groups and len(issues) == 1
        assert "Saved external study" in issues[0]


def test_unavailable_location_is_reported_without_fallback(tmp_path):
    main, pointer = _published(tmp_path)
    path = Path(pointer["view_path"])
    path.unlink()
    groups, issues = catalog.registered_groups(main)
    assert not groups and "registered result unavailable" in issues[0]


def test_normal_study_target_keeps_bound_external_store_and_lineage(tmp_path):
    import ifvg_workspace

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import StudySummary

    main, pointer = _published(tmp_path)
    study = StudySummary(
        key="study",
        kind="funded_comparison",
        name="Saved external study",
        question="cash",
        dates="January 2, 2026",
        status="Completed",
        scope="research",
        store_root=Path(pointer["external_store_root"]),
        state={
            "result_id": pointer["result_id"],
            "plan_id": pointer["plan_id"],
            "external_review_only": True,
            "catalog_binding": pointer,
            "versions": {pointer["result_id"]: pointer},
        },
    )
    target = ifvg_workspace.funded_result_target(study, {"store_root": main}, app="main")
    assert target["store_root"] == pointer["external_store_root"]
    assert target["catalog_binding"]["binding_id"] == pointer["binding_id"]
    assert target["versions"][pointer["result_id"]] == pointer


def test_view_preferences_restore_separately_and_clear_invalid_scope(tmp_path, monkeypatch):
    import ifvg_lab_nav as nav

    path = tmp_path / "preferences.json"
    monkeypatch.setattr(nav, "_preferences_path", lambda: path)
    fake = SimpleNamespace(
        session_state={
            nav.CONTEXT_KEY: {
                "version": {
                    "configuration": "A",
                    "firm_key": "mffu",
                    "lens": "cash",
                    "account": {"pair": "A|mffu", "number": 1},
                }
            }
        }
    )
    nav.save_view_preferences(fake)
    restarted = SimpleNamespace(session_state={})
    context = nav.funded_context("version", restarted)
    assert context["lens"] == "cash" and context["account"]["number"] == 1
    study = SimpleNamespace(
        configurations=("B",), firms=(("mffu", "MFFU"),), result={"tables": {}}, trades_by_pair={}
    )
    nav.validate_restored_context(study, context, restarted)
    assert "configuration" not in context and "account" not in context
    assert "configuration" in restarted.session_state[nav.LINK_NOTE]
    assert "lens" in context
