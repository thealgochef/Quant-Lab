"""Explicit external MFFU review links leave ordinary Lab stores unchanged."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))


def test_explicit_external_result_opens_verified_read_only_target(tmp_path, monkeypatch):
    import ifvg_lab_nav as nav

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_data
    from alpha_lab.propsim.funded import mffu_batch_review

    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "task_store"
    external.mkdir()
    marker = external / "unchanged.txt"
    marker.write_text("saved", encoding="utf-8")
    result_id = "a" * 64
    plan_id = "b" * 64
    monkeypatch.setenv("IFSM_MFFU_REVIEW_STORE_ROOT", str(external))
    monkeypatch.setenv("IFSM_MFFU_REVIEW_RESULT_ID", result_id)
    seen = []

    def open_study(root, selected_id):
        seen.append((root, selected_id))
        return SimpleNamespace(plan_id=plan_id, plan=object(), result={"verified": True})

    monkeypatch.setattr(funded_data, "open_funded_study", open_study)
    monkeypatch.setattr(mffu_batch_review, "_assert_result",
                        lambda plan, selected_plan, result: seen.append(
                            (plan, selected_plan, result)))
    monkeypatch.setattr(nav, "_saved_target", lambda *_: pytest.fail("ordinary store opened"))
    fake = SimpleNamespace(session_state={})

    nav._apply_link({"view": "funded", "app": "ifsm", "result": result_id},
                    {"repo_root": repo}, fake)

    target = fake.session_state[nav.FUNDED_TARGET]
    assert target == {
        "result_id": result_id, "plan_id": plan_id,
        "store_root": str(external.resolve()), "app": "ifsm",
        "study_key": result_id, "name": "IFSM MFFU context batch",
        "status": "Completed", "external_review_only": True,
    }
    assert seen[0] == (external.resolve(), result_id)
    assert seen[1][1:] == (plan_id, {"verified": True})
    assert marker.read_text(encoding="utf-8") == "saved"


def test_external_route_requires_exact_id_and_external_directory(tmp_path, monkeypatch):
    import ifvg_lab_nav as nav

    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "task_store"
    external.mkdir()
    result_id = "a" * 64
    roots = {"repo_root": repo}
    assert nav._external_mffu_review_target(result_id, "ifsm", roots) is None

    monkeypatch.setenv("IFSM_MFFU_REVIEW_STORE_ROOT", str(external))
    with pytest.raises(ValueError, match="both environment fields"):
        nav._external_mffu_review_target(result_id, "ifsm", roots)
    monkeypatch.setenv("IFSM_MFFU_REVIEW_RESULT_ID", "a" * 16)
    with pytest.raises(ValueError, match="full SHA-256 key"):
        nav._external_mffu_review_target(result_id, "ifsm", roots)
    monkeypatch.setenv("IFSM_MFFU_REVIEW_RESULT_ID", "b" * 64)
    assert nav._external_mffu_review_target(result_id, "ifsm", roots) is None
    monkeypatch.setenv("IFSM_MFFU_REVIEW_RESULT_ID", result_id)
    monkeypatch.setenv("IFSM_MFFU_REVIEW_STORE_ROOT", str(repo))
    with pytest.raises(ValueError, match="outside the repository"):
        nav._external_mffu_review_target(result_id, "ifsm", roots)
    assert nav._external_mffu_review_target(result_id, "main", roots) is None


def test_external_target_cannot_offer_publish_again():
    import ifvg_lab_detail_settings as settings

    target = {"external_review_only": True, "result_id": "a" * 64}
    assert settings.publish_again_route(target, object(), {}) is None
