"""Working-output defaults cannot repopulate the curated reports directory."""

import importlib
import sys
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.working_artifacts import (
    external_working_output,
    research_working_directory,
)


def test_task_working_path_stays_outside_repository_without_creating_files(tmp_path):
    repository = tmp_path / "checkout"
    path = research_working_directory(repository, "daily-close_20260922")
    assert path == tmp_path / "Claude-Quant-Lab-Research-Artifacts" / "daily-close_20260922"
    assert not path.is_relative_to(repository)
    assert not path.exists()


@pytest.mark.parametrize("task_id", ["", ".", "..", "../reports", "a/b", r"a\b", "C:temp"])
def test_task_working_path_refuses_escape(tmp_path, task_id):
    with pytest.raises(ValueError, match="task_id"):
        research_working_directory(tmp_path, task_id)


def test_explicit_working_output_rejects_repo_and_reports(tmp_path):
    repository = tmp_path / "checkout"
    for output in (repository, repository / "reports" / "dumps", repository / "scratch"):
        with pytest.raises(ValueError, match="outside the repository"):
            external_working_output(repository, output)


@pytest.mark.parametrize("directory", ["reports", "RePoRtS"])
def test_external_reports_directory_is_reserved_too(tmp_path, directory):
    with pytest.raises(ValueError, match="every reports directory"):
        external_working_output(tmp_path / "checkout", tmp_path / directory / "raw")


@pytest.mark.parametrize("reports_is_alias", [True, False])
def test_reports_alias_cannot_hide_supplied_or_resolved_directory(tmp_path, reports_is_alias):
    reports = tmp_path / "reports"
    working = tmp_path / "working"
    target, alias = (working, reports) if reports_is_alias else (reports, working)
    target.mkdir()
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows host does not permit creating directory symlinks")
    with pytest.raises(ValueError, match="every reports directory"):
        external_working_output(tmp_path / "checkout", alias / "raw")


def test_working_output_rejects_external_symlink_into_reports(tmp_path):
    repository = tmp_path / "checkout"
    reports = repository / "reports"
    reports.mkdir(parents=True)
    alias = tmp_path / "apparently_external"
    try:
        alias.symlink_to(reports, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows host does not permit creating directory symlinks")
    with pytest.raises(ValueError, match="outside the repository"):
        external_working_output(repository, alias / "raw_evidence")


@pytest.mark.parametrize(
    "module_name, flag, required",
    [
        ("prepare_ifvg_fsm_audit", "--parity-report", []),
        ("ifvg_fsm_audit_reports", "--output-dir", ["--audit-id", "1" * 64]),
        ("run_ifvg_tf_variant", "--output-dir", []),
    ],
)
def test_working_cli_rejects_reports_before_loading_data(
    monkeypatch, tmp_path, module_name, flag, required
):
    module = importlib.import_module(f"scripts.{module_name}")
    repository = tmp_path / "checkout"
    monkeypatch.setattr(module, "ROOT", repository)
    monkeypatch.setattr(
        sys, "argv", [module_name, *required, flag, str(repository / "reports" / "dump")]
    )
    with pytest.raises(ValueError, match="outside the repository"):
        module.main()
    assert not repository.exists()


@pytest.mark.parametrize("explicit_output", [False, True])
def test_audit_preparation_routes_parity_output_without_changing_saved_store(
    monkeypatch, tmp_path, explicit_output
):
    module = importlib.import_module("scripts.prepare_ifvg_fsm_audit")
    monkeypatch.setattr(module, "ROOT", tmp_path / "checkout")
    arguments = ["prepare_ifvg_fsm_audit.py"]
    requested = tmp_path / "chosen" / "parity.json"
    if explicit_output:
        arguments += ["--parity-report", str(requested)]
    monkeypatch.setattr(sys, "argv", arguments)
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(module, "prepare_ifvg_fsm_audit_persisted", prepare)
    monkeypatch.setattr(module, "summarize_result", lambda _: "prepared")
    assert module.main() == 0
    assert captured["repo_root"] == tmp_path / "checkout"
    assert captured["cached_artifacts_only"] is True
    expected = (
        requested
        if explicit_output
        else (
            tmp_path
            / "Claude-Quant-Lab-Research-Artifacts"
            / "ifvg_fsm_audit"
            / "IFVG_FSM_AUDITABILITY_PARITY_REPORT.json"
        )
    )
    assert captured["parity_report_path"] == expected
    assert not (tmp_path / "checkout" / "reports").exists()


def test_saved_audit_report_writer_respects_explicit_output(tmp_path):
    module = importlib.import_module("scripts.ifvg_fsm_audit_reports")
    module._write_json("summary.json", {"passed": True}, output_dir=tmp_path)
    assert (tmp_path / "summary.json").read_text().strip() == '{\n  "passed": true\n}'


@pytest.mark.parametrize("module_name", ["ifvg_fsm_audit_reports", "run_ifvg_tf_variant"])
def test_report_generator_defaults_are_external(module_name):
    module = importlib.import_module(f"scripts.{module_name}")
    assert not module.REPORT_DIR.is_relative_to(module.ROOT)
    assert module.REPORT_DIR.is_relative_to(
        Path(module.ROOT).parent / "Claude-Quant-Lab-Research-Artifacts"
    )
