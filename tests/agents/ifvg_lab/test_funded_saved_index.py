"""Pure saved-result indexing and exact FullRange display labels."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import ifvg_lab_funded as funded  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (  # noqa: E402
    study_from_result,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import (  # noqa: E402
    ConfigurationName,
)


def saved_result():
    completed = ["C04", "C03", "C05", "C06"]
    settings = [{"setting": "Supporting (parent) charts", "value": "one-minute, three-minute"}]
    result = {
        "tables": {"configurations": [{"configuration": key, "settings": settings}
                                      for key in completed]},
        "summaries_cents": {},
    }
    for key in ["C01", *completed, "C02"]:
        for firm in ["firm_b", "firm_a"]:
            result["summaries_cents"][f"{key}|{firm}"] = {
                "configuration": key, "firm_key": firm,
                "status": "Completed" if key in completed else "Not completed",
                **({"net_cash_earned_cents": 12345} if key in completed else
                   {"reason": "Saved execution failure"}),
            }
    return result


def test_saved_index_includes_failed_configurations_once_preserving_existing_order():
    result = saved_result()
    before = copy.deepcopy(result)
    study = study_from_result(result, result_id="synthetic")
    assert study.configurations == ("C04", "C03", "C05", "C06", "C01", "C02")
    assert {row["configuration"] for row in study.completed_at("firm_a")} == {
        "C03", "C04", "C05", "C06"}
    assert study.summary("C01", "firm_a")["status"] == "Not completed"
    assert "net_cash_earned_cents" not in study.summary("C01", "firm_a")
    assert result == before and study.result is result


def test_full_range_names_use_saved_parent_timeframes_for_completed_and_failed_rows(monkeypatch):
    import ifvg_lab_cache

    def forbidden(*args):
        raise AssertionError("FullRange display must use the loaded saved plan, not another store")

    monkeypatch.setattr(ifvg_lab_cache, "configuration_names", forbidden)
    parents = ["1m", "3m", "5m", "10m", "15m", "30m"]
    plan = SimpleNamespace(plan_schema="ifsm_correct_config_full_range_plan_v1", configurations=[
        SimpleNamespace(name=f"C{index:02d}", effective_section_json=json.dumps({
            "parent_timeframes": parents if index != 6 else ["5m", "1m"]}))
        for index in range(1, 7)])
    study = study_from_result(saved_result(), result_id="synthetic", plan=plan)
    names = funded.study_names_for(study)
    assert tuple(names) == study.configurations
    for key in ("C01", "C02", "C03", "C04", "C05"):
        assert "1, 3, 5, 10, 15, 30-minute parents" in names[key].line2
        assert "1- and 3-minute parents" not in names[key].line2
    assert "5, 1-minute parents" in names["C06"].line2
    assert names["C01"].line1 == "C01"


def test_other_plan_names_keep_existing_cached_path(monkeypatch):
    import ifvg_lab_cache

    expected = {"C03": ConfigurationName("saved", "old parent wording", "saved")}
    calls = []

    def cached(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(ifvg_lab_cache, "configuration_names", cached)
    study = study_from_result(saved_result(), result_id="synthetic")
    assert funded.study_names_for(study) is expected
    assert calls == [("", "synthetic")]
