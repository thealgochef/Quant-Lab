"""Package-less saved batches must not borrow another study's market evidence."""
from pathlib import Path
from types import SimpleNamespace

from alpha_lab.agents.data_infra.ifvg.presentation.lab.market import study_package_root


def test_registered_input_plan_has_no_package_lookup(monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.presentation import funded_trade_review

    def forbidden(*args):
        raise AssertionError("A package-less plan must not discover unrelated market bars")

    monkeypatch.setattr(funded_trade_review, "plan_strategy_package", forbidden)
    plan = SimpleNamespace(source=SimpleNamespace(
        source_kind="verified_task_b_registered_inputs", task_b_plan_id="reference-only"))
    assert study_package_root(plan) is None


def test_bound_package_plan_keeps_verified_lookup(monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.presentation import funded_trade_review

    calls = []
    expected = Path("verified-bound-package")
    plan = SimpleNamespace(source=SimpleNamespace(package_root_name="exact-package"))

    def verified_lookup(actual_plan, archive_root):
        calls.append((actual_plan, archive_root))
        return expected

    monkeypatch.setattr(funded_trade_review, "plan_strategy_package", verified_lookup)
    assert study_package_root(plan) == expected
    assert len(calls) == 1 and calls[0][0] is plan
