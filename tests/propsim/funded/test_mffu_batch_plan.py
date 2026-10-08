"""Contracts for the one frozen 64-intent MyFundedFutures plan."""

from __future__ import annotations

import json
from hashlib import sha256
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.propsim.funded import mffu_batch_plan as batch


def _matrix():
    rows = []

    def add(family, schedule, cap, context, exit_code, sizing, overhead, geometry):
        rows.append({
            "variant_id": f"MCB{len(rows) + 1:03d}", "family": family,
            "schedule": schedule, "daily_cap": cap, "entry_context": context,
            "exit": exit_code, "sizing": sizing, "overhead": overhead,
            "geometry": geometry, "firm": "myfundedfutures",
            "base_reference": f"{schedule}-T1-H14-P1-L-SO",
        })

    for schedule, cap, context, exit_code in product(
        ("S0", "S1"), ("U", "D1"), ("F0", "FE", "FL", "FEL"),
        ("XP", "XF", "XG"),
    ):
        add("core_48", schedule, cap, context, exit_code, "Q10", "O0", "G0")
    for schedule, sizing in product(("S0", "S1"), ("Q6", "QG")):
        add("sizing", schedule, "U", "F0", "XP", sizing, "O0", "G0")
    for schedule, cap in product(("S0", "S1"), ("U", "D1")):
        add("overhead", schedule, cap, "F0", "XP", "Q10", "O8", "G0")
    for schedule, geometry in product(("S0", "S1"), ("G05", "G075", "G10")):
        add("geometry", schedule, "U", "F0", "XP", "Q10", "O0", geometry)
    for schedule in ("S0", "S1"):
        add("state_exit", schedule, "U", "F0", "XE", "Q10", "O0", "G0")
    matrix = {"schema": "ifsm_64_candidate_intent_v1", "count": 64,
              "core_count": 48, "targeted_count": 16, "variants": rows}
    policy = {"schema": "ifsm_mffu_context_research_policy_intent_v1",
              "day_cap": {"U": None, "D1": 1},
              "data": {"batch_adapter": "mq_eod_asof_nominal_2200_chicago_v01",
                       "nominal_publication_local": "22:00"},
              "constants": {"firm": "myfundedfutures",
                            "micro_cost_per_contract_per_fill_usd": "0.514"},
              "overhead": {"O8": {"threshold_vendor_units": 300000,
                                   "threshold_comparator": "strictly_greater"}},
              "geometry": {"G05": {"implied_move_fraction": "0.05"},
                           "G075": {"implied_move_fraction": "0.075"},
                           "G10": {"implied_move_fraction": "0.10"}},
              "sizing": {"Q6": {"positive": 6},
                         "QG": {"positive": 6, "negative": 10}},
              "schedule": {
                  "S0": {"base_configuration": "S0-T1-H14-P1-L-SO"},
                  "S1": {"base_configuration": "S1-T1-H14-P1-L-SO"},
              }}
    return matrix, policy


def test_complete_matrix_and_targeted_membership():
    matrix, policy = _matrix()
    assert len(batch._validate_matrix(matrix, policy)) == 64
    matrix["variants"][-1]["schedule"] = "S0"
    matrix["variants"][-1]["base_reference"] = "S0-T1-H14-P1-L-SO"
    with pytest.raises(ValueError, match="state_exit"):
        batch._validate_matrix(matrix, policy)


def test_external_frozen_runtime_can_be_read_from_normal_dashboard(monkeypatch, tmp_path):
    runtime = tmp_path / "frozen"
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path / "normal_dashboard")
    paths = {str(runtime / relative): "a" * 64 for relative in batch.REQUIRED_RUNTIME_SOURCES}
    assert batch._runtime_source_root(paths) == runtime.as_posix()
    first = batch.REQUIRED_RUNTIME_SOURCES[0]
    paths[str(tmp_path / "foreign" / first)] = "a" * 64
    with pytest.raises(ValueError, match="required economic source"):
        batch._runtime_source_root(paths)
    del paths[str(runtime / first)]
    with pytest.raises(ValueError, match="one frozen runtime root"):
        batch._runtime_source_root(paths)


def test_resolved_section_keeps_inherited_fields_and_distinguishes_dynamic_policy(monkeypatch):
    monkeypatch.setattr(
        batch, "_resolve_section",
        lambda section: (section, canonical_contract_sha256(section)),
    )
    matrix, _policy = _matrix()
    base = {"effective_section_json": json.dumps({
        "profile_name": "saved", "holding_policy": "scheduled_daily_close_v1",
        "exit_policy": "scale_out_half_breakeven_hold_to_close_v1",
        "max_executed_trades_per_day": None,
    }), "effective_section_config_hash": "a" * 64}
    passive = batch._variant_from_intent(matrix["variants"][0], base)
    gamma_exit = batch._variant_from_intent(matrix["variants"][2], base)
    dynamic_size = batch._variant_from_intent(matrix["variants"][49], base)
    assert passive.base_section_config_hash == "a" * 64
    assert passive.quantity == 10
    assert json.loads(passive.effective_section_json)["ifsm_context_policy_version"] == (
        "mq_eod_asof_nominal_2200_chicago_v01"
    )
    assert json.loads(gamma_exit.effective_section_json)["exit_policy"] == (
        "gamma_conditional_1r_v1"
    )
    assert json.loads(gamma_exit.effective_section_json)["ifsm_context_policy_version"] == (
        "mq_eod_asof_nominal_2200_chicago_v01"
    )
    assert dynamic_size.possible_quantities == (6, 10)
    assert dynamic_size.quantity == 10
    assert len({passive.effective_behavior_hash, gamma_exit.effective_behavior_hash,
                dynamic_size.effective_behavior_hash}) == 3


def test_supplied_handoff_manifest_when_available():
    handoff = Path.home() / "Downloads" / "IFSM_MFFU_Context_Batch_v01.zip"
    if not handoff.exists():
        pytest.skip("user supplied handoff is outside the repository")
    matrix, policies, scope, _members, _archives, tables = batch._read_handoff(handoff)
    assert len(batch._validate_matrix(matrix, policies)) == 64
    assert scope["source_result_id"] == batch.REFERENCE_RESULT_ID
    assert len(scope["evaluation_dates"]) == 253
    assert set(tables) == set(batch.CONTEXT_TABLES)


def test_registered_loader_mutation_invalidates_source_bound_plan(monkeypatch, tmp_path):
    """A formerly omitted replay loader must be frozen without reading market data."""
    relative = "src/alpha_lab/agents/data_infra/ifvg/prepared_store.py"
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    loader = tmp_path / relative
    loader.parent.mkdir(parents=True)
    loader.write_text("loader_version = 1\n", encoding="utf-8")
    frozen_hash = sha256(loader.read_bytes()).hexdigest()
    bound_sources = set(batch._runtime_source_paths(()))
    for source in (
        relative,
        "src/alpha_lab/agents/data_infra/ifvg/day_artifacts.py",
        "src/alpha_lab/propsim/funded/comparison_run.py",
        "src/alpha_lab/propsim/funded/runner.py",
    ):
        assert str((tmp_path / source).resolve()) in bound_sources

    handoff = tmp_path / "handoff.zip"
    context = tmp_path / "context.zip"
    handoff.write_bytes(b"frozen handoff")
    context.write_bytes(b"frozen context")
    monkeypatch.setattr(batch, "_context_hashes", lambda _path: {})
    plan = SimpleNamespace(
        handoff_zip=handoff,
        handoff_zip_sha256=sha256(handoff.read_bytes()).hexdigest(),
        context_archive_path=context,
        context_archive_sha256=sha256(context.read_bytes()).hexdigest(),
        context_table_sha256={},
        runtime_source_file_sha256={str(loader.resolve()): frozen_hash},
    )
    loader.write_text("loader_version = 2\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="frozen source changed: .*prepared_store.py"):
        batch.verify_mffu_batch_plan(plan)


def test_final_source_mutation_invalidates_prepared_plan(monkeypatch, tmp_path):
    from alpha_lab.propsim.funded import core_identity

    handoff = Path.home() / "Downloads" / "IFSM_MFFU_Context_Batch_v01.zip"
    artifacts = Path.home() / "Documents" / "Claude-Quant-Lab-Research-Artifacts"
    task = artifacts / "ifsm-mffu-context-batch-20261007"
    core = task / "core"
    context = (task / "handoff" / "docs" / "ifsm-mffu-context-batch-v01" /
               "inputs" / "MenthorQ_Research_Data_v02.zip")
    reference = artifacts / "ifsm-correct-config-full-range-v01" / "study_store"
    if not all(path.exists() for path in (handoff, core, context, reference)):
        pytest.skip("task-owned external sources are not installed")
    monkeypatch.setattr(batch, "_resolve_section",
                        lambda section: (section, canonical_contract_sha256(section)))
    monkeypatch.setattr(core_identity, "core_source_identity",
                        lambda: core_identity.source_identity_at(core))
    final_worker_source = tmp_path / "worker.py"
    final_worker_source.write_text("version = 1\n", encoding="utf-8")
    plan = batch.build_mffu_batch_plan(
        handoff_zip=handoff, reference_store_root=reference,
        context_archive_path=context, runtime_source_files=(final_worker_source,),
        core_root=core,
    )
    batch.verify_mffu_batch_plan(plan.payload)
    store = tmp_path / "store"
    assert batch.save_mffu_batch_plan(store, plan) == plan.funded_comparison_plan_id
    from alpha_lab.propsim.funded.comparison_runner import load_plan

    assert len(load_plan(store, plan.funded_comparison_plan_id).configurations) == 64
    with pytest.raises(PermissionError, match="no owner approval"):
        batch.load_approved_mffu_batch_plan(store, plan.funded_comparison_plan_id)
    final_worker_source.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="frozen source changed"):
        batch.verify_mffu_batch_plan(plan.payload)
