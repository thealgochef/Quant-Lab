"""Actual saved-record regression witnesses; mutations below are synthetic negatives."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.funded_comparison_review import _configuration_rows
from alpha_lab.agents.data_infra.ifvg.presentation.lab import mffu_gamma as gamma

FIXTURE = Path(__file__).parent / "fixtures/b8db_reporting.json"


@pytest.fixture
def saved():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class RecordedIndex:
    """Exact previously captured adapter responses, not a replacement policy."""

    _level_times = ()

    def __init__(self, saved):
        self.saved = saved
        self.bundle_sha256 = saved["bundle_sha256"]
        self.table_sha256 = saved["table_sha256"]

    def snapshot(self, at):
        return deepcopy(self.saved["snapshots"][str(pd.Timestamp(at).value)])


def build(saved, name, mutate=None):
    case = deepcopy(next(c for c in saved["cases"] if c["name"] == name))
    if mutate:
        mutate(case)
    trade = case["trade"]
    result = {
        "tables": {"trades" if case["population"] == "funded" else "strategy_trades": [trade]},
        "mffu_batch": {
            "decision_context": case["decisions"],
            "reuse_context_annotations": case["annotations"],
            "plan": saved["plan"],
        },
    }
    study = SimpleNamespace(
        result=result,
        result_id=saved["source"]["result_id"],
        configurations=(trade["configuration"],),
        calendar=(trade["trading_day"],),
    )
    data = gamma.build_gamma(study, RecordedIndex(saved))
    return data, data["trades"][0]


@pytest.mark.parametrize("name", ["anchor_whole", "conditional_half", "ordinary_whole"])
def test_recorded_checkpoint_uses_own_verified_origin(saved, name):
    data, row = build(saved, name)
    assert row["first_checkpoint_context_role"] == "executed_saved_decision_receipt"
    selected = gamma.selected_rows(
        data, row["configuration_id"], row["population"], "first_1R_checkpoint"
    )
    assert selected[0]["context_role"] == row["first_checkpoint_context_role"]
    assert gamma.coverage(selected)["executed"] == 1
    assert gamma.checkpoint_cards(row)[-1]["context_role"] == row["first_checkpoint_context_role"]


@pytest.mark.parametrize("name", ["unconditional_partial", "reused_control"])
def test_annotations_stay_annotations(saved, name):
    data, row = build(saved, name)
    if row["first_checkpoint_utc"]:
        assert row["first_checkpoint_context_role"].startswith("reporting_annotation")
        selected = gamma.selected_rows(
            data, row["configuration_id"], row["population"], "first_1R_checkpoint"
        )
        assert gamma.coverage(selected)["executed"] == 0
        assert gamma.coverage(selected)["posthoc"] == 1
    else:
        assert row["first_checkpoint_context_role"] == "unavailable"


def test_exact_cursor_never_exposes_target_one_nanosecond_early(saved):
    data, row = build(saved, "anchor_whole")
    ns = 1750068615558271393
    assert pd.Timestamp(row["first_checkpoint_utc"]).value == ns
    before = pd.Timestamp(ns - 1, tz="UTC")
    at = pd.Timestamp(ns, tz="UTC")
    assert not gamma.selected_rows(data, "MCB003", basis="first_1R_checkpoint", cursor=before)
    assert all(
        c["checkpoint"] != "First 1R checkpoint" for c in gamma.checkpoint_cards(row, cursor=before)
    )
    entry_row = gamma.selected_rows(data, "MCB003", cursor=before)[0]
    assert entry_row.get("first_target_policy_receipt") is None
    assert entry_row.get("first_checkpoint_snapshot") is None
    assert len(gamma.selected_rows(data, "MCB003", basis="first_1R_checkpoint", cursor=at)) == 1
    assert gamma.checkpoint_cards(row, cursor=at)[-1]["at_utc"] == row["first_checkpoint_utc"]


def test_all_64_actual_standard_export_describes_micro_exposure(saved):
    result = {
        "tables": {"configurations": saved["configurations"]},
        "settings": saved["settings"],
        "mffu_batch": {"plan": saved["plan"]},
    }
    rows = _configuration_rows(result)
    assert len(rows) == 64
    assert all("Micro E-mini Nasdaq-100 exposure (MNQ)" in r["Traded product"] for r in rows)
    for r in rows[56:62]:
        description = r["Largest distance from the parent gap to the opposing gap"]
        assert "(1D Max" in description and "fallback" in description
        assert "ROUND_HALF_UP" in description


@pytest.mark.parametrize(
    "field,value",
    [
        ("trade_id", "wrong"),
        ("configuration", "MCB064"),
        ("stream", "strategy"),
        ("account_number", 999),
        ("firm_key", "other"),
        ("policy", "fixed_target_v1"),
        ("action", "partial"),
    ],
)
def test_wrong_scoped_receipt_never_becomes_executed(saved, field, value):
    def mutate(case):
        next(d for d in case["decisions"] if d["event"] == "first_target")[field] = value

    _, row = build(saved, "anchor_whole", mutate)
    assert row["first_checkpoint_context_role"] == "unresolved_checkpoint_evidence"
    assert row["first_target_policy_receipt"] is None


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "source", "decision_ns", "observation_ns", "future_report", "receipt_time"],
)
def test_ambiguous_contradictory_nearby_evidence_is_unresolved(saved, fault):
    def mutate(case):
        receipt = next(d for d in case["decisions"] if d["event"] == "first_target")
        embedded = case["trade"]["target_decision"]
        if fault == "duplicate":
            case["decisions"].append(dict(receipt, action="partial"))
        elif fault == "source":
            receipt["context"]["gamma_source_id"] = "f" * 64
        elif fault == "decision_ns":
            embedded["decision_ns"] -= 1
        elif fault == "observation_ns":
            embedded["target_observation_ns"] -= 1
        elif fault == "future_report":
            receipt["context"]["gamma_eligible_from_utc"] = "2025-06-17T03:00:00Z"
            embedded["context"]["receipt"] = deepcopy(receipt["context"])
        else:
            receipt["context"]["decision_ts_utc"] = "2025-06-16T10:10:15.558272+00:00"
            embedded["context"]["receipt"] = deepcopy(receipt["context"])

    _, row = build(saved, "anchor_whole", mutate)
    assert row["first_checkpoint_context_role"] == "unresolved_checkpoint_evidence"


def test_receipt_timezone_representation_preserves_exact_event(saved):
    def mutate(case):
        receipt = next(d for d in case["decisions"] if d["event"] == "first_target")
        receipt["context"]["decision_ts_utc"] = "2025-06-16T05:10:15.558271-05:00"
        case["trade"]["target_decision"]["context"]["receipt"] = deepcopy(receipt["context"])

    _, row = build(saved, "anchor_whole", mutate)
    assert row["first_checkpoint_origin"]["status"] == "verified"
    assert row["first_checkpoint_origin"]["observation_ns"] == 1750068615558271393
    assert row["first_checkpoint_origin"]["precision_difference_ns"] == 393


@pytest.mark.parametrize("reached", [False, True])
def test_no_target_is_not_annotation_and_missing_time_is_unavailable(saved, reached):
    def mutate(case):
        case["decisions"] = [d for d in case["decisions"] if d["event"] != "first_target"]
        case["trade"].pop("target_decision", None)
        case["trade"]["exit_utc"] = None
        case["trade"]["exit_kind"] = "target" if reached else "stop"

    _, row = build(saved, "anchor_whole", mutate)
    assert row["first_checkpoint_context_role"] == ("unavailable" if reached else "not_reached")


def test_cards_and_coverage_are_independent_of_entry_role(saved):
    _, row = build(saved, "anchor_whole")
    row["context_role"] = row["entry_context_role"] = "posthoc_reporting_annotation"
    cards = {c["checkpoint"]: c for c in gamma.checkpoint_cards(row)}
    assert cards["Entry"]["context_role"] == "posthoc_reporting_annotation"
    assert cards["First 1R checkpoint"]["context_role"] == "executed_saved_decision_receipt"
    row["lock_context_role"] = "unresolved_checkpoint_evidence"
    cards = {c["checkpoint"]: c for c in gamma.checkpoint_cards(row)}
    if "Supporting-pattern lock" in cards:
        assert cards["Supporting-pattern lock"]["context_role"] == "unresolved_checkpoint_evidence"
    rows = [
        {"context_role": role}
        for role in [
            "executed_saved_decision_receipt",
            "reporting_annotation_at_actual_funded_fill",
            "unavailable",
        ]
    ]
    counts = gamma.coverage(rows)
    assert (counts["executed"], counts["posthoc"], counts["unavailable"]) == (1, 1, 1)


def test_shared_export_labels_all_axes_and_ui_agree(saved):
    from alpha_lab.agents.data_infra.ifvg.funded_comparison_review import _chart_label
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        configuration_columns,
    )
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import (
        canonical_settings,
        result_variants,
    )

    result = {
        "tables": {"configurations": saved["configurations"]},
        "settings": saved["settings"],
        "mffu_batch": {"plan": saved["plan"]},
    }
    variants = result_variants(result)
    columns = configuration_columns(result)
    labels = []
    for row in _configuration_rows(result):
        key = row["configuration"]
        settings = dict(canonical_settings(variants[key]))
        for name, value in settings.items():
            assert row[name] == value
        label = _chart_label(SimpleNamespace(configuration=key, rank_text="1", label=key), columns)
        labels.append(label)
        assert key in label and "MNQ" in label
        if json.loads(variants[key].effective_section_json)["exit_policy"] in {
            "gamma_conditional_1r_v1",
            "early_positive_whole_1r_v1",
        }:
            assert "unknown" in row["Exit rule"] and "else half" in label
    assert len(set(labels)) == 64
    for i, fraction in zip(range(57, 63), ["5%", "7.5%", "10%"] * 2, strict=True):
        assert (
            fraction
            in dict(canonical_settings(variants[f"MCB{i:03d}"]))[
                "Largest distance from the parent gap to the opposing gap"
            ]
        )


def test_report_boundary_eligibility_uses_declared_time_not_near_match(saved):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_provenance import receipt_matches

    case = saved["cases"][0]
    context = deepcopy(case["trade"]["target_decision"]["context"]["receipt"])
    snapshot = deepcopy(case["trade"]["target_decision"]["context"]["asof"])
    # Synthetic source release is one microsecond after the decision.
    context["gamma_eligible_from_utc"] = "2025-06-16T10:10:15.558272Z"
    snapshot["gamma"]["nominal_eligible_from_utc"] = context["gamma_eligible_from_utc"]
    assert not receipt_matches(context, snapshot)
    context["decision_ts_utc"] = "2025-06-16T05:10:15.558272-05:00"
    assert receipt_matches(context, snapshot)


def test_unverified_ordinary_receipt_cannot_supply_future_card_context(saved):
    def mutate(case):
        receipt = next(d for d in case["decisions"] if d["event"] == "first_target")
        receipt["context"]["decision_ts_utc"] = "2025-08-04T12:57:00+00:00"

    _, row = build(saved, "ordinary_whole", mutate)
    assert row["first_checkpoint_origin"]["status"] == "unresolved"
    assert row["first_checkpoint_origin"]["reporting_evaluation_utc"] is None
    assert row["first_checkpoint_snapshot"] is None
    card = gamma.checkpoint_cards(row, cursor=row["first_checkpoint_utc"])[-1]
    assert card["snapshot"] is None and card["recorded_policy"] is None


def test_parent_lock_selection_does_not_reveal_a_future_trade(saved):
    data, row = build(saved, "anchor_whole")
    row["lock_policy_receipt"] = {"context": {"decision_ts_utc": "2025-06-16T10:06:00Z"}}
    assert not gamma.selected_rows(
        data, "MCB003", basis="parent_lock", cursor="2025-06-16T10:06:00Z"
    )
