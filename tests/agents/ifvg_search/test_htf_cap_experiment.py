"""No-market-data preflight, exact four profiles, and supplemental audit reconciliation."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.ifsm_replication import fixed_axis_values
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    assert_axes_authorized,
    resolve_axis_overrides,
)
from alpha_lab.agents.data_infra.ifvg.search.htf_cap_experiment import (
    MEMBERS,
    assert_effective_matrix,
    fixed_profile_drafts,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import canonicalize_section
from alpha_lab.agents.data_infra.ifvg.selection_audit import KINDS, validate_selection_evidence
from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft, save_draft

FIXTURE = Path(__file__).with_name("fixtures") / "htf_cap_experiment.json"


def matrix(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    configs, drafts = {}, []
    for name, draft in fixed_profile_drafts(fixture["control"], fixture["charter"]):
        save_draft(tmp_path, draft)
        loaded = load_draft(tmp_path, draft.draft_id)
        selected = fixed_axis_values(loaded)
        resolved = resolve_profile_config(
            {
                "profile_name": fixture["charter"]["baseline_profile_name"],
                "section_overrides": resolve_axis_overrides(selected),
            }
        )
        section = canonicalize_section(resolved.section).model_dump(mode="json")
        configs[name] = {**fixture["control"], "section": section}
        drafts.append(loaded)
    return fixture, configs, drafts


def test_exact_four_effective_profiles_and_distinct_cap_identities(tmp_path):
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection, ifvg_profile_hash

    fixture, configs, _ = matrix(tmp_path)
    assert assert_effective_matrix(configs, fixture["control"])["passed"]
    hashes = {
        ifvg_profile_hash(IfvgSmcSection.model_validate(c["section"])) for c in configs.values()
    }
    assert len(hashes) == 4


def test_exact_matrix_draft_to_serialized_worker_enumeration(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import ifvg_study_wizard as wizard

    from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import enumerate_children

    _, _, drafts = matrix(tmp_path / "drafts")
    pairs = set()
    for draft in drafts:
        fields = wizard._charter_fields(draft, {"store_root": tmp_path / "store"})
        # Synthetic authorization is confined to this in-memory configuration
        # unit test; the real launcher records separate owner approvals.
        fields["owner_authorization"] = wizard.SyntheticAuthorizationMarker()
        envelope = SearchCharterEnvelope.from_payload(wizard.SearchCharterPayload(**fields))
        loaded = SearchCharterEnvelope.model_validate_json(envelope.model_dump_json())
        children = enumerate_children(
            loaded, identity_resolver=lambda s: s.resolved_section_config_hash
        )
        assert len(children) == 1
        section = resolve_profile_config(
            {
                "profile_name": loaded.payload.baseline_profile_name,
                "section_overrides": children[0].section_overrides,
            }
        ).section
        assert section.parent_retest_timeout_1m_bars == 240
        pairs.add(
            (section.htf_selection_max_per_timeframe, section.opposing_parent_distance_ticks_max)
        )
    assert pairs == {(1, 80), (2, 80), (1, 160), (2, 160)}


@pytest.mark.parametrize("mutation", ["timeout", "entry_family", "extra", "evaluator"])
def test_effective_preflight_rejects_hidden_drift(tmp_path, mutation):
    fixture, configs, _ = matrix(tmp_path)
    config = configs[MEMBERS[1][0]]
    if mutation == "timeout":
        config["section"]["parent_retest_timeout_1m_bars"] = None
    elif mutation == "entry_family":
        config["section"]["entry_family"] = "ifvg_retest"
    elif mutation == "extra":
        configs["unexpected"] = copy.deepcopy(config)
    else:
        config["evaluator"] = {**config["evaluator"], "bootstrap_samples": 9}
    with pytest.raises(ValueError):
        assert_effective_matrix(configs, fixture["control"])


@pytest.mark.parametrize("value", ["0", "-1", "3", "2.5", "true"])
def test_unregistered_cap_values_refused(value):
    with pytest.raises(PermissionError):
        assert_axes_authorized(
            {"htf_selection_max_per_timeframe": f"htf_selection_max_per_timeframe.{value}"},
            require_ratified=False,
        )


def test_cap_two_requires_exact_owner_ratification():
    with pytest.raises(PermissionError):
        assert_axes_authorized(
            {"htf_selection_max_per_timeframe": "htf_selection_max_per_timeframe.2"}
        )


def evidence():
    encoded = "[]"
    ident = hashlib.sha256(encoded.encode()).hexdigest()
    return {
        KINDS[0]: pd.DataFrame(
            [
                {
                    "stamp_source_bar_id": "bar0",
                    "stamp_source_bar_cursor": "cursor0",
                    "universe_id": ident,
                    "selection_cap": 2,
                    "inventory_live_count": 0,
                    "inventory_view_ids_json": "[]",
                    "scan_mode": "free_slot_scan",
                    "observed_tap_ids_json": "[]",
                    "observed_selected_id": "",
                    "observed_drop_counts_json": "{}",
                    "conflict_predicate": "not_evaluated",
                    "direction_enabled_predicate": "not_evaluated",
                }
            ]
        ),
        KINDS[1]: pd.DataFrame([{"universe_id": ident, "zones_json": encoded}]),
        KINDS[2]: pd.DataFrame(),
    }


def test_selection_reconciliation_checks_actual_bar_population():
    assert validate_selection_evidence(evidence(), expected_bar_ids=["bar0"], taps=pd.DataFrame())[
        "passed"
    ]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "universe", "tap", "view"])
def test_negative_selection_reconciliation(bad):
    tables = evidence()
    if bad == "missing":
        tables[KINDS[0]] = tables[KINDS[0]].iloc[:0]
    elif bad == "duplicate":
        tables[KINDS[0]] = pd.concat([tables[KINDS[0]], tables[KINDS[0]]])
    elif bad == "universe":
        tables[KINDS[1]] = tables[KINDS[1]].iloc[:0]
    elif bad == "tap":
        tables[KINDS[0]].loc[0, "observed_tap_ids_json"] = '["invented"]'
        tables[KINDS[0]].loc[0, "observed_drop_counts_json"] = '{"selected":1}'
    else:
        tables[KINDS[0]].loc[0, "inventory_view_ids_json"] = '["invented"]'
    with pytest.raises(ValueError):
        validate_selection_evidence(tables, expected_bar_ids=["bar0"], taps=pd.DataFrame())
