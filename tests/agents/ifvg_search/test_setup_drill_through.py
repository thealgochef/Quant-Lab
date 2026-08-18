"""Exact-setup drill-through: provider contract + the full R2 chain (B-M2/B-M4).

The provider half proves ``resolve_selection(setup_id=…)``'s three exact
outcomes over a duck-typed verified pair (single-candidate resolve,
zero-candidate refusal toward setup mode, multi-candidate refusal with the
exact count) plus the exactly-one-ID rule. The chain half drives the R2 gate
bullet end to end: population-delta first divergence → ``queue_jump`` →
setup-mode routing → exact resolution over the same tables — no fallback at
any hop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import (
    MissingEvidenceError,
    resolve_selection,
)
from alpha_lab.agents.data_infra.ifvg.study.population_delta import (
    build_native_population_delta,
)
from tests.agents.ifvg_search.test_lineage import _entity_tables

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_verifier_tab as tab  # noqa: E402


def _ctx_for(tables: dict) -> SimpleNamespace:
    """A duck-typed ReplayContext: `_table` reads pair.v2.tables[table]."""

    candidates = tables.get(RecordTable.ENTRY_CANDIDATE, pd.DataFrame()).copy()
    if not candidates.empty and "setup_id" not in candidates.columns:
        candidates["setup_id"] = candidates["envelope_setup_id"]
    dossiers = pd.DataFrame(
        {
            "candidate_id": candidates.get("candidate_id", pd.Series(dtype=str)),
            "decision_id": [f"dec-of-{c}" for c in candidates.get("candidate_id", ())],
            "trade_id": [f"trade-of-{c}" for c in candidates.get("candidate_id", ())],
        }
    )
    v2 = SimpleNamespace(
        tables={
            RecordTable.ENTRY_CANDIDATE: candidates,
            RecordTable.GEOMETRY_DOSSIER: dossiers,
        }
    )
    return SimpleNamespace(pair=SimpleNamespace(v2=v2, v3=None))


def test_setup_id_resolves_only_the_unique_candidate() -> None:
    ctx = _ctx_for(_entity_tables())
    assert resolve_selection(ctx, setup_id="setup-A") == "cand-1"
    assert resolve_selection(ctx, setup_id="setup-B") == "cand-2"


def test_setup_id_zero_candidates_refuses_toward_setup_mode() -> None:
    tables = _entity_tables()
    candidates = tables[RecordTable.ENTRY_CANDIDATE]
    tables[RecordTable.ENTRY_CANDIDATE] = candidates[
        candidates["envelope_setup_id"] != "setup-A"
    ].reset_index(drop=True)
    ctx = _ctx_for(tables)
    with pytest.raises(MissingEvidenceError, match="no entry candidate"):
        resolve_selection(ctx, setup_id="setup-A")


def test_setup_id_multi_candidate_refuses_with_the_exact_count() -> None:
    tables = _entity_tables()
    candidates = tables[RecordTable.ENTRY_CANDIDATE].copy()
    extra = candidates.iloc[[0]].assign(candidate_id="cand-3")
    tables[RecordTable.ENTRY_CANDIDATE] = pd.concat(
        [candidates, extra], ignore_index=True
    )
    ctx = _ctx_for(tables)
    with pytest.raises(MissingEvidenceError, match="has 2 entry candidates"):
        resolve_selection(ctx, setup_id="setup-A")


def test_exactly_one_exact_id_kind_including_setup() -> None:
    ctx = _ctx_for(_entity_tables())
    with pytest.raises(MissingEvidenceError, match="exactly one exact ID"):
        resolve_selection(ctx, setup_id="setup-A", candidate_id="cand-1")
    with pytest.raises(MissingEvidenceError, match="exactly one exact ID"):
        resolve_selection(ctx)


def test_funnel_delta_drill_through_chain_end_to_end() -> None:
    """B-M4: divergence entity → queue_jump → setup-mode routing → exact
    resolution over the artifact's own tables; a missing id refuses cleanly."""

    baseline = _entity_tables()
    challenger = _entity_tables()
    # the challenger loses setup-B entirely (lifecycle + candidate rows) —
    # an exact-set divergence with nothing left to resolve against
    lifecycle = challenger[RecordTable.SETUP_LIFECYCLE]
    challenger[RecordTable.SETUP_LIFECYCLE] = lifecycle[
        lifecycle["envelope_setup_id"] != "setup-B"
    ].reset_index(drop=True)
    candidates = challenger[RecordTable.ENTRY_CANDIDATE]
    challenger[RecordTable.ENTRY_CANDIDATE] = candidates[
        candidates["envelope_setup_id"] != "setup-B"
    ].reset_index(drop=True)
    report = build_native_population_delta("setup", baseline, challenger)
    assert report.removed_keys == ("setup-B",)
    target = report.first_divergence.entity_key
    assert target == "setup-B"

    # the UI seam: queue → route into setup mode's exact resolver input
    tab.queue_jump("setup_id", target)
    tab._route_pending_setup_jump(SimpleNamespace())
    assert tab.st.session_state.pop(f"{tab._STATE_PREFIX}selection_mode") == "setup"
    routed = tab.st.session_state.pop(f"{tab._STATE_PREFIX}setup_jump")
    assert routed == target

    # the provider seam: the routed id resolves EXACTLY on the baseline
    # artifact (where the entity exists) …
    ctx = _ctx_for(baseline)
    assert resolve_selection(ctx, setup_id=routed) == "cand-2"
    # … and refuses cleanly on the challenger (where it was removed), with no
    # nearest/fuzzy fallback of any kind
    with pytest.raises(MissingEvidenceError, match="no entry candidate"):
        resolve_selection(_ctx_for(challenger), setup_id=routed)
