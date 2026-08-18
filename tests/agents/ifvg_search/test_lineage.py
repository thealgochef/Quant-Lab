"""Lineage suite (TEST_MATRIX §3.2 + Amendment P1-E collision rows).

Native determinism and cross-profile lineage validity run on the REAL
Strategy-Core reducer over the synthetic three-day walk (setups + audit tap
evidence). The candidate→decision→trade projection, one-to-one uniqueness,
collision refusal, and match-basis rules run on hand-built frames using the
exact v2 column vocabulary. No fuzzy path exists anywhere (source-scanned).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import (
    RecordTable,
    partition_capture_tables,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.lineage import (
    LINEAGE_ENTITY_KINDS,
    LineageUniquenessEnvelope,
    build_native_lineage_map,
    derive_lineage_validity,
    parse_fvg_id,
    persist_lineage_uniqueness,
)
from alpha_lab.agents.data_infra.ifvg.study.population_delta import (
    build_lineage_population_delta,
    build_native_population_delta,
)
from tests.agents.ifvg_search.conftest import (
    build_artifact_chain,
    run_synthetic_chain,
)

_CORE_A = "1" * 64
_CORE_B = "2" * 64


def _tables_of(chain) -> dict:
    frames = [r.rows for r in chain if len(r.rows)]
    trace = pd.concat(frames, ignore_index=True, sort=False)
    trace["trace_ordinal"] = range(len(trace))
    return partition_capture_tables(trace)


def _audit_tap_frame(chain) -> dict[str, pd.DataFrame]:
    """Tap evidence keyed the way the assembled audit tables key it.

    HTF-tap audit rows are TRACE-cut kinds (they ride ``day_result.rows``,
    exactly as ``assemble_fsm_audit_tables`` cuts them) — not audit-channel
    rows.
    """

    frames = [r.rows for r in chain if len(r.rows)]
    if not frames:
        return {}
    rows = pd.concat(frames, ignore_index=True, sort=False)
    taps = rows.loc[rows["kind"] == "htf_tap"] if "kind" in rows else pd.DataFrame()
    return {"ifvg_audit_htf_tap": taps.reset_index(drop=True)}


@pytest.fixture(scope="module")
def doc_chain(doc_default_cfg):
    return run_synthetic_chain(
        doc_default_cfg,
        build_artifact_chain(doc_default_cfg),
        audit_capture_mode="fsm_audit_v1",
    )


@pytest.fixture(scope="module")
def variant_chain(doc_default_cfg):
    """Same TF set + min_gap_ticks_capture; only a timeout clock differs."""

    variant = resolve_profile_config(
        {
            "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "section_overrides": {"parent_retest_timeout_1m_bars": 480},
        }
    )
    cfg = replace(doc_default_cfg, section=variant.section)
    return run_synthetic_chain(
        cfg, build_artifact_chain(cfg), audit_capture_mode="fsm_audit_v1"
    )


# ── §3.2 native determinism ──────────────────────────────────────────────────


def test_native_determinism_jaccard_one_per_entity_kind(doc_default_cfg, doc_chain) -> None:
    rerun = run_synthetic_chain(
        doc_default_cfg,
        build_artifact_chain(doc_default_cfg),
        audit_capture_mode="fsm_audit_v1",
    )
    tables_a, tables_b = _tables_of(doc_chain), _tables_of(rerun)
    for kind in LINEAGE_ENTITY_KINDS:
        report = build_native_population_delta(kind, tables_a, tables_b)
        assert report.match_basis == "native_id_exact"
        assert report.jaccard == 1.0, f"{kind} native determinism failed"
        assert report.added_keys == () and report.removed_keys == ()


# ── §3.2 lineage validity across profiles sharing the geometry axes ──────────


def test_lineage_keys_stable_across_profiles_sharing_geometry(
    doc_chain, variant_chain
) -> None:
    map_a = build_native_lineage_map(
        _tables_of(doc_chain), core_replay_id=_CORE_A, audit_tables=_audit_tap_frame(doc_chain)
    )
    map_b = build_native_lineage_map(
        _tables_of(variant_chain),
        core_replay_id=_CORE_B,
        audit_tables=_audit_tap_frame(variant_chain),
    )
    assert map_a.usable("setup"), map_a.incomplete_kinds
    assert map_b.usable("setup"), map_b.incomplete_kinds
    keys_a, keys_b = map_a.lineage_keys("setup"), map_b.lineage_keys("setup")
    assert keys_a, "the synthetic walk must produce setup lineage"
    # native ids embed the profile hash and CANNOT match across profiles…
    native_a = set(map_a.native_to_lineage["setup"])
    native_b = set(map_b.native_to_lineage["setup"])
    assert native_a.isdisjoint(native_b)
    # …while the shared market opportunities match EXACTLY on lineage keys
    assert keys_a == keys_b
    delta = build_lineage_population_delta("setup", map_a, map_b, lineage_valid=True)
    assert delta.match_basis == "profile_independent_lineage_exact"
    assert delta.jaccard == 1.0


def test_native_basis_refuses_cross_profile_tables(doc_chain, variant_chain) -> None:
    """F4: the native basis is SAME-PROFILE only — two different profile
    hashes disable the comparison instead of fabricating a Jaccard-0 report."""

    report = build_native_population_delta(
        "setup", _tables_of(doc_chain), _tables_of(variant_chain)
    )
    assert report.match_basis == "not_comparable"
    assert "different profile section hashes" in report.match_basis_reason
    assert report.common_keys == () and report.jaccard is None


def test_uniqueness_report_persists_immutably(tmp_path, doc_chain) -> None:
    """CS §4: collision evidence is PERSISTED, never in-memory-only."""

    from alpha_lab.agents.data_infra.ifvg.search.store import (
        has_envelope,
        load_verified_envelope,
    )

    lineage_map = build_native_lineage_map(
        _tables_of(doc_chain),
        core_replay_id=_CORE_A,
        audit_tables=_audit_tap_frame(doc_chain),
    )
    envelope, reused = persist_lineage_uniqueness(
        tmp_path, lineage_map.uniqueness_report
    )
    assert reused is False
    assert has_envelope(tmp_path, "lineage_reports", envelope.lineage_report_id)
    again, reused_again = persist_lineage_uniqueness(
        tmp_path, lineage_map.uniqueness_report
    )
    assert reused_again is True
    reloaded = load_verified_envelope(
        tmp_path, "lineage_reports", envelope.lineage_report_id, LineageUniquenessEnvelope
    )
    assert reloaded.payload == lineage_map.uniqueness_report


def test_lineage_validity_is_derived_from_the_axis_registry() -> None:
    """m8: validity is derived, never hand-asserted — geometry/TF/thesis axes
    break lineage keys; clock axes do not."""

    valid, reason = derive_lineage_validity(("parent_retest_timeout_1m_bars",))
    assert valid is True and reason is None
    for breaking in (
        "min_gap_ticks_capture",
        "htf_timeframes",
        "entry_family",
    ):
        valid, reason = derive_lineage_validity((breaking,))
        assert valid is False
        assert breaking in reason


def test_parse_fvg_id_refuses_unknown_shapes() -> None:
    seconds, direction = parse_fvg_id("300s:bullish:tbar-x")
    assert (seconds, direction) == (300, "bullish")
    with pytest.raises(ValueError, match="unparseable"):
        parse_fvg_id("not-an-fvg-id")


# ── hand-built candidate→decision→trade projection + collisions ─────────────


def _entity_tables(*, collide: bool = False) -> dict:
    lifecycle = pd.DataFrame(
        {
            "envelope_setup_id": ["setup-A", "setup-A", "setup-B", "setup-B"],
            "transition": [
                "setup_activated",
                "setup_ended",
                "setup_activated",
                "setup_ended",
            ],
            "event_cursor": ["c-001", "c-050", "c-002", "c-060"],
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["cand-1", "cand-2"],
            "envelope_setup_id": ["setup-A", "setup-B"],
            "htf_fvg_id": ["300s:bullish:bar-a", "600s:bearish:bar-b"],
            "entry_family": ["fresh", "fresh"],
            "trigger_cursor": ["c-010", "c-010" if collide else "c-020"],
            "envelope_ts_utc": ["2026-01-13T10:00:00Z", "2026-01-13T10:05:00Z"],
            "fvg_fvg_id": ["60s:bullish:bar-e1", "60s:bearish:bar-e2"],
        }
    )
    if collide:
        # two native candidates constructed onto ONE lineage key: same setup,
        # family, trigger cursor, entry fvg, and as-of instant
        candidates.loc[1, "envelope_setup_id"] = "setup-A"
        candidates.loc[1, "htf_fvg_id"] = "300s:bullish:bar-a"
        candidates.loc[1, "fvg_fvg_id"] = "60s:bullish:bar-e1"
        candidates.loc[1, "envelope_ts_utc"] = "2026-01-13T10:00:00Z"
    decisions = pd.DataFrame(
        {
            "decision_id": ["dec-1", "dec-2"],
            "candidate_id": ["cand-1", "cand-2"],
            "selected": [True, False],
            "drop_reason": [None, "outranked"],
            "envelope_ts_utc": ["2026-01-13T10:00:30Z", "2026-01-13T10:05:30Z"],
        }
    )
    trades = pd.DataFrame(
        {
            "trade_id": ["trade-1"],
            "decision_id": ["dec-1"],
            "entry_ts_utc": ["2026-01-13T10:01:00Z"],
        }
    )
    return {
        RecordTable.SETUP_LIFECYCLE: lifecycle,
        RecordTable.ENTRY_CANDIDATE: candidates,
        RecordTable.ELIGIBLE_DECISION: decisions,
        RecordTable.EXECUTED_TRADE: trades,
    }


def test_full_projection_is_one_to_one_and_pit(  ) -> None:
    lineage_map = build_native_lineage_map(_entity_tables(), core_replay_id=_CORE_A)
    for kind in LINEAGE_ENTITY_KINDS:
        assert lineage_map.uniqueness_report.per_entity_kind[kind] == "one_to_one"
    assert lineage_map.usable("candidate") and lineage_map.usable("trade")
    # decision kinds are typed from selected/drop_reason, never guessed
    decision_payloads = lineage_map.lineage_payloads["decision"].values()
    kinds = {payload.decision_kind for payload in decision_payloads}
    assert kinds == {"selected", "dropped:outranked"}
    trade_payload = next(iter(lineage_map.lineage_payloads["trade"].values()))
    assert trade_payload.entry_policy_id == "confirmation_close_next1m_stop_first_v1"


def test_collision_is_recorded_never_deduped() -> None:
    lineage_map = build_native_lineage_map(
        _entity_tables(collide=True), core_replay_id=_CORE_A
    )
    report = lineage_map.uniqueness_report
    assert report.per_entity_kind["candidate"] == "collisions_present"
    assert len(report.collisions) == 1
    record = report.collisions[0]
    assert record.entity_kind == "candidate"
    assert record.native_ids == ("cand-1", "cand-2")  # both persisted, none kept-first
    assert not lineage_map.usable("candidate")
    # a colliding side disables the population delta with the exact reason
    clean = build_native_lineage_map(_entity_tables(), core_replay_id=_CORE_B)
    delta = build_lineage_population_delta(
        "candidate", lineage_map, clean, lineage_valid=True
    )
    assert delta.match_basis == "not_comparable"
    assert "collisions" in delta.match_basis_reason
    assert delta.common_keys == () and delta.jaccard is None


def test_gap_detection_axis_change_disables_comparison() -> None:
    map_a = build_native_lineage_map(_entity_tables(), core_replay_id=_CORE_A)
    map_b = build_native_lineage_map(_entity_tables(), core_replay_id=_CORE_B)
    delta = build_lineage_population_delta(
        "setup",
        map_a,
        map_b,
        lineage_valid=False,
        lineage_invalid_reason=(
            "min_gap_ticks_capture changed: FVG detection geometry differs, "
            "lineage keys are not comparable"
        ),
    )
    assert delta.match_basis == "not_comparable"
    assert "min_gap_ticks_capture" in delta.match_basis_reason


def test_no_fuzzy_matching_code_exists() -> None:
    """Source scan (§3.2 match-basis row): no nearest-time / keep-last /
    fuzzy-geometry matching path exists in the lineage or delta modules."""

    import alpha_lab.agents.data_infra.ifvg.search.lineage as lineage_module
    import alpha_lab.agents.data_infra.ifvg.study.population_delta as delta_module

    for module in (lineage_module, delta_module):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        for token in ("nearest_time", "merge_asof", "keep_first", "keep_last", "fuzzy"):
            hits = [
                line
                for line in source.splitlines()
                if token in line and "never" not in line and "no nearest" not in line
            ]
            assert not hits, f"{module.__name__} contains {token!r}: {hits}"
