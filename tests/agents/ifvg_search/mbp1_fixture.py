"""Shared synthetic MBP-1 fixtures (R5B suites; no real source data).

One crafted trading day of top-of-book events with hand-controllable order
keys — same-timestamp bursts, sequence gaps, roll boundaries — plus exact
candidate stage anchors, so every cutoff/window/formula/missingness test
computes against known-by-construction evidence.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
    FEATURE_BLOCK_RESOLUTION_REGISTRY,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
    build_mbp1_source_artifact,
    normalize_mbp1_events,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    MIN_DAY_COVERAGE_FRACTION,
    R5B_WINDOW_SPECS,
    Mbp1SourceContract,
)

FIXTURE_DAY = "2026-01-13"
DAY_BASE_NS = int(pd.Timestamp(f"{FIXTURE_DAY}T14:00:00Z").value)

MBP1_RESOLVED_BLOCK = FEATURE_BLOCK_RESOLUTION_REGISTRY["IFVG_ORDER_FLOW_MBP1_V1"]


def synthetic_contract() -> Mbp1SourceContract:
    return Mbp1SourceContract(
        instrument="NQ",
        contract_roll_policy_id="front_month_open_interest_roll_v1",
        feature_window_specs=R5B_WINDOW_SPECS,
        coverage_policy={"min_day_coverage_fraction": MIN_DAY_COVERAGE_FRACTION},
    )


def ns_at(seconds: float) -> int:
    """Nanoseconds at ``seconds`` after the fixture day's 14:00Z base."""

    return DAY_BASE_NS + int(seconds * 1_000_000_000)


def iso_at(seconds: float) -> str:
    return pd.Timestamp(ns_at(seconds), unit="ns", tz="UTC").isoformat()


_PRICE_PER_TICK = int(0.25 / 1e-9)  # databento 1e-9 scale, NQ tick 0.25


def raw_event(
    *,
    ts_event: int,
    ts_recv: int | None = None,
    sequence: int,
    action: str = "A",
    side: str = "N",
    size: int = 1,
    bid_ticks: float = 20000.0,
    ask_ticks: float = 20001.0,
    bid_sz: int = 5,
    ask_sz: int = 5,
    bid_ct: int = 2,
    ask_ct: int = 2,
    instrument_id: int = 42,
    symbol: str = "NQH6",
) -> dict[str, Any]:
    """One raw Databento-shaped mbp-1 row with exact controllable keys."""

    return {
        "ts_recv": ts_recv if ts_recv is not None else ts_event + 5,
        "ts_event": ts_event,
        "rtype": 1,
        "publisher_id": 1,
        "instrument_id": instrument_id,
        "action": action,
        "side": side,
        "depth": 0,
        "price": int(bid_ticks * _PRICE_PER_TICK),
        "size": size,
        "flags": 0,
        "ts_in_delta": 0,
        "sequence": sequence,
        "bid_px_00": int(bid_ticks * _PRICE_PER_TICK),
        "ask_px_00": int(ask_ticks * _PRICE_PER_TICK),
        "bid_sz_00": bid_sz,
        "ask_sz_00": ask_sz,
        "bid_ct_00": bid_ct,
        "ask_ct_00": ask_ct,
        "symbol": symbol,
    }


def normalized_day(rows: list[dict[str, Any]], *, day: str = FIXTURE_DAY) -> pd.DataFrame:
    return normalize_mbp1_events(pd.DataFrame(rows), instrument="NQ", trading_day=day)


def default_day_events() -> pd.DataFrame:
    """A clean, gap-free event stream spanning 0…600 s after the base.

    Book state steps deterministically so snapshot/aggregate formulas have
    hand-computable values; two trades sit inside the (inversion, entry]
    window (one buy-aggressor, one sell-aggressor).
    """

    rows = [
        raw_event(ts_event=ns_at(0), sequence=100, bid_ticks=20000, ask_ticks=20001,
                  bid_sz=5, ask_sz=5, bid_ct=2, ask_ct=2),
        raw_event(ts_event=ns_at(30), sequence=101, action="C", bid_ticks=20000,
                  ask_ticks=20001, bid_sz=4, ask_sz=6, bid_ct=2, ask_ct=3),
        raw_event(ts_event=ns_at(70), sequence=102, bid_ticks=20001, ask_ticks=20002,
                  bid_sz=6, ask_sz=3, bid_ct=3, ask_ct=1),
        raw_event(ts_event=ns_at(130), sequence=103, action="C", bid_ticks=20001,
                  ask_ticks=20002, bid_sz=2, ask_sz=7, bid_ct=1, ask_ct=4),
        raw_event(ts_event=ns_at(190), sequence=104, bid_ticks=20000, ask_ticks=20002,
                  bid_sz=8, ask_sz=4, bid_ct=4, ask_ct=2),
        raw_event(ts_event=ns_at(250), sequence=105, action="T", side="B", size=3,
                  bid_ticks=20000, ask_ticks=20002, bid_sz=8, ask_sz=1, bid_ct=4,
                  ask_ct=1),
        raw_event(ts_event=ns_at(310), sequence=106, action="T", side="A", size=2,
                  bid_ticks=20000, ask_ticks=20001, bid_sz=6, ask_sz=5, bid_ct=3,
                  ask_ct=2),
        raw_event(ts_event=ns_at(400), sequence=107, bid_ticks=20001, ask_ticks=20002,
                  bid_sz=7, ask_sz=6, bid_ct=3, ask_ct=3),
        raw_event(ts_event=ns_at(500), sequence=108, action="C", bid_ticks=20001,
                  ask_ticks=20003, bid_sz=5, ask_sz=2, bid_ct=2, ask_ct=1),
        raw_event(ts_event=ns_at(600), sequence=109, bid_ticks=20002, ask_ticks=20003,
                  bid_sz=9, ask_sz=4, bid_ct=5, ask_ct=2),
    ]
    return normalized_day(rows)


def anchor_row(
    candidate_id: str,
    *,
    tap_s: float | None = 60.0,
    lock_s: float | None = 120.0,
    armed_s: float | None = 180.0,
    inversion_s: float | None = 240.0,
    entry_s: float | None = 360.0,
    day: str = FIXTURE_DAY,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "setup_id": f"setup_{candidate_id}",
        "trading_day": day,
        "tap_ts_utc": iso_at(tap_s) if tap_s is not None else None,
        "lock_ts_utc": iso_at(lock_s) if lock_s is not None else None,
        "armed_ts_utc": iso_at(armed_s) if armed_s is not None else None,
        "inversion_ts_utc": iso_at(inversion_s) if inversion_s is not None else None,
        "entry_ts_utc": iso_at(entry_s) if entry_s is not None else None,
    }


def default_anchors() -> pd.DataFrame:
    return pd.DataFrame([anchor_row("cand_a"), anchor_row("cand_b", entry_s=550.0)])


def build_fixture_source(
    events_by_day: dict[str, pd.DataFrame] | None = None,
):
    """(source envelope, event bytes, events_by_day) over the fixture day."""

    events = events_by_day or {FIXTURE_DAY: default_day_events()}
    envelope, event_bytes = build_mbp1_source_artifact(
        events,
        contract=synthetic_contract(),
        authorized_date_set_id="synthetic_fixture_days_v1",
        events_stored=True,
    )
    return envelope, event_bytes, events


def feature_artifact_for_frame(
    metrics_frame: pd.DataFrame,
    *,
    setup_ids: dict[str, str] | None = None,
    trading_day: str = FIXTURE_DAY,
    anchor_salt: str = "a",
):
    """A REAL (verified-bindable) feature artifact over a crafted frame.

    Builds the full pinned feature-table schema around the supplied metric
    columns (evidence columns valid=True / reason null), mints the envelope
    via ``from_payload`` with the ACTUAL table hashes — so the F1 rehash at
    the bundle-view seam verifies — and returns ``(envelope, full_frame)``.
    Upstream payload references are synthetic 64-hex ids: this helper
    exercises the frame↔envelope binding seam, not source provenance.
    """

    import hashlib

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_arrow_schemas import (
        MBP1_FEATURE_TABLE_SCHEMA,
        MBP1_FEATURE_TABLE_SCHEMA_HASH,
        MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA,
        MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA_HASH,
        mbp1_window_missing_reason_fields,
        mbp1_window_validity_fields,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_feature_materializer import (
        Mbp1FeatureArtifactEnvelope,
        Mbp1FeatureArtifactPayload,
        _table_bytes,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
        MBP1_FORMULA_VERSION,
        MBP1_MATERIALIZER_VERSION,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_stage_windows import (
        COMPLETED_BAR_CUTOFF_POLICY_ID,
    )

    full = metrics_frame.copy()
    ids = full["candidate_id"].astype(str)
    full["setup_id"] = [
        (setup_ids or {}).get(cid, f"setup_{cid}") for cid in ids
    ]
    full["trading_day"] = trading_day
    for name in mbp1_window_validity_fields():
        full[name] = True
    for name in mbp1_window_missing_reason_fields():
        full[name] = None
    ordered = [field.name for field in MBP1_FEATURE_TABLE_SCHEMA]
    full = full.loc[:, ordered]
    evidence = pd.DataFrame(
        columns=[field.name for field in MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA]
    )
    payload = Mbp1FeatureArtifactPayload(
        mbp1_source_artifact_id="1" * 64,
        resolved_feature_block_id=MBP1_RESOLVED_BLOCK.resolved_feature_block_id,
        candidate_anchor_hash=anchor_salt * 64,
        cutoff_policy_id=COMPLETED_BAR_CUTOFF_POLICY_ID,
        formula_version=MBP1_FORMULA_VERSION,
        materializer_version=MBP1_MATERIALIZER_VERSION,
        feature_table_schema_hash=MBP1_FEATURE_TABLE_SCHEMA_HASH,
        stage_window_evidence_schema_hash=MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA_HASH,
        candidate_count=int(len(full)),
    )
    envelope = Mbp1FeatureArtifactEnvelope.from_payload(
        payload,
        feature_table_sha256=hashlib.sha256(
            _table_bytes(full, MBP1_FEATURE_TABLE_SCHEMA)
        ).hexdigest(),
        stage_evidence_sha256=hashlib.sha256(
            _table_bytes(evidence, MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA)
        ).hexdigest(),
    )
    return envelope, full
