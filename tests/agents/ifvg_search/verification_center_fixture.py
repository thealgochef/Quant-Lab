"""Shared UI-2 Verification Center fixture: an isolated ``test`` store under a
tmp repo root laid out exactly like production (``<repo>/data/ifvg_datasets/
search_test/v1``), a real shortlist document over synthetic coverage rows and
a fake inventory, the conftest synthetic seed persisted as a profile-bound
snapshot, a SYNTHETIC-provenance seed-production authorization with a
hand-built run receipt, and a bound (test) ``VerificationAuthorizationRef``
written as a completed packet file.

Nothing here signs an owner artifact, reads raw source data, produces a real
seed or runs a verification; every artifact lives under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from strategy_core.strategies.ifvg_smc.state import IFVG_SEED_SCHEMA_VERSION, seed_hash

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
    DaySeedsRecord,
    SeedSnapshotPayload,
    save_seed_snapshot,
)
from alpha_lab.agents.data_infra.ifvg.search.seed_production import (
    SEED_PRODUCTION_CHAIN_POLICY_ID,
    SEED_PRODUCTION_RUN_STORE,
    SeedProductionRunEnvelope,
    SeedProductionRunPayload,
    synthetic_seed_production_authorization,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SEARCH_TEST_STORE_ROOT,
    save_or_reuse_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_test_namespace
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    current_supersession_head_witness,
)
from alpha_lab.agents.data_infra.ifvg.search.trading_calendar import (
    logical_trading_days,
    store_day_chain,
)
from alpha_lab.agents.data_infra.ifvg.search.verification_window import (
    build_logical_day_coverage,
    build_verification_window_shortlist,
    shortlist_document,
)
from tests.agents.ifvg_search.namespace_fixture import verification_authorization_ref

__all__ = [
    "PROFILE",
    "QL_IDENTITY",
    "SC_COMMIT",
    "SC_IDENTITY",
    "SEED_CHAIN",
    "WINDOW",
    "build_verification_store",
    "inventory_for_chain",
    "write_inventory_manifest",
    "write_shortlist_document",
    "write_signed_ref",
]

PROFILE = "ifvg_v2_doc_default_fresh_static_1r"
#: the conftest synthetic chain replays 2026-01-13 / 01-14 (/ 01-15); the seed
#: through 01-14 is continuous with a window starting Thursday 2026-01-15
SEED_CHAIN = ("2026-01-13", "2026-01-14")
WINDOW = ("2026-01-15", "2026-01-16")
QL_IDENTITY = "1" * 64
SC_COMMIT = "c" * 40
SC_IDENTITY = "2" * 64
_INVENTORY_LAST_DAY = "2026-01-23"


def inventory_for_chain(
    first: str = "2026-01-01", last: str = _INVENTORY_LAST_DAY
) -> dict[str, tuple[str, str]]:
    """A fake public-kind inventory over the complete store-day chain."""

    return {
        day: ("mbp1", hashlib.sha256(day.encode("utf-8")).hexdigest())
        for day in store_day_chain(first, last)
    }


def write_inventory_manifest(path: Path, inventory: dict[str, tuple[str, str]]) -> Path:
    """A manifest-shaped JSON (``identity.permitted_source_hashes``) the seed
    CLI and the center's loader accept; physical file stems follow the
    accepted inventory's naming."""

    entries = [[f"{day}/{kind}.parquet", sha] for day, (kind, sha) in inventory.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"identity": {"permitted_source_hashes": entries}}, indent=2),
        encoding="utf-8",
    )
    return path


def _frame(rows: dict[str, int], *, with_setups: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    ordinal = 0
    for day, count in rows.items():
        for _ in range(count):
            ordinal += 1
            record = {"envelope_trading_day": day, "envelope_setup_id": f"setup-{ordinal:04d}"}
            if with_setups:
                record["setup_id"] = f"setup-{ordinal:04d}"
                record["status"] = "resolved"
            records.append(record)
    return pd.DataFrame(records)


def write_shortlist_document(path: Path, *, window_length: int = 2) -> dict[str, Any]:
    """A REAL shortlist (typed contracts, lexicographic ranking) over synthetic
    coverage rows, persisted as the JSON document the CLI writes."""

    inventory = inventory_for_chain()
    counts = {"2026-01-13": 2, "2026-01-14": 3, "2026-01-15": 5, "2026-01-16": 4, "2026-01-20": 1}
    tables = {
        RecordTable.SETUP_LIFECYCLE: _frame({day: count * 4 for day, count in counts.items()}),
        RecordTable.ENTRY_CANDIDATE: _frame({day: count * 2 for day, count in counts.items()}),
        RecordTable.ELIGIBLE_DECISION: _frame(counts),
        RecordTable.EXECUTED_TRADE: _frame(counts, with_setups=True),
        RecordTable.CANDIDATE_LABEL: _frame(counts),
    }
    logical = logical_trading_days("2026-01-02", _INVENTORY_LAST_DAY)
    coverage = build_logical_day_coverage(
        logical_days=logical,
        inventory=inventory,
        tables=tables,
        funnel_days=frozenset(logical),
    )
    shortlist = build_verification_window_shortlist(
        coverage,
        inventory=inventory,
        evidence_source_dataset_id="1" * 64,
        evidence_source_manifest_sha256="2" * 64,
        audit_artifact_id="3" * 64,
        window_length=window_length,
    )
    document = shortlist_document(
        shortlist,
        generated_at_utc=datetime.now(UTC).isoformat(),
        owner_selection="NOT PERFORMED",
        register_program_allowlist_called=False,
        no_raw_source_reads=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def build_verification_store(
    repo_root: Path,
    synthetic_chain,
    *,
    with_seed: bool = True,
    with_authorization: bool = True,
    with_receipt: bool = True,
) -> dict[str, Any]:
    """The isolated test store at the canonical relative location under
    ``repo_root`` with the synthetic seed lane persisted (never produced)."""

    repo_root = Path(repo_root)
    store_root = repo_root / SEARCH_TEST_STORE_ROOT
    store_root.mkdir(parents=True, exist_ok=True)
    namespace = initialize_test_namespace(store_root)
    resolved = resolve_profile_config({"profile_name": PROFILE})
    inventory = inventory_for_chain()
    fixture: dict[str, Any] = {
        "repo_root": repo_root,
        "store_root": store_root,
        "namespace_id": namespace.store_namespace_id,
        "resolved": resolved,
        "section_hash": resolved.section_config_hash,
        "inventory": inventory,
        "snapshot": None,
        "authorization": None,
        "receipt": None,
    }
    seed = synthetic_chain[1].end_seed if (with_seed and synthetic_chain) else None
    if with_seed and seed is not None:
        snapshot = save_seed_snapshot(
            store_root,
            SeedSnapshotPayload(
                profile_name=PROFILE,
                resolved_section_config_hash=seed.profile_hash,
                seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
                seed_hash=seed_hash(seed),
                snapshot_through_day=SEED_CHAIN[-1],
                first_replay_day=WINDOW[0],
                entering_day_seeds=DaySeedsRecord(
                    prev_day=SEED_CHAIN[-1],
                    prev_full_hl=(100, 50),
                    prev_ny_day=None,
                    prev_ny_hl=None,
                ),
                chain_policy_id=SEED_PRODUCTION_CHAIN_POLICY_ID,
                chain_date_count=len(SEED_CHAIN),
                strategy_core_commit=SC_COMMIT,
            ),
            seed,
        )
        fixture["snapshot"] = snapshot
    if with_authorization:
        authorization = synthetic_seed_production_authorization(
            store_root,
            chain_replay_days=SEED_CHAIN,
            first_intended_verification_day=WINDOW[0],
            inventory=inventory,
            quant_lab_source_identity=QL_IDENTITY,
            strategy_core_commit=SC_COMMIT,
            strategy_core_source_identity=SC_IDENTITY,
        )
        fixture["authorization"] = authorization
        if with_receipt and seed is not None:
            audit_bytes = b'{"policy": "seed_production_explicit_chain_v1", "denied_dates": {}}\n'
            receipt = SeedProductionRunEnvelope.from_payload(
                SeedProductionRunPayload(
                    seed_production_authorization_id=authorization.seed_production_authorization_id,
                    store_namespace_id=namespace.store_namespace_id,
                    supersession_head_witness=current_supersession_head_witness(store_root),
                    seed_snapshot_id=fixture["snapshot"].seed_snapshot_id,
                    seed_hash=fixture["snapshot"].payload.seed_hash,
                    baseline_profile_name=PROFILE,
                    resolved_section_config_hash=seed.profile_hash,
                    ordered_seed_chain_replay_days=SEED_CHAIN,
                    chain_replay_day_count=len(SEED_CHAIN),
                    logical_trading_day_count=len(SEED_CHAIN),
                    snapshot_through_day=SEED_CHAIN[-1],
                    first_intended_verification_day=WINDOW[0],
                    access_audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),
                    quant_lab_source_identity=QL_IDENTITY,
                    strategy_core_commit=SC_COMMIT,
                    strategy_core_source_identity=SC_IDENTITY,
                    provenance="synthetic_test_authorization_v1",
                )
            )
            stored, _reused = save_or_reuse_envelope(
                store_root,
                SEED_PRODUCTION_RUN_STORE,
                receipt,
                extra_files={"access_audit.json": audit_bytes},
            )
            fixture["receipt"] = stored
    return fixture


def write_signed_ref(
    path: Path,
    store_root: Path,
    *,
    seed_snapshot_id: str,
    allowlist: tuple[str, ...] = WINDOW,
    coverage_matrix_artifact_id: str = "b" * 64,
    content_hash: str = "d" * 64,
    approved_by: str = "test-owner",
) -> Path:
    """A COMPLETED (test-fixture) verification authorization reference bound
    to ``store_root``, written as the file the owner would complete."""

    ref = verification_authorization_ref(
        store_root,
        approved_allowlist_hash=allowlist_sha256(tuple(allowlist)),
        coverage_matrix_artifact_id=coverage_matrix_artifact_id,
        seed_snapshot_id=seed_snapshot_id,
        approved_by=approved_by,
        approved_at="2026-09-04T00:00:00+00:00",
        content_hash=content_hash,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ref.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
