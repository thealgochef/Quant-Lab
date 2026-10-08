"""Verified one-minute reporting views of Task B's registered NQ inputs.

Only named prepared caches are decoded, under their existing access guard.
Execution sections and source pins are never installed or re-resolved here.
"""

from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256, file_sha256

VERSION = "registered_task_b_index_minutes_v1"
REPO_ROOT = Path(__file__).resolve().parents[7]
_COLUMNS = (
    "timeframe_ticks",
    "trading_day",
    "kind",
    "bar_id",
    "logical_open_ts_utc",
    "logical_close_ts_utc",
    "open_ticks",
    "high_ticks",
    "low_ticks",
    "close_ticks",
    "volume",
    "is_complete",
    "is_partial",
    "close_reason",
)


class MarketSourceError(ValueError):
    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


@lru_cache(maxsize=512)
def _source_digest(path: str, _size: int, _modified_ns: int) -> str:
    return file_sha256(Path(path))


def _bars_path(owner, day: str) -> Path:
    return (
        Path(owner.definition["artifact_cache_dir"])
        / "NQ"
        / day
        / (f"ifvg_tbars_{owner.definition['artifacts_tag']}.parquet")
    )


def source_binding(plan: Any) -> dict[str, Any]:
    source = plan.source
    if getattr(source, "kind", None) != "verified_task_b_registered_inputs":
        raise MarketSourceError("unsupported_adapter", "This source needs another market adapter.")
    dates = tuple(source.warmup_dates) + tuple(source.evaluation_dates)
    if any(day > "2026-06-10" for day in dates):
        raise PermissionError("reporting source includes a protected date")
    return {
        "version": VERSION,
        "source": source.model_dump(mode="json"),
        "task_b_store_root": plan.task_b_store_root,
        "task_b_plan_id": plan.task_b_plan_id,
        "prepared_registration_ids": dict(plan.prepared_registration_ids),
        "preparation_catalog_sha256": dict(plan.task_b_scope.preparation_catalog_sha256),
        "registry_paths": list(plan.task_b_scope.prepared_store_registry_paths),
        "dates": list(dates),
        "tick_size_points": 0.25,
        "price_source": "full_size_NQ_source_proxy",
        "continuous_adjustment": "none",
    }


def reporting_root(plan: Any) -> Path:
    identity = canonical_sha256(source_binding(plan))
    return (
        REPO_ROOT.parent
        / "Claude-Quant-Lab-Research-Artifacts"
        / "ifsm-mffu-repair-integration-v01/reporting/market"
        / identity
    )


def _verify_sources(plan: Any):
    from alpha_lab.agents.data_infra.ifvg.prepared_store import PreparedStoreReplayPolicy
    from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import require_store_namespace

    binding = source_binding(plan)
    try:
        namespace = require_store_namespace(Path(plan.task_b_store_root), expected_class="research")
        if namespace.store_namespace_id != plan.source.task_b_store_namespace_id:
            raise ValueError("registered input namespace differs from this result")
        manifest = Path(plan.task_b_store_root) / "charters" / plan.task_b_plan_id / "manifest.json"
        if file_sha256(manifest) != plan.source.task_b_plan_manifest_sha256:
            raise ValueError("registered Task B charter manifest has changed")
        charter = load_verified_envelope(
            Path(plan.task_b_store_root), "charters", plan.task_b_plan_id, SearchCharterEnvelope
        )
        if (
            charter.payload.task_b_execution != plan.task_b_scope
            or list(charter.payload.date_policy.replay_dates) != binding["dates"]
        ):
            raise ValueError("registered input scope or exact date membership has changed")
        policy = PreparedStoreReplayPolicy(
            binding["dates"], registry_paths=tuple(Path(path) for path in binding["registry_paths"])
        )
        if {str(reg.path): reg.registration_id for reg in policy.registrations} != (
            binding["prepared_registration_ids"]
        ):
            raise ValueError("prepared-store registrations differ from this result")
        catalogs = {
            str(Path(reg.definition["catalog_path"]).resolve()): file_sha256(
                reg.definition["catalog_path"]
            )
            for reg in policy.registrations
        }
        if catalogs != binding["preparation_catalog_sha256"]:
            raise ValueError("source-selected contract catalogs have changed")
        for day in binding["dates"]:
            policy.authorize_date(day)
            owner = next(reg for reg in policy.registrations if day in reg.owned_dates)
            access = policy.for_day(day)
            path = access.resolve_source_path(day, lambda _day, reg=owner: _bars_path(reg, _day))
            access.record_metadata_access(day)
            stat = path.stat()
            access.record_file_open(day)
            if (
                _source_digest(str(path), stat.st_size, stat.st_mtime_ns)
                != owner.days[day]["bars_sha256"]
            ):
                raise ValueError(f"registered prepared bars changed for {day}")
    except OSError as error:
        raise MarketSourceError(
            "inaccessible_bound_location", "A bound market-input location cannot be read."
        ) from error
    except (ValueError, PermissionError) as error:
        raise MarketSourceError("failed_integrity", str(error)) from error
    return binding, policy


def build_companion(plan: Any, *, output_root: Path | None = None) -> Path:
    """Create immutable full-session one-minute bars, never selected position minutes."""
    binding, policy = _verify_sources(plan)
    root = Path(output_root) if output_root is not None else reporting_root(plan)
    if root.resolve().is_relative_to(REPO_ROOT):
        raise ValueError("derived market reporting must be outside the repository")
    if (root / "manifest.json").exists():
        load_companion(plan, root=root)
        return root
    frames, receipts = [], []
    for day in binding["dates"]:
        policy.authorize_date(day)  # before constructing a date-derived path
        owner = next(reg for reg in policy.registrations if day in reg.owned_dates)
        receipt = owner.days[day]
        access = policy.for_day(day)
        path = access.resolve_source_path(day, lambda _day, reg=owner: _bars_path(reg, _day))
        try:
            access.record_metadata_access(day)
            metadata = pq.read_metadata(path).metadata or {}
            # The footer key is part of the established artifact protocol.
            from alpha_lab.agents.data_infra.ifvg.day_artifacts import _META_KEY

            saved_metadata = json.loads(metadata.get(_META_KEY, b"{}"))
            access.record_file_open(day)
            if file_sha256(path) != receipt["bars_sha256"] or saved_metadata != receipt["metadata"]:
                raise MarketSourceError(
                    "failed_integrity", f"Prepared one-minute source changed on {day}."
                )
            frame = pd.read_parquet(
                path, columns=list(_COLUMNS), filters=[("timeframe_ticks", "==", 60)]
            )
            access.record_rows_read(day, rows=len(frame))
        except OSError as error:
            raise MarketSourceError(
                "missing_required_date_segment", f"Prepared market segment is missing on {day}."
            ) from error
        frame = frame[frame["kind"].str.lower() == "time"].copy()
        if frame.empty or set(frame["trading_day"].astype(str)) != {day}:
            raise MarketSourceError(
                "missing_required_date_segment", f"No canonical time minutes on {day}."
            )
        if frame[["logical_open_ts_utc", "logical_close_ts_utc"]].isna().any().any():
            raise MarketSourceError(
                "failed_integrity", f"Source lacks logical minute boundaries on {day}."
            )
        frame["timeframe_seconds"] = 60
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[f"{column}_ticks"].astype(float) * 0.25
        catalog = json.loads(Path(owner.definition["catalog_path"]).read_text(encoding="utf-8"))
        contract = catalog.get("days", {}).get(day)
        if not contract or contract.get("selected_instrument_id") is None:
            raise MarketSourceError(
                "failed_integrity", f"Source-selected contract receipt is absent on {day}."
            )
        frame["selected_instrument_id"] = int(contract["selected_instrument_id"])
        frame["raw_symbol"] = contract.get("raw_symbol")
        frame["source_registration_id"] = owner.registration_id
        frame["source_bars_sha256"] = receipt["bars_sha256"]
        frames.append(frame)
        receipts.append(
            {
                "day": day,
                "registration_id": owner.registration_id,
                "bars_sha256": receipt["bars_sha256"],
                **contract,
                "minute_count": len(frame),
            }
        )
    minutes = pd.concat(frames, ignore_index=True).sort_values("logical_close_ts_utc")
    keys = ["trading_day", "logical_open_ts_utc", "logical_close_ts_utc"]
    duplicate = minutes[minutes.duplicated(keys, keep=False)]
    if not duplicate.empty:
        for _key, group in duplicate.groupby(keys):
            if len(group.drop_duplicates()) != 1:
                raise MarketSourceError(
                    "failed_integrity", "Conflicting source minutes cannot be deduplicated."
                )
        minutes = minutes.drop_duplicates(keys)
    selected = {row["day"]: row["selected_instrument_id"] for row in receipts}
    previous = None
    rolls = {}
    for day in binding["dates"]:
        rolls[day] = previous is not None and previous != selected[day]
        previous = selected[day]
    minutes["roll_flag"] = minutes["trading_day"].map(rolls)
    minutes = minutes[minutes["logical_close_ts_utc"] <= pd.Timestamp(plan.source.cutoff_utc)]
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "market_minutes.parquet"
    temporary = root / f".minutes-{uuid.uuid4().hex}.parquet"
    minutes.reset_index(drop=True).to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    manifest = {
        "schema": VERSION,
        "binding": binding,
        "source_receipts": receipts,
        "bars_sha256": file_sha256(destination),
        "bars_bytes": destination.stat().st_size,
        "row_count": len(minutes),
        "forbidden_accesses": sum(policy.audit.denied_dates.values()),
        "session_semantics": "original logical bars; completed and final partial status retained",
    }
    policy.assert_zero_forbidden_access()
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def load_companion(plan: Any, *, root: Path | None = None) -> pd.DataFrame:
    binding, policy = _verify_sources(plan)
    root = Path(root) if root is not None else reporting_root(plan)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if (
            manifest.get("schema") != VERSION
            or manifest["binding"] != binding
            or manifest["manifest_sha256"] != canonical_sha256(body)
        ):
            raise MarketSourceError(
                "failed_integrity", "Derived market source manifest has changed."
            )
        receipts = {row["day"]: row for row in manifest["source_receipts"]}
        if list(receipts) != binding["dates"]:
            raise MarketSourceError("failed_integrity", "Derived market source dates changed.")
        for day in binding["dates"]:
            policy.authorize_date(day)
            owner = next(reg for reg in policy.registrations if day in reg.owned_dates)
            if (
                receipts[day]["registration_id"] != owner.registration_id
                or receipts[day]["bars_sha256"] != owner.days[day]["bars_sha256"]
            ):
                raise MarketSourceError(
                    "failed_integrity", "Derived market receipt binds another source."
                )
        path = root / "market_minutes.parquet"
        if (
            path.stat().st_size != manifest["bars_bytes"]
            or file_sha256(path) != manifest["bars_sha256"]
        ):
            raise MarketSourceError(
                "failed_integrity", "Derived market bars failed integrity verification."
            )
        minutes = pd.read_parquet(path)
    except OSError as error:
        raise MarketSourceError(
            "inaccessible_bound_location",
            "The registered market reporting companion is inaccessible.",
        ) from error
    if (
        len(minutes) != manifest["row_count"]
        or not set(minutes["trading_day"].astype(str)) <= set(binding["dates"])
        or (minutes["logical_close_ts_utc"] > pd.Timestamp(plan.source.cutoff_utc)).any()
    ):
        raise MarketSourceError(
            "failed_integrity", "Derived market bars escaped the exact study scope."
        )
    policy.assert_zero_forbidden_access()
    minutes.attrs["source_provenance"] = manifest
    return minutes


def cache_signature(plan: Any) -> tuple[Any, ...]:
    """Invalidate cached screens if a bound manifest, catalog or named cache changes."""
    binding = source_binding(plan)
    root = reporting_root(plan)
    from alpha_lab.agents.data_infra.ifvg.prepared_store import load_prepared_store

    files = [root / "manifest.json", root / "market_minutes.parquet"]
    files.extend(Path(path) for path in binding["registry_paths"])
    files.extend(Path(path) for path in binding["preparation_catalog_sha256"])
    registrations = [load_prepared_store(Path(path)) for path in binding["registry_paths"]]
    for day in binding["dates"]:
        # Date is checked against the bound manifest before a named path exists.
        if day > "2026-06-10":
            raise PermissionError("protected reporting date")
        owner = next(reg for reg in registrations if day in reg.owned_dates)
        files.append(_bars_path(owner, day))
    try:
        return (
            canonical_sha256(binding),
            *((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files),
        )
    except OSError as error:
        raise MarketSourceError(
            "inaccessible_bound_location", "A registered market reporting location is inaccessible."
        ) from error
