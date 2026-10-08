"""Versioned immutable reporting publication, separate from economic results."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.lab import mffu_gamma, mffu_lenses

SCHEMA = "ifsm_mffu_reporting_companion_v1"


def _source_paths() -> dict[str, Path]:
    from alpha_lab.agents.data_infra.ifvg import menthorq_asof
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import (
        funded_data,
        mffu_matrix,
        mffu_provenance,
    )
    from alpha_lab.propsim.funded import position_walk, reporting_accounts, reporting_legs

    paths = {
        module.__name__: Path(module.__file__)
        for module in (
            mffu_gamma,
            mffu_lenses,
            reporting_legs,
            reporting_accounts,
            position_walk,
            menthorq_asof,
            funded_data,
            mffu_matrix,
            mffu_provenance,
        )
    }
    paths[__name__] = Path(__file__)
    return paths


def source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in _source_paths().items()
    }


def _sources_match(folder: Path, expected: Any) -> bool:
    """Verify the producer's exact bytes without executing its historical code.

    A saved report is an immutable read model. Later engine changes cannot
    invalidate it when the producer sources remain available by their saved
    hashes. Legacy publications can still use matching installed sources.
    """
    current = source_hashes()
    if not isinstance(expected, dict) or expected.keys() != current.keys():
        return False
    for module, digest in expected.items():
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            return False
        witness = folder / "producer_sources" / f"{digest}.py"
        if witness.exists():
            if hashlib.sha256(witness.read_bytes()).hexdigest() != digest:
                return False
        elif current[module] != digest:
            return False
    return True


def _save_source_witnesses(folder: Path, expected: dict[str, str]) -> None:
    sources = {name: path.read_bytes() for name, path in _source_paths().items()}
    if {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()} != expected:
        raise ValueError("reporting producer sources changed during publication")
    destination = folder / "producer_sources"
    destination.mkdir()
    for name, data in sources.items():
        with (destination / f"{expected[name]}.py").open("xb") as output:
            output.write(data)


def report_folder(store_root: Path, result_id: str) -> Path:
    if len(result_id) != 64 or any(c not in "0123456789abcdef" for c in result_id):
        raise ValueError("reporting result requires its complete verified ID")
    return Path(store_root) / "mffu_reporting" / result_id / mffu_lenses.VERSION


def build_report(
    study: Any, context_index: Any, *, vendor_metrics: dict | None = None
) -> dict[str, Any]:
    gamma = mffu_gamma.build_gamma(study, context_index, vendor_metrics)
    groups = {}
    for configuration in study.configurations:
        for population in ("funded", "strategy"):
            for basis in ("entry", "first_1R_checkpoint"):
                rows = mffu_gamma.selected_rows(gamma, configuration, population, basis)
                full_rows = mffu_gamma.selected_rows(gamma, configuration, population, "entry")
                checkpoint_coverage = dict(Counter(r["first_checkpoint_status"] for r in full_rows))
                groups[f"{configuration}|{population}|{basis}"] = {
                    "groups": mffu_gamma.grouped(rows),
                    "clock_groups": mffu_gamma.grouped(rows, by_clock=True),
                    "coverage": mffu_gamma.coverage(rows),
                    "outcomes": mffu_gamma.outcome_partition(rows),
                    "checkpoint_population_coverage": checkpoint_coverage,
                }
    gamma["precomputed"] = groups
    return {
        "schema": SCHEMA,
        "economic_result_id": study.result_id,
        "plan_id": study.plan_id,
        "reporting_version": mffu_lenses.VERSION,
        "scope": "full_study",
        "evaluation_dates": list(study.calendar),
        "warmup_dates": list(study.warmup),
        "lenses": mffu_lenses.build_rows(study),
        "gamma": gamma,
    }


def publish_report(study: Any, *, context_index: Any = None) -> Path:
    """Publish once after verifying source; does not alter saved results/approvals."""
    if study.store_root is None:
        raise ValueError("reporting publication requires a verified store")
    if context_index is None:
        from alpha_lab.agents.data_infra.ifvg.menthorq_asof import load_v02_eod_asof_zip

        context_index = load_v02_eod_asof_zip(
            study.plan.context_archive_path,
            cutoff_utc=mffu_lenses.stamp(study.plan.source.cutoff_utc).to_pydatetime(),
            expected_archive_sha256=study.plan.context_archive_sha256,
        )
    envelope_path = (
        study.store_root / "funded_comparison_results" / study.result_id / "envelope.json"
    )
    source_envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    source_hash = source_envelope["payload"]["result_json_sha256"]
    report = build_report(study, context_index, vendor_metrics=_vendor_metrics(study.plan))
    report["source_result_json_sha256"] = source_hash
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    hashes = source_hashes()
    envelope = {
        "schema": SCHEMA,
        "economic_result_id": study.result_id,
        "source_result_json_sha256": source_hash,
        "reporting_version": mffu_lenses.VERSION,
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "report_bytes": len(encoded),
        "source_bundle_sha256": context_index.bundle_sha256,
        "reporting_source_sha256": hashes,
    }
    folder = report_folder(study.store_root, study.result_id)
    if folder.exists():
        existing = load_report(study.store_root, study.result_id)
        if existing != report:
            raise ValueError(
                "immutable reporting version already exists with other content; "
                "advance reporting version"
            )
        return folder
    folder.mkdir(parents=True)
    _save_source_witnesses(folder, hashes)
    (folder / "report.json").write_bytes(encoded)
    (folder / "envelope.json").write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    load_report(study.store_root, study.result_id)
    return folder


def _vendor_metrics(plan: Any) -> dict:
    """Supplied percentile of the exact eligible report, never a study re-ranking."""
    from alpha_lab.agents.data_infra.ifvg.menthorq_asof import nominal_eligible_from

    member = "data/canonical/end_of_day/metrics/total_gamma_by_report_date.csv"
    expected = dict(plan.context_table_sha256)[member]
    with zipfile.ZipFile(plan.context_archive_path) as archive:
        names = [n for n in archive.namelist() if n.endswith("/" + member) or n == member]
        if len(names) != 1:
            raise ValueError("bound vendor gamma table is absent or ambiguous")
        raw = archive.read(names[0])
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("bound vendor percentile table bytes changed")
    output = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        day = date.fromisoformat(row["vendor_report_date"])
        if nominal_eligible_from(day) > mffu_lenses.stamp(plan.source.cutoff_utc):
            continue
        percentile = row.get("gex_percentile_1y")
        output[day.isoformat()] = {
            "source_sha256": row["source_sha256"],
            "source_json_pointer": row["source_json_pointer"],
            "gex_percentile_1y": float(percentile) if percentile else None,
        }
    return output


def load_report(store_root: Path, result_id: str) -> dict:
    """Only the expected namespace, checked against the economic envelope."""
    folder = report_folder(store_root, result_id)
    envelope = json.loads((folder / "envelope.json").read_text(encoding="utf-8"))
    encoded = (folder / "report.json").read_bytes()
    source = json.loads(
        (Path(store_root) / "funded_comparison_results" / result_id / "envelope.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        envelope.get("economic_result_id") != result_id
        or envelope.get("schema") != SCHEMA
        or envelope.get("reporting_version") != mffu_lenses.VERSION
        or not _sources_match(folder, envelope.get("reporting_source_sha256"))
        or envelope.get("source_result_json_sha256") != source["payload"]["result_json_sha256"]
        or envelope.get("report_bytes") != len(encoded)
        or envelope.get("report_sha256") != hashlib.sha256(encoded).hexdigest()
    ):
        raise ValueError("reporting companion identity or bytes changed")
    report = json.loads(encoded)
    if (
        report.get("economic_result_id") != result_id
        or report.get("reporting_version") != mffu_lenses.VERSION
    ):
        raise ValueError("reporting payload names another result or definition version")
    return report


def compact_review_tables(report: dict) -> dict[str, list[dict]]:
    """The UI's exact read model, flattened without repeating receipt/map dumps."""
    config_rows = []
    milestones = []
    intervals = []
    for source in report["lenses"]:
        row = {
            key: value
            for key, value in source.items()
            if key
            not in {
                "account_milestone_statuses",
                "no_receipt_intervals",
                "cushion_status_counts",
                "receipt_status_counts",
            }
        }
        config_rows.append(row)
        identity = {
            key: source[key]
            for key in (
                "configuration_id",
                "economic_result_id",
                "reporting_version",
                "firm",
                "scope",
            )
        }
        milestones.extend(
            {**identity, **entry} for entry in source.get("account_milestone_statuses", [])
        )
        intervals.extend({**identity, **entry} for entry in source.get("no_receipt_intervals", []))
    checkpoints = []
    gamma_groups, gamma_clock_groups, coverage, outcomes = [], [], [], []
    for key, tables in report["gamma"]["precomputed"].items():
        config, population, basis = key.split("|")
        identity = {
            "configuration_id": config,
            "population": population,
            "basis": basis,
            "economic_result_id": report["economic_result_id"],
            "reporting_version": report["reporting_version"],
        }
        gamma_groups.extend({**identity, **row} for row in tables["groups"])
        gamma_clock_groups.extend({**identity, **row} for row in tables["clock_groups"])
        coverage.append(
            {
                **identity,
                **tables["coverage"],
                "checkpoint_population_coverage": tables["checkpoint_population_coverage"],
            }
        )
        outcomes.extend({**identity, **row} for row in tables["outcomes"])
    for trade in report["gamma"]["trades"]:
        for basis, at, snapshot, role in (
            ("entry", trade["entry_utc"], trade["entry_snapshot"], trade["context_role"]),
            (
                "first_1R_checkpoint",
                trade.get("first_checkpoint_utc"),
                trade.get("first_checkpoint_snapshot"),
                trade.get("first_checkpoint_context_role"),
            ),
        ):
            if not at:
                continue
            gamma = (snapshot or {}).get("gamma") or {}
            checkpoints.append(
                {
                    **{
                        key: trade.get(key)
                        for key in (
                            "trade_key",
                            "trade_ref",
                            "configuration_id",
                            "population",
                            "account_number",
                            "seq",
                            "economic_result_id",
                            "reporting_version",
                            "trading_date",
                            "entry_utc",
                            "exit_utc",
                            "net_pnl_cents",
                            "net_initial_risk_units",
                            "had_partial",
                            "outcome",
                            "remaining_leg_net_cents",
                            "setup_id",
                            "first_checkpoint_branch",
                        )
                    },
                    "basis": basis,
                    "checkpoint_utc": at,
                    "context_role": role,
                    "receipt_context_utc": (
                        (trade.get("first_checkpoint_origin") or {}).get("receipt_context_utc")
                        if basis == "first_1R_checkpoint" else None
                    ),
                    "reporting_evaluation_utc": (snapshot or {}).get("decision_time_utc"),
                    "origin_validation_status": (
                        (trade.get("first_checkpoint_origin") or {}).get("status")
                        if basis == "first_1R_checkpoint" else None
                    ),
                    "origin_match_method": (
                        (trade.get("first_checkpoint_origin") or {}).get("match_method")
                        if basis == "first_1R_checkpoint" else None
                    ),
                    "gamma_category": mffu_gamma.category(gamma),
                    "chicago_clock": mffu_gamma.clock_bin(at),
                    "total_gamma_vendor_units": gamma.get("value"),
                    "positive_run_age": gamma.get("positive_run_age"),
                    "positive_run_lower_bound": gamma.get("positive_run_lower_bound"),
                    "gamma_status": gamma.get("status"),
                    "gamma_report_date": gamma.get("report_date"),
                    "gamma_source_sha256": gamma.get("source_sha256"),
                    "level_set_id": (snapshot or {}).get("level_set_id"),
                    "source_bundle_sha256": report["gamma"]["source_bundle_sha256"],
                }
            )
    return {
        "configuration_lenses": config_rows,
        "account_milestones": milestones,
        "receipt_intervals": intervals,
        "gamma_checkpoints": checkpoints,
        "gamma_groups": gamma_groups,
        "gamma_clock_groups": gamma_clock_groups,
        "gamma_coverage": coverage,
        "trade_outcomes": outcomes,
        "frozen_geometry": report["gamma"]["geometry"],
    }
