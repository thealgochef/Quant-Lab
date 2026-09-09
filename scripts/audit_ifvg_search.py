"""Audit every saved child file/row and optionally publish corrected evaluations.

No market replay or strategy change. Original content-addressed artifacts remain
unchanged; only the operational frontier locator is advanced after publication.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Direct CLI execution requires the source-path bootstrap above.
from alpha_lab.agents.data_infra.ifvg.audit_contracts import AuditTable  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256, file_sha256  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.preparation import _write_json_atomic  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.reporting import (  # noqa: E402
    build_candidate_report,
    build_decision_report,
    build_executed_trade_report,
    build_invariant_audit,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import OBJECTIVE_DIRECTIONS  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search.child_replay import (  # noqa: E402
    verify_exact_drill_targets,
)
from alpha_lab.agents.data_infra.ifvg.search.frontier import (  # noqa: E402
    ObjectiveSpec,
    build_frontier,
)
from alpha_lab.agents.data_infra.ifvg.search.gates import evaluate_strategy_gates  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: E402
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (  # noqa: E402
    SearchFrontierEnvelope,
    SearchFrontierPayload,
    _child_evaluation_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.review_evidence import (  # noqa: E402
    load_search_day_bars,
    load_search_review_evidence,
)
from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import (  # noqa: E402
    compute_strategy_metrics,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import load_charter  # noqa: E402


def json_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    ).encode()


def leaves(value, prefix="$ "):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, prefix + "." + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaves(child, prefix + f"[{index}]")
    else:
        yield prefix, value


def audit(search_id: str, output: Path, *, apply: bool = False):
    store = ROOT / "data/ifvg_datasets/search/v1"
    state_path = ROOT / "data/ifvg_search_jobs" / search_id / "search_state.json"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    if state["phase"] != "search_complete":
        raise ValueError("only completed searches can be audited/re-evaluated")
    charter = load_charter(store, search_id)
    if charter is None or charter.payload.authorized_firm_contract_ids:
        raise ValueError("this repair supports verified strategy-only charters")
    output.mkdir(parents=True, exist_ok=True)
    original = output / "original_search_state.json"
    if not original.exists():
        original.write_bytes(state_bytes)
    dates = set(charter.payload.date_policy.replay_dates)
    warmup = set(charter.payload.date_policy.warmup_dates)
    failures, files, fields, columns, trades_audit, summaries, publications = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    relevant = {search_id, *(c["core_replay_id"] for c in state["children"])}
    directories = {store / "charters" / search_id}
    # Follow exact IDs in envelopes; include dependent companions/memberships.
    envelopes = {p.parent: json.loads(p.read_text()) for p in store.glob("*/*/envelope.json")}
    changed = True
    while changed:
        changed = False
        for path, envelope in envelopes.items():
            strings = {v for _, v in leaves(envelope) if isinstance(v, str) and len(v) == 64}
            if path not in directories and (path.name in relevant or strings & relevant):
                directories.add(path)
                relevant.add(path.name)
                relevant.update(strings)
                changed = True
    for child in state["children"]:
        core_id = child["core_replay_id"]
        evidence = load_search_review_evidence(store, core_id)
        directories.add(evidence.dataset.exploration_dir)
        tables = evidence.dataset.tables
        for kind, frame in tables.items():
            if not frame.empty:
                day = pd.to_datetime(frame["trading_day"]).dt.strftime("%Y-%m-%d")
                if not day.isin(dates).all():
                    failures.append(f"{core_id}: {kind} has off-policy dates")
                if not frame["is_warmup"].eq(day.isin(warmup)).all():
                    failures.append(f"{core_id}: {kind} has inconsistent warmup flags")
        invariant = build_invariant_audit(
            tables, data_access_audit=evidence.dataset.reports["data_access_audit.json"]
        )
        if not invariant["passed"]:
            failures.append(f"{core_id}: invariant audit failed: {invariant}")
        if invariant != evidence.dataset.reports["invariant_audit.json"]:
            failures.append(f"{core_id}: saved invariant report differs from recomputation")
        audit_dirs = [
            p
            for p, env in envelopes.items()
            if p.parent.name == "fsm_audit_companions"
            and env["payload"]["core_replay_id"] == core_id
        ]
        audit_tables = {}
        for folder in audit_dirs:
            audit_tables = {
                kind: pd.read_parquet(folder / f"{kind.value}.parquet") for kind in AuditTable
            }
            coverage = json.loads((folder / "coverage_report.json").read_text())
            for kind, frame in audit_tables.items():
                if coverage["audit_rows_by_table"][kind.value] != len(frame):
                    failures.append(f"{core_id}: wrong audit count {kind}")
        links = verify_exact_drill_targets(tables, audit_tables, max_targets_per_kind=1_000_000)
        if not links["resolves"]:
            failures.append(f"{core_id}: exact drill links failed {links['failures']}")
        eval_hash = canonical_contract_sha256(
            {
                "core_replay_id": core_id,
                "cost_policy": charter.payload.cost_policy.model_dump(mode="json"),
            }
        )
        metrics = compute_strategy_metrics(
            tables,
            cost_points=charter.payload.cost_policy.cost_points_round_turn,
            evaluation_config_hash=eval_hash,
        )
        gates = evaluate_strategy_gates(metrics, charter.payload.objective_policy.feasibility_gates)
        raw_trades = tables[RecordTable.EXECUTED_TRADE]
        for day, day_trades in raw_trades.groupby("trading_day"):
            bars = load_search_day_bars(ROOT, evidence, str(day)[:10])
            bars = bars[bars.timeframe_ticks == 60].sort_values("logical_close_ts_utc")
            for _, trade in day_trades.iterrows():
                forward = bars[bars.logical_close_ts_utc > trade.entry_ts_utc]
                long = trade.direction == "LONG"
                stop = (
                    forward.low_ticks <= trade.stop_ticks
                    if long
                    else forward.high_ticks >= trade.stop_ticks
                )
                target = (
                    forward.high_ticks >= trade.target_ticks
                    if long
                    else forward.low_ticks <= trade.target_ticks
                )
                hits = forward[stop | target]
                expected = None
                if len(hits):
                    hit = hits.iloc[0]
                    expected = "stop" if bool(stop.loc[hit.name]) else "target"
                checks = {
                    "risk": trade.risk_ticks == abs(trade.entry_ticks - trade.stop_ticks),
                    "one_r_target": abs(trade.target_ticks - trade.entry_ticks) == trade.risk_ticks,
                    "realized_ticks": trade.realized_ticks
                    == (trade.risk_ticks if trade.resolution == "target" else -trade.risk_ticks),
                    "realized_r": np.isclose(
                        trade.realized_r, trade.realized_ticks / trade.risk_ticks
                    ),
                    "entry_geometry": trade.entry_ticks == trade.geometry_entry_bar_close_ticks,
                    "stop_geometry": trade.stop_ticks
                    == trade.geometry_manipulation_swing_ticks
                    + (-1 if long else 1) * trade.geometry_sl_buffer_ticks,
                    "causality": trade.geometry_tap_bar_logical_close_ts_utc
                    < trade.geometry_parent_confirmed_ts_utc
                    <= trade.geometry_lock_bar_logical_close_ts_utc
                    < trade.geometry_opposing_confirmed_ts_utc
                    <= trade.geometry_inversion_bar_logical_close_ts_utc
                    < trade.entry_ts_utc
                    < trade.resolution_ts_utc,
                    "resolution": expected == trade.resolution,
                    "resolution_time": len(hits) > 0
                    and hit.logical_close_ts_utc == trade.resolution_ts_utc,
                    "bar_count": len(hits) > 0
                    and int((forward.logical_close_ts_utc <= hit.logical_close_ts_utc).sum())
                    == int(trade.bars_after_entry_to_resolution),
                }
                bad = [k for k, v in checks.items() if not v]
                if bad:
                    failures.append(f"{core_id}/{trade.trade_id}: {bad}")
                trades_audit.append(
                    {
                        "configuration": child["ordinal"],
                        "core_replay_id": core_id,
                        "trade_id": trade.trade_id,
                        "candidate_id": trade.candidate_id,
                        "trading_day": str(day)[:10],
                        "warmup": bool(trade.is_warmup),
                        "entry_utc": str(trade.entry_ts_utc),
                        "resolution_utc": str(trade.resolution_ts_utc),
                        "entry": trade.entry_ticks * 0.25,
                        "stop": trade.stop_ticks * 0.25,
                        "target": trade.target_ticks * 0.25,
                        "outcome": trade.resolution,
                        "net_r": (
                            trade.realized_ticks
                            - charter.payload.cost_policy.cost_points_round_turn / 0.25
                        )
                        / trade.risk_ticks,
                        "checks": {key: bool(value) for key, value in checks.items()},
                    }
                )
        # The original reports deliberately describe all replay rows. Verify
        # them independently, then publish explicitly scoped research reports.
        cfg_hash = evidence.dataset.manifest["identity"]["evaluation_config_hash"]
        report_builders = {
            "candidate_report.json": lambda ts, cfg_hash=cfg_hash: build_candidate_report(
                ts[RecordTable.ENTRY_CANDIDATE],
                ts[RecordTable.CANDIDATE_LABEL],
                evaluation_config_hash=cfg_hash,
                max_candidates_per_day=None,
            ),
            "decision_report.json": lambda ts, cfg_hash=cfg_hash: build_decision_report(
                ts[RecordTable.ENTRY_CANDIDATE],
                ts[RecordTable.ELIGIBLE_DECISION],
                evaluation_config_hash=cfg_hash,
            ),
            "executed_trade_report.json": lambda ts, cfg_hash=cfg_hash: build_executed_trade_report(
                ts[RecordTable.EXECUTED_TRADE],
                cost_points=charter.payload.cost_policy.cost_points_round_turn,
                evaluation_config_hash=cfg_hash,
                tick_size=charter.payload.cost_policy.tick_size,
            ),
        }
        scoped = {key: frame.loc[~frame.is_warmup].copy() for key, frame in tables.items()}
        reports = {}
        for name, builder in report_builders.items():
            old = builder(tables)
            if canonical_sha256(old) != canonical_sha256(evidence.dataset.reports[name]):
                failures.append(f"{core_id}: original {name} differs from recomputation")
            reports[name] = builder(scoped)
        scope = {
            "metrics_policy_id": "post_warmup_zero_peak_v2",
            "source": evidence.reference,
            "evaluation_dates": sorted(dates - warmup),
            "warmup_dates": sorted(warmup),
            "excluded_warmup_trade_ids": raw_trades.loc[raw_trades.is_warmup, "trade_id"].tolist(),
            "raw_counts": {k.value: len(v) for k, v in tables.items()},
            "research_counts": {k.value: len(v) for k, v in scoped.items()},
            "original_reports_scope": "full_replay_including_warmup",
            "corrected_reports_scope": "post_warmup_research",
        }
        envelope = _child_evaluation_envelope(core_id, charter.payload.cost_policy)
        publications.append(
            (
                envelope,
                {
                    "strategy_metrics.json": json_bytes(metrics.model_dump(mode="json")),
                    "evaluation_scope.json": json_bytes(scope),
                    **{k: json_bytes(v) for k, v in reports.items()},
                },
            )
        )
        summary = {
            "ordinal": child["ordinal"],
            "axis_values": child["axis_value_ids"],
            "core_replay_id": core_id,
            "costed_evaluation_id": envelope.costed_evaluation_id,
            "dataset_id": evidence.dataset.reference.artifact_id,
            "metrics": metrics.model_dump(mode="json", exclude={"trade_stats"}),
            "gates": gates.model_dump(mode="json"),
            "exact_links_checked": links["target_count"],
            "scope": scope,
        }
        summaries.append(summary)
        print(
            f"Child {child['ordinal']}: {metrics.executed_trades} trades, "
            f"{metrics.independent_days} days, E[R]={metrics.net_expectancy_r:.6f}, "
            f"gates={gates.passed}",
            flush=True,
        )
    # Fully parse every JSON leaf and every Parquet row/column, including the
    # large audit event streams; no head()/row sampling in the file audit.
    for folder in sorted(directories):
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            expected = manifest["manifest_payload_sha256"]
            if (
                canonical_sha256(
                    {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
                )
                != expected
            ):
                failures.append(f"manifest mismatch: {folder}")
            for item in manifest["artifacts"]:
                path = (folder.parent if folder.name == "exploration" else folder) / item["path"]
                if file_sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                    failures.append(f"file mismatch: {path}")
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            info = {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            if path.suffix == ".json":
                content = path.read_text(encoding="utf-8")
                data = json.loads(content)
                info["lines"] = len(content.splitlines())
                all_fields = list(leaves(data))
                info["leaf_fields"] = len(all_fields)
                fields.extend({"path": info["path"], "field": k, "value": v} for k, v in all_fields)
            elif path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                info.update(rows=len(frame), columns=len(frame.columns))
                for name in frame:
                    col = frame[name]
                    columns.append(
                        {
                            "path": info["path"],
                            "column": name,
                            "dtype": str(col.dtype),
                            "rows": len(col),
                            "nulls": int(col.isna().sum()),
                            "unique": int(col.nunique()),
                            "all_values_scanned": True,
                        }
                    )
            files.append(info)
    feasible = {}
    for summary in summaries:
        if summary["gates"]["passed"]:
            feasible[summary["core_replay_id"]] = {
                k: v
                for k, v in summary["metrics"].items()
                if k in OBJECTIVE_DIRECTIONS and v is not None
            }
    frontier = build_frontier(
        feasible,
        objectives=tuple(
            ObjectiveSpec(metric=k, direction=OBJECTIVE_DIRECTIONS[k])
            for k in charter.payload.objective_policy.pareto_objectives
        ),
        lexicographic_tie_breaks=charter.payload.objective_policy.lexicographic_tie_breaks,
    )
    result = {
        "search_id": search_id,
        "failures": failures,
        "children": summaries,
        "frontier": frontier.model_dump(mode="json"),
        "coverage": {
            "files": len(files),
            "json_leaf_fields": len(fields),
            "parquet_rows": sum(f.get("rows", 0) for f in files),
            "parquet_columns": len(columns),
            "trade_rows_checked": len(trades_audit),
        },
    }
    (output / "audit.json").write_bytes(json_bytes(result))
    for name, rows in (
        ("files", files),
        ("json_fields", fields),
        ("parquet_columns", columns),
        ("trades", trades_audit),
    ):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, default=str, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    if apply:
        if failures:
            raise ValueError(
                f"refusing publication: {len(failures)} audit failures; inspect audit.json"
            )
        for envelope, sidecars in publications:
            save_or_reuse_envelope(store, "costed_evaluations", envelope, extra_files=sidecars)
        final, _ = save_or_reuse_envelope(
            store,
            "frontiers",
            SearchFrontierEnvelope.from_payload(
                SearchFrontierPayload(search_id=search_id, frontier=frontier)
            ),
        )
        if state_path.read_bytes() != state_bytes:
            raise ValueError("search state changed during audit; published immutable evidence only")
        state["phase_notes"]["frontier_id"] = final.frontier_id
        state["phase_notes"]["metrics_policy_id"] = "post_warmup_zero_peak_v2"
        state["phase_notes"]["evaluation_correction"] = (
            "Warmup excluded; drawdown includes initial zero peak. Original artifacts retained."
        )
        for child, summary in zip(state["children"], summaries, strict=True):
            child["explanation"] = summary["gates"]["human_explanation"]
        _write_json_atomic(state_path, state)
    print(
        json.dumps(
            {"coverage": result["coverage"], "failures": failures, "applied": apply}, indent=2
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    audit(args.search_id, args.output, apply=args.apply)
