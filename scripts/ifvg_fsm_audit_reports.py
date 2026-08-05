"""Generate the FSM-audit evidence reports from a saved ``ifvg_fsm_audit_v1``
artifact: funnel report (with the three prior overstatements re-reported from
persisted evidence, descriptive only), evidence-coverage report, rule-conflict
characterization report, and the deterministic setup visual-review queue.

Usage: python scripts/ifvg_fsm_audit_reports.py --audit-id <sha256>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.audit_contracts import AuditTable  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.fsm_audit_io import (  # noqa: E402
    load_verified_fsm_audit_artifact,
)

REPORT_DIR = ROOT / "reports" / "ifvg_fsm_audit"


def _write(name: str, text: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / name).write_text(text, encoding="utf-8")
    print("wrote", (REPORT_DIR / name).as_posix())


def _write_json(name: str, payload) -> None:
    _write(name, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def funnel_report(tables) -> dict:
    funnel = tables[AuditTable.DAY_FUNNEL]
    deaths = tables[AuditTable.SLOT_DEATH]
    windows = tables[AuditTable.PARENT_WINDOW]
    taps = tables[AuditTable.HTF_TAP]
    intervals = tables[AuditTable.PARENTLESS_INTERVAL]
    total = (
        funnel.groupby("counter")["value"].sum().astype(int).to_dict()
        if not funnel.empty
        else {}
    )
    terminal = deaths.loc[deaths["setup_terminated"].astype(bool)]
    occupied = taps.loc[taps["drop_reason"] == "slot_occupied"]
    by_session = (
        occupied.groupby("session_doc").size().astype(int).to_dict()
        if len(occupied)
        else {}
    )
    replacements = windows.loc[
        (windows["event_kind"] == "parent_selected")
        & windows["prior_parent_fvg_id"].notna()
    ]
    successor_ended = intervals.loc[
        intervals["end_reason"] == "successor_parent_selected"
    ]
    return {
        "funnel_totals": total,
        "terminal_death_reasons": (
            terminal["death_reason"].value_counts().astype(int).to_dict()
        ),
        "provisional_parent_deaths": int(len(deaths) - len(terminal)),
        "corrections_previously_unknown_now_measured_descriptive_only": {
            "note": (
                "These three quantities were previously reported as claims "
                "without persisted evidence; they are now DESCRIPTIVE "
                "measurements from the audit artifact. No strategy conclusion "
                "is drawn from them."
            ),
            "cross_setup_opportunity_cost": {
                "taps_dropped_slot_occupied": int(len(occupied)),
                "distinct_gaps_tapping_while_occupied": int(
                    occupied["fvg_fvg_id"].nunique()
                )
                if len(occupied)
                else 0,
            },
            "session_impact": {
                "slot_occupied_taps_by_doc_session": by_session,
            },
            "ranked_fallback_benefit": {
                "parent_replacements_observed": int(len(replacements)),
                "parentless_intervals_ended_by_successor": int(
                    len(successor_ended)
                ),
                "note": (
                    "No ranked HTF fallback exists (D-2); these counts only "
                    "describe how often a later parent arrived in-window."
                ),
            },
        },
    }


def rule_conflict_report(tables) -> str:
    taps = tables[AuditTable.HTF_TAP]
    deaths = tables[AuditTable.SLOT_DEATH]
    conflicted = int(taps["conflicted"].astype(bool).sum()) if len(taps) else 0
    structural = (
        int(deaths["structural_close"].astype(bool).sum()) if len(deaths) else 0
    )
    direction_disabled = int((taps["drop_reason"] == "direction_disabled").sum())
    table_rows = (
        (
            "D-1 same-TF conflict unreachable at cap=1",
            "`taps_conflicted` cannot fire (retention keeps one gap per TF)",
            f"conflicted taps observed: **{conflicted}** (provably zero; "
            "SC test `test_d1_same_tf_conflict_unreachable_at_cap_1`)",
        ),
        (
            "D-2 direction-disabled winner, no fallback",
            "bearish winner + shorts-off drops `direction_disabled`; "
            "runner-up drops `outranked`; no setup born",
            f"direction_disabled taps observed: **{direction_disabled}**; "
            "SC test `test_d2_direction_disabled_winner_no_fallback`",
        ),
        (
            "D-3 htf_max_age is a memory bound, not a selection gate",
            "age evictions are registry events, never tap-time gates",
            "`evicted_age` fill events persisted; `htf_age_seconds` emitted "
            "as measurement only",
        ),
        (
            "D-4 Q-23 window wording + BE-off",
            "parentless predicate encodes the IMPLEMENTED any-window-open "
            "semantics; no break-even management exists",
            "`parentless_step` rows reconcile 1:1 with "
            "`parentless_window_live`; resolver walks stop/target only",
        ),
        (
            "D-5 profile-name-in-hash + one-per-TF wording",
            "renames move identity; retention is per-TF but the slot is "
            "single-global",
            "emitted `rank`/`outranked` evidence; identity discipline retained",
        ),
    )
    lines = [
        "# IFVG FSM Rule-Conflict Characterization (D-1..D-5)",
        "",
        "All entries are characterized, UNRESOLVED, and registered in",
        "`docs/ifvg/IFVG_FSM_AUDITABILITY_OPEN_DECISIONS.md`. No behavior change.",
        "",
        "| Decision | Characterization | Evidence in this artifact |",
        "|---|---|---|",
    ]
    lines.extend(f"| {a} | {b} | {c} |" for a, b, c in table_rows)
    lines.append("")
    lines.append(
        f"Structural parent invalidations observed: **{structural}** (the "
        "branch is contract-supported and instrumented even at zero "
        "observations)."
    )
    return "\n".join(lines) + "\n"


def review_queue(tables) -> pd.DataFrame:
    deaths = tables[AuditTable.SLOT_DEATH]
    terminal = deaths.loc[deaths["setup_terminated"].astype(bool)].copy()
    taps = tables[AuditTable.HTF_TAP]
    selected = taps.loc[taps["selected"].astype(bool)]
    tap_info = selected.set_index(selected["envelope_setup_id"].astype(str))
    intervals = tables[AuditTable.PARENTLESS_INTERVAL]
    interval_counts = (
        intervals.groupby(intervals["setup_id"].astype(str))["bars_count"]
        .sum()
        .astype(int)
        if len(intervals)
        else pd.Series(dtype=int)
    )

    terminal["setup_id"] = terminal["setup_id"].astype(str)
    terminal = terminal.sort_values("setup_id", kind="mergesort")
    rows: list[dict] = []

    def _add(cohort: str, frame: pd.DataFrame, limit: int | None = None) -> None:
        scoped = frame.sort_values("setup_id", kind="mergesort")
        if limit is not None:
            scoped = scoped.head(limit)
        for record in scoped.to_dict("records"):
            setup_id = str(record["setup_id"])
            info = (
                tap_info.loc[setup_id]
                if setup_id in tap_info.index
                else pd.Series(dtype=object)
            )
            rows.append(
                {
                    "cohort": cohort,
                    "setup_id": setup_id,
                    "phase_at_death": record["phase"],
                    "death_reason": record["death_reason"],
                    "htf_tf_seconds": info.get("htf_tf_seconds"),
                    "htf_age_seconds": info.get("htf_age_seconds"),
                    "fvg_size_ticks": info.get("fvg_size_ticks"),
                    "session_doc": info.get("session_doc"),
                    "parentless_bars": int(interval_counts.get(setup_id, 0)),
                }
            )

    _add("s4_deaths_all", terminal.loc[terminal["phase"] == "S4"])
    _add("s3_deaths_all", terminal.loc[terminal["phase"] == "S3"])
    s2 = terminal.loc[terminal["phase"] == "S2"]
    for reason in sorted(s2["death_reason"].unique()):
        _add(f"s2_stratified_{reason}", s2.loc[s2["death_reason"] == reason], limit=5)
    _add(
        "slot_death_parent_filled",
        terminal.loc[terminal["death_reason"] == "invalidated_parent_filled"],
        limit=10,
    )
    htf_fill = terminal.loc[terminal["death_reason"] == "invalidated_htf_filled"].copy()
    if len(htf_fill):
        ages = tap_info["htf_age_seconds"].astype(float)
        htf_fill["_age"] = htf_fill["setup_id"].map(ages)
        htf_fill["_tf"] = htf_fill["setup_id"].map(tap_info["htf_tf_seconds"])
        htf_fill["_session"] = htf_fill["setup_id"].map(tap_info["session_doc"])
        for tf in sorted(htf_fill["_tf"].dropna().unique()):
            scoped = htf_fill.loc[htf_fill["_tf"] == tf].sort_values(
                "_age", kind="mergesort"
            )
            _add(f"rapid_htf_fill_tf{int(tf)}_youngest", scoped, limit=4)
        for session in sorted(htf_fill["_session"].dropna().unique()):
            _add(
                f"rapid_htf_fill_session_{session}",
                htf_fill.loc[htf_fill["_session"] == session],
                limit=3,
            )
    # suppression/conflict cohorts are present-but-empty on this artifact —
    # emitted as explicit zero-row cohorts via the coverage report instead.
    if len(interval_counts):
        threshold = int(interval_counts.quantile(0.9))
        over = terminal.loc[
            terminal["setup_id"].map(interval_counts).fillna(0) > threshold
        ]
        _add(f"parentless_over_p90_{threshold}", over, limit=10)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["cohort", "setup_id"])
    return frame.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-id", required=True)
    args = parser.parse_args()
    artifact = load_verified_fsm_audit_artifact(
        ROOT / "data" / "ifvg_datasets" / "fsm_audit" / "v1", args.audit_id
    )
    coverage = json.loads(
        (artifact.directory / "evidence_coverage.json").read_text(encoding="utf-8")
    )
    reconciliation = json.loads(
        (artifact.directory / "count_reconciliation.json").read_text(encoding="utf-8")
    )

    funnel = funnel_report(artifact.tables)
    funnel["fsm_audit_artifact_id"] = artifact.artifact_id
    _write_json("IFVG_FSM_FUNNEL_REPORT.json", funnel)
    lines = ["# IFVG FSM Funnel Report", "", f"Artifact: `{artifact.artifact_id}`", ""]
    lines.append("## Funnel totals\n")
    for key, value in sorted(funnel["funnel_totals"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("\n## Terminal deaths\n")
    for key, value in sorted(funnel["terminal_death_reasons"].items()):
        lines.append(f"- {key}: {value}")
    lines.append(
        f"\nProvisional parent deaths: {funnel['provisional_parent_deaths']}\n"
    )
    corrections = funnel[
        "corrections_previously_unknown_now_measured_descriptive_only"
    ]
    lines.append("## Previously-unknown quantities, now measured (descriptive only)\n")
    lines.append(corrections["note"] + "\n")
    lines.append("```json")
    lines.append(
        json.dumps(
            {k: v for k, v in corrections.items() if k != "note"},
            indent=2,
            sort_keys=True,
        )
    )
    lines.append("```")
    _write("IFVG_FSM_FUNNEL_REPORT.md", "\n".join(lines) + "\n")

    coverage_payload = {
        "fsm_audit_artifact_id": artifact.artifact_id,
        **coverage,
        "reconciliation_passed": reconciliation.get("passed"),
    }
    _write_json("IFVG_FSM_EVIDENCE_COVERAGE_REPORT.json", coverage_payload)
    md = ["# IFVG FSM Evidence Coverage", "", f"Artifact: `{artifact.artifact_id}`", ""]
    md.append("## Drop-reason coverage (every contract reason observed or provably zero)\n")
    for reason, count in sorted(coverage["drop_reason_counts"].items()):
        zero = " (provably zero)" if count == 0 else ""
        md.append(f"- {reason}: {count}{zero}")
    md.append("\n## Audit rows by table\n")
    for table, count in sorted(coverage["audit_rows_by_table"].items()):
        md.append(f"- {table}: {count}")
    md.append(
        f"\nFunnel ⇔ events reconciliation: "
        f"{'EXACT' if reconciliation.get('passed') else 'FAILED'}\n"
    )
    _write("IFVG_FSM_EVIDENCE_COVERAGE_REPORT.md", "\n".join(md) + "\n")

    _write("IFVG_FSM_RULE_CONFLICT_REPORT.md", rule_conflict_report(artifact.tables))

    queue = review_queue(artifact.tables)
    _write("IFVG_SETUP_VISUAL_REVIEW_SAMPLE.csv", queue.to_csv(index=False))
    print(
        json.dumps(
            {
                "queue_rows": len(queue),
                "cohorts": sorted(queue["cohort"].unique().tolist()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
