"""One-off timeframe-variant IFVG replay: HTF 1H only, parents 5m + 15m.

Provisional config measurement over the same 138 permitted development days
(sealed/June-11 untouched). Entry executions were already restricted to the
``fresh_fvg_continuation`` family by the base profile; the ONLY changes are
``section_overrides = {htf_timeframes: (1H,), parent_timeframes: (5m, 15m)}``.

This is an EXPERIMENT lane: a new profile hash and a new day-artifact
identity (the timeframe set is part of the artifact tag, so day artifacts
rebuild on first run). Nothing here saves an immutable accepted artifact,
touches catalogs, or claims parity with the accepted dataset.

Usage:
    python scripts/run_ifvg_tf_variant.py --probe 2   # timing probe only
    python scripts/run_ifvg_tf_variant.py             # full run + reports
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.dataset import build_ifvg_v2_capture  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.development_access import (  # noqa: E402
    FROZEN_WARMUP_DATES,
    PERMITTED_DEVELOPMENT_DATES,
    DevelopmentDataAccess,
    DevelopmentReplayPolicy,
)
from alpha_lab.agents.data_infra.ifvg.experiment import (  # noqa: E402
    json_safe,
    run_ifvg_v2_evaluation,
)
from alpha_lab.agents.data_infra.ifvg.preparation import _discover_sources  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config  # noqa: E402

VARIANT_NAME = "tf_variant_htf1h_parent5m15m"
SECTION_OVERRIDES = {
    "htf_timeframes": ["1H"],
    "parent_timeframes": ["5m", "15m"],
}
REPORT_DIR = ROOT / "reports" / "ifvg_tf_variants" / VARIANT_NAME


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe",
        type=int,
        default=0,
        help="Replay only the first N chain days and report per-day timing.",
    )
    args = parser.parse_args()

    resolved = resolve_profile_config(
        {
            "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "section_overrides": SECTION_OVERRIDES,
        }
    )
    section = resolved.section
    assert tuple(section.htf_timeframes) == ("1H",)
    assert tuple(section.parent_timeframes) == ("5m", "15m")
    assert section.entry_family == "fresh_fvg_continuation"
    base_cfg = IfvgCaptureConfig()
    cfg = replace(
        base_cfg,
        section=section,
        session_scheme=base_cfg.session_scheme,
        data_dir=ROOT / "data" / "databento",
    )
    print(
        json.dumps(
            {
                "variant": VARIANT_NAME,
                "profile_hash": resolved.section_config_hash,
                "artifacts_tag": cfg.artifacts_tag(),
                "capture_tag": cfg.capture_tag(),
                "timeframes_seconds": list(cfg.timeframes_seconds()),
            }
        ),
        flush=True,
    )

    discovery = DevelopmentDataAccess()
    source_files = _discover_sources(discovery, data_dir=cfg.data_dir, symbol=cfg.symbol)
    if tuple(d for d in FROZEN_WARMUP_DATES if d in source_files) != FROZEN_WARMUP_DATES:
        raise RuntimeError("the frozen ten-date warmup is not fully available")
    all_dates = tuple(d for d in PERMITTED_DEVELOPMENT_DATES if d in source_files)
    replay_dates = all_dates[: args.probe] if args.probe else all_dates
    # the policy always carries the full permitted allowlist (it requires the
    # frozen warmup); a probe truncates only the replayed prefix.
    policy = DevelopmentReplayPolicy(all_dates, development_audit=discovery.audit)

    started = time.perf_counter()
    day_times: list[tuple[str, float]] = []
    last_mark = started

    def progress(completed: int, total: int, day: str) -> None:
        nonlocal last_mark
        now = time.perf_counter()
        day_times.append((day, now - last_mark))
        last_mark = now
        print(
            json.dumps(
                {
                    "completed": completed,
                    "total": total,
                    "source_date": day,
                    "day_seconds": round(day_times[-1][1], 1),
                }
            ),
            flush=True,
        )

    capture = build_ifvg_v2_capture(
        replay_dates,
        cfg,
        resolved,
        access_policy=policy,
        cached_artifacts_only=False,
        progress_fn=progress,
    )
    elapsed = time.perf_counter() - started
    timing = {
        "days": len(replay_dates),
        "rebuilt_day_artifacts": len(capture.rebuilt_days),
        "cached_day_artifacts": len(capture.cached_artifact_days),
        "total_seconds": round(elapsed, 1),
        "mean_seconds_per_day": round(elapsed / max(len(replay_dates), 1), 1),
    }
    print(json.dumps({"timing": timing}), flush=True)

    if args.probe:
        print(
            json.dumps(
                {
                    "projection_full_run_minutes": round(
                        (elapsed / max(len(replay_dates), 1)) * len(all_dates) / 60,
                        1,
                    )
                }
            ),
            flush=True,
        )
        return 0

    audit_dict = policy.audit.as_dict(allowlist=policy.allowlist)
    reports = run_ifvg_v2_evaluation(
        capture.tables,
        resolved_profile=resolved,
        data_access_audit=audit_dict,
        tick_size=cfg.tick_size,
        old_artifact_mutations=0,
    )
    funnel_totals: dict[str, int] = {}
    for counters in capture.day_funnels.values():
        for key, value in counters.items():
            funnel_totals[key] = funnel_totals.get(key, 0) + int(value)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": VARIANT_NAME,
        "status": "provisional config measurement — NOT a strategy verdict",
        "section_overrides": SECTION_OVERRIDES,
        "profile_hash": resolved.section_config_hash,
        "evaluation_config_hash": resolved.evaluation_config_hash,
        "replay_days": len(replay_dates),
        "timing": timing,
        "funnel_totals": funnel_totals,
        "candidate_report": reports["candidate_report"],
        "decision_report": reports["decision_report"],
        "executed_trade_report": reports["executed_trade_report"],
        "invariant_audit_passed": reports["invariant_audit"]["passed"],
        "protected_counters": audit_dict.get("protected_counters"),
    }
    out = REPORT_DIR / "IFVG_TF_VARIANT_REPORT.json"
    out.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(out)}), flush=True)
    summary = {
        "setups_born": funnel_totals.get("setups_born"),
        "entry_candidates_fresh": funnel_totals.get(
            "entry_candidates_fresh_fvg_continuation"
        ),
        "executions_opened": funnel_totals.get("executions_opened"),
        "resolved_target": funnel_totals.get("resolved_target"),
        "resolved_stop": funnel_totals.get("resolved_stop"),
        "invariants_passed": reports["invariant_audit"]["passed"],
    }
    print(json.dumps({"summary": summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
