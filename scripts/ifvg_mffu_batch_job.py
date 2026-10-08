"""Prepare, approve, run, and inspect the bounded 64-intent MFFU comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from research_workspace import require_external_work_paths


def _source_paths(core_root: Path) -> None:
    core_src = str(core_root.resolve(strict=True) / "src")
    lab_src = str(Path(__file__).resolve().parents[1] / "src")
    inherited = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    os.environ["PYTHONPATH"] = os.pathsep.join([core_src, lab_src, *inherited])
    sys.path[:0] = [core_src, lab_src]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "approve", "run", "publish", "status"))
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--plan-id")
    parser.add_argument("--handoff-zip", type=Path)
    parser.add_argument("--reference-store-root", type=Path)
    parser.add_argument("--context-archive", type=Path)
    parser.add_argument("--owner-statement")
    parser.add_argument("--approved-on")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--result-id")
    parser.add_argument("--reports-root", type=Path)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--findings-file", type=Path)
    parser.add_argument("--matrix-png", type=Path)
    parser.add_argument("--result-png", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command != "status":
        work_paths = {"--store-root": args.store_root, "--state-root": args.state_root}
        if args.staging_root is not None:
            work_paths["--staging-root"] = args.staging_root
        try:
            require_external_work_paths(Path(__file__).resolve().parents[1], work_paths)
        except ValueError as error:
            parser.error(str(error))
    _source_paths(args.core_root)
    if args.command == "prepare":
        from alpha_lab.propsim.funded.mffu_batch_plan import (
            build_mffu_batch_plan,
            save_mffu_batch_plan,
        )
        from alpha_lab.propsim.funded.mffu_batch_run import validate_mffu_runtime

        for field in ("handoff_zip", "reference_store_root", "context_archive"):
            if getattr(args, field) is None:
                raise SystemExit(f"prepare requires --{field.replace('_', '-')}")
        envelope = build_mffu_batch_plan(
            handoff_zip=args.handoff_zip,
            reference_store_root=args.reference_store_root,
            context_archive_path=args.context_archive,
            runtime_source_files=(
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("research_workspace.py"),
            ),
            core_root=args.core_root,
        )
        validate_mffu_runtime(envelope.payload)
        plan_id = save_mffu_batch_plan(args.store_root, envelope)
        working = args.state_root / plan_id
        working.mkdir(parents=True, exist_ok=True)
        (working / "resolved_plan.json").write_text(
            envelope.payload.model_dump_json(indent=2), encoding="utf-8",
        )
        with (working / "reuse_ledger.csv").open("w", newline="", encoding="utf-8") as file:
            ledger = csv.writer(file)
            ledger.writerow(("variant_id", "family", "base_reference",
                             "effective_section_config_hash", "effective_behavior_hash",
                             "planned_action", "reason"))
            for row in envelope.payload.variants:
                candidate = row.variant_id in {"MCB001", "MCB025"}
                ledger.writerow((
                    row.variant_id, row.family, row.base_reference,
                    row.effective_section_config_hash, row.effective_behavior_hash,
                    "verify_reuse_else_replay" if candidate else "replay",
                    "repaired fixed partial reference requires sealed compatibility proof"
                    if candidate else "declared behavior differs from repaired references",
                ))
        print(json.dumps({"funded_comparison_plan_id": plan_id, "variants": 64,
                          "status": "prepared_unapproved"}, sort_keys=True))
        return
    if not args.plan_id:
        raise SystemExit(f"{args.command} requires --plan-id")
    if args.command == "approve":
        from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
        from alpha_lab.propsim.funded.comparison_plan import PLAN_STORE
        from alpha_lab.propsim.funded.comparison_study import record_owner_approval
        from alpha_lab.propsim.funded.mffu_batch_plan import (
            MffuBatchPlanEnvelope,
            verify_mffu_batch_plan,
        )
        from alpha_lab.propsim.funded.mffu_batch_run import validate_mffu_runtime

        if not args.owner_statement or not args.approved_on:
            raise SystemExit("approve requires --owner-statement and --approved-on")
        envelope = load_verified_envelope(
            args.store_root, PLAN_STORE, args.plan_id, MffuBatchPlanEnvelope,
        )
        verify_mffu_batch_plan(envelope.payload)
        validate_mffu_runtime(envelope.payload, verify_inputs=False)
        approval_id = record_owner_approval(
            args.store_root, args.plan_id, approved_on=args.approved_on,
            channel="codex_conversation", statement=args.owner_statement,
            scope=("The exact 64-intent MyFundedFutures context batch, one managed run, "
                   "10 warmup and 253 evaluation dates, with frozen sources and fees"),
        )
        print(json.dumps({"funded_comparison_plan_id": args.plan_id,
                          "funded_comparison_approval_id": approval_id}, sort_keys=True))
        return
    if args.command == "run":
        from alpha_lab.propsim.funded.mffu_batch_run import run_mffu_batch

        print(json.dumps(run_mffu_batch(
            plan_id=args.plan_id, store_root=args.store_root,
            state_root=args.state_root, workers=args.workers,
        ), sort_keys=True))
        return
    if args.command == "publish":
        from alpha_lab.propsim.funded.mffu_batch_review import publish_mffu_result_review

        needed = ("result_id", "reports_root", "staging_root", "findings_file",
                  "matrix_png", "result_png")
        for field in needed:
            if getattr(args, field) is None:
                raise SystemExit(f"publish requires --{field.replace('_', '-')}")
        published = publish_mffu_result_review(
            plan_id=args.plan_id, result_id=args.result_id,
            store_root=args.store_root, state_root=args.state_root,
            staging_root=args.staging_root,
            reports_root=args.reports_root,
            review_findings=args.findings_file.read_text(encoding="utf-8"),
            screenshots={"matrix": args.matrix_png, "result": args.result_png},
        )
        print(json.dumps({"path": str(published.path), "receipt": published.receipt},
                         sort_keys=True))
        return
    from alpha_lab.propsim.funded.runner import read_state

    print(json.dumps(read_state(args.state_root, args.plan_id) or {}, sort_keys=True))


if __name__ == "__main__":
    main()
