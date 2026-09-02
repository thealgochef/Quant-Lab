"""Phase 4 operator runner — the real ≤5-day bounded verification (plan §6).

Subcommands (importing this module launches nothing):

* ``preflight`` — exact-load the persisted ``VerificationRunEnvelope`` of the
  store, then run every §6.1 check BEFORE any source path is constructed
  (namespace, witness, real authorization, logical window, mapping, program
  allowlist, seed). Prints the typed preflight record or the typed refusal.
* ``run`` — preflight, then execute the REAL baseline pipeline twice through
  the registered ``pipeline_baseline_verification_v1`` executor (attempt 1
  fresh, attempt 2 verified reuse), assemble the immutable
  ``R1BaselineGateReport`` (§6.2) and ``BoundedReleaseControlFlowReport``
  (§6.3), and write the evidence folder files of §6.4. Every gate is
  reported truthfully; any authorization, namespace, head-witness, source,
  date, seed, neutrality, immutable-evidence or forbidden-access failure
  aborts with a typed reason. Without the owner's persisted authorization the
  executor refuses at construction — no config, policy or source path exists.

Research profitability, strategy, payout, feature-selection or promotion
gates are never applied; the seed snapshot is consumed here, never created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EVIDENCE_FILES = (
    "VERIFICATION_RUN.md",
    "R1_BASELINE_GATES.json",
    "BOUNDED_RELEASE_CONTROL_FLOW.json",
    "ACCESS_AUDIT.jsonl",
    "REUSE_RECEIPT.json",
    "MANIFEST_AND_SIDECAR_HASHES.json",
)


def _print(payload: Any) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, default=str))


def _load_run(store_root: Path):
    """The ONE persisted verification run of the store (catalogued; stores
    are never listed) — refused when absent."""

    from alpha_lab.agents.data_infra.ifvg.search.executors import (  # noqa: PLC0415
        _verification_run_envelope,
    )

    return _verification_run_envelope(store_root)


def _preflight(args) -> tuple[int, dict[str, Any]]:
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config  # noqa: PLC0415
    from alpha_lab.agents.data_infra.ifvg.search.bounded_verification import (  # noqa: PLC0415
        BoundedVerificationRefusalError,
        preflight_bounded_verification,
    )

    store_root = Path(args.store_root)
    repo_root = Path(args.repo_root)
    try:
        run = _load_run(store_root)
        resolved = resolve_profile_config({"profile_name": run.payload.baseline_profile_id})
        record = preflight_bounded_verification(
            store_root=store_root,
            repo_root=repo_root,
            verification_run=run,
            authorization=run.payload.verification_authorization,
            pipeline_semantic_id=run.payload.pipeline_semantic_id,
            baseline_profile_id=run.payload.baseline_profile_id,
            baseline_section_config_hash=resolved.section_config_hash,
        )
    except BoundedVerificationRefusalError as error:
        return 2, {"status": "refused", "reason": error.reason, "detail": str(error)}
    except PermissionError as error:
        return 2, {"status": "refused", "reason": "fail_before_path", "detail": str(error)}
    return 0, {"status": "preflight_passed", "preflight": record.model_dump(mode="json")}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_hashes(store_root: Path, store: str, artifact_id: str) -> dict[str, str]:
    directory = store_root / store / artifact_id
    if not directory.is_dir():
        return {}
    return {name.name: _sha256_file(name) for name in sorted(directory.iterdir()) if name.is_file()}


def _run(args) -> tuple[int, dict[str, Any]]:
    code, preflight = _preflight(args)
    if code != 0:
        return code, preflight
    from alpha_lab.agents.data_infra.ifvg.search.bounded_verification import (  # noqa: PLC0415
        build_bounded_release_control_flow_report,
        build_r1_baseline_gate_report,
        save_bounded_release_control_flow_report,
        save_r1_baseline_gate_report,
        store_behavior_proofs,
    )
    from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
        SearchCharterEnvelope,
    )
    from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (  # noqa: PLC0415
        load_executed_trade_table,
    )
    from alpha_lab.agents.data_infra.ifvg.search.executors import (  # noqa: PLC0415
        pipeline_baseline_verification_entry,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        PipelineResultEnvelope,
        PipelineSemanticIdentity,
        StageStatus,
        WorkerPolicy,
        read_pipeline_state,
        run_pipeline,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        load_verified_envelope,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (  # noqa: PLC0415
        SupersessionHeadWitness,
    )
    from alpha_lab.agents.data_infra.ifvg.search.verification import (  # noqa: PLC0415
        ControlFlowGateReport,
    )

    store_root = Path(args.store_root)
    state_root = Path(args.state_root)
    run = _load_run(store_root)
    pipeline_id = run.payload.pipeline_semantic_id
    semantic = load_verified_envelope(
        store_root, "pipeline_specs", pipeline_id, PipelineSemanticIdentity
    )
    charter = load_verified_envelope(
        store_root, "charters", semantic.payload.search_charter_id, SearchCharterEnvelope
    )
    # the registered REAL executor: refuses at construction without the
    # owner's persisted authorization (fail-before-path)
    wiring = pipeline_baseline_verification_entry(charter, semantic, store_root=store_root)
    policy = WorkerPolicy(max_workers=1, max_tasks_per_child=1, memory_budget_bytes=2 << 30)
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def _result_after_attempt():
        state_now = read_pipeline_state(state_root, pipeline_id)
        result_id_now = dict(state_now.get("publication") or {}).get("pipeline_result_id")
        if not result_id_now:
            raise PermissionError(
                "the attempt produced no pipeline result (S15 did not complete); the bounded "
                "verification cannot be reported"
            )
        envelope = load_verified_envelope(
            store_root, "search_results", str(result_id_now), PipelineResultEnvelope
        )
        return state_now, str(result_id_now), envelope

    attempts = []
    # adversarial RA-04: attempt 1's gate report is CAPTURED before attempt 2 runs
    attempts.append(
        run_pipeline(
            semantic, charter, store_root=store_root, state_root=state_root, wiring=wiring,
            worker_policy=policy, operational_retry_reason=None,
        )
    )
    _state1, first_result_id, first_result = _result_after_attempt()
    first_gates: ControlFlowGateReport = first_result.payload.control_flow_gates
    attempts.append(
        run_pipeline(
            semantic, charter, store_root=store_root, state_root=state_root, wiring=wiring,
            worker_policy=policy,
            operational_retry_reason="bounded verification reuse attempt",
        )
    )
    state, result_id, pipeline_result = _result_after_attempt()
    gates: ControlFlowGateReport = pipeline_result.payload.control_flow_gates
    children = list(state.get("children") or [])
    reused_zero_replay = bool(children) and all(
        row.get("state") == "reused" and int(row.get("replay_invocations", 1)) == 0
        for row in children
    )
    second_statuses = attempts[1].stage_statuses
    all_reused = all(
        status in (StageStatus.REUSED.value, StageStatus.BLOCKED.value)
        for stage, status in second_statuses.items()
    )
    # the executed-trade table ids are the STATE's recorded evidence (R6.1-FIX:
    # the children rows carry the table id S02 persisted / verified)
    table_ids = [
        str(row["executed_trade_table_id"])
        for row in children
        if row.get("executed_trade_table_id")
    ]
    table_loads = bool(table_ids)
    for table_id in table_ids:
        try:
            load_executed_trade_table(store_root, table_id)
        except Exception:  # noqa: BLE001 — recorded as a failed proof, never swallowed silently
            table_loads = False
    # adversarial RA-04: the store-behavior proofs are GATHERED on the fixture's
    # own tables through scratch copies under the evidence directory
    probe_root = evidence_dir / "_scratch_store_probes"
    gathered = {proof: bool(table_ids) for proof in (
        "identical_bytes_reuse",
        "different_bytes_fail_closed",
        "missing_or_corrupt_manifest_fails_closed",
        "corrupt_sidecar_fails_closed",
    )}
    probe_observations: dict[str, Any] = {}
    for table_id in table_ids:
        table_proofs, table_evidence = store_behavior_proofs(
            store_root, table_id, probe_root / table_id[:16]
        )
        for proof, held in table_proofs.items():
            gathered[proof] = gathered[proof] and bool(held)
        probe_observations[table_id] = table_evidence["observations"]
    neutrality = {row["core_replay_id"]: row.get("neutrality") for row in children}
    def _mode_digest(key: str) -> str:
        facts = {k: (v or {}).get(key) for k, v in neutrality.items()}
        return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()

    audit_disabled = _mode_digest("audit_disabled_core_table_hashes")
    audit_enabled = _mode_digest("audit_enabled_core_table_hashes")
    witness = SupersessionHeadWitness.model_validate(
        run.payload.verification_authorization.supersession_head_witness.model_dump(mode="json")
    )
    proofs = {
        # the second attempt re-derived the identical pipeline result (every
        # native id repeated) and every child was verified reuse
        "native_ids_repeat": all_reused and first_result_id == result_id,
        "core_table_hashes_repeat": audit_disabled == audit_enabled,
        "executed_trade_table_exact_loads": table_loads and bool(table_ids),
        "second_invocation_reused_with_zero_replay": reused_zero_replay and all_reused,
        **gathered,
    }
    gate_report = build_r1_baseline_gate_report(
        verification_run_id=run.verification_run_id,
        pipeline_semantic_id=pipeline_id,
        store_namespace_id=run.payload.verification_authorization.store_namespace_id,
        supersession_head_witness=witness,
        first_attempt_gates=first_gates,
        second_attempt_gates=gates,
        audit_disabled_core_table_hashes_sha256=audit_disabled,
        audit_enabled_core_table_hashes_sha256=audit_enabled,
        proofs=proofs,
        evidence_refs={
            "first_pipeline_result_id": first_result_id,
            "pipeline_result_id": str(result_id),
            "executed_trade_table_ids": ",".join(table_ids),
            "store_probe_observations": json.dumps(probe_observations, sort_keys=True),
            "supplementary_store_suite": "tests/agents/ifvg_search/test_store_sidecar_probe.py "
            "tests/agents/ifvg_search/test_executed_trade_table.py "
            "tests/agents/ifvg_search/test_pipeline_evidence_integrity.py",
        },
    )
    save_r1_baseline_gate_report(store_root, gate_report)
    bounded = build_bounded_release_control_flow_report(
        store_root=store_root,
        state=state,
        store_namespace_id=run.payload.verification_authorization.store_namespace_id,
        supersession_head_witness=witness,
        verification_run_id=run.verification_run_id,
        mbp1_diagnostic_id=args.mbp1_diagnostic_id,
    )
    save_bounded_release_control_flow_report(store_root, bounded)
    (evidence_dir / "R1_BASELINE_GATES.json").write_text(
        json.dumps(gate_report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "BOUNDED_RELEASE_CONTROL_FLOW.json").write_text(
        json.dumps(bounded.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "REUSE_RECEIPT.json").write_text(
        json.dumps(
            {
                "attempts": [a.attempt.model_dump(mode="json") for a in attempts],
                "second_attempt_stage_statuses": second_statuses,
                "children": children,
            },
            indent=2, sort_keys=True, default=str,
        ) + "\n",
        encoding="utf-8",
    )
    hashes = {
        f"search_results/{result_id}": _entry_hashes(store_root, "search_results", str(result_id)),
        **{f"executed_trade_tables/{tid}": _entry_hashes(store_root, "executed_trade_tables", tid)
           for tid in table_ids},
        f"r1_baseline_gate_reports/{gate_report.r1_baseline_gate_report_id}": _entry_hashes(
            store_root, "r1_baseline_gate_reports", gate_report.r1_baseline_gate_report_id
        ),
    }
    (evidence_dir / "MANIFEST_AND_SIDECAR_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (evidence_dir / "ACCESS_AUDIT.jsonl").open("a", encoding="utf-8") as sink:
        for row in children:
            sink.write(json.dumps({"core_replay_id": row.get("core_replay_id"),
                                   "access_audit": row.get("access_audit")}, sort_keys=True,
                                  default=str) + "\n")
    stamps = dict(pipeline_result.payload.verification_stamps or {})
    (evidence_dir / "VERIFICATION_RUN.md").write_text(
        "\n".join(
            [
                "# R1 bounded verification run",
                "",
                f"- verification_run_id: `{run.verification_run_id}`",
                f"- pipeline_semantic_id: `{pipeline_id}`",
                f"- allowlist: {list(run.payload.allowlist)} (hash `{run.payload.allowlist_hash}`)",
                f"- R1 baseline gates passed: {gate_report.payload.passed}",
                f"- bounded release control-flow passed: {bounded.payload.passed}",
                f"- stamps: {json.dumps(stamps, sort_keys=True)}",
                "- full_pipeline_not_run=true; verification_only; not_for_research_interpretation",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0 if (gate_report.payload.passed and bounded.payload.passed) else 1, {
        "status": "completed",
        "r1_baseline_gates_passed": gate_report.payload.passed,
        "bounded_release_control_flow_passed": bounded.payload.passed,
        "r1_baseline_gate_report_id": gate_report.r1_baseline_gate_report_id,
        "bounded_release_control_flow_report_id": bounded.bounded_release_control_flow_report_id,
        "evidence_dir": str(evidence_dir),
        "full_pipeline_not_run": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--store-root", default=str(ROOT / "data/ifvg_datasets/search_test/v1"))
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--state-root", default=str(ROOT / "data/ifvg_pipeline_jobs"))
    parser.add_argument("--evidence-dir", default=str(ROOT / "R1-VERIFICATION-EVIDENCE"))
    parser.add_argument("--mbp1-diagnostic-id", default=None)
    args = parser.parse_args(argv)
    try:
        code, payload = _preflight(args) if args.command == "preflight" else _run(args)
    except PermissionError as error:
        _print({"status": "refused", "reason": "fail_before_path", "detail": str(error)})
        return 2
    _print(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
