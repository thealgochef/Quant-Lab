"""Detached CLI shim for the 16-stage pipeline runner (R5; FUX §30).

Same contracts as ``ifvg_search_job.py``: the Streamlit UI NEVER runs a
pipeline in-process — ``start`` launches the ``worker`` subcommand detached
and returns immediately; the monitor polls the atomic
``pipeline_state.json``; ``cancel`` writes the safe-boundary sentinel;
``publish-gates`` / ``activate`` are the explicit verify-then-activate
publication actions (activation refuses verification scope). Importing this
module launches nothing.

The worker executes ONLY registry-resolved runner entries
(``--runner-entry-key`` is the only form the UI passes; raw
``module:function`` strings are refused before any import).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PIPELINE_JOB_ROOT = Path("data/ifvg_pipeline_jobs")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ENTRY = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


def _validated_pipeline_id(value: str) -> str:
    if not _HEX64.match(value or ""):
        raise SystemExit("pipeline id must be a 64-hex pipeline_semantic_id")
    return value


def _state_payload(state_root: Path, pipeline_id: str) -> dict | None:
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        read_pipeline_state,
    )

    return read_pipeline_state(state_root, pipeline_id)


def _resolve_runner_entry(entry: str):
    from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (  # noqa: PLC0415
        assert_runner_entry_registered,
    )

    if not _ENTRY.match(entry or ""):
        raise SystemExit("runner entry must be module:function")
    assert_runner_entry_registered(entry)
    module_name, function_name = entry.split(":", 1)
    module = import_module(module_name)
    factory = getattr(module, function_name, None)
    if factory is None:
        raise SystemExit(f"runner entry {entry!r} does not resolve to a callable")
    return factory


def _entry_from_args(args) -> str | None:
    if args.runner_entry_key:
        from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (  # noqa: PLC0415
            resolve_registered_runner_entry,
        )

        return resolve_registered_runner_entry(args.runner_entry_key)
    return args.runner_entry


def _worker(args) -> int:
    from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
        SearchCharterEnvelope,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        PipelineSemanticIdentity,
        StageStatus,
        WorkerPolicy,
        run_pipeline,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        load_verified_envelope,
    )

    pipeline_id = _validated_pipeline_id(args.pipeline_id)
    store_root = Path(args.store_root)
    state_root = Path(args.state_root)
    entry = _entry_from_args(args)
    if not entry:
        raise SystemExit(
            "a registered --runner-entry-key (or registered --runner-entry) "
            "is required"
        )
    factory = _resolve_runner_entry(entry)
    semantic = load_verified_envelope(
        store_root, "pipeline_specs", pipeline_id, PipelineSemanticIdentity
    )
    charter = load_verified_envelope(
        store_root,
        "charters",
        semantic.payload.search_charter_id,
        SearchCharterEnvelope,
    )
    wiring = factory(charter, semantic, store_root=store_root)
    result = run_pipeline(
        semantic,
        charter,
        store_root=store_root,
        state_root=state_root,
        wiring=wiring,
        worker_policy=WorkerPolicy(
            max_workers=args.max_workers,
            max_tasks_per_child=1,
            memory_budget_bytes=2 << 30,
        ),
        operational_retry_reason=args.retry_reason,
    )
    print(
        json.dumps(
            {
                "pipeline_semantic_id": result.pipeline_semantic_id,
                "stage_statuses": result.stage_statuses,
            },
            sort_keys=True,
        )
    )
    terminal = {
        StageStatus.COMPLETED.value,
        StageStatus.REUSED.value,
        StageStatus.BLOCKED.value,
    }
    return 0 if set(result.stage_statuses.values()) <= terminal else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "start",
            "status",
            "cancel",
            "resume",
            "worker",
            "publish-gates",
            "activate",
        ),
    )
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument(
        "--store-root", default=str(ROOT / "data/ifvg_datasets/search_test/v1")
    )
    parser.add_argument("--state-root", default=str(ROOT / PIPELINE_JOB_ROOT))
    parser.add_argument(
        "--runner-entry",
        default=None,
        help="registered module:function (raw unregistered strings are refused)",
    )
    parser.add_argument(
        "--runner-entry-key",
        default=None,
        help="registered runner-entry key (the only form the UI passes)",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--retry-reason", default=None)
    args = parser.parse_args(argv)

    pipeline_id = _validated_pipeline_id(args.pipeline_id)
    state_root = Path(args.state_root)

    if args.command == "status":
        print(
            json.dumps(
                _state_payload(state_root, pipeline_id), sort_keys=True, default=str
            )
        )
        return 0
    if args.command == "cancel":
        from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
            request_pipeline_cancel,
        )

        sentinel = request_pipeline_cancel(state_root, pipeline_id)
        print(
            json.dumps(
                {"status": "cancellation_requested", "sentinel": str(sentinel)},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "publish-gates":
        from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
            run_publication_gates,
        )

        gates = run_publication_gates(
            state_root, pipeline_id, store_root=Path(args.store_root)
        )
        print(json.dumps({"gates": gates}, sort_keys=True))
        return 0 if all(gates.values()) else 1
    if args.command == "activate":
        from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
            PublicationError,
            activate_pipeline_result,
        )

        try:
            result_id = activate_pipeline_result(
                state_root, pipeline_id, store_root=Path(args.store_root)
            )
        except PublicationError as error:
            print(json.dumps({"status": "refused", "reason": str(error)}))
            return 1
        print(json.dumps({"status": "activated", "pipeline_result_id": result_id}))
        return 0
    if args.command == "worker":
        return _worker(args)

    # start / resume share one detached-launch path; idempotency lives in the
    # runner (verified stage reuse), never in the shim.
    job_dir = state_root / pipeline_id
    job_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--pipeline-id",
        pipeline_id,
        "--store-root",
        str(Path(args.store_root)),
        "--state-root",
        str(state_root),
        "--max-workers",
        str(args.max_workers),
    ]
    if args.runner_entry_key:
        command += ["--runner-entry-key", args.runner_entry_key]
    elif args.runner_entry:
        command += ["--runner-entry", args.runner_entry]
    if args.retry_reason:
        command += ["--retry-reason", args.retry_reason]
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0
    )
    with (job_dir / "job.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(  # noqa: S603 - exact local interpreter/script only
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
    print(
        json.dumps(
            {"status": "started", "pid": process.pid, "pipeline_id": pipeline_id}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
