"""Detached process controls for one frozen FSM configuration search.

The Streamlit workspace NEVER runs replays in-process (IMPLEMENTATION_PLAN §6):
``start`` launches this script's ``--worker`` mode detached and returns
immediately; the monitor polls the orchestrator's atomic
``search_state.json`` checkpoints; ``cancel`` writes the safe-boundary
sentinel the orchestrator honors at child boundaries only.

Execution scope is fail-closed and REGISTRY-GATED (the R2→R4 obligation):
the worker refuses to run children unless the replay wiring is named by a
REGISTERED runner entry — either ``--runner-entry-key <registered key>``
(the only form the UI passes) or a ``--runner-entry module:function`` whose
exact string is a registered value (`search/runner_registry.py`). No
user-shaped string reaches ``importlib``. The real full-scope executors
land with the R5 pipeline (`PHASED_DELIVERY.md`); nothing here can silently
start a full-development replay, and importing this module launches
nothing. ``resume`` re-enters the same idempotent worker: completed
children reuse their published identities, so a killed run continues from
its last atomic checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEARCH_JOB_ROOT = Path("data/ifvg_search_jobs")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ENTRY = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


def _validated_search_id(value: str) -> str:
    if not _HEX64.fullmatch(value or ""):
        raise SystemExit("search id must be a 64-hex search_id")
    return value


def _state_payload(state_root: Path, search_id: str) -> dict | None:
    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (  # noqa: PLC0415
        read_search_state,
    )

    return read_search_state(state_root, search_id)


def _resolve_runner_entry(entry: str):
    """Registered ``module:function`` → the callable providing the wiring.

    The callable must return a mapping with ``identity_resolver`` and
    ``child_runner`` (optionally ``prewarm`` / ``cost_points``). It is the
    ONLY way this job can execute replays; no built-in real-data wiring
    exists before the R5 pipeline executors. The entry must be an exact
    registered value (`runner_registry.py`) — raw strings are refused
    before any import occurs.
    """

    if not _ENTRY.fullmatch(entry or ""):
        raise SystemExit("--runner-entry must be 'module:function'")
    from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (  # noqa: PLC0415
        RunnerEntryError,
        assert_runner_entry_registered,
    )

    try:
        assert_runner_entry_registered(entry)
    except RunnerEntryError as error:
        raise SystemExit(str(error)) from None
    module_name, function_name = entry.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise SystemExit(f"runner entry {entry!r} is not callable")
    return factory


def _entry_from_args(args) -> str | None:
    """The effective registered entry string from key/raw arguments."""

    if getattr(args, "runner_entry_key", None):
        from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (  # noqa: PLC0415
            RunnerEntryError,
            resolve_registered_runner_entry,
        )

        try:
            return resolve_registered_runner_entry(args.runner_entry_key)
        except RunnerEntryError as error:
            raise SystemExit(str(error)) from None
    return args.runner_entry


def _worker(args) -> int:
    from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
        SearchCharterEnvelope,
    )
    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (  # noqa: PLC0415
        run_search,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        load_verified_envelope,
    )

    store_root = Path(args.store_root)
    state_root = Path(args.state_root)
    charter = load_verified_envelope(
        store_root, "charters", args.search_id, SearchCharterEnvelope
    )
    entry = _entry_from_args(args)
    if not entry:
        raise SystemExit(
            "no runner entry was provided: replay execution is refused "
            "(the R5 pipeline registers the real executors; synthetic runs "
            "pass --runner-entry-key or a registered --runner-entry)"
        )
    # DEV-R5-10 closure (safety F4): the worker's --store-root is passed to
    # the factory explicitly, matching the pipeline shim's contract — the
    # real search entry no longer defaults to the canonical namespace.
    wiring = _resolve_runner_entry(entry)(charter, store_root=store_root)
    result = run_search(
        charter,
        store_root=store_root,
        state_root=state_root,
        identity_resolver=wiring["identity_resolver"],
        child_runner=wiring["child_runner"],
        cost_points=wiring.get("cost_points"),
        prewarm=wiring.get("prewarm"),
    )
    print(json.dumps({"search_id": result.search_id, "phase": result.phase}))
    return 0 if result.phase == "search_complete" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("start", "status", "cancel", "resume", "worker")
    )
    parser.add_argument("--search-id", required=True)
    parser.add_argument("--store-root", default=str(ROOT / "data/ifvg_datasets/search_test/v1"))
    parser.add_argument("--state-root", default=str(ROOT / SEARCH_JOB_ROOT))
    parser.add_argument(
        "--runner-entry",
        default=None,
        help=(
            "registered module:function returning the replay wiring "
            "(refused unless it is an exact registered value)"
        ),
    )
    parser.add_argument(
        "--runner-entry-key",
        default=None,
        help="registered runner-entry key (the only form the UI passes)",
    )
    args = parser.parse_args(argv)
    search_id = _validated_search_id(args.search_id)
    state_root = Path(args.state_root)

    if args.command == "status":
        print(json.dumps(_state_payload(state_root, search_id), sort_keys=True, default=str))
        return 0

    if args.command == "cancel":
        from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (  # noqa: PLC0415
            request_safe_cancel,
        )

        sentinel = request_safe_cancel(state_root, search_id)
        print(json.dumps({"status": "cancellation_requested", "sentinel": str(sentinel)}))
        return 0

    if args.command == "worker":
        return _worker(args)

    # start / resume: detached worker launch; the monitor reads the state
    # files. Resume re-enters the same idempotent worker (completed children
    # reuse their published identities from the last atomic checkpoint).
    job_dir = state_root / search_id
    job_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--search-id",
        search_id,
        "--store-root",
        str(Path(args.store_root)),
        "--state-root",
        str(state_root),
    ]
    if args.runner_entry_key:
        command += ["--runner-entry-key", args.runner_entry_key]
    elif args.runner_entry:
        command += ["--runner-entry", args.runner_entry]
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
    print(json.dumps({"status": "started", "pid": process.pid, "search_id": search_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
