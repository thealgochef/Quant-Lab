"""One-time explicit store-namespace migration (HARDENING-BACKEND §4.1; F-11).

Research-versus-test authority is SEMANTIC: a store carries an immutable
``STORE_NAMESPACE.json`` envelope (``namespace_class``, a stable
``store_instance_id``, the supersession-chain genesis anchor) that every
owner decision, supersession record, authorization bundle, seed-production
and verification authorization references by content id. An unmarked
existing store cannot authorize a write or a real launch.

``init`` shows the operator the intended class and target root, then
initializes the namespace ONCE (idempotent on an identical replay; a
different class or instance for an already-marked store is refused). The
class is an explicit argument — it is never inferred from the path. The
FIRST initialization of an unmarked store requires an explicit
``--store-instance-id`` (32 lowercase hex characters) that the operator
chooses and RECORDS before running (HARDENING-BACKEND-FIX.1 §2): the
initializer never generates one, so an initialization interrupted between
its two publications stays recoverable by replaying the identical request;
without it the store is refused untouched (``store_instance_id_required``).
A ``test`` namespace under a research-looking path is refused as an
incoherent deployment (defense in depth). ``show`` prints the verified
namespace and the current supersession-head witness. Importing this module
launches nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _show(root: Path) -> dict:
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (  # noqa: PLC0415
        load_store_namespace,
    )
    from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (  # noqa: PLC0415
        current_supersession_head_witness,
    )

    namespace = load_store_namespace(root)
    witness = current_supersession_head_witness(root)
    return {
        "store_root": str(root),
        "store_namespace_id": namespace.store_namespace_id,
        "namespace_class": namespace.payload.namespace_class,
        "store_instance_id": namespace.payload.store_instance_id,
        "authority_genesis_id": namespace.payload.authority_genesis_id,
        "supersession_head_witness": witness.model_dump(mode="json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "show"))
    parser.add_argument("--store-root", required=True)
    parser.add_argument(
        "--namespace-class",
        choices=("research", "test"),
        default=None,
        help="the SEMANTIC class the operator states for this store (init only)",
    )
    parser.add_argument(
        "--store-instance-id",
        default=None,
        help=(
            "explicit 32-lowercase-hex instance id; REQUIRED for the first initialization "
            "of an unmarked store (choose it, record it, then pass it — it is the only key "
            "that recovers an interrupted initialization; never generated for you); "
            "optional for an idempotent replay of an already-marked store"
        ),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="init: acknowledge the displayed class/root and perform the migration",
    )
    args = parser.parse_args(argv)
    root = Path(args.store_root)

    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (  # noqa: PLC0415
        StoreNamespaceError,
        assert_namespace_deployment_coherent,
        initialize_store_namespace,
        namespace_class_of,
        path_looks_like_research_store,
    )

    if args.command == "show":
        try:
            print(json.dumps(_show(root), sort_keys=True, indent=2))
        except StoreNamespaceError as error:
            print(json.dumps({"status": "refused", "reason": error.reason, "detail": str(error)}))
            return 2
        return 0

    if args.namespace_class is None:
        print(json.dumps({"status": "refused", "reason": "namespace_class_required"}))
        return 2
    if args.store_instance_id is not None and (
        re.fullmatch(r"[0-9a-f]{32}", args.store_instance_id) is None
    ):
        # review FIX.1-R2: a malformed explicit id is a typed CLI refusal before
        # anything is read or published, never a traceback
        print(
            json.dumps(
                {
                    "status": "refused",
                    "reason": "store_instance_id_malformed",
                    "detail": "--store-instance-id must be exactly 32 lowercase hexadecimal "
                    "characters",
                }
            )
        )
        return 2
    try:
        existing = namespace_class_of(root)
    except StoreNamespaceError as error:
        print(json.dumps({"status": "refused", "reason": error.reason, "detail": str(error)}))
        return 2
    instance_id_required = existing is None and args.store_instance_id is None
    intent = {
        "status": "intent",
        "store_root": str(root.resolve()),
        "intended_namespace_class": args.namespace_class,
        "store_instance_id": args.store_instance_id,
        "store_instance_id_required": instance_id_required,
        "path_looks_like_research_store": path_looks_like_research_store(root),
        "currently_marked_as": existing,
        "note": (
            "the class is an explicit operator statement, never inferred from the path; "
            "re-run with --confirm to initialize (idempotent on an identical replay)"
            + (
                "; this store is unmarked: its FIRST initialization requires an explicit "
                "--store-instance-id (32 lowercase hex characters) that you choose and record "
                "before running — the initializer never generates one, so an interrupted "
                "initialization stays recoverable by the identical request"
                if instance_id_required
                else ""
            )
        ),
    }
    if not args.confirm:
        print(json.dumps(intent, sort_keys=True, indent=2))
        return 0
    if args.namespace_class == "test" and path_looks_like_research_store(root):
        print(
            json.dumps(
                {
                    "status": "refused",
                    "reason": "store_namespace_deployment_incoherent",
                    "detail": "a test namespace under a research-looking path is refused",
                }
            )
        )
        return 2
    try:
        namespace = initialize_store_namespace(
            root,
            namespace_class=args.namespace_class,
            store_instance_id=args.store_instance_id,
        )
        assert_namespace_deployment_coherent(root, namespace)
    except StoreNamespaceError as error:
        print(json.dumps({"status": "refused", "reason": error.reason, "detail": str(error)}))
        return 2
    except ValueError as error:  # an argument the initializer refuses (backstop; typed)
        print(json.dumps({"status": "refused", "reason": "invalid_argument", "detail": str(error)}))
        return 2
    print(json.dumps({"status": "initialized", **_show(root)}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
