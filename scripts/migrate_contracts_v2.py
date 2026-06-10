"""Migrate deployed model-bundle strategy.json files to contract v2 (E1/E2), in place.

For every bundle directory under the models root that contains a strategy.json:

* SKIP bundles already at ``trade_lab_contract_v2`` (idempotent re-runs).
* SKIP bundles whose ``engine_version`` is not ``strategy_core_engine_v3``:
  the platform axis (``strategy_core_platform_v1``) RENAMES engine v3 — granting
  it to an engine-v1/v2/absent bundle would FORGE the structural binding (the
  v2-engine bundle is SC-schema-valid and would activate under semantics it was
  not built with). Non-v3 bundles stay at contract v1 and keep failing closed at
  the loader's first check, the same rejection class they had pre-E.
* Otherwise back up the original to ``strategy.json.pre_v2.bak`` (never
  overwritten once present — the backup is the pre-migration record) and
  rewrite, preserving every other field byte-for-byte:
    - ``contract_version`` -> the strategy_core ``CONTRACT_VERSION`` (v2)
    - ``engine_version``   -> renamed key ``platform_version`` with the
      strategy_core ``PLATFORM_VERSION`` value (the E1 axis rename)
    - ``strategy_id``      -> ``"touch_reversal"`` (the REGISTRY ROUTER KEY; the
      bundle's identity remains its directory name)
    - + ``strategy_version`` resolved from the registered plugin
    - ``supported_by_runtime`` left AS-IS (the migration does not adjudicate it)
* Regenerate the ``model.cbm.sha256`` sidecar IF one exists (the sidecar hashes
  the model binary, which this migration never touches — regeneration is a
  consistency formality, not a correctness need).

Run (from the QL repo root, PYTHONPATH=src or the editable install):
    python scripts/migrate_contracts_v2.py [models_root]
models_root defaults to $TRADE_LAB_MODELS_PATH, then ./models.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Explicit registration import so get_strategy can resolve the router id.
import strategy_core.strategies.touch_reversal  # noqa: F401
from strategy_core import CONTRACT_VERSION, PLATFORM_VERSION
from strategy_core.strategies.registry import get_strategy

ROUTER_STRATEGY_ID = "touch_reversal"

#: The engine-axis value the platform axis renames. ONLY bundles built under this
#: engine legitimately carry PLATFORM_VERSION ("strategy_core_platform_v1").
MIGRATABLE_ENGINE_VERSION = "strategy_core_engine_v3"


def migrate_bundle(bundle: Path) -> str:
    strategy_file = bundle / "strategy.json"
    payload = json.loads(strategy_file.read_text(encoding="utf-8"))

    before = {
        "contract_version": payload.get("contract_version"),
        "engine_version": payload.get("engine_version"),
        "platform_version": payload.get("platform_version"),
        "strategy_id": payload.get("strategy_id"),
        "strategy_version": payload.get("strategy_version"),
        "supported_by_runtime": payload.get("supported_by_runtime"),
    }

    if payload.get("contract_version") == CONTRACT_VERSION:
        return f"{bundle.name}: SKIP (already {CONTRACT_VERSION}) | before={before}"

    if payload.get("engine_version") != MIGRATABLE_ENGINE_VERSION:
        return (
            f"{bundle.name}: SKIP (engine_version={before['engine_version']!r} is not "
            f"{MIGRATABLE_ENGINE_VERSION!r} — not migratable; stays fail-closed at "
            f"contract_version) | before={before}"
        )

    backup = bundle / "strategy.json.pre_v2.bak"
    if not backup.exists():
        backup.write_text(strategy_file.read_text(encoding="utf-8"), encoding="utf-8")

    # Rebuild the dict preserving key order, swapping engine_version in place for
    # platform_version and inserting strategy_version right after strategy_id.
    migrated: dict = {}
    for key, value in payload.items():
        if key == "contract_version":
            migrated[key] = CONTRACT_VERSION
        elif key == "engine_version":
            migrated["platform_version"] = PLATFORM_VERSION
        elif key == "strategy_id":
            migrated[key] = ROUTER_STRATEGY_ID
            migrated["strategy_version"] = get_strategy(ROUTER_STRATEGY_ID).strategy_version
        else:
            migrated[key] = value

    strategy_file.write_text(json.dumps(migrated, indent=2, default=str), encoding="utf-8")

    sidecar = bundle / "model.cbm.sha256"
    sidecar_note = "no sidecar"
    if sidecar.exists():
        digest = hashlib.sha256((bundle / "model.cbm").read_bytes()).hexdigest()
        sidecar.write_text(digest, encoding="utf-8")
        sidecar_note = "sidecar regenerated"

    after = {
        "contract_version": migrated.get("contract_version"),
        "platform_version": migrated.get("platform_version"),
        "strategy_id": migrated.get("strategy_id"),
        "strategy_version": migrated.get("strategy_version"),
        "supported_by_runtime": migrated.get("supported_by_runtime"),
    }
    return (
        f"{bundle.name}: MIGRATED ({sidecar_note}; backup={backup.name})\n"
        f"  before={before}\n  after ={after}"
    )


def main() -> int:
    root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("TRADE_LAB_MODELS_PATH", "models")
    )
    if not root.is_dir():
        print(f"models root not found: {root}")
        return 1
    bundles = sorted(p for p in root.iterdir() if p.is_dir() and (p / "strategy.json").exists())
    print(f"models root: {root.resolve()} | bundles with strategy.json: {len(bundles)}")
    for bundle in bundles:
        print(migrate_bundle(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
