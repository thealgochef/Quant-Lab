"""Migrate deployed model-bundle strategy.json files to contract v3 (E3), in place.

Contract v3 is the envelope/section split: the flat v2 contract decomposes into
the platform-consumed ENVELOPE (flat keys) + ONE strategy-owned ``section``
subtree typed by the plugin's ``SectionModel``. For every bundle directory under
the models root that contains a strategy.json:

* SKIP bundles already at ``trade_lab_contract_v3`` (idempotent re-runs).
* SKIP bundles whose ``contract_version`` is not ``trade_lab_contract_v2``: ONLY
  v2 bundles migrate (the proven migration-#1 narrowing, D-E-c). The store's
  legacy/engine-v1/engine-v2 bundles never reached contract v2, so restructuring
  them would grant a shape they were never validated under; they stay at their
  old version and keep failing closed at the loader's first check — the same
  rejection class they have had since pre-E.
* Otherwise back up the original to ``strategy.json.pre_v3.bak`` (never
  overwritten once present — the backup is the pre-migration record) and
  restructure, preserving every kept field byte-for-byte:
    - ``contract_version``  -> the strategy_core ``CONTRACT_VERSION`` (v3)
    - the five section fields (``session_scheme``/``level_scheme``/
      ``touch_rule``/``feature_windows``/``research_session_experiment``) MOVE
      verbatim into the new ``section`` subtree (appended as the last key — the
      wire shape is flat envelope keys + one section subtree)
    - ``feature_set.interaction_features``/``feature_set.approach_features``
      MOVE out of the envelope's feature_set (now the SHELL) into the section
    - ``label_policy`` gains ``barrier_mode: "fixed_points"`` (inserted after
      ``resolution``) — every deployed touch bundle was labeled with the
      fixed-points barrier, the schema default
* Regenerate the ``model.cbm.sha256`` sidecar IF one exists (the sidecar hashes
  the model binary, which this migration never touches).

Run (from the QL repo root, PYTHONPATH=src or the editable install):
    python scripts/migrate_contracts_v3.py [models_root]
models_root defaults to $TRADE_LAB_MODELS_PATH, then ./models.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from strategy_core import CONTRACT_VERSION

MIGRATABLE_CONTRACT_VERSION = "trade_lab_contract_v2"

#: The five flat fields that move verbatim into the section subtree (E3
#: classification by consumer: these are plugin-consumed).
SECTION_FIELDS = (
    "session_scheme",
    "level_scheme",
    "touch_rule",
    "feature_windows",
    "research_session_experiment",
)
#: The feature-partition keys that move out of feature_set into the section.
PARTITION_FIELDS = ("interaction_features", "approach_features")


def _restructure(payload: dict) -> dict:
    """Flat v2 dict -> v3 envelope + section, preserving key order and bytes."""
    section: dict = {}

    migrated: dict = {}
    for key, value in payload.items():
        if key == "contract_version":
            migrated[key] = CONTRACT_VERSION
        elif key in SECTION_FIELDS:
            section[key] = value  # moved verbatim; envelope position dropped
        elif key == "feature_set":
            shell = {}
            for fs_key, fs_value in value.items():
                if fs_key in PARTITION_FIELDS:
                    section[fs_key] = fs_value  # partition moves to the section
                else:
                    shell[fs_key] = fs_value
            migrated[key] = shell
        elif key == "label_policy":
            label = {}
            for lp_key, lp_value in value.items():
                label[lp_key] = lp_value
                if lp_key == "resolution":
                    label["barrier_mode"] = "fixed_points"
            if "barrier_mode" not in label:  # defensive: resolution key absent
                label["barrier_mode"] = "fixed_points"
            migrated[key] = label
        else:
            migrated[key] = value

    migrated["section"] = section
    return migrated


def migrate_bundle(bundle: Path) -> str:
    strategy_file = bundle / "strategy.json"
    payload = json.loads(strategy_file.read_text(encoding="utf-8"))

    before = {
        "contract_version": payload.get("contract_version"),
        "engine_version": payload.get("engine_version"),
        "platform_version": payload.get("platform_version"),
        "strategy_id": payload.get("strategy_id"),
        "strategy_version": payload.get("strategy_version"),
        "has_section": "section" in payload,
    }

    if payload.get("contract_version") == CONTRACT_VERSION:
        return f"{bundle.name}: SKIP (already {CONTRACT_VERSION}) | before={before}"

    if payload.get("contract_version") != MIGRATABLE_CONTRACT_VERSION:
        return (
            f"{bundle.name}: SKIP (contract_version={before['contract_version']!r} is not "
            f"{MIGRATABLE_CONTRACT_VERSION!r} — not migratable; stays fail-closed at "
            f"contract_version) | before={before}"
        )

    backup = bundle / "strategy.json.pre_v3.bak"
    if not backup.exists():
        backup.write_text(strategy_file.read_text(encoding="utf-8"), encoding="utf-8")

    migrated = _restructure(payload)
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
        "barrier_mode": migrated.get("label_policy", {}).get("barrier_mode"),
        "section_keys": sorted(migrated.get("section", {})),
        "feature_set_keys": sorted(migrated.get("feature_set", {})),
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
