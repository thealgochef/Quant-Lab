"""Build the immutable ``ifvg_replay_chart_v1`` companion artifact.

Usage::

    python scripts/ifvg_build_replay_chart.py build  --profile ifvg_v2_doc_default_fresh_static_1r
    python scripts/ifvg_build_replay_chart.py status --profile ifvg_v2_doc_default_fresh_static_1r

``build`` resolves the profile's context-ready v2/v3 pair from the pair
catalog (explicit ``--v2``/``--v3`` override when several pairs exist), loads
it fully verified, resamples the verified 1m label source to all replay
timeframes, corroborates against the per-day engine tbars oracle, and
publishes atomically.  Re-running against an existing artifact re-verifies it
and exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha_lab.agents.data_infra.ifvg.artifact_io import (  # noqa: E402
    load_verified_ifvg_pair,
)
from alpha_lab.agents.data_infra.ifvg.config import (  # noqa: E402
    V2_DATASET_DIR,
    V3_DATASET_DIR,
)
from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (  # noqa: E402
    REPLAY_CHART_CATALOG,
    REPLAY_CHART_STORE,
    ArtifactPairRef,
    build_replay_chart_artifact,
    find_replay_artifact,
    load_verified_replay_chart_artifact,
    read_replay_chart_catalog,
)

PAIR_CATALOG = ROOT / "data/ifvg_datasets/context_pair_catalog_v1.json"
DEFAULT_PROFILE = "ifvg_v2_doc_default_fresh_static_1r"


def _resolve_pair_ids(profile: str, v2: str | None, v3: str | None) -> tuple[str, str]:
    if v2 and v3:
        return v2, v3
    if v2 or v3:
        raise SystemExit("--v2 and --v3 must be given together or not at all")
    try:
        catalog = json.loads(PAIR_CATALOG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"pair catalog is unreadable: {error}") from error
    entry = catalog.get(profile)
    if not isinstance(entry, dict):
        raise SystemExit(f"profile is not in the pair catalog: {profile}")
    if entry.get("preparation_status") != "context_ready":
        raise SystemExit(
            f"profile pair is not context_ready: {entry.get('preparation_status')}"
        )
    return str(entry["v2_artifact_id"]), str(entry["v3_artifact_id"])


def _load_pair(profile: str, v2_id: str, v3_id: str):
    print(f"verifying pair v2={v2_id[:12]}… v3={v3_id[:12]}… (profile {profile})")
    pair = load_verified_ifvg_pair(
        v2_root=ROOT / V2_DATASET_DIR,
        v2_artifact_id=v2_id,
        v3_root=ROOT / V3_DATASET_DIR,
        v3_artifact_id=v3_id,
    )
    pair_ref = ArtifactPairRef.from_verified_pair(pair)
    if pair_ref.profile_name != profile:
        raise SystemExit(
            f"pair carries profile {pair_ref.profile_name!r}, not {profile!r}"
        )
    return pair, pair_ref


def cmd_build(args: argparse.Namespace) -> int:
    v2_id, v3_id = _resolve_pair_ids(args.profile, args.v2, args.v3)
    pair, pair_ref = _load_pair(args.profile, v2_id, v3_id)
    destination = build_replay_chart_artifact(
        pair,
        repo_root=ROOT,
        corroborate=not args.no_corroborate,
    )
    artifact_id = destination.name
    print(f"replay-chart artifact ready: {artifact_id}")
    report_path = destination / "corroboration_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        corroborated = len(report.get("corroborated_days", ()))
        uncorroborated = report.get("uncorroborated_days", ())
        print(f"corroborated days: {corroborated}; uncorroborated: {len(uncorroborated)}")
        if uncorroborated:
            print(f"  uncorroborated (no oracle file): {', '.join(uncorroborated)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    v2_id, v3_id = _resolve_pair_ids(args.profile, args.v2, args.v3)
    _, pair_ref = _load_pair(args.profile, v2_id, v3_id)
    catalog = read_replay_chart_catalog(ROOT / REPLAY_CHART_CATALOG)
    artifact_id = find_replay_artifact(catalog, pair_ref)
    if artifact_id is None:
        print("no replay-chart artifact for this exact pair; run `build`")
        return 1
    artifact = load_verified_replay_chart_artifact(
        ROOT / REPLAY_CHART_STORE, artifact_id, expected_pair=pair_ref
    )
    print(f"replay-chart artifact verified: {artifact.artifact_id}")
    print(f"  bars_tf rows: {len(artifact.bars_tf)}")
    print(f"  candidate ranges: {len(artifact.candidate_ranges)}")
    print(f"  anchor_240m_status: {artifact.anchor_240m_status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("build", cmd_build), ("status", cmd_status)):
        sub = subparsers.add_parser(name)
        sub.add_argument("--profile", default=DEFAULT_PROFILE)
        sub.add_argument("--v2", default=None, help="explicit v2 artifact ID override")
        sub.add_argument("--v3", default=None, help="explicit v3 artifact ID override")
        if name == "build":
            sub.add_argument(
                "--no-corroborate",
                action="store_true",
                help="skip the engine-tbars oracle comparison (records it as skipped)",
            )
        sub.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
