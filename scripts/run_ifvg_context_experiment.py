"""Run and catalog one deterministic IFVG context experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_lab.agents.data_infra.ifvg.artifact_io import (  # noqa: E402
    load_verified_ifvg_pair,
)
from alpha_lab.agents.data_infra.ifvg.config import (  # noqa: E402
    V2_DATASET_DIR,
    V3_DATASET_DIR,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (  # noqa: E402
    ArtifactPreparationStatus,
    ContextFeatureTier,
    IfvgContextExperimentConfig,
    IfvgContextExperimentDatasetConfig,
    IfvgContextLabelConfig,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_service import (  # noqa: E402
    CONTEXT_COHORT_REGISTRY,
    context_cohort_filters,
    run_and_catalog_context_experiment,
)
from alpha_lab.agents.data_infra.ifvg.context_feature_view import (  # noqa: E402
    build_candidate_feature_view,
)
from alpha_lab.agents.data_infra.ifvg.context_run_store import (  # noqa: E402
    CONTEXT_RUN_CATALOG,
    CONTEXT_RUN_STORE,
    CONTEXT_VIEW_STORE,
)
from alpha_lab.agents.data_infra.ifvg.preparation import (  # noqa: E402
    PAIR_CATALOG_PATH,
)


def _cataloged_pair(profile: str):
    path = ROOT / PAIR_CATALOG_PATH
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        entry = catalog[profile]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"no cataloged IFVG pair for profile {profile!r}") from error
    if entry.get("preparation_status") != ArtifactPreparationStatus.CONTEXT_READY.value:
        raise ValueError(f"IFVG pair for profile {profile!r} is not context_ready")
    pair = load_verified_ifvg_pair(
        v2_root=ROOT / V2_DATASET_DIR,
        v2_artifact_id=str(entry["v2_artifact_id"]),
        v3_root=ROOT / V3_DATASET_DIR,
        v3_artifact_id=str(entry["v3_artifact_id"]),
    )
    if (
        pair.v2.reference.manifest_payload_sha256
        != entry.get("v2_manifest_payload_sha256")
        or pair.v3.reference.manifest_payload_sha256
        != entry.get("v3_manifest_payload_sha256")
    ):
        raise ValueError("cataloged IFVG pair manifest identity changed")
    return pair


def _label_config(args: argparse.Namespace) -> IfvgContextLabelConfig:
    fixed = args.fixed_stop_ticks is not None or args.fixed_target_ticks is not None
    if fixed:
        if args.reward_r is not None:
            raise ValueError("choose either --reward-r or fixed stop/target ticks")
        if args.fixed_stop_ticks is None or args.fixed_target_ticks is None:
            raise ValueError(
                "fixed labels require both --fixed-stop-ticks and --fixed-target-ticks"
            )
        return IfvgContextLabelConfig(
            label_family="fixed_sl_tp",
            fixed_stop_ticks=args.fixed_stop_ticks,
            fixed_target_ticks=args.fixed_target_ticks,
        )
    return IfvgContextLabelConfig(reward_r=1.0 if args.reward_r is None else args.reward_r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="ifvg_v2_doc_default_fresh_static_1r",
    )
    parser.add_argument(
        "--tier",
        choices=[tier.value for tier in ContextFeatureTier],
        default=ContextFeatureTier.M2.value,
    )
    parser.add_argument(
        "--cohort",
        choices=tuple(CONTEXT_COHORT_REGISTRY),
        default="prior_research",
    )
    parser.add_argument("--reward-r", type=float, choices=(1.0, 1.5, 2.0))
    parser.add_argument("--fixed-stop-ticks", type=int)
    parser.add_argument("--fixed-target-ticks", type=int)
    parser.add_argument("--display-name")
    parser.add_argument("--notes")
    args = parser.parse_args()

    pair = _cataloged_pair(args.profile)
    view = build_candidate_feature_view(pair)
    config = IfvgContextExperimentConfig(
        dataset=IfvgContextExperimentDatasetConfig(
            artifact_pair=pair.reference,
            profile_name=args.profile,
        ),
        feature_tier=ContextFeatureTier(args.tier),
        label=_label_config(args),
        observation_filters=context_cohort_filters(view, args.cohort),
    )
    cataloged = run_and_catalog_context_experiment(
        pair,
        config,
        display_name=args.display_name,
        notes=args.notes,
        view_store=ROOT / CONTEXT_VIEW_STORE,
        run_store=ROOT / CONTEXT_RUN_STORE,
        catalog_path=ROOT / CONTEXT_RUN_CATALOG,
    )
    print(
        json.dumps(
            {
                "run_id": cataloged.stored_run.result.run_id,
                "run_manifest_payload_sha256": cataloged.run_manifest_sha256,
                "view_id": cataloged.stored_run.result.view_id,
                "view_manifest_payload_sha256": cataloged.view_manifest_sha256,
                "status": cataloged.stored_run.result.status,
                "reused_run": cataloged.reused_run,
                "reused_view": cataloged.reused_view,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
