"""Prepare the reviewable VERIFIED PAIR for the timeframe variant
(HTF 1H only; parents 5m + 15m; continuation-only executions — the base
profile already enforces the last one).

Produces an immutable v2 dataset + v3 context artifact + replay-chart
artifact, and catalogs the pair under its own stamped profile name
``ifvg_v2_tf1h_5m15m_fresh_static_1r`` so it appears in the Replay/Verifier
"Verified artifact pair" dropdown for candidate/decision/trade review.
Day artifacts for this timeframe set are already cached by the experiment
run. Accepted baseline artifacts are untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategy_core.strategies.ifvg_smc.context_config import (  # noqa: E402
    ContextFeatureConfig,
)

from alpha_lab.agents.data_infra.ifvg.preparation import (  # noqa: E402
    prepare_ifvg_development_pair_persisted,
)
from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (  # noqa: E402
    build_replay_chart_artifact,
)

VARIANT_PROFILE_NAME = "ifvg_v2_tf1h_5m15m_fresh_static_1r"
SECTION_OVERRIDES = {
    "profile_name": VARIANT_PROFILE_NAME,
    "htf_timeframes": ["1H"],
    "parent_timeframes": ["5m", "15m"],
}
# the context observer must declare a subset of the section's timeframes.
VARIANT_CONTEXT = ContextFeatureConfig(
    normalized_timeframes=("1m", "5m", "15m", "60m"),
    mtf_timeframes=("5m", "15m", "60m"),
)


def main() -> int:
    def progress(completed: int, total: int, day: str) -> None:
        print(
            json.dumps({"completed": completed, "total": total, "source_date": day}),
            flush=True,
        )

    prepared = prepare_ifvg_development_pair_persisted(
        repo_root=ROOT,
        profile_name="ifvg_v2_doc_default_fresh_static_1r",
        cached_artifacts_only=True,
        section_overrides=SECTION_OVERRIDES,
        context_config=VARIANT_CONTEXT,
        job_label=VARIANT_PROFILE_NAME,
        progress_fn=progress,
    )
    chart_dir = build_replay_chart_artifact(prepared.pair, repo_root=ROOT)
    chart_manifest = json.loads(
        (chart_dir / "manifest.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "status": prepared.preparation_state.status.value,
                "variant_profile": VARIANT_PROFILE_NAME,
                "v2_artifact_id": prepared.pair.reference.v2.artifact_id,
                "v3_artifact_id": prepared.pair.reference.v3.artifact_id,
                "replay_chart_artifact_id": chart_manifest["replay_chart_artifact_id"],
                "protected_counters": prepared.access_audit.get("protected_counters"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
