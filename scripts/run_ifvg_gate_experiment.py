"""First honest IFVG gate — thin CLI over the experiment engine.

The logic lives in ``alpha_lab.agents.data_infra.ifvg.experiment``; this script
runs the DEFAULT config with the model block on and renders the same
``IFVG_GATE_REPORT.md`` sections (parity check for the refactor: the numbers
must reproduce the pre-refactor report exactly) + writes the pooled-OOS
parquet as before.

Usage:
    PYTHONPATH=src python scripts/run_ifvg_gate_experiment.py [--dataset PATH]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.experiment import (  # noqa: E402
    IfvgExperimentConfig,
    IfvgModelConfig,
    run_ifvg_experiment,
    save_experiment,
)


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _render(result: dict, dataset_name: str, capture_tag: str) -> str:
    model = result["model"] or {}
    features = result["features"]
    lines = ["# IFVG first honest gate (label_r10 win/other)", ""]
    lines.append(
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"dataset `{dataset_name}` | capture_tag `{capture_tag}`"
    )
    lines.append("")
    n_rows, n_days = model.get("n_rows", 0), model.get("n_days", 0)
    n_feat, n_cat = features["n"], len(features["categorical"])
    lines.append(f"Rows {n_rows}, days {n_days}, features {n_feat} ({n_cat} cat).")
    lines += [
        "",
        "## Feature matrix (explicit, post dtype exclusion)",
        "",
        f"{n_feat} features — identity/label/outcome columns, datetime64 columns"
        " (excluded by dtype, IFVG-FIX F3) and the parent_fvg_id identifier are out:",
        "",
    ]
    cats = set(features["categorical"])
    lines += [f"- {c}{'  (cat)' if c in cats else ''}" for c in features["all"]]
    lines.append("")
    if model.get("skipped_reason"):
        lines.append(f"Model skipped: {model['skipped_reason']}")
        return "\n".join(lines)
    for split in model["splits"]:
        if split.get("skipped"):
            lines.append(
                f"- split {split['frac']}: skipped (train {split['train_n']}, "
                f"test {split['test_n']})"
            )
        else:
            lines.append(
                f"- split {split['frac']}: train {split['train_n']} / test {split['test_n']} | "
                f"OOS win_rate {split['win_rate']:.3f} net_R {split['mean_net_r']:+.3f}"
            )
    if "pooled" not in model:
        return "\n".join(lines)

    lines += ["", "## Baselines vs gate (pooled OOS)", ""]
    p = model["pooled"]
    lines.append(
        f"- take-everything: n={p['n']} win_rate={p['win_rate']:.3f} "
        f"mean_net_R={p['mean_net_r']:+.3f}"
    )
    d = model["doc_defaults_baseline"]
    lines.append(
        f"- doc-defaults filter: n={d['n']} win_rate={_fmt(d['win_rate'], '.3f')} "
        f"mean_net_R={_fmt(d['mean_net_r'], '+.3f')}"
    )
    lines += [
        "",
        "### Gate coverage sweep (P(win) threshold)",
        "",
        "| thr | n | coverage | win_rate | mean_net_R |",
        "|---|---|---|---|---|",
    ]
    for row in model["coverage"]:
        if row["n"]:
            lines.append(
                f"| {row['thr']} | {row['n']} | {row['coverage']:.2f} "
                f"| {row['win_rate']:.3f} | {row['mean_net_r']:+.3f} |"
            )
        else:
            lines.append(f"| {row['thr']} | 0 | 0.00 | - | - |")
    lines += [
        "",
        "### Calibration (quartile bins of p_win)",
        "",
        "| bin | n | mean p | actual win rate |",
        "|---|---|---|---|",
    ]
    for row in model["calibration"]:
        lines.append(
            f"| {row['bin']} | {row['n']} | {row['mean_p']:.3f} | {row['actual']:.3f} |"
        )
    lines += [
        "",
        f"Brier score: {model['brier']:.4f} (base-rate reference: {model['base_rate']:.3f})",
        "",
    ]
    lines += ["", "## Per-session OOS (mandatory breakout, engine scheme)", ""]
    for row in model["per_session"]:
        lines.append(
            f"- {row['session']}: n={row['n']} win_rate={row['win_rate']:.3f} "
            f"net_R={row['mean_net_r']:+.3f}"
        )
    lines += ["", "## Per-family OOS", ""]
    for row in model["per_family"]:
        lines.append(
            f"- {row['family']}: n={row['n']} win_rate={row['win_rate']:.3f} "
            f"net_R={row['mean_net_r']:+.3f}"
        )

    mean_ps = [row["mean_p"] for row in model["calibration"]]
    actuals = [row["actual"] for row in model["calibration"]]
    lines += [
        "",
        "## Verdict (this run)",
        "",
        f"- Brier {model['brier']:.4f} vs constant-base-rate reference "
        f"{model['ref_brier']:.4f} (base rate {model['base_rate']:.3f}).",
        f"- Calibration: quartile mean p_win spans {min(mean_ps):.3f} -> "
        f"{max(mean_ps):.3f}; actual win rate spans "
        f"{min(actuals):.3f} -> {max(actuals):.3f}.",
        f"- Pooled OOS (overlapping splits, n={p['n']}): win_rate {p['win_rate']:.3f}, "
        f"mean_net_R {p['mean_net_r']:+.3f}.",
        "",
        "_Measurement facts only — works/doesn't-work judgements are reserved for the"
        " owner after the build is complete and the sealed one-shot validation runs._",
        "",
        "_Small-N caveat: shallow fixed-hyperparameter model, no tuning, no feature"
        " selection; calibration and expectancy-at-coverage are the readouts that"
        " matter. The funnel and label reports are the window's primary results._",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    cfg = IfvgCaptureConfig()
    path = (
        Path(args.dataset)
        if args.dataset
        else Path(cfg.data_dir) / cfg.symbol / f"ifvg_entry_dataset_{cfg.capture_tag()}.parquet"
    )
    config = IfvgExperimentConfig(model=IfvgModelConfig())
    result = run_ifvg_experiment(config, dataset_path=path, capture_cfg=cfg)

    model = result["model"] or {}
    print(
        f"dataset {path.name}: {model.get('n_rows', 0)} eval rows over "
        f"{model.get('n_days', 0)} days | experiment {result['meta']['experiment_hash']}"
    )
    oos = result.get("_oos_frame")
    if oos is not None:
        oos_path = (
            Path(cfg.data_dir) / cfg.symbol / f"ifvg_oos_predictions_{cfg.capture_tag()}.parquet"
        )
        oos.drop(columns=["bin"], errors="ignore").to_parquet(oos_path, index=False)
    Path("IFVG_GATE_REPORT.md").write_text(
        _render(result, path.name, cfg.capture_tag()), encoding="utf-8"
    )
    print("wrote IFVG_GATE_REPORT.md")
    # Baseline seeding (plan A4): the default-config run persists as a saved
    # history entry (idempotent — same config hash redeploys the same dir).
    run_dir = save_experiment(
        config, result, name="baseline", note="default-config gate run (CLI)"
    )
    print(f"saved baseline experiment -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
