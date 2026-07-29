"""First honest IFVG gate: CatBoost over the entry dataset, walk-forward purged.

Small-N discipline (expected 150-450 rows): shallow depth, no tuning loops,
calibration-first reporting, and the two baselines the gate must beat OOS at
matched coverage — take-everything and doc-defaults-as-filter. The funnel +
label reports are the window's primary results; this gate is "first honest",
not final.

Eval: day-level walk-forward — train on days [0, k), purge 2 trading days,
test on the remainder; repeated over 3 expanding splits. Warmup rows and
``no_forward`` rows excluded. Target: ``label_r10 == "win"`` (binary), scored
against ``realized_r_net_r10`` expectancy at coverage sweeps.

Usage:
    PYTHONPATH=src python scripts/run_ifvg_gate_experiment.py [--dataset PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_lab.agents.data_infra.ifvg.config import (  # noqa: E402
    SEALED_HOLDOUT_START,
    IfvgCaptureConfig,
)
from alpha_lab.agents.data_infra.ifvg.funnel_report import doc_default_pass  # noqa: E402

_EXCLUDE_PREFIXES = ("label_", "realized_", "bars_to_res_", "mfe_r", "mae_r")
_IDENTITY = {
    "setup_id",
    "trading_day",
    "entry_ts_utc",
    "selected",
    "drop_reason",
    "is_warmup",
    "days_of_htf_history",
    "profile_hash",
    "strategy_version",
    "label_window_end",
    "entry_slippage_next_open_pts",
    "entry_ticks",
    "stop_ticks",
    "parent_fvg_id",  # string identifier (IFVG-FIX F1), not a measurement
}
_CATEGORICAL = ("entry_family", "direction", "session_engine", "session_doc",
                "tap_nearest_level_kind", "inv_sweep_swept_kinds")


def _feature_columns(ds: pd.DataFrame) -> list[str]:
    cols = []
    for col in ds.columns:
        if col in _IDENTITY or any(col.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        # Raw timestamps are a monotone train-earlier-than-test signal under
        # expanding splits — exclude every datetime64 column BY DTYPE so future
        # timestamp columns can never leak in by name (IFVG-FIX F3).
        if pd.api.types.is_datetime64_any_dtype(ds[col]):
            continue
        cols.append(col)
    return cols


def _expectancy(sub: pd.DataFrame) -> tuple[int, float, float]:
    if sub.empty:
        return 0, float("nan"), float("nan")
    return len(sub), float((sub["label_r10"] == "win").mean()), float(
        sub["realized_r_net_r10"].mean()
    )


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
    ds = pd.read_parquet(path)
    core = ds[
        (~ds["is_warmup"])
        & (ds["label_r10"] != "no_forward")
        & (ds["trading_day"] < SEALED_HOLDOUT_START)  # sealed holdout untouched
    ].copy()
    core = core.sort_values("entry_ts_utc").reset_index(drop=True)
    days = sorted(core["trading_day"].unique())
    print(f"dataset {path.name}: {len(ds)} rows -> {len(core)} eval rows over {len(days)} days")
    if len(core) < 40 or len(days) < 12:
        print("TOO FEW ROWS for even a first gate — funnel/label reports are the result.")
        return 0

    features = _feature_columns(core)
    cats = [c for c in _CATEGORICAL if c in features]
    for col in cats:
        core[col] = core[col].fillna("none").astype(str)
    numeric = [c for c in features if c not in cats]
    for col in numeric:
        core[col] = pd.to_numeric(core[col], errors="coerce")
    core["target"] = (core["label_r10"] == "win").astype(int)

    from catboost import CatBoostClassifier, Pool

    lines = ["# IFVG first honest gate (label_r10 win/other)", ""]
    lines.append(f"Rows {len(core)}, days {len(days)}, features {len(features)} ({len(cats)} cat).")
    lines += [
        "",
        "## Feature matrix (explicit, post dtype exclusion)",
        "",
        f"{len(features)} features — identity/label/outcome columns, datetime64 columns"
        " (excluded by dtype, IFVG-FIX F3) and the parent_fvg_id identifier are out:",
        "",
    ]
    lines += [f"- {c}{'  (cat)' if c in cats else ''}" for c in features]
    lines.append("")
    oos_frames = []
    splits = [0.5, 0.65, 0.8]
    for frac in splits:
        k = int(len(days) * frac)
        train_days = set(days[:k])
        test_days = set(days[k + 2 :])  # 2-day purge
        train = core[core["trading_day"].isin(train_days)]
        test = core[core["trading_day"].isin(test_days)]
        if len(train) < 30 or len(test) < 10:
            lines.append(f"- split {frac}: skipped (train {len(train)}, test {len(test)})")
            continue
        model = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.08,
            loss_function="Logloss",
            random_seed=7,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(Pool(train[features].fillna(-1), train["target"], cat_features=cats))
        proba = model.predict_proba(Pool(test[features].fillna(-1), cat_features=cats))[:, 1]
        fold = test.copy()
        fold["p_win"] = proba
        fold["split"] = frac
        oos_frames.append(fold)
        n, wr, net = _expectancy(fold)
        lines.append(
            f"- split {frac}: train {len(train)} / test {n} | "
            f"OOS win_rate {wr:.3f} net_R {net:+.3f}"
        )

    if not oos_frames:
        Path("IFVG_GATE_REPORT.md").write_text("\n".join(lines))
        print("no viable splits; wrote IFVG_GATE_REPORT.md")
        return 0
    oos = pd.concat(oos_frames, ignore_index=True)
    oos_path = Path(cfg.data_dir) / cfg.symbol / f"ifvg_oos_predictions_{cfg.capture_tag()}.parquet"
    oos.to_parquet(oos_path, index=False)

    lines += ["", "## Baselines vs gate (pooled OOS)", ""]
    n, wr, net = _expectancy(oos)
    lines.append(f"- take-everything: n={n} win_rate={wr:.3f} mean_net_R={net:+.3f}")
    dd = oos[doc_default_pass(oos)]
    n, wr, net = _expectancy(dd)
    lines.append(f"- doc-defaults filter: n={n} win_rate={wr:.3f} mean_net_R={net:+.3f}")
    lines += [
        "",
        "### Gate coverage sweep (P(win) threshold)",
        "",
        "| thr | n | coverage | win_rate | mean_net_R |",
        "|---|---|---|---|---|",
    ]
    for thr in (0.4, 0.5, 0.6, 0.7):
        sub = oos[oos["p_win"] >= thr]
        n, wr, net = _expectancy(sub)
        cov = n / max(1, len(oos))
        lines.append(
            f"| {thr} | {n} | {cov:.2f} | {wr:.3f} | {net:+.3f} |"
            if n
            else f"| {thr} | 0 | 0.00 | - | - |"
        )

    # Calibration (quartile bins) + Brier.
    oos["bin"] = pd.qcut(oos["p_win"], q=min(4, oos["p_win"].nunique()), duplicates="drop")
    lines += [
        "",
        "### Calibration (quartile bins of p_win)",
        "",
        "| bin | n | mean p | actual win rate |",
        "|---|---|---|---|",
    ]
    for interval, group in oos.groupby("bin", observed=True):
        lines.append(
            f"| {interval} | {len(group)} | {group['p_win'].mean():.3f} "
            f"| {group['target'].mean():.3f} |"
        )
    brier = float(((oos["p_win"] - oos["target"]) ** 2).mean())
    lines += ["", f"Brier score: {brier:.4f} (base-rate reference: {oos['target'].mean():.3f})", ""]

    lines += ["", "## Per-session OOS (mandatory breakout, engine scheme)", ""]
    for session, group in oos.groupby("session_engine"):
        n, wr, net = _expectancy(group)
        lines.append(f"- {session}: n={n} win_rate={wr:.3f} net_R={net:+.3f}")
    lines += ["", "## Per-family OOS", ""]
    for family, group in oos.groupby("entry_family"):
        n, wr, net = _expectancy(group)
        lines.append(f"- {family}: n={n} win_rate={wr:.3f} net_R={net:+.3f}")

    lines += [
        "",
        "_Small-N caveat: shallow fixed-hyperparameter model, no tuning, no feature"
        " selection; calibration and expectancy-at-coverage are the readouts that"
        " matter. The funnel and label reports are the window's primary results._",
    ]
    Path("IFVG_GATE_REPORT.md").write_text("\n".join(lines))
    print("wrote IFVG_GATE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
