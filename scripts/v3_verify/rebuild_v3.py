"""ENGINE v3: regenerate the labeled dataset under the new hash (d8e239c7).

Calls the PRODUCTION ``build_utility_dataset`` over the model's exact 225 dates with
the model's exact config (THIS_CONFIG, the hash-3d2f8466 v2 config), now under engine
v3. This is the literal dataset regeneration: it writes ``ml_utility_<newhash>.parquet``
per date into the store and returns the concatenated labeled feature matrix. NO model
is trained (we stop at the dataset). Reports the new hash, the labeled-row count, the
class balance, and the KEPT (labeled) touch counts by session x level_type.

PYTHONPATH=src.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import strategy_core as sc
from strategy_core import classify_session

from alpha_lab.agents.data_infra.ml.config import (
    DashboardUtilityConfig, MLPipelineConfig, ModelConfig, WalkForwardConfig,
)
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import build_utility_dataset

DATA_DIR = Path("C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento")
OUT = Path(__file__).parent / "dataset_v3.parquet"

# THIS model's exact config (the one that hashed to 3d2f8466 under v2). UNCHANGED —
# only the engine moved v2 -> v3, so the hash changes via the engine_version key.
CFG = MLPipelineConfig(
    training_mode="dashboard_utility",
    walk_forward=WalkForwardConfig(train_days=30, test_days=7, gap_days=1, expanding=False),
    model=ModelConfig(iterations=500, depth=4, learning_rate=0.03, loss_function="MultiClass",
                      auto_class_weights="Balanced", rfecv_enabled=False, rfecv_min_features=5),
    dashboard_utility=DashboardUtilityConfig(
        tp_points=15.0, sl_points=15.0, trap_mfe_min=5.0, interaction_window_minutes=5,
        level_proximity_pts=0.5, bar_type="147t", include_approach_features=True,
        approach_window_minutes=15),
    tick_size=0.25, instrument="NQ")


def main() -> int:
    ev = json.load(open("models/NQ_20260602_232808/evaluation.json"))
    dates = sorted(ev["dates_used"])
    h = CFG.dataset_config_hash()
    print(f"engine={sc.ENGINE_VERSION}  new dataset_config_hash={h}  "
          f"(v2 was 3d2f8466; differs={h != '3d2f8466'})")
    print(f"regenerating labeled dataset over {len(dates)} dates "
          f"[{dates[0]} .. {dates[-1]}] ...", flush=True)

    df = build_utility_dataset(dates, DATA_DIR, CFG)
    print(f"\nlabeled rows (v3) = {len(df)}  (v2 had 384)")
    if df.empty:
        print("EMPTY dataset"); return 1

    df = df[df["date"].isin(set(dates))].copy()
    print("class balance:", df["label"].value_counts().to_dict())

    df["event_ts"] = pd.to_datetime(df["event_ts"])
    sess = [classify_session(pd.Timestamp(t).tz_convert("UTC").to_pydatetime()).session
            for t in df["event_ts"]]
    df["_session"] = sess
    print("\nKEPT (labeled) touches by SESSION x LEVEL_TYPE:")
    print(pd.crosstab(df["_session"], df["level_type"], margins=True).to_string())

    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  (per-date caches: ml_utility_{h}.parquet)")
    print("REBUILD DONE — model NOT trained (stopped at dataset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
