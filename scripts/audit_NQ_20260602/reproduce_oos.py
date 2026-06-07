# ruff: noqa: E501,SIM115
"""AUDIT: reproduce the 33-fold walk-forward OOS for NQ_20260602_232808 and compute the
honest edge (B2/F).  READ-ONLY: trains throwaway fold models in memory, touches nothing.

STEP A  reproduce production OOS on the labeled 384 -> cross-check confusion == 172/57/42/63
        (anchors trust in every dashboard number).  Also recompute threshold_table (E2),
        feature_stability (E3), Brier baseline (E4), fold precision dist (D4).
STEP B  score the FULL entered population (kept + no_resolution, from enrich_dates) by
        assigning each touch to the fold whose [test_start,test_end] window contains its
        event_ts and using THAT fold's model (trained on labeled-only train rows) -> per-touch
        prob_reversal.  Gate 0.70 & ny_rth.  Honest per-trade-taken PnL net of costs (F).

PYTHONPATH=src.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alpha_lab.agents.data_infra.ml.config import (
    DashboardUtilityConfig,
    MLPipelineConfig,
    ModelConfig,
    WalkForwardConfig,
)
from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer
from alpha_lab.agents.data_infra.ml.walk_forward import WalkForwardSplitter

STORE = "C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento/NQ"
ENRICHED = Path(__file__).parent / "enriched"
FEATS = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
    "app_avg_trade_size",
    "app_large_trade_vol_pct",
    "app_max_spread",
]
POINT_VALUE = 20.0
COMMISSION_RT = (2.14 + 0.50) * 2.0  # $5.28
COMMISSION_PTS = COMMISSION_RT / POINT_VALUE

CFG = MLPipelineConfig(
    training_mode="dashboard_utility",
    walk_forward=WalkForwardConfig(train_days=30, test_days=7, gap_days=1, expanding=False),
    model=ModelConfig(
        iterations=500,
        depth=4,
        learning_rate=0.03,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        rfecv_enabled=False,
        rfecv_min_features=5,
    ),
    dashboard_utility=DashboardUtilityConfig(
        tp_points=15.0,
        sl_points=15.0,
        trap_mfe_min=5.0,
        interaction_window_minutes=5,
        level_proximity_pts=0.5,
        bar_type="147t",
        include_approach_features=True,
        approach_window_minutes=15,
    ),
    tick_size=0.25,
    instrument="NQ",
)


def load_labeled_384():
    ev = json.load(open("models/NQ_20260602_232808/evaluation.json"))
    dfs = []
    for d in ev["dates_used"]:
        p = os.path.join(STORE, d, "ml_utility_3d2f8466.parquet")
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    return df


def train_folds(valid):
    """Return (splits, fold_models, oos_labeled_df with prob_reversal/raw_pred/fold)."""
    valid = valid.reset_index(drop=True)
    features = valid[FEATS]
    y = valid["label_encoded"].astype(int)
    ts = (
        pd.to_datetime(valid["timestamp"])
        if "timestamp" in valid
        else pd.to_datetime(valid["event_ts"])
    )
    splits = WalkForwardSplitter(CFG.walk_forward).split(ts)

    fw_min = max(5, CFG.labeling.forward_window // 500)
    purge = pd.Timedelta(minutes=fw_min)
    fold_cfg = CFG.model.model_copy(update={"rfecv_enabled": False})

    fold_models = {}
    rows = []
    fold_importances = []
    n_purged = 0
    for s in splits:
        train_ts = ts.iloc[s.train_indices]
        safe = s.test_start - purge
        ptrain = s.train_indices[(train_ts <= safe).values]
        n_purged += len(s.train_indices) - len(ptrain)
        xtr, ytr = features.iloc[ptrain][FEATS], y.iloc[ptrain]
        if ytr.nunique() < 2:
            continue
        tm = ExtremaModelTrainer(fold_cfg).train(xtr, ytr)
        fold_models[s.fold] = (tm.model, s.test_start, s.test_end)
        fold_importances.append(tm.feature_importances)
        xte = features.iloc[s.test_indices][FEATS]
        classes = list(np.asarray(tm.model.classes_).astype(int))
        proba = tm.model.predict_proba(xte)
        praw = tm.model.predict(xte).flatten().astype(int)
        pr = proba[:, classes.index(0)] if 0 in classes else np.zeros(len(xte))
        sub = valid.iloc[s.test_indices].copy()
        sub["fold"] = s.fold
        sub["prob_reversal"] = pr
        sub["raw_pred"] = praw
        rows.append(sub)
    oos = pd.concat(rows, ignore_index=True)
    return splits, fold_models, oos, fold_importances, n_purged


def confusion(oos):
    yt = (oos["label_encoded"].astype(int) == 0).astype(int).values
    yp = (oos["raw_pred"].astype(int) == 0).astype(int).values
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn)


def honest_block(t, label):
    t = t[t["honest_gross_pts"].notna()].copy()
    n = len(t)
    if n == 0:
        return {"label": label, "n": 0}
    gp = t["honest_gross_pts"].astype(float).values
    net = gp - COMMISSION_PTS
    wins = gp[gp > 0]
    losses = gp[gp < 0]
    boot = []
    rng = np.random.default_rng(42)
    for _ in range(10000):
        boot.append(net[rng.integers(0, n, n)].mean())
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {
        "label": label,
        "n": int(n),
        "hit_rate": float((gp > 0).mean()),
        "exit_reasons": t["honest_exit_reason"].value_counts().to_dict(),
        "expectancy_gross_pts": float(gp.mean()),
        "expectancy_net_pts": float(net.mean()),
        "expectancy_net_usd": float(net.mean() * POINT_VALUE),
        "net_ci95_pts": ci,
        "total_net_usd": float(net.sum() * POINT_VALUE),
        "avg_win_pts": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_pts": float(losses.mean()) if len(losses) else 0.0,
    }


def main():
    out = {}
    valid = load_labeled_384()
    print("labeled rows:", len(valid))
    splits, models, oos, fimp, n_purged = train_folds(valid)
    cm = confusion(oos)
    out["reproduction"] = {
        "n_oos": len(oos),
        "confusion": cm,
        "precision": cm["tp"] / (cm["tp"] + cm["fp"]),
        "recall": cm["tp"] / (cm["tp"] + cm["fn"]),
        "accuracy": (cm["tp"] + cm["tn"]) / len(oos),
        "n_purged_total": int(n_purged),
        "target_confusion": dict(tp=172, fp=57, tn=42, fn=63),
    }
    print("REPRO confusion:", cm, "target tp=172 fp=57 tn=42 fn=63")

    # E2: threshold table on reproduced probs
    yt = (oos["label_encoded"].astype(int) == 0).astype(int).values
    pr = oos["prob_reversal"].values
    tt = []
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        m = pr >= thr
        tt.append(
            {
                "threshold": thr,
                "n": int(m.sum()),
                "coverage": float(m.mean()),
                "precision": float(yt[m].mean()) if m.sum() else 0.0,
                "exp_15_15_pts": float(15 * (2 * yt[m].mean() - 1)) if m.sum() else 0.0,
            }
        )
    out["threshold_table_reproduced"] = tt

    # E3 feature stability
    fs = []
    for i in range(len(fimp)):
        for j in range(i + 1, len(fimp)):
            a = [fimp[i].get(f, 0.0) for f in FEATS]
            b = [fimp[j].get(f, 0.0) for f in FEATS]
            rho, _ = spearmanr(a, b)
            if np.isfinite(rho):
                fs.append(rho)
    out["feature_stability_reproduced"] = float(np.mean(fs)) if fs else None

    # E4 Brier baseline
    base = yt.mean()
    out["brier"] = {
        "model": float(np.mean((pr - yt) ** 2)),
        "base_rate": float(base),
        "baseline_constant_baserate": float(base * (1 - base) ** 2 + (1 - base) * base**2),
    }

    # D4 fold precision dist (reproduced)
    fps = []
    for _f, g in oos.groupby("fold"):
        ytf = (g["label_encoded"].astype(int) == 0).astype(int).values
        ypf = (g["raw_pred"].astype(int) == 0).astype(int).values
        d = ((ytf == 1) & (ypf == 1)).sum()
        dn = ((ytf == 0) & (ypf == 1)).sum()
        fps.append(d / (d + dn) if (d + dn) > 0 else 0.0)
    fps = np.array(fps)
    out["fold_precision_repro"] = {
        "mean": float(fps.mean()),
        "median": float(np.median(fps)),
        "std": float(np.std(fps)),
        "below_0.5": int((fps < 0.5).sum()),
        "le_0.5": int((fps <= 0.5).sum()),
        "n": len(fps),
    }

    # ---- STEP B: score full entered population from enrichment ----
    efiles = sorted(glob.glob(str(ENRICHED / "*.parquet")))
    if efiles:
        dates_used = set(json.load(open("models/NQ_20260602_232808/evaluation.json"))["dates_used"])
        en = pd.concat([pd.read_parquet(f) for f in efiles], ignore_index=True)
        en = en[en["date"].isin(dates_used)].copy()  # match the model's 225 dates_used
        en["event_ts"] = pd.to_datetime(en["event_ts"])
        out["enriched_dates"] = len(efiles)
        out["drop_accounting"] = en["drop_reason"].value_counts().to_dict()
        out["kept_total"] = int((en["drop_reason"] == "kept").sum())
        # entered = has entry+forward (kept or no_resolution); has honest PnL
        entered = en[en["honest_gross_pts"].notna() & en[FEATS].notna().all(axis=1)].copy()
        # assign each entered touch to the fold whose test window covers event_ts; score
        probs = np.full(len(entered), np.nan)
        infold = np.full(len(entered), -1)
        evts = entered["event_ts"]
        for fold, (model, t0, t1) in models.items():
            t0u = pd.Timestamp(t0)
            t1u = pd.Timestamp(t1)
            if t0u.tz is None:
                # align tz
                t0u = t0u.tz_localize(evts.dt.tz)
                t1u = t1u.tz_localize(evts.dt.tz)
            m = (evts >= t0u) & (evts <= t1u)
            if m.any():
                classes = list(np.asarray(model.classes_).astype(int))
                pp = model.predict_proba(entered.loc[m, FEATS])
                probs[m.values] = pp[:, classes.index(0)]
                infold[m.values] = fold
        entered["prob_reversal"] = probs
        entered["in_test_fold"] = infold
        scored = entered[entered["in_test_fold"] >= 0].copy()
        out["n_entered_in_test"] = len(scored)

        rth = scored[scored["session_rth"]].copy()
        gate = 0.70
        gated_all = scored[(scored["prob_reversal"] >= gate)]
        gated_rth = rth[rth["prob_reversal"] >= gate]
        # RTH base rate (reversal among entered RTH that resolved -> label 0 vs 1/2)
        rth_resolved = rth[rth["drop_reason"] != "no_resolution"]
        out["rth"] = {
            "n_entered_rth": len(rth),
            "rth_base_rate_resolved": float((rth_resolved["label_encoded"] == 0).mean())
            if len(rth_resolved)
            else None,
            "n_gated_rth_0.70": len(gated_rth),
            "gated_rth_precision_resolved": float(
                (
                    gated_rth[gated_rth["drop_reason"] != "no_resolution"]["label_encoded"] == 0
                ).mean()
            )
            if len(gated_rth[gated_rth["drop_reason"] != "no_resolution"])
            else None,
            "n_gated_rth_no_resolution": int((gated_rth["drop_reason"] == "no_resolution").sum()),
        }
        out["F_headline"] = {
            "per_trade_taken_RTH_gated": honest_block(gated_rth, "RTH gate0.70 per-trade-taken"),
            "resolved_only_RTH_gated": honest_block(
                gated_rth[gated_rth["drop_reason"] != "no_resolution"], "RTH gate0.70 resolved-only"
            ),
            "baseline_all_RTH_entered_nogate": honest_block(rth, "RTH all entered no-gate"),
            "blended_gated_allsession": honest_block(
                gated_all, "ALL-session gate0.70 per-trade-taken"
            ),
        }
        out["cost_scheme"] = {
            "slippage_ticks_per_side": 1.0,
            "slippage_pts_rt": 0.5,
            "commission_rt_usd": COMMISSION_RT,
            "commission_pts": COMMISSION_PTS,
            "point_value": POINT_VALUE,
        }
    else:
        out["enriched_dates"] = 0

    Path(__file__).parent.joinpath("oos_results.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
