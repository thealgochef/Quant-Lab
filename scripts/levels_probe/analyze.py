"""Analyze the levels probe (READ-ONLY): PART 2 mechanics + PART 3 regime-vs-selection.
Joins probe touches (all detected) with the audit enrichment (labels/drops). PYTHONPATH=src."""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).parent / "out"
ENR = Path(__file__).parent.parent / "audit_NQ_20260602" / "enriched"
DU = set(json.load(open("models/NQ_20260602_232808/evaluation.json"))["dates_used"])

zones = pd.concat([pd.read_parquet(f) for f in glob.glob(str(OUT / "zones_*.parquet"))], ignore_index=True)
touch = pd.concat([pd.read_parquet(f) for f in glob.glob(str(OUT / "touches_*.parquet"))], ignore_index=True)
zones = zones[zones.date.isin(DU)].copy(); touch = touch[touch.date.isin(DU)].copy()
en = pd.concat([pd.read_parquet(f) for f in glob.glob(str(ENR / "*.parquet"))], ignore_index=True)
en = en[en.date.isin(DU)].copy()
en["event_ts"] = pd.to_datetime(en["event_ts"]).astype(str)
touch["event_ts"] = pd.to_datetime(touch["event_ts"]).astype(str)
# join label/drop onto probe touches by (date, rep_price, event_ts)
en["k"] = en.date + "|" + en.event_ts.str[:19]
touch["k"] = touch.date + "|" + touch.event_ts.str[:19]
lab = en[["k", "label_encoded", "drop_reason", "honest_gross_pts", "session_rth"]].drop_duplicates("k")
t = touch.merge(lab, on="k", how="left")
t["reb"] = (t["label_encoded"] == 0).astype("float")  # NaN if not kept
out = {}

# ---- PART 2A: detected-touch counts by session x level_type, and kept ----
out["detected_total"] = int(len(t))
out["detected_by_session"] = t.session.value_counts().to_dict()
out["detected_by_session_x_level"] = pd.crosstab(t.level_type, t.session).to_dict()
kept = t[t.drop_reason == "kept"]
out["kept_total"] = int(len(kept))
out["kept_by_session"] = kept.session.value_counts().to_dict()

# ---- PART 2B: per-day zones formed / touched / surviving to RTH ----
zg = zones.groupby("date")
perday = pd.DataFrame({
    "zones_formed": zg.size(),
    "zones_touched": zg.apply(lambda g: (g.n_straddles > 0).sum()),
    "zones_untouched_to_rth": zg.apply(lambda g: ((g.first_straddle_session == "ny_rth") | (g.n_straddles == 0)).sum()),
    "zones_first_touch_rth": zg.apply(lambda g: (g.first_straddle_session == "ny_rth").sum()),
})
out["perday_zone_stats_describe"] = perday.describe().to_dict()
out["perday_zones_formed_hist"] = perday.zones_formed.value_counts().sort_index().to_dict()
out["perday_first_touch_rth_hist"] = perday.zones_first_touch_rth.value_counts().sort_index().to_dict()

# ---- PART 2C: THE KEY NUMBER ----
out["zones_total"] = int(len(zones))
out["zones_with_any_rth_straddle"] = int((zones.n_rth_straddles > 0).sum())
out["zones_first_touch_in_rth (recorded RTH touches)"] = int((zones.first_straddle_session == "ny_rth").sum())
out["zones_RTH_swept_but_consumed_overnight"] = int(zones.first_straddle_overnight_but_rth_swept.sum())
out["pct_RTH_sweeps_consumed"] = float(
    zones.first_straddle_overnight_but_rth_swept.sum() /
    max(1, (zones.n_rth_straddles > 0).sum()))
# lookahead
out["lookahead_touches_total"] = int(zones.lookahead_first_before_avail.sum())
out["lookahead_by_first_session"] = zones[zones.lookahead_first_before_avail].first_straddle_session.value_counts().to_dict()
out["self_touch_raw_extreme_zones"] = int(zones.rep_is_raw_session_extreme_standalone.sum())
# how many recorded touches are on a raw standalone session extreme, by session
out["recorded_touch_is_raw_extreme_by_session"] = (
    t.assign(raw=t.rep_is_raw_extreme).groupby("session").raw.sum().to_dict())

# ---- PART 3: E1-E4 RTH vs overnight (on KEPT touches, which carry labels) ----
kept = kept.copy()
kept["grp"] = np.where(kept.session == "ny_rth", "RTH", "overnight")
def dist(col):
    return kept.groupby("grp")[col].describe()[["count", "mean", "50%", "std"]].to_dict("index")
out["E1_travel"] = {c: dist(c) for c in ["net_disp30_pts", "approach_range30_pts", "dist_session_open_pts", "dist_rth_open_pts"]}
out["E2_vol"] = {c: dist(c) for c in ["vol_atr14_pts", "ret_std30_pts"]}
out["E4_volume"] = {c: dist(c) for c in ["vol_touch", "vol_mean30"]}
# E3 rebound by session, and by vol bucket (atr14 terciles), and the controlled comparison
out["E3_rebound_by_session"] = kept.groupby("session").reb.agg(["mean", "count"]).to_dict("index")
kept["volb"] = pd.qcut(kept.vol_atr14_pts, 3, labels=["lo", "mid", "hi"], duplicates="drop")
out["E3_rebound_by_volbucket"] = kept.groupby("volb").reb.agg(["mean", "count"]).to_dict("index")
out["E3_rebound_by_session_x_volbucket"] = (
    kept.groupby(["grp", "volb"]).reb.agg(["mean", "count"]).reset_index().to_dict("records"))
# rebound by level_type (the tautology check: session-extreme levels vs pdh/pdl)
out["E3_rebound_by_level_type"] = kept.groupby("level_type").reb.agg(["mean", "count"]).to_dict("index")
# rebound: raw-extreme self-touch vs not
out["E3_rebound_raw_extreme_vs_not"] = kept.groupby(kept.rep_is_raw_extreme).reb.agg(["mean", "count"]).to_dict("index")

Path(OUT.parent / "analysis.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
