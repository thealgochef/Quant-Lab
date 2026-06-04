"""DIAGNOSTIC report (read-only): honest no-model NY baseline on the v3 dataset.

Reads scripts/v3_verify/ny_enriched/*.parquet (from ny_baseline_enrich.py) and computes
the four diagnostic blocks. NO model, NO 0.70 gate — every honest NY touch is taken, so
the result is directly comparable to a future gated number (only the gate differs).
Cost model VERBATIM from audit_NQ_20260602/reproduce_oos.py. PYTHONPATH=src.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENRICHED = Path(__file__).parent / "ny_enriched"
OUT = Path(__file__).parent / "ny_baseline_results.json"

# Cost model — verbatim (reproduce_oos.py:34-36).
POINT_VALUE = 20.0
COMMISSION_RT = (2.14 + 0.50) * 2.0  # $5.28
COMMISSION_PTS = COMMISSION_RT / POINT_VALUE  # 0.264 pts

RTH_BASE_RATE_PRIOR = 0.525  # phase9 audit: RTH base rate resolved (~0.52 to beat)

PDH_PDL = {"PDH", "PDL"}
SESSION_LEVELS = {"asia_high", "asia_low", "london_high", "london_low"}


def _group(lt: str) -> str:
    if lt in PDH_PDL:
        return "pdh_pdl"
    if lt in SESSION_LEVELS:
        return "session"
    return f"other:{lt}"  # merged-zone first-name fallthrough (flagged, not hidden)


def _ci_block(gross: np.ndarray) -> dict:
    """n, net pt/trade, hit rate, bootstrap 95% CI (10000, seed 42) — honest_block logic."""
    n = len(gross)
    if n == 0:
        return {"n": 0}
    net = gross - COMMISSION_PTS
    rng = np.random.default_rng(42)
    boot = np.array([net[rng.integers(0, n, n)].mean() for _ in range(10000)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {
        "n": int(n),
        "hit_rate": float((gross > 0).mean()),
        "expectancy_gross_pts": float(gross.mean()),
        "expectancy_net_pts": float(net.mean()),
        "expectancy_net_usd": float(net.mean() * POINT_VALUE),
        "net_ci95_pts": ci,
        "ci_half_width_pts": float((ci[1] - ci[0]) / 2.0),
        "total_net_usd": float(net.sum() * POINT_VALUE),
    }


def main() -> int:
    files = sorted(glob.glob(str(ENRICHED / "*.parquet")))
    if not files:
        print("NO enriched files — run ny_baseline_enrich.py first"); return 1
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ev = json.load(open("models/NQ_20260602_232808/evaluation.json"))
    dates_used = set(ev["dates_used"])
    df = df[df["date"].isin(dates_used)].copy()
    df["grp"] = df["level_type"].map(_group)

    out = {"engine": "strategy_core_engine_v3", "dataset_hash": "d8e239c7",
           "cost_model": {"commission_rt_usd": COMMISSION_RT, "commission_pts": COMMISSION_PTS,
                          "slippage_pts_per_side": 0.25, "point_value": POINT_VALUE,
                          "tp_points": 15.0, "sl_points": 15.0},
           "n_dates": int(df["date"].nunique())}

    print("=" * 78)
    print("HONEST NO-MODEL NY BASELINE — v3 dataset (hash d8e239c7), NO gate, NO model")
    print("=" * 78)

    # ── 1. Honest NY touch universe ─────────────────────────────────────────
    total_ny = len(df)
    lookahead = int(df["lookahead"].sum())
    by_lt = df["level_type"].value_counts().to_dict()
    by_grp = df["grp"].value_counts().to_dict()
    elig = df[df["eligible"] == True]  # noqa: E712
    n_elig = len(elig)
    out["block1_universe"] = {
        "total_ny_touches": total_ny,
        "lookahead_in_ny": lookahead,
        "by_level_type": by_lt,
        "by_group": by_grp,
        "n_decision_eligible": n_elig,
        "eligible_by_group": elig["grp"].value_counts().to_dict(),
        "eligible_by_level_type": elig["level_type"].value_counts().to_dict(),
        "drop_reasons_full_ny": df["drop_reason"].value_counts(dropna=False).to_dict(),
    }
    print(f"\n[1] NY touch universe (availability-ENFORCED):")
    print(f"    total NY touches               = {total_ny}")
    print(f"    look-ahead within NY           = {lookahead}  (MUST be 0)")
    print(f"    by level_type                  = {by_lt}")
    print(f"    by group  pdh_pdl / session    = {by_grp}")
    print(f"    decision-time ELIGIBLE (n ceiling, decision<16:40 ET) = {n_elig}")
    print(f"       eligible by group           = {elig['grp'].value_counts().to_dict()}")

    # ── 2. Label distribution over ELIGIBLE NY touches ──────────────────────
    def _label_stats(sub: pd.DataFrame) -> dict:
        labs = sub["label"].value_counts(dropna=False).to_dict()
        resolved = sub[sub["label"].isin(
            ["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"])]
        n_res = len(resolved)
        rate = float((resolved["label"] == "tradeable_reversal").mean()) if n_res else None
        rate_incl = float((sub["label"] == "tradeable_reversal").mean()) if len(sub) else None
        return {"labels": labs, "n_eligible": len(sub), "n_resolved": n_res,
                "tradeable_base_rate_resolved": rate,
                "tradeable_share_incl_noresolution": rate_incl}
    out["block2_labels"] = {
        "RTH_base_rate_to_beat": RTH_BASE_RATE_PRIOR,
        "overall": _label_stats(elig),
        "pdh_pdl": _label_stats(elig[elig["grp"] == "pdh_pdl"]),
        "session": _label_stats(elig[elig["grp"] == "session"]),
    }
    print(f"\n[2] Label distribution over ELIGIBLE NY touches "
          f"(base rate to beat ~{RTH_BASE_RATE_PRIOR}):")
    for k in ("overall", "pdh_pdl", "session"):
        b = out["block2_labels"][k]
        print(f"    {k:8s}: n_elig={b['n_eligible']:4d} n_resolved={b['n_resolved']:4d} "
              f"tradeable_base_rate_resolved={b['tradeable_base_rate_resolved']} "
              f"labels={b['labels']}")

    # ── 3. Unconditional baseline (every eligible NY touch with a fill+forward) ─
    traded = elig[elig["honest_gross_pts"].notna()].copy()
    out["block3_baseline"] = {
        "all_eligible_ny": _ci_block(traded["honest_gross_pts"].to_numpy(float)),
        "pdh_pdl_only": _ci_block(
            traded[traded["grp"] == "pdh_pdl"]["honest_gross_pts"].to_numpy(float)),
        "session_levels_only": _ci_block(
            traded[traded["grp"] == "session"]["honest_gross_pts"].to_numpy(float)),
        "exit_reasons": traded["honest_exit_reason"].value_counts().to_dict(),
        "n_eligible_no_fill_or_forward": int(len(elig) - len(traded)),
    }
    print(f"\n[3] Unconditional baseline (take EVERY eligible NY touch, net of costs, tp15/sl15):")
    for k in ("all_eligible_ny", "pdh_pdl_only", "session_levels_only"):
        b = out["block3_baseline"][k]
        if b.get("n", 0) == 0:
            print(f"    {k:20s}: n=0"); continue
        print(f"    {k:20s}: n={b['n']:4d}  net={b['expectancy_net_pts']:+.3f} pt/trade "
              f"(${b['expectancy_net_usd']:+.2f})  hit={b['hit_rate']:.3f}  "
              f"CI95={[round(x,2) for x in b['net_ci95_pts']]}")

    # ── 4. Power note ───────────────────────────────────────────────────────
    base = out["block3_baseline"]["all_eligible_ny"]
    hw = base.get("ci_half_width_pts")
    out["block4_power"] = {
        "n": base.get("n"),
        "ci_half_width_pts": hw,
        "min_distinguishable_edge_pts": hw,
        "note": "An edge smaller than the CI half-width is indistinguishable from zero "
                "at this n; halving the half-width needs ~4x the trades.",
    }
    print(f"\n[4] Power: n={base.get('n')}  CI half-width = "
          f"{round(hw,3) if hw is not None else None} pt  -> any per-trade edge below "
          f"~{round(hw,2) if hw is not None else None} pt is indistinguishable from zero.")

    # ── Decision frame (numbers only; the call is the user's) ───────────────
    n_elig_traded = base.get("n", 0)
    sess = out["block2_labels"]["session"]["tradeable_base_rate_resolved"]
    ci = base.get("net_ci95_pts", [0, 0])
    straddles = (ci[0] <= 0 <= ci[1])
    frame = []
    frame.append(f"eligible NY tradeable n = {n_elig_traded} "
                 f"({'SMALL <60' if n_elig_traded < 60 else 'healthy >=60'})")
    frame.append(f"unconditional net CI95 straddles zero = {straddles}")
    frame.append(f"session-level reversal rate = {sess} vs ~{RTH_BASE_RATE_PRIOR} prior")
    if n_elig_traded < 60 and straddles and (sess is None or sess <= 0.55):
        rec = "PARK — small n + CI straddles zero + reversal ~0.52: a retrain most likely reproduces the n=11 ambiguity."
    elif n_elig_traded >= 60 and ((sess is not None and sess > 0.55) or (ci[0] > 0)):
        rec = "RETRAIN worth running — healthy n AND (session reversal > ~0.52 OR baseline positive); the 0.70 gate may sharpen it."
    else:
        rec = "MIXED — see numbers; the call is the user's (does not cleanly meet either rule)."
    out["decision_frame"] = {"signals": frame, "suggested_read": rec}
    print(f"\n[DECISION FRAME] (report only; the call is the user's)")
    for s in frame:
        print(f"    - {s}")
    print(f"    => {rec}")

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
