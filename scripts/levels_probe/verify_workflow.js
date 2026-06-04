export const meta = {
  name: 'levels-probe-verify',
  description: 'Adversarially verify the levels/touch investigation findings by independent recomputation',
  phases: [{ title: 'Verify', detail: 'one skeptic per finding recomputes from probe artifacts + production code' }],
}

const CTX = `
ADVERSARIAL verifier of ONE finding from a READ-ONLY investigation into why model
NQ_20260602_232808 has scarce/weak NY-RTH touches. Try to REFUTE by recomputing from primary
artifacts. CONFIRM only if your independent number matches. Environment:
cd C:/Users/gonza/Documents/Claude-Quant-Lab ; ALL python needs PYTHONPATH=src (run:
cd /c/Users/gonza/Documents/Claude-Quant-Lab && PYTHONPATH=src python -c "...").

Artifacts (recompute from these; PREFER primary source over trusting the summary):
- Probe outputs (reuse production build_zones/is_touch/detect_touches; one row per zone-day / per
  recorded touch): scripts/levels_probe/out/zones_*.parquet , touches_*.parquet
  zones cols: date,zone_idx,rep_price,side,names,n_levels,available_from_et,n_straddles,
  first_straddle_et,first_straddle_session,has_rth_straddle,rth_straddle_first_et,
  lookahead_first_before_avail,rep_is_raw_session_extreme_standalone,n_rth_straddles,
  first_straddle_overnight_but_rth_swept
  touches cols: date,event_ts,session,level_type,direction,rep_price,n_levels,lookahead,
  rep_is_raw_extreme,vol_atr14_pts,ret_std30_pts,net_disp30_pts,approach_range30_pts,
  dist_session_open_pts,dist_rth_open_pts,vol_touch,vol_mean30
- Audit enrichment (labels/drops, byte-faithful to the cached 384): scripts/audit_NQ_20260602/enriched/*.parquet
  (label_encoded, drop_reason; filter both to evaluation.json['dates_used'] = 225 dates)
- Pre-computed summary (you may cross-check, but recompute yourself): scripts/levels_probe/analysis.json
- PRODUCTION code to quote (file:line): strategy_core/src/strategy_core/decisions/{zones.py,touch.py},
  constants.py; CQL src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py
  (_compute_levels_for_date, _slice_session, _build_bars_for_date, _detect/_build_zones legacy),
  engine_decision.py; strategy_contract.py (available_from_guard emit).
Note: the probe's recorded touches (first straddle per zone) reproduce detect_touches and match the
enriched detected-touch counts exactly (already validated). 'overnight' = asia+london+premarket.

Return a strict verdict. CONFIRM=your number matches; REFUTE=does not (give discrepancy);
UNVERIFIABLE=artifacts insufficient (say what's missing); PARTIAL=mostly right w/ caveat. Always
give the concrete number you computed and the file:line/command. Be adversarial.
`

const SCHEMA = { type:'object', additionalProperties:false,
  required:['item','verdict','claimed','verified_number','note'],
  properties:{ item:{type:'string'}, verdict:{type:'string',enum:['CONFIRM','REFUTE','UNVERIFIABLE','PARTIAL']},
    claimed:{type:'string'}, verified_number:{type:'string'}, source:{type:'string'}, note:{type:'string'} } }

const ITEMS = [
  {key:'P1_code', claim:`PART 1 code: (1a) levels built in _compute_levels_for_date (dashboard_utility_builder.py:397-426): PDH/PDL = prior-day NY-RTH high/low (prev_ny_hl, lines 408-410); asia_high/low & london_high/low = THIS day's session slices via _slice_session (382-394) over an ET-indexed bar frame (_ensure_et_index 364-372), session edges _ASIA_START=18:00.._NY_RTH_END=16:15 (44-51) evaluated on bars.index.time in US/Eastern. (1c) build_zones (strategy_core/decisions/zones.py:23-97) merges levels within 3pt (ZONE_PROXIMITY_PTS) chained against the last-appended level, rep=mean. (1d) detect_touches (touch.py:49-95) fires the FIRST bar whose [low,high] straddles rep (is_touch, bar_low<=rep<=bar_high, line 46/82), flips zone.touched (83), never re-fires (79); 'per day' = the 18:00-ET Globex day from _build_bars_for_date bounds [prev 18:00 ET, cur 18:00 ET) (builder:319-320). Verify each line reference says what is claimed.`},
  {key:'P1b_guard', claim:`PART 1b: the contract's available_from_guard is NOT enforced. strategy_contract.py:153 emits available_from_guard from constants.LEVEL_AVAILABLE_FROM_GUARD (=True) as a DESCRIPTOR only; neither _compute_levels_for_date nor detect_touches gates touch detection by a level's availability time, so detect_touches scans ALL Globex-day bars incl. the asia/london forming sessions. Empirical: 713 of 910 recorded touches (78%) and 273 of 384 kept training touches (71%) are look-ahead (first straddle before the zone's available_from: asia avail 01:00 ET, london 08:00 ET, pdh/pdl from day start). Recompute lookahead counts from zones_*/touches_* and confirm no availability gate exists in the touch path.`},
  {key:'P2C_key', claim:`PART 2C (THE number): over 225 dates, 1071 zones; 617 zones had >=1 RTH straddle (price swept the level during 09:30-16:15 ET); only 64 were first-touched IN RTH (recorded RTH touches); 553 were first-touched OVERNIGHT but also swept in RTH and thus discarded by first-touch-per-day dedup = 553/617 = 89.6% of RTH level-sweeps invisible to the model. Recompute n_rth_straddles>0 count, first_straddle_session=='ny_rth' count, and first_straddle_overnight_but_rth_swept sum from zones_*.parquet (filtered to dates_used).`},
  {key:'P2AB_counts', claim:`PART 2A/B: detected touches=910 by session asia600/london227/ny_rth64/premarket17/post2; kept=384 (london227/asia79/ny_rth61/premarket17) — the ~17 not in 3 named sessions are premarket (08:00-09:30 ET), RTH detected 64 vs kept 61 (3 late-RTH flatten-drops). Per day: ~5.8 zones formed (median 6; merging rare = 35/1071 zones have n_levels>=2), and on 124 of 185 zone-days (67%) ZERO zones are first-touched in RTH (mean 0.35/day). Recompute from zones_*/touches_* + enriched drop_reason.`},
  {key:'P3_selection', claim:`PART 3 (WHY): the overnight-vs-RTH reversal gap is SELECTION (look-ahead session-extreme touches), not regime. Rebound (label_encoded==0) by level_type: london_high 0.867/london_low 0.811 vs PDH 0.500/PDL 0.526/asia_high 0.512 (the ~0.80 overnight rate is the london session-extreme self-touches). Controlling for volatility (atr14 terciles): in the LOW-vol bucket, RTH rebound 0.488 (n=41) vs overnight 0.747 (n=87) — the gap SURVIVES vol control, so it is not regime/vol-driven. RTH touches are almost all PDH/PDL reverting ~0.52 (coin flip). Recompute rebound by level_type and the session x vol-bucket table from touches_* joined to enriched label_encoded.`},
  {key:'P3_context', claim:`PART 3 E1/E2/E4 context (RTH vs overnight, kept touches): E4 volume RTH higher (vol_touch median 253 vs 207; vol_mean30 227 vs 203) -> liquid RTH / thinner overnight CONFIRMED. E2 per-147t-bar range atr14 RTH median 6.0 < overnight 8.14 (overnight bars span more wall-clock time per 147 trades -> wider range; a tick-bar-clock artifact, NOT higher true RTH vol). E1 travel mixed: overnight has larger recent 30-bar range (median 50.25 vs 37.75, session-extreme context) while RTH touches sit far from the RTH open (dist_rth_open median 130 vs 49). Recompute these medians from touches_*.parquet grouped RTH vs overnight.`},
]

phase('Verify')
const results = await parallel(ITEMS.map(it => () =>
  agent(`${CTX}\n\n=== ITEM ${it.key} ===\nVerify:\n${it.claim}`,
        { label:`verify:${it.key}`, phase:'Verify', schema:SCHEMA })))
return { results: results.filter(Boolean) }
