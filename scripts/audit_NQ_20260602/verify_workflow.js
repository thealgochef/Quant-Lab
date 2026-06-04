export const meta = {
  name: 'audit-adversarial-verify',
  description: 'Adversarially verify each finding of the NQ_20260602_232808 honesty audit by independent recomputation',
  phases: [
    { title: 'Verify', detail: 'one skeptic per audit item recomputes from raw artifacts and tries to refute' },
  ],
}

const CTX = `
You are an ADVERSARIAL auditor verifying ONE finding about CatBoost model NQ_20260602_232808
(research project Claude-Quant-Lab). READ-ONLY. Your job is to try to REFUTE the claim by
recomputing from raw artifacts. CONFIRM only if your independent number matches; otherwise REFUTE.

Environment: cd C:/Users/gonza/Documents/Claude-Quant-Lab ; ALL python needs PYTHONPATH=src.
Run python via:  cd /c/Users/gonza/Documents/Claude-Quant-Lab && PYTHONPATH=src python -c "..."
(Use bash tool. Windows; use forward slashes.)

Shared verified artifacts (already computed, byte-reconciled to production — you MAY read them,
but PREFER to recompute the specific number yourself from the primary source):
- Model bundle: models/NQ_20260602_232808/{evaluation.json,metadata.json,strategy.json,model.cbm}
- Per-date production dataset (the 384 kept, hash 3d2f8466):
  C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento/NQ/<date>/ml_utility_3d2f8466.parquet
- Audit enrichment (every touch incl. drops + honest PnL), 227 per-date parquets:
  scripts/audit_NQ_20260602/enriched/*.parquet   (cols: date,event_ts,decision_ts,direction,
  level_type,representative_price,session_rth,decision_rth,drop_reason,label,label_encoded,
  max_mfe,max_mae,entry_price,honest_exit_reason,honest_gross_pts,int_*,app_*)
  -> filter to the 225 dates in evaluation.json['dates_used'] before counting.
- OOS reproduction results: scripts/audit_NQ_20260602/oos_results.json
- Key code: src/alpha_lab/agents/data_infra/ml/{config.py,walk_forward.py,model_trainer.py,
  model_evaluator.py,engine_decision.py,dashboard_utility_builder.py}; scripts/ml_training_tab.py;
  Strategy-core/src/strategy_core/decisions/{outcomes.py,honest_entry.py,features.py}; constants.py

Return STRICT verdict. CONFIRM = your recomputed number matches the claim. REFUTE = it does not
(give the discrepancy). UNVERIFIABLE = artifacts insufficient (say exactly what's missing). PARTIAL =
claim mostly right but with a caveat. Always include the concrete number(s) you computed and the
file/line or code that produced them. Do not be agreeable — if the claim is wrong, say REFUTE.
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['item', 'verdict', 'claimed', 'verified_number', 'note'],
  properties: {
    item: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRM', 'REFUTE', 'UNVERIFIABLE', 'PARTIAL'] },
    claimed: { type: 'string', description: 'the claim you were asked to verify' },
    verified_number: { type: 'string', description: 'the concrete number(s) YOU computed' },
    source: { type: 'string', description: 'file:line or code/command that produced your number' },
    note: { type: 'string', description: 'one-to-three sentences; any caveat or discrepancy' },
  },
}

const ITEMS = [
  {key:'A1_rfecv', claim:`A1 RFECV conflict is NOT a contradiction: run_walk_forward_training (scripts/ml_training_tab.py ~234-251) runs RFECV ONCE at pipeline level when config.model.rfecv_enabled is True (evaluation.json full_config.model.rfecv_enabled=true), then trains every fold AND the final model with config.model.model_copy(update={rfecv_enabled:False}) (line ~249, ~400). metadata.json reflects the FINAL model's (copied) config, hence rfecv_enabled=false. Feature selection DID run; selected_features = all 6 (rfecv_min_features=5, so 6>=5 kept). Verify the two configs are different objects and the claim holds.`},
  {key:'A2A3_folds_dates', claim:`A2: the 33 folds are correct for the 225 dates at train=30/test=7/gap=1/expanding=false (reproduce WalkForwardSplitter on the 384 'timestamp' column -> expect 33 splits, 334 union OOS test rows, 0 single-class-train folds). The UI's '~216 folds from 1543d' is the slider mis-estimating off the FULL store calendar span (2021-12-02..2026-02-22 ~1543 days), not the selected 263-day span. A3: 85 store date-dirs are older than 2025-06-02 (excluded by range selection only); within [2025-06-02,2026-02-22] only 2 store dirs are not in dates_used (2025-11-20 known data gap, 2026-02-14 Saturday) — no silent quality drop. Verify fold count, n_skipped=0, and the date reconciliation by listing C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento/NQ.`},
  {key:'A4_hash', claim:`A4: dataset_config_hash() for THIS config (dashboard_utility tp=15,sl=15,trap_mfe_min=5,interaction=5,level_proximity=0.5,bar_type=147t,include_approach=True,approach=15; tick 0.25; engine v2 trade_price/realistic_at_decision) reproduces EXACTLY '3d2f8466' (the model's provenance hash). The sl=30 variant hashes to a DIFFERENT value. Therefore the dataset was built with sl=15 (matching strategy.json/full_config). Recompute the hash via MLPipelineConfig(...).dataset_config_hash().`},
  {key:'A5_gates', claim:`A5: check_quality_gates (scripts/ml_training_tab.py ~436-481) has exactly 6 gates. Five PASS: precision>=0.55 (0.751), permutation p<0.05 (0.003992), ROC-AUC>0.55 (0.635), Brier<0.25 (0.2469), n>=200 (334). ONE FAILS: 'Fold stability (std<0.15)' because np.std(fold precisions)=0.1613 >= 0.15. That is the lone gate behind 'Some quality gates failed.' The model was saved anyway (the save path only st.warning's; the saved bundle exists on disk). Recompute the fold-precision std from evaluation.json['fold_metrics'] and confirm exactly which gates pass/fail.`},
  {key:'B1B2_drops', claim:`B1: over the 225 dates the enrichment finds 910 detected touches -> 525 flatten-drop, 1 feature_drop (interaction <5 trades), 384 kept; and ZERO no_resolution, no_fill, no_forward, cutoff. kept reconciles to the production 384. B2: because there are 0 no_resolution drops, the 'per trade taken' expectancy EQUALS 'resolved' expectancy — the no_resolution optimism hole is EMPTY for this tp15/sl15/forward-to-16:15 config. Verify by value_counts of drop_reason over enriched parquets filtered to dates_used, and confirm kept==384.`},
  {key:'B3B4_flatten_tz', claim:`B3/B4: flatten (decision ET time>=15:55) and forward/resolution cutoff (16:15 ET) are consistent — an entered trade (decision<15:55) always has >=20min to resolve to 16:15, so you can't enter a trade you can't hold (strategy_core/decisions/honest_entry.py:138-141). Timezone sanity: ALL 525 flatten touches have decision ET time>=15:55 and ALL 384 kept have decision ET time<15:55 (clean ET partition, no UTC smear). The flatten bucket is dominated by EVENING/overnight touches (decision ET hours 18-23, ~520) not near-close (only ~5 at hour 16). Verify the time partition and the hour histogram from the enriched parquets.`},
  {key:'F_headline', claim:`F (THE go/no-go): on the DEPLOYMENT population (ny_rth only, gate prob_reversal>=0.70, decision-time touch+5m trade entry, net of costs = 1 tick/side slippage 0.5pt RT + commission $5.28RT), the full-OOS honest edge is n=11 trades, hit 0.545 (6 tp/5 sl), gross +1.11 pts, NET +0.85 pts/trade (+$17), 95% bootstrap CI [-7.3,+9.0] pts (straddles zero). RTH base rate 0.509-0.525 and RTH gated precision 0.545 vs the BLENDED 0.704 base / 0.805 gated-precision on 133 all-session trades. The blended numbers are ~83% non-RTH (overnight) and are NOT the deployment number. Verify from oos_results.json['F_headline'] and by independently recomputing the RTH gated honest PnL from enriched+OOS probs (the OOS confusion 172/57/42/63 is reproducible byte-exact, anchoring the gating).`},
  {key:'C_leakage', claim:`C1-C5 NO LEAKAGE: approach features query trades/quotes over [touch-15m, touch) END-EXCLUSIVE (engine_decision._query_trades/_query_quotes: ts_event < touch); interaction features over [touch, touch+5m]=[touch,decision] (tick_store.query_tick_feature_rows: ts_event<=end, never beyond decision); forward/label bars close STRICTLY > decision and < 16:15 (strategy_core/decisions/honest_entry.py:151-155); entry = most-recent trade print ts<=decision (engine_decision._trade_price_at); the 6 feature functions (strategy_core/decisions/features.py) read ONLY their window's trades/quotes, no session-level or post-decision aggregate. An empirical trace (2025-06-05 PDL LONG @14:16:51 ET, 2025-07-02 PDH SHORT) confirms approach.max<touch, interaction.max<=decision, first forward bar close>decision. Verify by reading the cited code and re-running scripts/audit_NQ_20260602/trace_leakage.py on any RTH date.`},
  {key:'D1_purge', claim:`D1: n_purged_total=0 is LEGITIMATE, not a silent no-op. The labels resolve intraday (same-day by 16:15 ET); the walk-forward gap (gap_days=1) places every test fold >=1 calendar day after train_end, and the splitter's train_mask already ends ~1 day before test_start. So no training touch's same-day label window can reach a test fold. The production purge buffer is max(5, forward_window//500)=10 min (scripts/ml_training_tab.py:263) — far smaller than the true ~6.75h intraday horizon, so the buffer itself is mis-sized, but the day-gap (not the buffer) is what actually prevents leakage; hence 0 purged is correct here. Reproduced n_purged=0. Verify the purge code and that gap>=1day exceeds the intraday horizon.`},
  {key:'D2_permutation', claim:`D2: permutation p=0.003992 (model_evaluator._permutation_test, (count+1)/(500+1)) is computed for THIS single config only — it permutes labels against fixed predictions for ONE model. It does NOT account for config selection across the multi-config sweep run in this lab; the family-wise/selection-adjusted significance of the SELECTED config is necessarily weaker (a multiple-comparisons / selection-bias inflation). p=0.004 must NOT be presented as the family-wise result. This is an interpretation claim — confirm the per-config computation and that no sweep-wide correction is applied anywhere in the code.`},
  {key:'D3D4_overfit_variance', claim:`D3: train_accuracy 0.797 (metadata train_metrics, the FINAL model evaluated IN-SAMPLE on all 384 — model_trainer.py:87-88) vs OOS accuracy 0.641 -> gap 0.156; the train figure is in-sample by construction so optimistic; gap is moderate for depth4/500it on 384 samples. D4: fold precision (33 folds) mean 0.746, median 0.75, IQR[0.667,0.857], std 0.161; 1 fold <0.50 (0.333) and 4 folds <=0.50; aggregate pooled precision 0.751 is broad, NOT propped by a few strong folds. Verify from evaluation.json fold_metrics + metadata train_metrics; reproduce fold precisions if you wish.`},
  {key:'E_metric_coherence', claim:`E1: expectancy_15_30=3.80 is INCOHERENT — compute_utility_metrics (scripts/ml_training_tab.py:494-508) reuses the SAME sl=15 OOS confusion (tp=172,fp=57) and merely re-prices the false positives at -30 ((172*15-57*30)/229=3.80); it does NOT re-resolve outcomes at sl=30. Recommend hiding it. E2: expectancy_15_15=7.53 is at the argmax/0.5 gate (229 predicted-positive); at the config-correct 0.70 gate the IDEALIZED expectancy_15_15 is ~9.14 pts (133 trades, precision 0.8045) — but the HONEST RTH-only 0.70-gate number is ~+0.85 net (see F). E3: feature_stability=0.152 is the mean pairwise Spearman rank-corr of the 6 features' importances across the 33 folds (ml_training_tab.py:358-374); reproduced 0.152; low (~uncorrelated importance ranks) — expected for 6 features on ~10-sample folds, signals no stable feature ranking. E4: the low-bucket calibration inversion (mean_pred 0.132 -> observed 0.604) is base-rate compression (70% prior) amplified by auto_class_weights=Balanced pulling probabilities toward 50/50; Brier 0.247 is actually WORSE than a constant base-rate predictor's Brier 0.208 -> poor calibration, not a bug. Verify each via the cited code and a recompute where cheap.`},
]

phase('Verify')
const results = await parallel(ITEMS.map(it => () =>
  agent(`${CTX}\n\n=== ITEM ${it.key} ===\nVerify this claim:\n${it.claim}`,
        { label: `verify:${it.key}`, phase: 'Verify', schema: SCHEMA })
))

return { results: results.filter(Boolean) }
