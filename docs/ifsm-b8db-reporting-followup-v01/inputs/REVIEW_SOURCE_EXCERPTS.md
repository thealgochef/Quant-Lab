# Source excerpts — delivered b8db source

Line numbers refer to the extracted source ZIP, not the user's later working tree.

## source/task_core/src/strategy_core/strategies/ifvg_smc/reducer.py · 2320–2360

```text
2320:         inp: IfvgStepInput,
2321:         s: _Setup,
2322:         out: list[IfvgEmission],
2323:     ) -> None:
2324:         bar = inp.bar_1m
2325:         if s.inversion_ordinal is None or s.inversion_ordinal >= self._ordinal:
2326:             return
2327:         triggers: list[tuple[str, Fvg | None, str, bool, bool | None, bool | None]] = []
2328:         wanted = self._trade_direction_to_gap(s.direction)
2329:         for gap in inp.new_fvgs.get(60, ()):
2330:             if gap.direction is wanted:
2331:                 confirmed, fully, satisfied = self._causality(
2332:                     gap,
2333:                     trigger_ts=s.inversion_ts_utc,
2334:                     policy=self._cfg.causality_entry,
2335:                 )
2336:                 triggers.append(
2337:                     (
2338:                         "fresh_fvg_continuation",
2339:                         gap,
2340:                         gap.fvg_id,
2341:                         satisfied and gap.confirmed_ts_utc <= bar.availability_ts_utc,
2342:                         confirmed,
2343:                         fully,
2344:                     )
2345:                 )
2346:         if self._retest_triggered(bar, s):
2347:             triggers.append(
2348:                 (
2349:                     "ifvg_retest",
2350:                     None,
2351:                     f"{self._cfg.retest_trigger}|{bar_cursor(bar)}",
2352:                     True,
2353:                     None,
2354:                     None,
2355:                 )
2356:             )
2357:         for family, entry_gap, evidence_id, causality_ok, confirmed, fully in triggers:
2358:             self._emit_candidate(
2359:                 inp,
2360:                 s,
```

## source/task_core/src/strategy_core/strategies/ifvg_smc/reducer.py · 2615–2642

```text
2615:                 )
2616:             )
2617:             if (
2618:                 "out_of_session" in blocks
2619:                 and self._cfg.outside_session_policy
2620:                 == OutsideSessionPolicy.RESET_SETUP_AS_MISSED.value
2621:                 and s.phase == "S4"
2622:             ):
2623:                 self._terminate_pretrade(s, "missed_out_of_session", bar, out)
2624:             elif (
2625:                 self._cfg.ifsm_context_policy_version is not None
2626:                 and s.phase == "S4"
2627:                 # Only an otherwise executable opportunity can be refused by
2628:                 # these policies. Inactive/unratified diagnostics and causal,
2629:                 # geometric or account blocks retain their existing lifecycle.
2630:                 and blocks
2631:                 and all(reason in {
2632:                     "ifsm_early_positive", "ifsm_positive_london",
2633:                     "ifsm_overhead_gex", "daily_execution_cap",
2634:                 } for reason in blocks)
2635:             ):
2636:                 self._terminate_pretrade(s, "ifsm_policy_refused_entry", bar, out)
2637:             return
2638: 
2639:         self._assert_decision_consistency(s, entry_gap, entry, stop, risk)
2640:         decision_id = make_decision_id(candidate_id, self._cfg.profile_hash)
2641:         trade_id = make_trade_id(decision_id, bar_cursor(bar))
2642:         decision = EligibleDecisionRecord(
```

## source/quant_lab/src/alpha_lab/agents/data_infra/ifvg/presentation/lab/mffu_gamma.py · 330–363

```text
330:             base["first_checkpoint_time_reason"] = (
331:                 trade.get("scale_out_timestamp_status") or "exact first checkpoint time not saved"
332:                 if had_partial and not checkpoint_time
333:                 else None
334:             )
335:             base["first_checkpoint_branch"] = (
336:                 "actual partial"
337:                 if had_partial
338:                 else "whole at first target"
339:                 if checkpoint_time
340:                 else "not reached"
341:             )
342:             basis_time = (
343:                 _decision_stamp(target_receipt)
344:                 if population == "strategy" and target_receipt
345:                 else checkpoint_time
346:             )
347:             base["first_checkpoint_snapshot"] = snapshot(basis_time) if checkpoint_time else None
348:             base["first_checkpoint_context_role"] = (
349:                 "executed_saved_decision_receipt"
350:                 if checkpoint_time
351:                 and target_receipt
352:                 and stamp(_decision_stamp(target_receipt)).value == stamp(checkpoint_time).value
353:                 else "reporting_annotation_at_actual_funded_fill"
354:                 if population == "funded"
355:                 else "reporting_annotation_at_ordinary_candle_time"
356:             )
357:             rows.append(base)
358:             if lock:
359:                 lock_snapshot = base["lock_snapshot"]
360:                 level_set = maps.get(lock_snapshot.get("level_set_id"), {})
361:                 items = level_set.get("items") or {}
362:                 upper, lower = (items.get(name, {}).get("price") for name in ("1D Max", "1D Min"))
363:                 move = (
```

## source/quant_lab/src/alpha_lab/agents/data_infra/ifvg/presentation/lab/mffu_gamma.py · 533–547

```text
533: 
534: 
535: def coverage(rows: Sequence[Mapping[str, Any]]) -> dict:
536:     return {
537:         "actual_trades": len(rows),
538:         "executed": sum(r["context_role"] == "executed_saved_context" for r in rows),
539:         "posthoc": sum(r["context_role"] != "executed_saved_context" for r in rows),
540:         "unknown_reasons": dict(
541:             Counter(
542:                 (r.get("selected_snapshot") or {}).get("gamma", {}).get("status", "no snapshot")
543:                 for r in rows
544:                 if r.get("gamma_category") == "unknown"
545:             )
546:         ),
547:         "report_dates": sorted(
```

## source/quant_lab/scripts/ifvg_lab_mffu_views.py · 890–916

```text
890:                     + "; "
891:                     + trade["first_checkpoint_context_role"].replace("_", " ")
892:                 )
893:             st_module.write("Context role: " + trade["context_role"].replace("_", " "))
894:             if receipt:
895:                 st_module.write(
896:                     f"Recorded decision: {receipt.get('action')} · "
897:                     f"policy {receipt.get('policy')} · reasons "
898:                     f"{', '.join(receipt.get('reasons') or []) or 'none'}."
899:                 )
900:         st_module.caption(
901:             "Identical-price names share a line. Signed EOD exposure is supplied "
902:             "in vendor units; null range exposure is unavailable, never zero. "
903:             "No trading-signal arrows or inferred touch history are added."
904:         )
905:     return (
906:         gamma.level_segments(data, start=window[0], end=window[1], cursor=moment, names=names)
907:         if overlay and names
908:         else []
909:     )
910: 
911: 
912: _LEVEL_NAMES = (
913:     "HVL",
914:     "1D Max",
915:     "1D Min",
916:     "Call Resistance",
```

## Actual current standard configuration row

```json
{
  "configuration": "MCB062",
  "configuration_label": "MCB062",
  "status": "Completed",
  "reason": "",
  "axes": "{\"base_reference\": \"S1-T1-H14-P1-L-SO\", \"daily_cap\": \"U\", \"entry_context\": \"F0\", \"exit\": \"XP\", \"family\": \"geometry\", \"firm\": \"myfundedfutures\", \"geometry\": \"G10\", \"matched_fixed_partial_reference\": \"MCB025\", \"overhead\": \"O0\", \"planned_state\": \"not_implemented_or_run_by_this_handoff\", \"purpose\": \"Only opposing distance scales with 5%, 7.5%, or 10% of eligible implied move at parent lock\", \"same_cap_fixed_partial_reference\": \"MCB025\", \"schedule\": \"S1\", \"sizing\": \"Q10\", \"variant_id\": \"MCB062\"}",
  "Entry hours": "All open-market hours",
  "Direction": "Long only",
  "Higher-timeframe gap charts": "one-hour and four-hour",
  "Supporting (parent) charts": "one-minute, three-minute, five-minute, ten-minute, fifteen-minute and thirty-minute",
  "Largest distance from the parent gap to the opposing gap": "80 ticks (20 points)",
  "Smallest opposing gap": "1 tick (0.25 points)",
  "Profit target": "equal to the initial risk (1 to 1)",
  "Stop": "Beyond the setup's swing extreme plus 1 tick",
  "Trades per day": "one position at a time, no daily limit",
  "Daily close": "All positions closed by 3:55 PM Chicago on every trading day (earlier on shortened days). A trading day runs from the 5:00 PM Chicago reopen, so a position may stay open past midnight inside one trading day; none is held through the daily close, a closed market or a weekend",
  "Exit rule": "Half exits at the target (1R); the stop of the rest moves to the entry price and it is held to that stop or the daily close",
  "Position size": "10 x Micro E-mini Nasdaq-100 (MNQ) per trade, $0.514 per contract per fill",
  "Traded product": "E-mini Nasdaq-100 (NQ), 10 contracts",
  "Signal source": "E-mini Nasdaq-100 (NQ) one-minute candles (the strategy study's own data)",
  "Open-position marks and loss-limit checks": "E-mini Nasdaq-100 (NQ) recorded exchange trades",
  "Execution prices": "E-mini Nasdaq-100 (NQ) recorded exchange trades used as a proxy for micro fills; no micro trade data was used (disclosed limitation)"
}
```
