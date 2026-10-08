# Consolidated bounded follow-up — b8db repair

This is review feedback for the owner to send; it does not itself authorize a new study or broaden the repair.

The independent review verified corrected financial arithmetic, the 28 changed histories/26 cash totals, all 43,130 gamma checkpoint values under the nominal policy and 3,754 geometry rows. The reviewer reran 121 portable Core tests (8 external parity cases deselected). Main discovery and registered-input/gamma/lens integration are materially implemented. Preserve b8db and original 7a8c records and all completed work.

## B8-01 — checkpoint-specific gamma origin

Inspect `mffu_gamma.py` build_gamma's first-checkpoint role, selected_rows/coverage/checkpoint_cards and `ifvg_lab_mffu_views.py` render_trade_gamma. The exact delivered source body repro is in verification/probe_reporting.py with results CHECKPOINT_PROVENANCE_PROBE.json.

Actual example MCB003 trade e2189e7d-1151-541d-a60e-0f1e992b587a: observation 1750068615558271393 nanoseconds vs receipt datetime 2025-06-16T10:10:15.558271+00:00. The 393-nanosecond representation difference incorrectly drives executed-vs-annotation classification even with a matched saved receipt. All 1,908 funded conditional target_decision records have this precision distinction (3–999 nanoseconds), while their decision_ns and exported checkpoint timestamp agree. Do not round actual timestamps backward or invent a receipt to make equality pass.

Link the actual saved target decision/fill; keep receipt context time, observation time and reporting evaluation time separate. Validate source context and causal event membership. Decide provenance per checkpoint; entry context_role must not be printed as first-target origin. Keep pure reporting annotations honestly labeled, including unconditional XP and reused control enrichment. Keep funded print and ordinary candle timing contracts distinct.

Focused regression expectations: exact-ns whole-target and partial-target recorded decisions; a genuine annotation-only XP target; an annotated reused control; before/at the first target cursor; report-eligibility boundary; mismatch must fail or remain explicitly unavailable, never fabricated. Numeric context groups, posted fees, trade and cash histories must remain unchanged unless a separately demonstrated error requires another decision.

## B8-02 — finish standard export labels centrally

The NEW corrected standard `configurations.csv` still calls all 64 micro quantities E-mini NQ and all 6 dynamic allowances fixed 80 ticks / 20 points. The generated top-20 cash PNG also retains near-identical policy labels and generic conditional whole-exit wording. Integrated UI descriptions are improved; use the same definition source for standard exports.

Show micro exposure separately from the NQ-price proxy, dynamic formula/frozen timing separately from fixed fallback, and ID plus distinctive schedule/exit/geometry in chart labels. Preserve original archives and publish a new report/export version. Verify all 64 rows, all 6 geometry rows and conditional exits, not only the leader. Reconcile the unchanged money and check representative rendered output.

## Completion and bounded validation

Do not rerun the 64 financial histories, mutate b8db economics, change Core rules, reopen fee rounding or create new ML/geometry hypotheses for these reporting fixes. Use local focused tests and a small normal-dashboard proof. Keep source/result/report identities distinct and publish new review artifacts with concise before/after evidence. No new full-suite request is implied by this review. Report precise limitations rather than widening verification into an unrelated project.

The review environment's missing PyArrow prevented portable Quant collection; the source package lists it as a required dependency. That is not a new application repair request. No additional market upload is needed for these two findings.
