# One-hour / four-hour gap invalidation

As of September 22, 2026, ordinary Quant-Lab and the IFSM study application use
the same Core commit, `7c7111e398c083cf8e966e2e0c5aac8a41cc12c0`, recorded in
`pyproject.toml` and `research/core/current.json`. Both gap-invalidation choices
are available in this engine. Follow [current Core setup](../research/core/README.md)
to update the installed package, prepare its verified external source checkout
and start `python scripts/run_ifsm_research_ui.py`. Preparation fetches the
current public commit; the launcher does not fall back to an older sibling Core.
Restart existing study screens and workers after upgrading. Incompatible saved-job
resumes are refused and their progress is preserved. Trade-Lab's dependency is
outside this upgrade.

The original September 18 research used an isolated process-local Core and kept
the ordinary installed pin unchanged. That installation arrangement is historical
and superseded by the current shared pin. Original source identities, archived
checkouts, approvals and study results remain immutable; the upgrade does not
change an old configuration's selected invalidation policy.

In **My studies → New study → Evaluate → Configuration**, beside the larger-chart
selection limit, **One-hour / four-hour gap invalidation** offers:

- **One-minute wick reaches the far edge (original rule)**
- **Candle closes beyond the gap on its own timeframe**

The field is `htf_gap_invalidation_policy`. Its values are
`execution_wick_full_fill_v1` and `own_timeframe_close_v1`. Missing fields in old
configurations resolve to the original rule. Selecting a single fixed policy
saves one economic configuration; comparing both requires explicitly selecting
both. Drafts, resolved sections, approvals and workers retain the selected value.

Opening, refreshing or saving a draft without a change never adds this field
(September 23, 2026 repair; earlier code added it when some legacy Evaluate
drafts were opened). A configuration saved without it keeps resolving to the
original rule, and choosing the rule already shown is not a change. The shown
rule is written into the configuration explicitly only when the owner changes
one of its fixed settings.

## Price, time and role contract

An upward starting gap becomes invalid only when a later finalized candle on
its own timeframe closes strictly below its lower boundary. A downward gap
requires a close strictly above its upper boundary. Equality, a wick, a
one-minute close or a one-hour close against a four-hour gap is insufficient.
No previous overlap, candle color or extra confirmation is required.

The existing aggregation and trading-day bar anchor remain unchanged. The
source candle's logical availability must be strictly later than formation and
no later than the actual decision cursor. A completed close is applied before
new setup processing. Pending bars retain their original availability across
closed intervals and day-seed checkpoints.

`COMPLETE` with complete/nonpartial flags represents a final candle.
`END_OF_DAY` alone does not: the builder also uses it for arbitrary data tails.
A tail is final only when the governing exchange schedule proves that the next
minute after its last observed minute is closed, and all remaining minutes
through its logical close are closed. Missing eligible observations do not prove finality. The
unchanged logical availability still controls delivery.

Physical wick touches, penetration and full traversal remain recorded even
when the gap stays valid. Policy validity controls starting-gap discovery and
selection. Age eviction still removes discovery eligibility; an active selected
gap continues to receive own-timeframe validity checks. Price invalidation is
permanent. Smaller-pattern fills, inversion rules, stops, targets, timeouts,
selection caps, ranking and direction rules keep their existing meanings.

## Evidence and compatibility

The audit companion adds `gap_validity_events.parquet`: exact source and
decision candles, gap geometry, selected policy, physical traversal, policy
validity before/after, finalization and active setup links. A separate source
verifier checks retained events without calling the engine's validity helper.
Export verification additionally reconstructs the applicable closing checks.
`invalidated_htf_own_timeframe_close` is a distinct setup terminal reason and
participates in funnel reconciliation and chart range selection.

Policy state is part of day seeds and source-bound replay identities. Legacy
wick seeds cannot seed a close-based replay because wick-deleted gaps are
missing. Historical results and original source checkouts remain immutable.
Changing policy or source requires a separate authorized study and a rebuilt
chronological chain where compatible complete state cannot be proved.

The bounded acceptance question uses exact sequence-supply run
`dbfc0b964d32e4eed1a24503b054c8f31d5746f8e49459b5c2cec43ea762a3f3`, D2, and
input bundle `f3443d0de5c153da9b0ca8dffeb243b700887947fab24d8267a86a8f830d0816`.
Only the invalidation policy changes between two profiles. The same 107
evaluation dates, ten warmup dates and final cutoff of June 10, 2026,
4:00 PM America/Chicago apply. No new inputs, holdout access, fitting or live
promotion is included. The light peer-review deliverable is
`reports/IFVG_Gap_Invalidation_Audit_Light/`. The original working evidence was
relocated to
`../Claude-Quant-Lab-Research-Artifacts/archived-reports/ifvg_gap_invalidation_20260918/`;
the former `reports/ifvg_gap_invalidation_20260918/` path is historical. Keep new
working stores, raw dumps and verification outputs outside the repository, as
described in [the report delivery policy](../AGENTS.md#report-delivery-and-working-artifacts-2026-09-22).
