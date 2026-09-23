# Replicate the completed IFSM studies

**ParentDistance160 was the strongest observed tradeoff:** 58 valid executed trades, 12.4129R total net return and 0.2140R mean net expectancy. The corrected reference had 44 trades, 8.9122R total and 0.2025R mean. ParentDistance160 increased drawdown from 2.2148R to 3.1819R and still failed the five-traded-day underwater gate at seven days. No completed configuration passed every applicable research gate.

The 74-trade configuration had the highest frequency, but only 6.1853R total net return, 9.4973R drawdown and 34 underwater days. Reaching 70 did not produce an acceptable result. All results remain exploratory on the same inspected 107 evaluation dates.

## Set values in a new study

1. Open **IFVG Lab → My studies → New study**. Choose **Evaluate**, click **Configure study**, then continue to the existing **Configuration** step.
2. Select the fresh-entry static-1R baseline and edit the fields there. To reproduce ParentDistance160, choose parent retest **240 processed 1m bars**, opposing timeout **90 processed 1m bars**, and parent-to-HTF distance **160 ticks**. Keep HTF registry age **15 days** and other baseline values. The table below records the changes for every completed configuration. There is no home-screen replication component or preset picker.
3. In the existing Strategy gates and Validation steps, review the original gates, 107 evaluation dates, ten excluded warmup dates and seed **7**. The evidence catalog preserves the complete dates and settings; setting the strategy fields alone does not select the historical evaluation scope. Review the costs and exact study request before saving the normal study approval and clicking **Run study**.

Install the current pinned dependency and prepare its source checkout from the
repository folder, then start the research screen:

```powershell
python -m pip install -e ".[dev]"
python -m pip install --force-reinstall --no-deps "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@7c7111e398c083cf8e966e2e0c5aac8a41cc12c0"
python scripts/prepare_ifsm_research_core.py
python scripts/run_ifsm_research_ui.py
```

Open [the IFSM research UI](http://localhost:8502). This process uses the verified
current Core, the same exact commit as the ordinary Quant-Lab installation. It
keeps output in `data/ifsm_ui_replication/`. Restart an already running screen
after upgrading. `--check` performs a read-only runtime/configuration check
without starting the UI or creating a replay.

Preparation fetches the exact public commit recorded in `research/core/current.json`
into a verified checkout outside the repository. It does not fall back to the
older `Strategy-Core-daily-close` sibling. Original source manifests, bundles,
checkouts and study identities remain historical evidence. Reusing these settings
on the current engine creates a new replay, not the original execution; incompatible
checkpoints require rebuilding state from the authorized input history. See
[current Core setup](../research/core/README.md) for verification and custom locations.

Both runtimes use the same existing study Configuration screen. Optional lifecycle fields appear only when the loaded engine implements them. Saved unsupported values are refused rather than dropped. A new replay under current Quant-Lab code receives its own source-bound identity; the catalog's historical search IDs remain the identities of the original executions. Selecting or saving values does not launch a study.

## Exact shared settings

The starting named profile is **Doc-default (fresh entries, static 1R)**. To reproduce the corrected reference, set:

| Setting | Named profile default | Corrected reference |
|---|---|---|
| Parent retest staleness timeout | No timeout | 240 processed one-minute bars |
| Opposing-FVG timeout | No timeout | 90 processed one-minute bars |
| HTF registry age | 15 days | 15 days, explicitly preserved |

The parent-retest clock starts when a provisional parent is selected and restarts on replacement. The opposing clock starts when the parent retest locks the setup. These differ from whole-setup lifetime, which starts at the original HTF activation, spans S1–S4, never resets on replacement and does not affect open positions.

Keep the executable family **fresh continuation**, long-only, static 1R target, one-tick stop buffer, break-even disabled, and one active setup/position. The declared diagnostic family list also contains retests; this does not enable retest execution. Preserve full-fill/structural invalidation, chronological confirmed-after selection, and next-one-minute-bar resolution with stop-first same-bar precedence.

All saved IFSM entry sessions are enabled in `America/New_York`: Asia 16:00–01:45, London 02:00–07:00, NY 08:00–14:00, with an 18:00 trading-day boundary and 17:00–18:00 closed window. Keep waiting outside sessions. HTF timeframes are 1H/4H; parent timeframes are 3m/5m/10m/15m/30m. The parent reaction window is 40 parent-timeframe bars, inversion wait is unbounded and post-inversion entry expiry is 80 processed one-minute bars unless the recipe states otherwise.

Costs remain 0.514 NQ price points round turn, tick size 0.25 and $20/point. The original explicit date list and input bundle are in the catalog; no June 11 or sealed data is included. The unchanged gate minimum is 30 executions; the user's frequency objective is 70. Underwater duration must be at most five traded days. The remaining gate thresholds and full effective sections are preserved in the catalog.

## All completed configurations and their changes

Every change below is **on top of the corrected reference's shared settings**. `P` means `parent_htf_distance_ticks_max`, `O` means `opposing_parent_distance_ticks_max`, and `L` means `setup_timeout_1m_bars`. P/O are interval distance caps: 80 NQ ticks = 20 price points; 160 ticks = 40 points. They are not zone-width values. L is measured in processed one-minute bars and defaults to no limit.

| Configuration | Exact change from corrected | Trades | Total net R | Mean net R | Closed DD R | Underwater days |
|---|---|---:|---:|---:|---:|---:|
| Corrected reference | None | 44 | 8.9122 | 0.2025 | 2.2148 | 7 |
| Lifetime180 | L = 180 | 50 | 6.6516 | 0.1330 | 4.4269 | 13 |
| Lifetime240 | L = 240 | 53 | 5.5687 | 0.1051 | 5.4771 | 15 |
| ParentRetest60 | `parent_retest_timeout_1m_bars`: 240 → 60 | 41 | 7.9500 | 0.1939 | 2.1490 | 6 |
| ParentReaction20 | `parent_reaction_window_parent_bars`: 40 → 20 | 45 | 5.8405 | 0.1298 | 3.3983 | 12 |
| ParentDistance160 | P: 80 → 160 | 58 | 12.4129 | 0.2140 | 3.1819 | 7 |
| OpposingDistance160 | O: 80 → 160 | 48 | 8.8750 | 0.1849 | 3.2219 | 9 |
| Lifetime240_Parent160 | P = 160; L = 240 | 65 | 7.2523 | 0.1116 | 6.5082 | 16 |
| Lifetime180_Parent160 | P = 160; L = 180 | 62 | 8.3352 | 0.1344 | 5.4581 | 14 |
| Parent160_Opposing160 | P = 160; O = 160 | 66 | 12.3266 | 0.1868 | 6.2070 | 11 |
| Lifetime240_Parent160_Opposing160 | P = 160; O = 160; L = 240 | 74 | 6.1853 | 0.0836 | 9.4973 | 34 |
| PreserveSelected | P = 160; `parent_replacement_policy` = `preserve_selected` | 29 | 0.0781 | 0.0027 | 5.3344 | 16 |
| ShallowFirstRetest | P = 160; `parent_retest_depth_policy` = `strictly_before_ce` | 56 | 10.4181 | 0.1860 | 3.3484 | 11 |

The default parent replacement policy is `highest_tf_newest`; the default first retest policy is `any_live_touch`. PreserveSelected keeps an eligible selected provisional parent, retaining its original retest clock; ignored arrivals are not saved as fallbacks. ShallowFirstRetest rejects and clears a parent if its first later live wick touch reaches the midpoint or deeper (`2 × penetration >= width`). A shallower touch locks immediately. It neither delays the lock nor changes the risk anchor.

## Which changes helped?

- **Parent distance 160** improved frequency, total return, mean expectancy, profit factor and occupation versus corrected. After accounting for displaced trades, the additional net contribution was +3.5007R. Drawdown increased and the underwater gate still failed.
- **Parent retest 60** slightly reduced drawdown, underwater duration and occupation, while losing trades and total return. Its six underwater days still failed the gate.
- **Lifetime 180/240** freed the setup slot and added trades, but weakened total return, expectancy and drawdown. The 180 versions compared better than the 240 versions within the tested pairs; neither justified a quality claim. Adding lifetime240 to both distance changes added eight trades—one winner and seven losers—for −6.1413R.
- **Reaction window 20** reduced occupation and added one trade, with worse performance. **Opposing distance 160** added four trades but slightly reduced total net return and worsened the main quality measures.
- **Both distances 160** added eight trades relative to parent distance 160 alone, while reducing total return by 0.0863R and increasing drawdown from 3.1819R to 6.2070R.
- **PreserveSelected** and **ShallowFirstRetest** did not improve their ParentDistance160 parent. Their total return fell by 12.3348R and 1.9949R respectively. ShallowFirstRetest reduced rapid pre-opposing parent failures, but lost more profitable executions than it added and increased full-sequence occupation.

All 13 configurations passed the existing execution correctness checks and failed the underwater gate. PreserveSelected also failed the minimum execution count, profit-factor and session-stability gates. Every expectancy confidence interval included zero. These are inspected-period results; no configuration is approved for activation.

## Evidence and identities

The self-contained [recipe catalog](../src/alpha_lab/agents/data_infra/ifvg/ifsm_replay_recipes.json) contains all original sections, exact field differences, 107+10 dates, costs, gates, seeds, original search/Core/source identities, metrics and source-artifact SHA-256 receipts. Historical metadata fields are excluded from UI configuration comparisons, while their original values remain preserved.

The [verified inventory report](../reports/ifsm_ui_replication_20260909/VERIFIED_CONFIGURATIONS.md) adds entry-day counts, profit factor, occupation and detailed replication notes. Its builder used supported verified readers and launched no studies. The two repaired-source reference replays matched all 108 preserved economic and geometry fields of the original 44/58-trade references.

## One-hour / four-hour gap choice (September 18, 2026)

The normal launch command above loads the verified daily-close development Core,
which retains the original selectable wick rule and existing lifecycle settings.
In Configuration, beside the selection cap, choose One-hour / four-hour gap
invalidation. The own-timeframe closing option requires a later finalized candle
strictly beyond its far boundary. Restart the study screen after updating. See
IFVG_GAP_INVALIDATION_CHOICE.md for exact semantics and evidence.
