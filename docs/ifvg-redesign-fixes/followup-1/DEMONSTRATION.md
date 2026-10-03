# Setup timing: before and at completion

Trade review's "How the setup formed" card, its setup-chart key and markers, and its moment picker, as produced by the application's own functions on the delivered source (tree `76bb3b0c…`, run from an isolated copy) and on the corrected source. All times are Chicago. Nothing was re-run or saved. Section 1 uses a hand-built record, the test suite's synthetic `_record` fixture, with synthetic one-minute candles for the chart markers. Section 2 reads the saved setup records of the saved funded study.

## 1. The owner's boundary case (synthetic record)

- The record is this configuration's own. Its tap candle opens at 5:18 PM and closes at 5:19 PM, and its parent gap is confirmed at 7:00 PM.
- The **opposing gap is confirmed at 7:05 PM**, the same minute the **close-through candle opens**. That candle **closes at 7:06 PM**, and the entry is at 7:07 PM.
- The moment picker offers these setup moments before the entry, unchanged by the correction:

  - 5:19 PM · the 5:18 PM candle has tapped the four-hour gap
  - 7:00 PM · the parent gap has formed
  - 7:05 PM · the opposing gap has formed
  - 7:06 PM · the 7:05 PM candle has closed through the opposing gap

### At the selectable 7:05 PM moment: before completion

Before the correction, the text already said a candle had closed through the opposing gap, but the key and the chart had not reached that point. After it, all three agree: the close-through is not known yet.

#### 7:05 PM

**Before the correction** (delivered tree `76bb3b0c…`):

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | Four-hour gap, 24,451.50–25,086.75 | Jan 6, 12:00 AM |
| 2 | Price taps into it | Apr 12, 5:18 PM |
| 3 | Five-minute parent gap forms | 7:00 PM |
| 4 | Opposing 1-minute gap, 3 ticks, inside the 80-tick limit | 7:05 PM |
| 5 | A candle closes through it | 7:05 PM |

Setup key: 1. Five-minute parent gap, 24,937.25–24,966.50, confirmed 7:00 PM; 2. Opposing one-minute gap, 24,961.75–24,962.50, confirmed 7:05 PM
Setup chart markers: 1, 2

**After the correction:**

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | Four-hour gap, 24,451.50–25,086.75 | Jan 6, 12:00 AM |
| 2 | The 5:18 PM candle taps into it | Apr 12, 5:19 PM |
| 3 | Five-minute parent gap forms | 7:00 PM |
| 4 | Opposing 1-minute gap, 3 ticks, inside the 80-tick limit | 7:05 PM |

Setup key: 1. Five-minute parent gap, 24,937.25–24,966.50, confirmed 7:00 PM; 2. Opposing one-minute gap, 24,961.75–24,962.50, confirmed 7:05 PM
Setup chart markers: 1, 2

### At 7:06 PM: at completion

The close-through appears when its candle closes. The candle keeps its 7:05 PM name, and the time shown is 7:06 PM, when it became known.

#### 7:06 PM

**Before the correction** (delivered tree `76bb3b0c…`):

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | Four-hour gap, 24,451.50–25,086.75 | Jan 6, 12:00 AM |
| 2 | Price taps into it | Apr 12, 5:18 PM |
| 3 | Five-minute parent gap forms | 7:00 PM |
| 4 | Opposing 1-minute gap, 3 ticks, inside the 80-tick limit | 7:05 PM |
| 5 | A candle closes through it | 7:05 PM |

Setup key: 1. Five-minute parent gap, 24,937.25–24,966.50, confirmed 7:00 PM; 2. Opposing one-minute gap, 24,961.75–24,962.50, confirmed 7:05 PM; 3. The 7:05 PM candle closes through it
Setup chart markers: 1, 2, 3

**After the correction:**

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | Four-hour gap, 24,451.50–25,086.75 | Jan 6, 12:00 AM |
| 2 | The 5:18 PM candle taps into it | Apr 12, 5:19 PM |
| 3 | Five-minute parent gap forms | 7:00 PM |
| 4 | Opposing 1-minute gap, 3 ticks, inside the 80-tick limit | 7:05 PM |
| 5 | The 7:05 PM candle closes through it | 7:06 PM |

Setup key: 1. Five-minute parent gap, 24,937.25–24,966.50, confirmed 7:00 PM; 2. Opposing one-minute gap, 24,961.75–24,962.50, confirmed 7:05 PM; 3. The 7:05 PM candle closes through it
Setup chart markers: 1, 2, 3

### Full history

The time next to the tap and the close-through is now their candle's close (5:19 PM and 7:06 PM), not the candle's opening minute.

#### Full history

**Before the correction** (delivered tree `76bb3b0c…`):

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | Four-hour gap, 24,451.50–25,086.75 | Jan 6, 12:00 AM |
| 2 | Price taps into it | Apr 12, 5:18 PM |
| 3 | Five-minute parent gap forms | 7:00 PM |
| 4 | Opposing 1-minute gap, 3 ticks, inside the 80-tick limit | 7:05 PM |
| 5 | A candle closes through it | 7:05 PM |
| 6 | Entry | 7:07 PM |

Setup key: 1. Five-minute parent gap, 24,937.25–24,966.50, confirmed 7:00 PM; 2. Opposing one-minute gap, 24,961.75–24,962.50, confirmed 7:05 PM; 3. The 7:05 PM candle closes through it; 4. Entry 24,971.00 at 7:07 PM · initial stop 24,949.75; 5. Half out at 24,992.25 (1R), 7:10 PM · stop on the rest moves to entry

**After the correction:**

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | Four-hour gap, 24,451.50–25,086.75 | Jan 6, 12:00 AM |
| 2 | The 5:18 PM candle taps into it | Apr 12, 5:19 PM |
| 3 | Five-minute parent gap forms | 7:00 PM |
| 4 | Opposing 1-minute gap, 3 ticks, inside the 80-tick limit | 7:05 PM |
| 5 | The 7:05 PM candle closes through it | 7:06 PM |
| 6 | Entry | 7:07 PM |

Setup key: 1. Five-minute parent gap, 24,937.25–24,966.50, confirmed 7:00 PM; 2. Opposing one-minute gap, 24,961.75–24,962.50, confirmed 7:05 PM; 3. The 7:05 PM candle closes through it; 4. Entry 24,971.00 at 7:07 PM · initial stop 24,949.75; 5. Half out at 24,992.25 (1R), 7:10 PM · stop on the rest moves to entry

## 2. The same boundary in the saved study

This check covers every funded trade of the configurations that are members of the verified strategy study, the ones with their own setup records, at both firms.

- **Checked:** 2 member configurations and 374 funded trades, all with their own records.
- **Boundary trades:** 84 of them have the opposing gap confirmed at the minute the close-through candle opens, before the entry. That is 41 at TakeProfitTrader, 43 at MyFundedFutures.
- **Before the correction:** the moment picker offered that minute on 84 of the 84 trades. There the text said a candle had closed through on 84 of them, while the key listed it on 0.
- **After it:** the minute is still offered on 84. The text shows the close-through there on 0, and the key on 0.

The first of them is configuration S0_D80_W1_P1 at MyFundedFutures, Account 1, entry Jan 15, 2:22 AM. Its opposing gap is confirmed at Jan 15, 2:20 AM, and its close-through candle opens Jan 15, 2:20 AM, closes Jan 15, 2:21 AM. The moment picker offers 2:20 AM: yes.

#### At 2:20 AM (the opposing gap's confirmation)

**Before the correction** (delivered tree `76bb3b0c…`):

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | One-hour gap, 25,647.50–25,677.75 | Jan 15, 2:00 AM |
| 2 | Price taps into it | Jan 15, 2:02 AM |
| 3 | Five-minute parent gap forms | 2:15 AM |
| 4 | Opposing 1-minute gap, 2 ticks, inside the 80-tick limit | 2:20 AM |
| 5 | A candle closes through it | 2:20 AM |

Setup key: 1. Five-minute parent gap, 25,696.25–25,717.75, confirmed 2:15 AM; 2. Opposing one-minute gap, 25,714.50–25,715.00, confirmed 2:20 AM

**After the correction:**

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | One-hour gap, 25,647.50–25,677.75 | Jan 15, 2:00 AM |
| 2 | The 2:02 AM candle taps into it | Jan 15, 2:03 AM |
| 3 | Five-minute parent gap forms | 2:15 AM |
| 4 | Opposing 1-minute gap, 2 ticks, inside the 80-tick limit | 2:20 AM |

Setup key: 1. Five-minute parent gap, 25,696.25–25,717.75, confirmed 2:15 AM; 2. Opposing one-minute gap, 25,714.50–25,715.00, confirmed 2:20 AM

#### At 2:21 AM (the close-through candle's close)

**Before the correction** (delivered tree `76bb3b0c…`):

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | One-hour gap, 25,647.50–25,677.75 | Jan 15, 2:00 AM |
| 2 | Price taps into it | Jan 15, 2:02 AM |
| 3 | Five-minute parent gap forms | 2:15 AM |
| 4 | Opposing 1-minute gap, 2 ticks, inside the 80-tick limit | 2:20 AM |
| 5 | A candle closes through it | 2:20 AM |

Setup key: 1. Five-minute parent gap, 25,696.25–25,717.75, confirmed 2:15 AM; 2. Opposing one-minute gap, 25,714.50–25,715.00, confirmed 2:20 AM; 3. The 2:20 AM candle closes through it

**After the correction:**

| # | How the setup formed | Time shown |
|---|---|---|
| 1 | One-hour gap, 25,647.50–25,677.75 | Jan 15, 2:00 AM |
| 2 | The 2:02 AM candle taps into it | Jan 15, 2:03 AM |
| 3 | Five-minute parent gap forms | 2:15 AM |
| 4 | Opposing 1-minute gap, 2 ticks, inside the 80-tick limit | 2:20 AM |
| 5 | The 2:20 AM candle closes through it | 2:21 AM |

Setup key: 1. Five-minute parent gap, 25,696.25–25,717.75, confirmed 2:15 AM; 2. Opposing one-minute gap, 25,714.50–25,715.00, confirmed 2:20 AM; 3. The 2:20 AM candle closes through it

The leader configuration (S1-T1-H14-P1-L-SO) is not a member. Its trades show related context, which offers no setup moments, so the boundary could not be selected there. Its full-history times still changed (5:19 PM and 7:06 PM on the April 12 trade).

