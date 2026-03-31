import { describe, it, expect } from "vitest";
import { computeEMA, computeKAMA, computeVWAP } from "../../src/utils/indicators";
import type { OHLCVBar } from "../../src/types/chart";

/** Helper: create a bar with close price and default O/H/L/V. */
function bar(time: number, close: number, volume = 100): OHLCVBar {
  return { time, open: close, high: close + 1, low: close - 1, close, volume };
}

describe("computeEMA", () => {
  it("computes EMA(13) matching expected values for a known price series", () => {
    // 20 bars with linearly increasing close prices (100, 101, ..., 119)
    const bars = Array.from({ length: 20 }, (_, i) => bar(1000 + i * 60, 100 + i));

    const ema = computeEMA(bars, 13);

    // EMA starts at index 12 (13th bar), seeded with SMA of first 13 closes
    expect(ema.length).toBe(8); // 20 - 13 + 1 = 8 points

    // First point = SMA of bars 0-12: (100+101+...+112) / 13 = 106
    expect(ema[0]!.value).toBeCloseTo(106, 4);
    expect(ema[0]!.time).toBe(1000 + 12 * 60);

    // Verify EMA trending upward (since prices are rising)
    for (let i = 1; i < ema.length; i++) {
      expect(ema[i]!.value).toBeGreaterThan(ema[i - 1]!.value);
    }

    // Verify last EMA is between SMA and last price (EMA lags)
    const lastPrice = 119;
    expect(ema[ema.length - 1]!.value).toBeLessThan(lastPrice);
    expect(ema[ema.length - 1]!.value).toBeGreaterThan(106);

    // Verify alpha = 2 / (13 + 1) = 0.142857
    // Second point: EMA = 113 * alpha + 106 * (1 - alpha) = 107.0
    const alpha = 2 / 14;
    const expected2 = 113 * alpha + 106 * (1 - alpha);
    expect(ema[1]!.value).toBeCloseTo(expected2, 4);
  });
});

describe("computeVWAP", () => {
  it("anchors to session open and resets on new session", () => {
    // Session 1: 3 bars on 2026-03-02 (timestamps in UTC)
    const session1Start = Math.floor(new Date("2026-03-02T14:00:00Z").getTime() / 1000);
    // Session 2: 2 bars on 2026-03-03
    const session2Start = Math.floor(new Date("2026-03-03T14:00:00Z").getTime() / 1000);

    const bars: OHLCVBar[] = [
      { time: session1Start, open: 100, high: 102, low: 98, close: 101, volume: 1000 },
      { time: session1Start + 60, open: 101, high: 103, low: 99, close: 102, volume: 500 },
      { time: session1Start + 120, open: 102, high: 110, low: 100, close: 108, volume: 2000 },
      // New session (different date)
      { time: session2Start, open: 200, high: 205, low: 195, close: 202, volume: 1000 },
      { time: session2Start + 60, open: 202, high: 210, low: 198, close: 205, volume: 800 },
    ];

    const vwap = computeVWAP(bars);
    expect(vwap.length).toBe(5);

    // First bar VWAP = typical price = (102 + 98 + 101) / 3 = 100.333...
    const tp0 = (102 + 98 + 101) / 3;
    expect(vwap[0]!.value).toBeCloseTo(tp0, 4);

    // Second bar: cumulative VWAP
    const tp1 = (103 + 99 + 102) / 3;
    const expectedVwap1 = (tp0 * 1000 + tp1 * 500) / 1500;
    expect(vwap[1]!.value).toBeCloseTo(expectedVwap1, 4);

    // Third bar: still session 1
    const tp2 = (110 + 100 + 108) / 3;
    const expectedVwap2 = (tp0 * 1000 + tp1 * 500 + tp2 * 2000) / 3500;
    expect(vwap[2]!.value).toBeCloseTo(expectedVwap2, 4);

    // Fourth bar: session 2 starts — VWAP resets
    const tp3 = (205 + 195 + 202) / 3;
    expect(vwap[3]!.value).toBeCloseTo(tp3, 4);

    // Fifth bar: session 2 continues
    const tp4 = (210 + 198 + 205) / 3;
    const expectedVwap4 = (tp3 * 1000 + tp4 * 800) / 1800;
    expect(vwap[4]!.value).toBeCloseTo(expectedVwap4, 4);
  });
});

describe("computeKAMA", () => {
  it("adapts smoothing to volatility — trending vs choppy", () => {
    // Trending series: price goes 100, 101, 102, ..., 120 (smooth uptrend)
    const trendBars = Array.from({ length: 21 }, (_, i) => bar(1000 + i * 60, 100 + i));

    // Choppy series: price oscillates ±5 but stays near 100
    const choppyBars = Array.from({ length: 21 }, (_, i) =>
      bar(2000 + i * 60, 100 + (i % 2 === 0 ? 5 : -5)),
    );

    const kamaTrend = computeKAMA(trendBars, 10, 2, 30);
    const kamaChoppy = computeKAMA(choppyBars, 10, 2, 30);

    // Both should produce points starting at index `period` (21 - 10 = 11 points)
    expect(kamaTrend.length).toBe(11);
    expect(kamaChoppy.length).toBe(11);

    // In trending market: KAMA tracks price closely
    // Last trending price = 120, KAMA should be close to it
    const lastKamaTrend = kamaTrend[kamaTrend.length - 1]!.value;
    expect(lastKamaTrend).toBeGreaterThan(115); // Close to 120

    // In choppy market: KAMA barely moves from seed
    // KAMA seed = bars[10].close = 100 + 10 = 110 → wait, choppy seed
    // Choppy: bars[10].close = 100 + 5 = 105 (even index)
    // KAMA should stay near that since ER ≈ 0
    const lastKamaChoppy = kamaChoppy[kamaChoppy.length - 1]!.value;
    // Choppy KAMA moves much less than trending KAMA
    const trendMovement = Math.abs(lastKamaTrend - kamaTrend[0]!.value);
    const choppyMovement = Math.abs(lastKamaChoppy - kamaChoppy[0]!.value);
    expect(trendMovement).toBeGreaterThan(choppyMovement * 2);
  });
});
