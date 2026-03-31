import type { OHLCVBar, IndicatorPoint } from "../types/chart";

/**
 * Compute Exponential Moving Average.
 * Uses standard formula: EMA_t = price * alpha + EMA_{t-1} * (1 - alpha)
 * where alpha = 2 / (period + 1).
 */
export function computeEMA(bars: OHLCVBar[], period: number): IndicatorPoint[] {
  if (bars.length === 0 || period < 1) return [];

  const alpha = 2 / (period + 1);
  const result: IndicatorPoint[] = [];

  // Seed with SMA of first `period` bars
  if (bars.length < period) {
    // Not enough bars for a full SMA seed — use what we have
    let sum = 0;
    for (let i = 0; i < bars.length; i++) {
      sum += bars[i]!.close;
    }
    const sma = sum / bars.length;
    result.push({ time: bars[bars.length - 1]!.time, value: sma });
    return result;
  }

  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += bars[i]!.close;
  }
  let ema = sum / period;
  result.push({ time: bars[period - 1]!.time, value: ema });

  for (let i = period; i < bars.length; i++) {
    ema = bars[i]!.close * alpha + ema * (1 - alpha);
    result.push({ time: bars[i]!.time, value: ema });
  }

  return result;
}

/**
 * Compute Kaufman Adaptive Moving Average (KAMA).
 *
 * KAMA adapts its smoothing constant based on the efficiency ratio:
 *   ER = direction / volatility
 *   SC = (ER * (fast_alpha - slow_alpha) + slow_alpha)^2
 *
 * Parameters:
 *   period — lookback for ER calculation (default 10)
 *   fastPeriod — fast EMA period (default 2)
 *   slowPeriod — slow EMA period (default 30)
 */
export function computeKAMA(
  bars: OHLCVBar[],
  period = 10,
  fastPeriod = 2,
  slowPeriod = 30,
): IndicatorPoint[] {
  if (bars.length <= period) return [];

  const fastAlpha = 2 / (fastPeriod + 1);
  const slowAlpha = 2 / (slowPeriod + 1);
  const result: IndicatorPoint[] = [];

  // Seed KAMA with the close at index `period`
  let kama = bars[period]!.close;
  result.push({ time: bars[period]!.time, value: kama });

  for (let i = period + 1; i < bars.length; i++) {
    // Direction: absolute price change over `period` bars
    const direction = Math.abs(bars[i]!.close - bars[i - period]!.close);

    // Volatility: sum of absolute bar-to-bar changes over `period` bars
    let volatility = 0;
    for (let j = i - period + 1; j <= i; j++) {
      volatility += Math.abs(bars[j]!.close - bars[j - 1]!.close);
    }

    // Efficiency ratio (0 to 1)
    const er = volatility === 0 ? 0 : direction / volatility;

    // Smoothing constant
    const sc = (er * (fastAlpha - slowAlpha) + slowAlpha) ** 2;

    // KAMA update
    kama = kama + sc * (bars[i]!.close - kama);
    result.push({ time: bars[i]!.time, value: kama });
  }

  return result;
}

/**
 * Compute Session-Anchored VWAP.
 *
 * VWAP resets at each new session. A session boundary is detected when
 * the bar's date (UTC) changes.
 *
 * Formula: cumulative(typical_price * volume) / cumulative(volume)
 * where typical_price = (high + low + close) / 3
 */
export function computeVWAP(bars: OHLCVBar[]): IndicatorPoint[] {
  if (bars.length === 0) return [];

  const result: IndicatorPoint[] = [];
  let cumTPV = 0; // cumulative typical_price * volume
  let cumVol = 0; // cumulative volume
  let currentDate = getDateFromTimestamp(bars[0]!.time);

  for (const bar of bars) {
    const barDate = getDateFromTimestamp(bar.time);

    // New session — reset accumulators
    if (barDate !== currentDate) {
      cumTPV = 0;
      cumVol = 0;
      currentDate = barDate;
    }

    const typicalPrice = (bar.high + bar.low + bar.close) / 3;
    cumTPV += typicalPrice * bar.volume;
    cumVol += bar.volume;

    const vwap = cumVol === 0 ? bar.close : cumTPV / cumVol;
    result.push({ time: bar.time, value: vwap });
  }

  return result;
}

/** Extract YYYY-MM-DD string from a Unix timestamp (seconds). */
function getDateFromTimestamp(timestamp: number): string {
  const d = new Date(timestamp * 1000);
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(d.getUTCDate()).padStart(2, "0")}`;
}
