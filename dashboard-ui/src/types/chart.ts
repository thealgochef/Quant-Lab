/** Chart-related types. */

export interface OHLCVBar {
  time: number; // Unix timestamp in seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorPoint {
  time: number;
  value: number;
}

export type Timeframe = "987t" | "2000t";

export interface TimeframeConfig {
  label: string;
  tickCount: number;
  priceRange: number; // ± points from current price for fixed Y-axis
  barWidth: number; // pixels per candle
}
