import type { Timeframe, TimeframeConfig } from "../types/chart";

/** Color palette — matches Phase 6 spec. */
export const COLORS = {
  darkest: "#0f0f1a",
  panel: "#1a1a2e",
  hover: "#25253d",
  primary: "#e0e0e0",
  secondary: "#8888aa",
  green: "#00c853",
  red: "#ff1744",
  orange: "#ff9100",
  blue: "#2979ff",
  yellow: "#ffd600",

  // Chart specific
  candleUp: "#00c853",
  candleDown: "#ff1744",
  ema13: "#2979ff",
  ema48: "#ff9100",
  ema200: "#8888aa",
  kama: "#ffd600",
  vwap: "#ba68c8",
} as const;

/** Timeframe configurations for tick-bar charts. */
export const TIMEFRAME_CONFIGS: Record<Timeframe, TimeframeConfig> = {
  "987t": { label: "987t", tickCount: 987, priceRange: 75, barWidth: 6 },
  "2000t": { label: "2000t", tickCount: 2000, priceRange: 100, barWidth: 8 },
};

export const ALL_TIMEFRAMES: Timeframe[] = ["987t", "2000t"];

/** Overlay keys matching backend overlay_config. */
export const OVERLAY_LABELS: Record<string, string> = {
  ema_13: "EMA 13",
  ema_48: "EMA 48",
  ema_200: "EMA 200",
  vwap: "VWAP",
  levels: "Levels",
  predictions: "Predictions",
};

/** Prediction class → marker color mapping. */
export const PREDICTION_COLORS: Record<string, string> = {
  tradeable_reversal: COLORS.green,
  trap_reversal: COLORS.orange,
  aggressive_blowthrough: COLORS.red,
};
