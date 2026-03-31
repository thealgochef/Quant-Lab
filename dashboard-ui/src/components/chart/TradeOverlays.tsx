/**
 * TradeOverlays — renders ALL chart markers and TP/SL price lines.
 *
 * Merges prediction markers, entry markers, and exit markers into a
 * single setMarkers() call (lightweight-charts only supports one marker
 * set per series). Also draws TP/SL price lines for open positions.
 *
 * Pure logic component (no DOM).
 */
import { useEffect, useRef } from "react";
import type {
  ISeriesApi,
  SeriesMarker,
  IPriceLine,
  Time,
  UTCTimestamp,
} from "lightweight-charts";
import { useConfigStore } from "../../stores/configStore";
import { useTradingStore } from "../../stores/tradingStore";
import { COLORS, PREDICTION_COLORS } from "../../utils/constants";

interface Props {
  candlestickSeries: ISeriesApi<"Candlestick"> | null;
}

// Exit reason → marker config
const EXIT_CONFIG: Record<string, { color: string; text: string }> = {
  tp: { color: COLORS.green, text: "TP" },
  sl: { color: COLORS.red, text: "SL" },
  flatten: { color: COLORS.yellow, text: "FLAT" },
  manual: { color: COLORS.blue, text: "MAN" },
  dll: { color: COLORS.red, text: "DLL" },
  blown: { color: COLORS.red, text: "BLOWN" },
};

// Prediction class → short label
const PRED_LABELS: Record<string, string> = {
  tradeable_reversal: "REV",
  trap_reversal: "TRAP",
  aggressive_blowthrough: "BLOW",
};

export default function TradeOverlays({ candlestickSeries }: Props) {
  const openPositions = useTradingStore((s) => s.openPositions);
  const todaysTrades = useTradingStore((s) => s.todaysTrades);
  const todaysPredictions = useTradingStore((s) => s.todaysPredictions);
  const overlays = useConfigStore((s) => s.overlays);
  const tpSlLinesRef = useRef<IPriceLine[]>([]);

  // ── All markers: predictions + entries + exits ─────────────
  useEffect(() => {
    if (!candlestickSeries) return;

    const markers: SeriesMarker<Time>[] = [];

    // Prediction markers (moved from ChartOverlays)
    if (overlays["predictions"]) {
      for (const p of todaysPredictions) {
        if (!p.timestamp || !p.predicted_class) continue;
        const predClass = p.predicted_class as string;
        const direction = p.trade_direction as string;
        const isLong = direction === "long";
        const isResolved = p.prediction_correct != null;
        const isCorrect = p.prediction_correct as boolean | undefined;

        // Color: use outcome color if resolved, else prediction class color
        let color: string;
        if (isResolved) {
          color = isCorrect ? COLORS.green : COLORS.red;
        } else {
          color = PREDICTION_COLORS[predClass] ?? COLORS.blue;
        }

        const ts = Math.floor(new Date(p.timestamp as string).getTime() / 1000) as UTCTimestamp;
        const label = PRED_LABELS[predClass] ?? predClass.slice(0, 4).toUpperCase();
        const suffix = isResolved ? (isCorrect ? " \u2713" : " \u2717") : "";

        markers.push({
          time: ts as Time,
          position: isLong ? "belowBar" : "aboveBar",
          color,
          shape: isLong ? "arrowUp" : "arrowDown",
          text: label + suffix,
        });
      }
    }

    // Trade entry + exit markers
    for (const trade of todaysTrades) {
      const direction = trade.direction as string | undefined;
      const entryPrice = trade.entry_price as number | undefined;
      const entryTime = trade.entry_time as string | undefined;
      const isLong = direction === "long";

      // Entry marker
      if (entryTime && entryPrice != null) {
        const ts = (new Date(entryTime).getTime() / 1000) as UTCTimestamp;
        markers.push({
          time: ts as Time,
          position: isLong ? "belowBar" : "aboveBar",
          color: isLong ? COLORS.green : COLORS.red,
          shape: isLong ? "arrowUp" : "arrowDown",
          text: isLong ? "BUY" : "SELL",
        });
      }

      // Exit marker (only for closed trades)
      const exitTime = trade.exit_time as string | undefined;
      const exitReason = (trade.exit_reason as string | undefined) ?? "";
      if (exitTime) {
        const ts = (new Date(exitTime).getTime() / 1000) as UTCTimestamp;
        const cfg = EXIT_CONFIG[exitReason] ?? { color: COLORS.secondary, text: "EXIT" };
        markers.push({
          time: ts as Time,
          position: isLong ? "aboveBar" : "belowBar",
          color: cfg.color,
          shape: "circle",
          text: cfg.text,
        });
      }
    }

    // Lightweight Charts requires markers sorted by time
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    candlestickSeries.setMarkers(markers);
  }, [candlestickSeries, todaysTrades, todaysPredictions, overlays]);

  // ── TP/SL price lines for open positions ───────────────────
  useEffect(() => {
    if (!candlestickSeries) return;

    // Remove old TP/SL lines
    for (const line of tpSlLinesRef.current) {
      candlestickSeries.removePriceLine(line);
    }
    tpSlLinesRef.current = [];

    // Draw TP/SL for each open position
    for (const pos of openPositions) {
      if (pos.tp_price != null) {
        const tpLine = candlestickSeries.createPriceLine({
          price: pos.tp_price,
          color: COLORS.green,
          lineWidth: 1,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: `TP ${pos.account_id}`,
        });
        tpSlLinesRef.current.push(tpLine);
      }

      if (pos.sl_price != null) {
        const slLine = candlestickSeries.createPriceLine({
          price: pos.sl_price,
          color: COLORS.red,
          lineWidth: 1,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: `SL ${pos.account_id}`,
        });
        tpSlLinesRef.current.push(slLine);
      }
    }
  }, [candlestickSeries, openPositions]);

  return null; // Pure logic component — no DOM output
}
