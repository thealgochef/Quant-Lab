import { useEffect, useRef } from "react";
import type {
  IChartApi,
  ISeriesApi,
  IPriceLine,
  LineData,
  Time,
} from "lightweight-charts";
import { useConfigStore } from "../../stores/configStore";
import { useTradingStore } from "../../stores/tradingStore";
import { computeEMA, computeKAMA, computeVWAP } from "../../utils/indicators";
import { COLORS } from "../../utils/constants";
import type { OHLCVBar, IndicatorPoint } from "../../types/chart";

interface Props {
  chart: IChartApi | null;
  bars: OHLCVBar[];
  candlestickSeries?: ISeriesApi<"Candlestick"> | null;
}

function toLineData(points: IndicatorPoint[]): LineData<Time>[] {
  return points.map((p) => ({ time: p.time as Time, value: p.value }));
}

// Level type → display color
const LEVEL_COLORS: Record<string, string> = {
  pdh: "#ff9100",      // orange — Previous Day High
  pdl: "#ff9100",      // orange — Previous Day Low
  asia_high: "#ba68c8", // purple — Asia session
  asia_low: "#ba68c8",
  london_high: "#2979ff", // blue — London session
  london_low: "#2979ff",
  manual: "#ffd600",    // yellow — manual
};

export default function ChartOverlays({ chart, bars, candlestickSeries }: Props) {
  const overlays = useConfigStore((s) => s.overlays);
  const levels = useTradingStore((s) => s.activeLevels);

  const seriesRefs = useRef<Map<string, ISeriesApi<"Line">>>(new Map());
  const levelLinesRef = useRef<IPriceLine[]>([]);

  // Indicator line series (EMA, VWAP, KAMA)
  useEffect(() => {
    if (!chart || bars.length === 0) return;

    const existing = seriesRefs.current;

    function ensureSeries(key: string, color: string, data: LineData<Time>[]) {
      let series = existing.get(key);
      if (!series) {
        series = chart!.addLineSeries({
          color,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        existing.set(key, series);
      }
      series.setData(data);
      series.applyOptions({ visible: true });
    }

    function hideSeries(key: string) {
      const series = existing.get(key);
      if (series) {
        series.applyOptions({ visible: false });
      }
    }

    // EMA 13
    if (overlays["ema_13"]) {
      ensureSeries("ema_13", COLORS.ema13, toLineData(computeEMA(bars, 13)));
    } else {
      hideSeries("ema_13");
    }

    // EMA 48
    if (overlays["ema_48"]) {
      ensureSeries("ema_48", COLORS.ema48, toLineData(computeEMA(bars, 48)));
    } else {
      hideSeries("ema_48");
    }

    // EMA 200
    if (overlays["ema_200"]) {
      ensureSeries("ema_200", COLORS.ema200, toLineData(computeEMA(bars, 200)));
    } else {
      hideSeries("ema_200");
    }

    // VWAP
    if (overlays["vwap"]) {
      ensureSeries("vwap", COLORS.vwap, toLineData(computeVWAP(bars)));
    } else {
      hideSeries("vwap");
    }

    // KAMA (shown alongside EMA 13)
    if (overlays["ema_13"]) {
      ensureSeries("kama", COLORS.kama, toLineData(computeKAMA(bars)));
    } else {
      hideSeries("kama");
    }
  }, [chart, bars, overlays]);

  // Level price lines on the candlestick series
  useEffect(() => {
    if (!candlestickSeries) return;

    // Remove old level lines
    for (const line of levelLinesRef.current) {
      candlestickSeries.removePriceLine(line);
    }
    levelLinesRef.current = [];

    // Don't render if levels overlay is off
    if (!overlays["levels"]) return;

    // Create price lines for each active zone
    for (const zone of levels) {
      if (zone.is_touched) continue;

      // Determine color from level types
      const levelTypes = zone.levels.map((l) => l.type);
      const firstType = levelTypes[0] ?? "";
      const color =
        LEVEL_COLORS[firstType] ??
        (zone.side === "high" ? COLORS.red : COLORS.green);

      // Build label from level types (e.g. "PDH" or "ASIA_H, LDN_H")
      const label = levelTypes
        .map((t) => t.replace("_high", "_H").replace("_low", "_L").toUpperCase())
        .join(", ");

      const priceLine = candlestickSeries.createPriceLine({
        price: zone.price,
        color,
        lineWidth: 1,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: label,
      });
      levelLinesRef.current.push(priceLine);
    }
  }, [candlestickSeries, levels, overlays]);

  // Prediction markers moved to TradeOverlays (setMarkers is per-series,
  // only one call can be active — TradeOverlays merges all marker types).

  return null; // Pure logic component — no DOM output
}
