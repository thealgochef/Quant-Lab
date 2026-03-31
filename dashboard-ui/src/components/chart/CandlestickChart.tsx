import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  ColorType,
  type CandlestickData,
  type Time,
} from "lightweight-charts";
import { COLORS } from "../../utils/constants";
import type { Timeframe } from "../../types/chart";

interface Props {
  bars: CandlestickData<Time>[];
  timeframe: Timeframe;
  autoScroll: boolean;
  onChartReady?: (chart: IChartApi, series: ISeriesApi<"Candlestick">) => void;
}

export default function CandlestickChart({ bars, timeframe, autoScroll, onChartReady }: Props) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // Create chart on mount
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: COLORS.darkest },
        textColor: COLORS.secondary,
        fontFamily: "'JetBrains Mono', monospace",
      },
      grid: {
        vertLines: { color: "#1a1a2e" },
        horzLines: { color: "#1a1a2e" },
      },
      crosshair: {
        vertLine: { color: COLORS.secondary, width: 1, style: 3 },
        horzLine: { color: COLORS.secondary, width: 1, style: 3 },
      },
      timeScale: {
        borderColor: COLORS.hover,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 10,
      },
      rightPriceScale: {
        borderColor: COLORS.hover,
        autoScale: true,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
    });

    const series = chart.addCandlestickSeries({
      upColor: COLORS.candleUp,
      downColor: COLORS.candleDown,
      borderUpColor: COLORS.candleUp,
      borderDownColor: COLORS.candleDown,
      wickUpColor: COLORS.candleUp,
      wickDownColor: COLORS.candleDown,
    });

    chartRef.current = chart;
    seriesRef.current = series;

    onChartReady?.(chart, series);

    // Handle resize
    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        chart.applyOptions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    resizeObserver.observe(chartContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Reset on timeframe change so the next data load auto-fits
  const prevBarCount = useRef(0);
  const initialLoadDone = useRef(false);
  useEffect(() => {
    initialLoadDone.current = false;
    prevBarCount.current = 0;
  }, [timeframe]);

  // Update bars when data changes — incremental when possible
  useEffect(() => {
    if (!seriesRef.current || bars.length === 0) return;

    // Incremental: exactly one new bar appended (WS bar_update)
    const lastBar = bars[bars.length - 1];
    if (bars.length === prevBarCount.current + 1 && lastBar) {
      seriesRef.current.update(lastBar);
    } else {
      seriesRef.current.setData(bars);
      // Only auto-fit on the very first load (or timeframe change which
      // resets initialLoadDone via the bars=[]/prevBarCount=0 path).
      // Subsequent polling updates preserve the user's zoom/scroll.
      if (!initialLoadDone.current) {
        chartRef.current?.timeScale().fitContent();
        initialLoadDone.current = true;
      }
    }
    prevBarCount.current = bars.length;
  }, [bars]);

  // Auto-scroll to latest bar
  useEffect(() => {
    if (autoScroll && chartRef.current && bars.length > 0) {
      chartRef.current.timeScale().scrollToRealTime();
    }
  }, [bars, autoScroll]);

  return (
    <div
      ref={chartContainerRef}
      data-testid="candlestick-chart"
      className="h-full w-full"
    />
  );
}
