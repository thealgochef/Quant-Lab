import { useState, useRef, useCallback, useEffect } from "react";
import type {
  CandlestickData,
  IChartApi,
  ISeriesApi,
  Time,
  UTCTimestamp,
} from "lightweight-charts";
import CandlestickChart from "../chart/CandlestickChart";
import ChartOverlays from "../chart/ChartOverlays";
import TradeOverlays from "../chart/TradeOverlays";
import TimeframeSelector from "../chart/TimeframeSelector";
import ReplayControls from "../panels/ReplayControls";
import ActiveLevelsPanel from "../panels/ActiveLevelsPanel";
import ObservationPanel from "../panels/ObservationPanel";
import PredictionPanel from "../panels/PredictionPanel";
import PositionsPanel from "../panels/PositionsPanel";
import SessionStatsPanel from "../panels/SessionStatsPanel";
import OverlayToggles from "../panels/OverlayToggles";
import AccountControls from "../panels/AccountControls";
import ConnectionStatus from "../panels/ConnectionStatus";
import { useTradingStore } from "../../stores/tradingStore";
import { formatPrice } from "../../utils/formatters";
import { fetchOHLCV } from "../../hooks/useApi";
import type { OHLCVBar, Timeframe } from "../../types/chart";

interface Props {
  sendMessage: (msg: Record<string, unknown>) => void;
  reconnectTo: (url: string) => void;
}

const LIVE_WS = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;

export default function TradingView({ sendMessage, reconnectTo }: Props) {
  const [timeframe, setTimeframe] = useState<Timeframe>("987t");
  const [autoScroll, setAutoScroll] = useState(true);
  const [panelWidth, setPanelWidth] = useState(420);
  const [bars, setBars] = useState<CandlestickData<Time>[]>([]);
  const [ohlcvBars, setOhlcvBars] = useState<OHLCVBar[]>([]);
  const [chartApi, setChartApi] = useState<IChartApi | null>(null);
  const [candleSeries, setCandleSeries] = useState<ISeriesApi<"Candlestick"> | null>(null);
  const resizing = useRef(false);

  const [replayDropdown, setReplayDropdown] = useState(false);
  const [replayStarting, setReplayStarting] = useState(false);

  const price = useTradingStore((s) => s.price);
  const bid = useTradingStore((s) => s.bid);
  const ask = useTradingStore((s) => s.ask);
  const connectionStatus = useTradingStore((s) => s.connectionStatus);
  const replayMode = useTradingStore((s) => s.replayMode);
  const latestBar = useTradingStore((s) => s.latestBar);

  // Append streaming bar_update to chart
  useEffect(() => {
    if (!latestBar || latestBar.timeframe !== timeframe) return;
    const bar = latestBar.bar;
    const ts = (new Date(bar.timestamp).getTime() / 1000) as UTCTimestamp;
    const candle: CandlestickData<Time> = {
      time: ts,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    };
    const ohlcv: OHLCVBar = {
      time: ts as number,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      volume: bar.volume,
    };
    setBars((prev) => [...prev, candle]);
    setOhlcvBars((prev) => [...prev, ohlcv]);
  }, [latestBar, timeframe]);

  // On timeframe change, notify backend
  const handleTimeframeChange = useCallback((tf: Timeframe) => {
    setTimeframe(tf);
    sendMessage({ type: "subscribe_timeframe", data: { timeframe: tf } });
  }, [sendMessage]);

  // Fetch OHLCV bars from backend, refresh every 5 seconds
  useEffect(() => {
    if (connectionStatus !== "connected") return;

    let mounted = true;

    async function loadBars() {
      try {
        const resp = await fetchOHLCV(timeframe);
        if (!mounted) return;
        const candles: CandlestickData<Time>[] = resp.bars.map((bar) => ({
          time: (new Date(bar.timestamp as string).getTime() / 1000) as UTCTimestamp,
          open: bar.open as number,
          high: bar.high as number,
          low: bar.low as number,
          close: bar.close as number,
        }));
        const ohlcv: OHLCVBar[] = resp.bars.map((bar) => ({
          time: new Date(bar.timestamp as string).getTime() / 1000,
          open: bar.open as number,
          high: bar.high as number,
          low: bar.low as number,
          close: bar.close as number,
          volume: (bar.volume as number) || 0,
        }));
        setBars(candles);
        setOhlcvBars(ohlcv);
      } catch {
        // Backend may not have data yet
      }
    }

    loadBars();
    const interval = setInterval(loadBars, 5000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [timeframe, connectionStatus]);

  const handleChartReady = useCallback(
    (chart: IChartApi, series: ISeriesApi<"Candlestick">) => {
      setChartApi(chart);
      setCandleSeries(series);
    },
    [],
  );

  const handleMouseDown = useCallback(() => {
    resizing.current = true;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizing.current) return;
      const newWidth = window.innerWidth - e.clientX;
      setPanelWidth(Math.max(300, Math.min(600, newWidth)));
    };
    const handleMouseUp = () => {
      resizing.current = false;
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
  }, []);

  return (
    <div className="flex h-full">
      {/* Chart area */}
      <div className="flex flex-1 flex-col bg-darkest">
        {/* Top bar: timeframe selector + price display + auto-scroll toggle */}
        <div className="flex items-center justify-between border-b border-hover px-2 py-1">
          <TimeframeSelector active={timeframe} onChange={handleTimeframeChange} />
          <div className="flex items-center gap-4">
            {price !== null && (
              <div className="flex items-center gap-2 font-mono text-sm">
                <span className="text-primary">{formatPrice(price)}</span>
                {bid !== null && (
                  <span className="text-xs text-green">{formatPrice(bid)}</span>
                )}
                {ask !== null && (
                  <span className="text-xs text-red">{formatPrice(ask)}</span>
                )}
              </div>
            )}
            <button
              onClick={() => setAutoScroll(!autoScroll)}
              className={`rounded px-1.5 py-0.5 text-xs ${
                autoScroll ? "bg-blue text-white" : "text-secondary hover:text-primary"
              }`}
              title={autoScroll ? "Auto-scroll locked" : "Auto-scroll unlocked"}
            >
              {autoScroll ? "Lock" : "Unlock"}
            </button>
            <div className="relative">
              <button
                onClick={() => setReplayDropdown(!replayDropdown)}
                className={`rounded px-1.5 py-0.5 text-xs ${
                  replayMode ? "bg-yellow text-darkest" : "text-secondary hover:text-primary"
                }`}
              >
                {replayMode ? "Replay" : "Replay"}
              </button>
              {replayDropdown && (
                <div className="absolute right-0 top-full z-50 mt-1 w-80 rounded border border-hover bg-panel p-3 shadow-lg">
                  <div className="mb-2 flex items-center gap-2">
                    <label className="text-xs text-secondary">Start:</label>
                    <input
                      id="replay-start"
                      type="date"
                      defaultValue="2025-07-10"
                      className="rounded bg-darkest px-1.5 py-0.5 text-xs text-primary"
                    />
                    <label className="text-xs text-secondary">End:</label>
                    <input
                      id="replay-end"
                      type="date"
                      defaultValue="2025-07-10"
                      className="rounded bg-darkest px-1.5 py-0.5 text-xs text-primary"
                    />
                  </div>
                  <div className="flex gap-2">
                    <button
                      disabled={replayStarting}
                      onClick={async () => {
                        setReplayStarting(true);
                        try {
                          const startEl = document.getElementById("replay-start") as HTMLInputElement;
                          const endEl = document.getElementById("replay-end") as HTMLInputElement;
                          const res = await fetch("/api/data/replay/start", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                              start: startEl?.value || "2025-07-10",
                              end: endEl?.value || "2025-07-10",
                              speed: 10,
                            }),
                          });
                          const data = await res.json();
                          if (data.ok) {
                            // Give the replay server time to start and preload
                            await new Promise((r) => setTimeout(r, 12000));
                            reconnectTo(data.ws_url);
                          }
                        } finally {
                          setReplayStarting(false);
                          setReplayDropdown(false);
                        }
                      }}
                      className="flex-1 rounded bg-yellow px-2 py-1 text-xs font-medium text-darkest hover:brightness-110 disabled:opacity-50"
                    >
                      {replayStarting ? "Starting..." : "Start Replay"}
                    </button>
                    <button
                      onClick={async () => {
                        await fetch("/api/data/replay/stop", { method: "POST" });
                        reconnectTo(LIVE_WS);
                        setReplayDropdown(false);
                      }}
                      className="flex-1 rounded bg-blue px-2 py-1 text-xs font-medium text-white hover:brightness-110"
                    >
                      Back to Live
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Replay controls (only in replay mode) */}
        {replayMode && <ReplayControls sendMessage={sendMessage} />}

        {/* Chart */}
        <div className="flex-1">
          <CandlestickChart
            bars={bars}
            timeframe={timeframe}
            autoScroll={autoScroll}
            onChartReady={handleChartReady}
          />
          <ChartOverlays chart={chartApi} bars={ohlcvBars} candlestickSeries={candleSeries} />
          <TradeOverlays candlestickSeries={candleSeries} />
        </div>
      </div>

      {/* Resize handle */}
      <div
        className="w-1 cursor-col-resize bg-hover hover:bg-blue"
        onMouseDown={handleMouseDown}
      />

      {/* Side panels */}
      <div
        className="shrink-0 overflow-y-auto border-l border-hover bg-panel"
        style={{ width: panelWidth }}
      >
        <div className="space-y-3 p-2">
          <ConnectionStatus />
          <AccountControls />
          <PositionsPanel />
          <PredictionPanel />
          <ObservationPanel />
          <ActiveLevelsPanel />
          <SessionStatsPanel />
          <OverlayToggles />
        </div>
      </div>
    </div>
  );
}
