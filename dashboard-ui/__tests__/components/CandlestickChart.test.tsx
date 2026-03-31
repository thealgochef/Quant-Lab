import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock lightweight-charts since it requires a real DOM canvas
const mockSetData = vi.fn();
const mockApplyOptions = vi.fn();
const mockAddCandlestickSeries = vi.fn(() => ({
  setData: mockSetData,
  applyOptions: mockApplyOptions,
}));
const mockRemove = vi.fn();
const mockTimeScale = vi.fn(() => ({ scrollToRealTime: vi.fn() }));
const mockPriceScale = vi.fn(() => ({ applyOptions: vi.fn() }));

vi.mock("lightweight-charts", () => ({
  createChart: vi.fn(() => ({
    addCandlestickSeries: mockAddCandlestickSeries,
    addLineSeries: vi.fn(() => ({ setData: vi.fn(), applyOptions: vi.fn() })),
    remove: mockRemove,
    applyOptions: vi.fn(),
    timeScale: mockTimeScale,
    priceScale: mockPriceScale,
  })),
  ColorType: { Solid: "Solid" },
}));

// Must import after mock
import { render, screen } from "@testing-library/react";
import CandlestickChart from "../../src/components/chart/CandlestickChart";
import type { CandlestickData, Time } from "lightweight-charts";

describe("CandlestickChart", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("creates chart with correct bar count from OHLCV data", () => {
    const bars: CandlestickData<Time>[] = Array.from({ length: 50 }, (_, i) => ({
      time: (1709400000 + i * 60) as Time,
      open: 21000 + i,
      high: 21005 + i,
      low: 20995 + i,
      close: 21002 + i,
    }));

    render(<CandlestickChart bars={bars} timeframe="987t" autoScroll={true} />);

    // Chart container rendered
    expect(screen.getByTestId("candlestick-chart")).toBeInTheDocument();

    // setData was called with the correct number of bars
    expect(mockSetData).toHaveBeenCalledTimes(1);
    expect(mockSetData).toHaveBeenCalledWith(bars);
    expect(bars).toHaveLength(50);
  });
});
