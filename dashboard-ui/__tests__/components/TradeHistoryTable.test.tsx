import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TradeHistoryTable from "../../src/components/analysis/TradeHistoryTable";

// Mock the API to return trade data
vi.mock("../../src/hooks/useApi", () => ({
  fetchTrades: vi.fn(() =>
    Promise.resolve({
      trades: [
        {
          account_id: "APEX-001",
          direction: "long",
          pnl: 300,
          exit_reason: "tp",
          entry_price: 21000,
          exit_price: 21015,
          entry_time: "2026-03-02T14:30:00Z",
          predicted_class: "tradeable_reversal",
        },
        {
          account_id: "APEX-002",
          direction: "short",
          pnl: -100,
          exit_reason: "sl",
          entry_price: 21050,
          exit_price: 21055,
          entry_time: "2026-03-02T15:00:00Z",
          predicted_class: "trap",
        },
        {
          account_id: "APEX-003",
          direction: "long",
          pnl: 150,
          exit_reason: "tp",
          entry_price: 20990,
          exit_price: 21000,
          entry_time: "2026-03-02T15:30:00Z",
          predicted_class: "tradeable_reversal",
        },
      ],
    }),
  ),
}));

describe("TradeHistoryTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("filters by prediction class", async () => {
    const user = userEvent.setup();
    render(<TradeHistoryTable />);

    // Wait for data to load
    await screen.findByText("APEX-001");

    // All 3 trades visible initially
    expect(screen.getByText("APEX-001")).toBeInTheDocument();
    expect(screen.getByText("APEX-002")).toBeInTheDocument();
    expect(screen.getByText("APEX-003")).toBeInTheDocument();

    // Click "trap" filter
    await user.click(screen.getByTestId("filter-trap"));

    // Only the trap trade should be visible
    expect(screen.getByText("APEX-002")).toBeInTheDocument();
    expect(screen.queryByText("APEX-001")).not.toBeInTheDocument();
    expect(screen.queryByText("APEX-003")).not.toBeInTheDocument();
  });
});
