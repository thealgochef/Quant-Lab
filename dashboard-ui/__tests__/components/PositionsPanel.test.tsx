import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import PositionsPanel from "../../src/components/panels/PositionsPanel";
import { useTradingStore } from "../../src/stores/tradingStore";

describe("PositionsPanel", () => {
  beforeEach(() => {
    useTradingStore.setState({ openPositions: [], price: null });
  });

  it("renders position cards with P&L", () => {
    useTradingStore.setState({
      price: 21050.0,
      openPositions: [
        {
          account_id: "APEX-001",
          direction: "long",
          entry_price: 21040.0,
          contracts: 1,
          entry_time: "2026-03-02T14:30:00Z",
          unrealized_pnl: 200.0,
        },
        {
          account_id: "APEX-002",
          direction: "short",
          entry_price: 21060.0,
          contracts: 1,
          entry_time: "2026-03-02T14:30:00Z",
          unrealized_pnl: 200.0,
        },
      ],
    });

    render(<PositionsPanel />);

    const cards = screen.getAllByTestId("position-card");
    expect(cards).toHaveLength(2);

    // First position — long from 21040, current 21050 → +$200 (10 pts * $20)
    expect(cards[0]).toHaveTextContent("APEX-001");
    expect(cards[0]).toHaveTextContent("LONG");
    expect(cards[0]).toHaveTextContent("21040.00");

    // Check P&L is positive
    const pnlElements = screen.getAllByTestId("pnl");
    expect(pnlElements[0]).toHaveTextContent("+$200.00");
  });
});
