import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import ObservationPanel from "../../src/components/panels/ObservationPanel";
import { useTradingStore } from "../../src/stores/tradingStore";

describe("ObservationPanel", () => {
  beforeEach(() => {
    useTradingStore.setState({ activeObservation: null });
  });

  it("displays countdown timer when observation is active", () => {
    // Set an observation ending 3 minutes from now
    const endTime = new Date(Date.now() + 3 * 60 * 1000).toISOString();

    useTradingStore.setState({
      activeObservation: {
        event_id: "evt_1",
        direction: "short",
        level_price: 21045.75,
        start_time: new Date().toISOString(),
        end_time: endTime,
        status: "active",
        trades_accumulated: 42,
      },
    });

    render(<ObservationPanel />);

    // Should show countdown
    const countdown = screen.getByTestId("countdown");
    expect(countdown).toHaveTextContent("remaining");
    // The countdown should show something like "2:59 remaining"
    expect(countdown.textContent).toMatch(/\d+:\d{2}/);

    // Should show event details
    expect(screen.getByText(/evt_1/)).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });
});
