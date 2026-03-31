import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import ConnectionStatus from "../../src/components/panels/ConnectionStatus";
import { useTradingStore } from "../../src/stores/tradingStore";

describe("ConnectionStatus", () => {
  beforeEach(() => {
    useTradingStore.setState({ connectionStatus: "disconnected" });
  });

  it("shows correct color and text for each connection state", () => {
    // Disconnected
    const { unmount } = render(<ConnectionStatus />);
    expect(screen.getByTestId("status-text")).toHaveTextContent("Disconnected");
    expect(screen.getByTestId("status-dot")).toHaveClass("bg-red");
    unmount();

    // Connected
    useTradingStore.setState({ connectionStatus: "connected" });
    const { unmount: unmount2 } = render(<ConnectionStatus />);
    expect(screen.getByTestId("status-text")).toHaveTextContent("Connected");
    expect(screen.getByTestId("status-dot")).toHaveClass("bg-green");
    unmount2();

    // Reconnecting
    useTradingStore.setState({ connectionStatus: "reconnecting" });
    render(<ConnectionStatus />);
    expect(screen.getByTestId("status-text")).toHaveTextContent("Reconnecting");
    expect(screen.getByTestId("status-dot")).toHaveClass("bg-yellow");
  });
});
