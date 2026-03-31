import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import PredictionPanel from "../../src/components/panels/PredictionPanel";
import { useTradingStore } from "../../src/stores/tradingStore";

describe("PredictionPanel", () => {
  beforeEach(() => {
    useTradingStore.setState({ lastPrediction: null });
  });

  it("displays predicted class with correct color", () => {
    useTradingStore.setState({
      lastPrediction: {
        predicted_class: "tradeable_reversal",
        is_executable: true,
        event_id: "evt_42",
      },
    });

    render(<PredictionPanel />);

    const classEl = screen.getByTestId("predicted-class");
    expect(classEl).toHaveTextContent("tradeable reversal");
    expect(classEl).toHaveClass("text-green");
    expect(screen.getByText("Executable")).toBeInTheDocument();
  });
});
