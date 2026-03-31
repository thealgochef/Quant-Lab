import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import ActiveLevelsPanel from "../../src/components/panels/ActiveLevelsPanel";
import { useTradingStore } from "../../src/stores/tradingStore";

describe("ActiveLevelsPanel", () => {
  beforeEach(() => {
    useTradingStore.setState({ activeLevels: [] });
  });

  it("renders level items with prices and types", () => {
    useTradingStore.setState({
      activeLevels: [
        {
          zone_id: "z1",
          price: 21045.75,
          side: "HIGH",
          is_touched: false,
          levels: [{ type: "pdh", price: 21045.75, is_manual: false }],
        },
        {
          zone_id: "z2",
          price: 20980.0,
          side: "LOW",
          is_touched: true,
          levels: [{ type: "pdl", price: 20980.0, is_manual: false }],
        },
      ],
    });

    render(<ActiveLevelsPanel />);

    const items = screen.getAllByTestId("level-item");
    expect(items).toHaveLength(2);

    // First level — active (not touched)
    expect(items[0]).toHaveTextContent("21045.75");
    expect(items[0]).toHaveTextContent("pdh");
    expect(items[0]).toHaveTextContent("HIGH");

    // Second level — touched (grayed out via opacity)
    expect(items[1]).toHaveTextContent("20980.00");
    expect(items[1]).toHaveClass("opacity-40");
  });
});
