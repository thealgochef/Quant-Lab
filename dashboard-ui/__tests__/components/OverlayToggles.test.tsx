import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import OverlayToggles from "../../src/components/panels/OverlayToggles";
import { useConfigStore } from "../../src/stores/configStore";

// Mock the API so updateOverlays doesn't make a real fetch (and revert)
vi.mock("../../src/hooks/useApi", () => ({
  updateOverlays: vi.fn(() => Promise.resolve({ overlays: {} })),
}));

describe("OverlayToggles", () => {
  beforeEach(() => {
    useConfigStore.setState({
      overlays: {
        ema_13: true,
        ema_48: true,
        ema_200: true,
        vwap: false,
        levels: true,
      },
    });
  });

  it("toggles update config store on click", async () => {
    const user = userEvent.setup();
    render(<OverlayToggles />);

    // VWAP starts unchecked
    const vwapCheckbox = screen.getByTestId("overlay-vwap") as HTMLInputElement;
    expect(vwapCheckbox.checked).toBe(false);

    // Click it
    await user.click(vwapCheckbox);

    // Store should reflect the change
    expect(useConfigStore.getState().overlays["vwap"]).toBe(true);
  });
});
