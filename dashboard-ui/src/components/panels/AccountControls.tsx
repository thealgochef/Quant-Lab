import { useState } from "react";
import { closeAllPositions, manualEntry } from "../../hooks/useApi";
import { useTradingStore } from "../../stores/tradingStore";

export default function AccountControls() {
  const price = useTradingStore((s) => s.price);
  const [closing, setClosing] = useState(false);

  async function handleCloseAll() {
    setClosing(true);
    try {
      await closeAllPositions("manual_close_all");
    } finally {
      setClosing(false);
    }
  }

  async function handleManualEntry(direction: "long" | "short") {
    if (price === null) return;
    await manualEntry(direction);
  }

  return (
    <div className="space-y-2" data-testid="account-controls">
      <h3 className="text-xs font-semibold uppercase text-secondary">Controls</h3>
      <button
        onClick={handleCloseAll}
        disabled={closing}
        className="w-full rounded bg-red px-3 py-1.5 text-xs font-semibold text-white hover:opacity-90 disabled:opacity-50"
      >
        {closing ? "Closing..." : "Close All Positions"}
      </button>
      <div className="flex gap-1">
        <button
          onClick={() => handleManualEntry("long")}
          disabled={price === null}
          className="flex-1 rounded bg-green px-2 py-1 text-xs font-semibold text-white hover:opacity-90 disabled:opacity-50"
        >
          Buy
        </button>
        <button
          onClick={() => handleManualEntry("short")}
          disabled={price === null}
          className="flex-1 rounded bg-red px-2 py-1 text-xs font-semibold text-white hover:opacity-90 disabled:opacity-50"
        >
          Sell
        </button>
      </div>
    </div>
  );
}
