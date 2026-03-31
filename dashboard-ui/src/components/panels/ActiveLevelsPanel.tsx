import { useState } from "react";
import { useTradingStore } from "../../stores/tradingStore";
import { addManualLevel, deleteManualLevel } from "../../hooks/useApi";
import { formatPrice } from "../../utils/formatters";

export default function ActiveLevelsPanel() {
  const levels = useTradingStore((s) => s.activeLevels);
  const [manualPrice, setManualPrice] = useState("");

  async function handleAddLevel() {
    const price = parseFloat(manualPrice);
    if (isNaN(price)) return;
    await addManualLevel(price);
    setManualPrice("");
  }

  async function handleDeleteLevel(price: number) {
    await deleteManualLevel(price);
  }

  return (
    <div className="space-y-2" data-testid="active-levels-panel">
      <h3 className="text-xs font-semibold uppercase text-secondary">Active Levels</h3>

      <div className="space-y-1">
        {levels.map((zone) => (
          <div
            key={zone.zone_id}
            className={`flex items-center justify-between rounded px-2 py-1 text-xs ${
              zone.is_touched ? "opacity-40" : "bg-hover"
            }`}
            data-testid="level-item"
          >
            <div className="flex items-center gap-2">
              <span className="font-mono text-primary">{formatPrice(zone.price)}</span>
              <span className="text-secondary">
                {zone.levels.map((l) => l.type).join(", ")}
              </span>
              <span className={`text-xs ${zone.side === "HIGH" ? "text-red" : "text-green"}`}>
                {zone.side}
              </span>
            </div>
            {zone.levels.some((l) => l.is_manual) && (
              <button
                onClick={() => handleDeleteLevel(zone.price)}
                className="text-secondary hover:text-red"
                title="Remove manual level"
              >
                x
              </button>
            )}
          </div>
        ))}
        {levels.length === 0 && (
          <div className="text-xs text-secondary">No active levels</div>
        )}
      </div>

      {/* Manual level input */}
      <div className="flex gap-1">
        <input
          type="text"
          value={manualPrice}
          onChange={(e) => setManualPrice(e.target.value)}
          placeholder="Price"
          className="w-full rounded bg-darkest px-2 py-1 font-mono text-xs text-primary outline-none focus:ring-1 focus:ring-blue"
        />
        <button
          onClick={handleAddLevel}
          className="whitespace-nowrap rounded bg-blue px-2 py-1 text-xs font-medium text-white hover:opacity-90"
        >
          Add
        </button>
      </div>
    </div>
  );
}
