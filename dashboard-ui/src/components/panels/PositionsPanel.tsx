import { useTradingStore } from "../../stores/tradingStore";
import { formatPrice, formatPnl } from "../../utils/formatters";

export default function PositionsPanel() {
  const positions = useTradingStore((s) => s.openPositions);
  const price = useTradingStore((s) => s.price);

  if (positions.length === 0) {
    return (
      <div className="space-y-1" data-testid="positions-panel">
        <h3 className="text-xs font-semibold uppercase text-secondary">Open Positions</h3>
        <div className="text-xs text-secondary">No open positions</div>
      </div>
    );
  }

  return (
    <div className="space-y-1" data-testid="positions-panel">
      <h3 className="text-xs font-semibold uppercase text-secondary">
        Open Positions ({positions.length})
      </h3>
      <div className="space-y-1">
        {positions.map((pos) => {
          // Live P&L calculation
          const unrealizedPnl =
            price !== null
              ? pos.direction === "long"
                ? (price - pos.entry_price) * 20 * pos.contracts // NQ $20/point
                : (pos.entry_price - price) * 20 * pos.contracts
              : pos.unrealized_pnl;
          const pnl = formatPnl(unrealizedPnl);

          return (
            <div
              key={pos.account_id}
              className="rounded bg-hover p-2"
              data-testid="position-card"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs text-secondary">{pos.account_id}</span>
                <span
                  className={`text-xs font-semibold ${
                    pos.direction === "long" ? "text-green" : "text-red"
                  }`}
                >
                  {pos.direction.toUpperCase()}
                </span>
              </div>
              <div className="mt-0.5 flex items-center justify-between">
                <span className="font-mono text-xs text-primary">
                  @ {formatPrice(pos.entry_price)}
                </span>
                <span className={`font-mono text-xs font-semibold ${pnl.className}`} data-testid="pnl">
                  {pnl.text}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
