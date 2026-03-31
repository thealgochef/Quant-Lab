import { useTradingStore } from "../../stores/tradingStore";
import { formatPercent } from "../../utils/formatters";

export default function SessionStats() {
  const stats = useTradingStore((s) => s.sessionStats);
  const trades = useTradingStore((s) => s.todaysTrades);
  const predictions = useTradingStore((s) => s.todaysPredictions);

  const winningTrades = trades.filter((t) => (t["pnl"] as number) > 0);
  const losingTrades = trades.filter((t) => (t["pnl"] as number) < 0);
  const avgWin =
    winningTrades.length > 0
      ? winningTrades.reduce((s, t) => s + (t["pnl"] as number), 0) / winningTrades.length
      : 0;
  const avgLoss =
    losingTrades.length > 0
      ? losingTrades.reduce((s, t) => s + (t["pnl"] as number), 0) / losingTrades.length
      : 0;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Session Statistics</h3>

      <div className="grid grid-cols-2 gap-4">
        <div className="rounded bg-hover p-3">
          <div className="text-xs text-secondary">Signals Fired</div>
          <div className="mt-1 font-mono text-lg font-semibold text-primary">
            {stats.signals_fired}
          </div>
        </div>
        <div className="rounded bg-hover p-3">
          <div className="text-xs text-secondary">Accuracy</div>
          <div className="mt-1 font-mono text-lg font-semibold text-blue">
            {formatPercent(stats.accuracy)}
          </div>
        </div>
        <div className="rounded bg-hover p-3">
          <div className="text-xs text-secondary">Avg Win</div>
          <div className="mt-1 font-mono text-lg font-semibold text-green">
            ${avgWin.toFixed(2)}
          </div>
        </div>
        <div className="rounded bg-hover p-3">
          <div className="text-xs text-secondary">Avg Loss</div>
          <div className="mt-1 font-mono text-lg font-semibold text-red">
            ${avgLoss.toFixed(2)}
          </div>
        </div>
      </div>

      <div className="text-xs text-secondary">
        Total predictions: {predictions.length} | Total trades: {trades.length}
      </div>
    </div>
  );
}
