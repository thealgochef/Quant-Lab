import { useTradingStore } from "../../stores/tradingStore";
import { formatPercent } from "../../utils/formatters";

export default function SessionStatsPanel() {
  const stats = useTradingStore((s) => s.sessionStats);

  return (
    <div className="space-y-1" data-testid="session-stats-panel">
      <h3 className="text-xs font-semibold uppercase text-secondary">Session Stats</h3>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <div className="text-secondary">Signals Fired</div>
        <div className="font-mono text-primary">{stats.signals_fired}</div>

        <div className="text-secondary">Wins</div>
        <div className="font-mono text-green">{stats.wins}</div>

        <div className="text-secondary">Losses</div>
        <div className="font-mono text-red">{stats.losses}</div>

        <div className="text-secondary">Accuracy</div>
        <div className="font-mono text-primary">{formatPercent(stats.accuracy)}</div>
      </div>
    </div>
  );
}
