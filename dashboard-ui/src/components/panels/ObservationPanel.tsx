import { useEffect, useState } from "react";
import { useTradingStore } from "../../stores/tradingStore";

export default function ObservationPanel() {
  const obs = useTradingStore((s) => s.activeObservation);
  const [remaining, setRemaining] = useState<string | null>(null);

  useEffect(() => {
    if (!obs) {
      setRemaining(null);
      return;
    }

    function tick() {
      const endTime = new Date(obs!.end_time).getTime();
      const now = Date.now();
      const diff = Math.max(0, endTime - now);
      const mins = Math.floor(diff / 60000);
      const secs = Math.floor((diff % 60000) / 1000);
      setRemaining(`${mins}:${String(secs).padStart(2, "0")}`);
    }

    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [obs]);

  if (!obs) {
    return (
      <div className="space-y-1" data-testid="observation-panel">
        <h3 className="text-xs font-semibold uppercase text-secondary">Observation</h3>
        <div className="text-xs text-secondary">Waiting for level touch</div>
      </div>
    );
  }

  return (
    <div className="space-y-1" data-testid="observation-panel">
      <h3 className="text-xs font-semibold uppercase text-secondary">Observation</h3>
      <div className="rounded bg-hover p-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-secondary">Event: {obs.event_id}</span>
          <span
            className="font-mono text-sm font-semibold text-blue"
            data-testid="countdown"
          >
            {remaining} remaining
          </span>
        </div>
        <div className="mt-1 flex gap-3 text-xs text-secondary">
          <span>Direction: <span className="text-primary">{obs.direction}</span></span>
          <span>Trades: <span className="text-primary">{obs.trades_accumulated}</span></span>
        </div>
        {/* Progress bar */}
        <div className="mt-1 h-1 w-full rounded-full bg-darkest">
          <div
            className="h-full rounded-full bg-blue transition-all"
            style={{ width: `${Math.max(0, 100 - (parseFloat(remaining?.split(":")[0] ?? "5") / 5) * 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
