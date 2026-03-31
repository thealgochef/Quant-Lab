import { useEffect, useState } from "react";
import { fetchPerformance } from "../../hooks/useApi";
import type { PerformanceData } from "../../types/api";
import { formatPercent } from "../../utils/formatters";

export default function PerformanceCharts() {
  const [perf, setPerf] = useState<PerformanceData | null>(null);

  useEffect(() => {
    fetchPerformance().then(setPerf).catch(() => {});
  }, []);

  if (!perf) {
    return <div className="text-xs text-secondary">Loading performance data...</div>;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Performance Summary</h3>
      <div className="grid grid-cols-3 gap-4">
        <StatCard label="Total Trades" value={String(perf.total_trades)} />
        <StatCard label="Win Rate" value={formatPercent(perf.win_rate)} color="text-green" />
        <StatCard
          label="Prediction Accuracy"
          value={formatPercent(perf.prediction_accuracy)}
          color="text-blue"
        />
        <StatCard label="Wins" value={String(perf.wins)} color="text-green" />
        <StatCard label="Losses" value={String(perf.losses)} color="text-red" />
        <StatCard
          label="Total P&L"
          value={`$${perf.total_pnl.toFixed(2)}`}
          color={perf.total_pnl >= 0 ? "text-green" : "text-red"}
        />
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded bg-hover p-3">
      <div className="text-xs text-secondary">{label}</div>
      <div className={`mt-1 font-mono text-lg font-semibold ${color ?? "text-primary"}`}>
        {value}
      </div>
    </div>
  );
}
