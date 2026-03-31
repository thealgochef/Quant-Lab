import { useEffect, useState } from "react";
import { fetchEquityCurve } from "../../hooks/useApi";
import { formatPrice } from "../../utils/formatters";

interface Snapshot {
  account_id: string;
  balance: number;
  profit: number;
  tier: number;
  status: string;
}

export default function EquityCurves() {
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);

  useEffect(() => {
    fetchEquityCurve()
      .then((data) => setSnapshots(data.snapshots as unknown as Snapshot[]))
      .catch(() => {});
  }, []);

  if (snapshots.length === 0) {
    return <div className="text-xs text-secondary">No equity data available</div>;
  }

  // Deduplicate: keep only latest snapshot per account_id
  const latestByAccount = new Map<string, Snapshot>();
  for (const s of snapshots) {
    latestByAccount.set(s.account_id, s);
  }
  const dedupedSnapshots = Array.from(latestByAccount.values());

  // Aggregate
  const totalBalance = dedupedSnapshots.reduce((sum, s) => sum + s.balance, 0);
  const totalProfit = dedupedSnapshots.reduce((sum, s) => sum + s.profit, 0);

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">Equity Curves</h3>

      <div className="flex gap-4">
        <div className="rounded bg-hover p-3">
          <div className="text-xs text-secondary">Total Balance</div>
          <div className="mt-1 font-mono text-lg font-semibold text-primary">
            ${formatPrice(totalBalance)}
          </div>
        </div>
        <div className="rounded bg-hover p-3">
          <div className="text-xs text-secondary">Total Profit</div>
          <div className={`mt-1 font-mono text-lg font-semibold ${totalProfit >= 0 ? "text-green" : "text-red"}`}>
            ${formatPrice(totalProfit)}
          </div>
        </div>
      </div>

      <div className="space-y-1">
        {dedupedSnapshots.map((s) => (
          <div key={s.account_id} className="flex items-center justify-between rounded bg-hover px-3 py-2 text-xs">
            <span className="text-secondary">{s.account_id}</span>
            <div className="flex gap-4">
              <span className="font-mono text-primary">${formatPrice(s.balance)}</span>
              <span className={`font-mono ${s.profit >= 0 ? "text-green" : "text-red"}`}>
                {s.profit >= 0 ? "+" : ""}${formatPrice(s.profit)}
              </span>
              <span className="text-secondary">Tier {s.tier}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
