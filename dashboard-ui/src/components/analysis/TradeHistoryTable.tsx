import { useEffect, useState } from "react";
import { fetchTrades } from "../../hooks/useApi";
import { formatPrice, formatPnl, formatTime } from "../../utils/formatters";

interface TradeRow {
  account_id?: string;
  direction?: string;
  pnl?: number;
  exit_reason?: string;
  entry_price?: number;
  exit_price?: number;
  entry_time?: string;
  predicted_class?: string;
  [key: string]: unknown;
}

type FilterKey = "all" | "tradeable_reversal" | "trap" | "blowthrough";

export default function TradeHistoryTable() {
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [filter, setFilter] = useState<FilterKey>("all");

  useEffect(() => {
    fetchTrades().then((data) => setTrades(data.trades as TradeRow[])).catch(() => {});
  }, []);

  const filtered = filter === "all"
    ? trades
    : trades.filter((t) => t.predicted_class === filter);

  return (
    <div data-testid="trade-history-table">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-xs text-secondary">Filter:</span>
        {(["all", "tradeable_reversal", "trap", "blowthrough"] as FilterKey[]).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            data-testid={`filter-${f}`}
            className={`rounded px-2 py-0.5 text-xs ${
              filter === f ? "bg-blue text-white" : "text-secondary hover:text-primary"
            }`}
          >
            {f === "all" ? "All" : f.replace(/_/g, " ")}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-hover text-left text-secondary">
              <th className="px-2 py-1">Time</th>
              <th className="px-2 py-1">Account</th>
              <th className="px-2 py-1">Direction</th>
              <th className="px-2 py-1">Entry</th>
              <th className="px-2 py-1">Exit</th>
              <th className="px-2 py-1">P&L</th>
              <th className="px-2 py-1">Reason</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((trade, i) => {
              const pnl = formatPnl(trade.pnl ?? 0);
              return (
                <tr key={i} className="border-b border-hover/50 hover:bg-hover">
                  <td className="px-2 py-1 font-mono">
                    {trade.entry_time ? formatTime(trade.entry_time) : "—"}
                  </td>
                  <td className="px-2 py-1">{trade.account_id ?? "—"}</td>
                  <td className={`px-2 py-1 font-semibold ${
                    trade.direction === "long" ? "text-green" : "text-red"
                  }`}>
                    {trade.direction?.toUpperCase() ?? "—"}
                  </td>
                  <td className="px-2 py-1 font-mono">
                    {trade.entry_price != null ? formatPrice(trade.entry_price) : "—"}
                  </td>
                  <td className="px-2 py-1 font-mono">
                    {trade.exit_price != null ? formatPrice(trade.exit_price) : "—"}
                  </td>
                  <td className={`px-2 py-1 font-mono font-semibold ${pnl.className}`}>
                    {pnl.text}
                  </td>
                  <td className="px-2 py-1 text-secondary">{trade.exit_reason ?? "—"}</td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7} className="px-2 py-4 text-center text-secondary">
                  No trades{filter !== "all" ? ` matching "${filter}"` : ""}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
