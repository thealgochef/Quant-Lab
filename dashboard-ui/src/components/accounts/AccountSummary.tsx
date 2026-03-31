import type { PortfolioSummary } from "../../types/api";
import { formatPrice } from "../../utils/formatters";

interface Props {
  summary: PortfolioSummary | null;
}

export default function AccountSummary({ summary }: Props) {
  if (!summary) return null;

  const roi = summary.total_invested > 0 ? summary.total_profit / summary.total_invested : 0;

  return (
    <div className="grid grid-cols-5 gap-4">
      <SummaryCard label="Total Invested" value={`$${formatPrice(summary.total_invested)}`} />
      <SummaryCard label="Total Payouts" value={`$${formatPrice(summary.total_payouts)}`} />
      <SummaryCard
        label="Total Profit"
        value={`$${formatPrice(summary.total_profit)}`}
        color={summary.total_profit >= 0 ? "text-green" : "text-red"}
      />
      <SummaryCard
        label="ROI"
        value={`${(roi * 100).toFixed(1)}%`}
        color={roi >= 0 ? "text-green" : "text-red"}
      />
      <SummaryCard label="Active / Total" value={`${summary.active_count} / ${summary.total_accounts}`} />
    </div>
  );
}

function SummaryCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded bg-hover p-3">
      <div className="text-xs text-secondary">{label}</div>
      <div className={`mt-1 font-mono text-base font-semibold ${color ?? "text-primary"}`}>
        {value}
      </div>
    </div>
  );
}
