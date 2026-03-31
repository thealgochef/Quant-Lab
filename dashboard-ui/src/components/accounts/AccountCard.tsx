import { useState } from "react";
import type { AccountInfo } from "../../types/api";
import { formatPrice, formatPnl } from "../../utils/formatters";
import { requestPayout } from "../../hooks/useApi";

interface Props {
  account: AccountInfo;
}

const STATUS_BADGE: Record<string, string> = {
  active: "bg-green/20 text-green",
  blown: "bg-red/20 text-red",
  retired: "bg-secondary/20 text-secondary",
};

export default function AccountCard({ account }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [payoutAmount, setPayoutAmount] = useState("");
  const [payoutError, setPayoutError] = useState<string | null>(null);

  const pnl = formatPnl(account.daily_pnl);
  const badgeClass = STATUS_BADGE[account.status] ?? STATUS_BADGE["active"]!;

  async function handlePayout() {
    const amount = parseFloat(payoutAmount);
    if (isNaN(amount) || amount <= 0) return;
    try {
      setPayoutError(null);
      await requestPayout(account.account_id, amount);
      setPayoutAmount("");
    } catch (err) {
      setPayoutError(err instanceof Error ? err.message : "Payout failed");
    }
  }

  return (
    <div className="rounded bg-hover">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-primary">{account.label}</span>
          <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${badgeClass}`}>
            {account.status}
          </span>
          <span className="text-xs text-secondary">Group {account.group}</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono text-sm text-primary">
            ${formatPrice(account.balance)}
          </span>
          <span className={`font-mono text-xs ${pnl.className}`}>{pnl.text}</span>
          <span className="text-xs text-secondary">{expanded ? "▼" : "▶"}</span>
        </div>
      </button>

      {expanded && (
        <div className="border-t border-darkest px-3 py-2">
          <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
            <Detail label="Eval Cost" value={`$${formatPrice(account.eval_cost ?? 0)}`} />
            <Detail label="Activation" value={`$${formatPrice(account.activation_cost ?? 0)}`} />
            <Detail label="Tier" value={String(account.tier)} />
            <Detail label="Max Contracts" value={String(account.max_contracts ?? "—")} />
            <Detail label="Peak Balance" value={`$${formatPrice(account.peak_balance ?? 0)}`} />
            <Detail label="Liquidation" value={`$${formatPrice(account.liquidation_threshold ?? 0)}`} />
            <Detail label="Safety Net" value={account.safety_net_reached ? "Yes" : "No"} />
            <Detail label="Payout #" value={String(account.payout_number ?? 0)} />
            <Detail label="Qualifying Days" value={String(account.qualifying_days ?? 0)} />
          </div>

          {/* Payout form */}
          <div className="mt-2 flex items-center gap-2">
            <input
              type="text"
              value={payoutAmount}
              onChange={(e) => setPayoutAmount(e.target.value)}
              placeholder="Payout amount"
              className="w-32 rounded bg-darkest px-2 py-1 font-mono text-xs text-primary outline-none focus:ring-1 focus:ring-blue"
            />
            <button
              onClick={handlePayout}
              className="rounded bg-blue px-2 py-1 text-xs font-medium text-white hover:opacity-90"
            >
              Request Payout
            </button>
            {payoutError && <span className="text-xs text-red">{payoutError}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <>
      <span className="text-secondary">{label}</span>
      <span className="col-span-2 font-mono text-primary">{value}</span>
    </>
  );
}
