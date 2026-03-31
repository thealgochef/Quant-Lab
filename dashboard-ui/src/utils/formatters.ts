/** Price formatting — NQ trades in 0.25 tick increments. */
export function formatPrice(price: number): string {
  return price.toFixed(2);
}

/** P&L formatting with sign and color class. */
export function formatPnl(pnl: number): { text: string; className: string } {
  const sign = pnl >= 0 ? "+" : "";
  return {
    text: `${sign}$${pnl.toFixed(2)}`,
    className: pnl >= 0 ? "text-green" : "text-red",
  };
}

/** Format percentage (0.6667 → "66.67%"). */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Format timestamp to HH:MM:SS. */
export function formatTime(isoString: string): string {
  const d = new Date(isoString);
  return d.toLocaleTimeString("en-US", { hour12: false });
}

/** Format timestamp to short date. */
export function formatDate(isoString: string): string {
  const d = new Date(isoString);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
