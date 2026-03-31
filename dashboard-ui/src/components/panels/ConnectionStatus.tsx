import { useTradingStore } from "../../stores/tradingStore";

const STATUS_CONFIG: Record<string, { color: string; label: string }> = {
  connected: { color: "bg-green", label: "Connected" },
  reconnecting: { color: "bg-yellow", label: "Reconnecting" },
  disconnected: { color: "bg-red", label: "Disconnected" },
};

export default function ConnectionStatus() {
  const status = useTradingStore((s) => s.connectionStatus);
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG["disconnected"]!;

  return (
    <div className="flex items-center gap-2" data-testid="connection-status">
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${config.color}`}
        data-testid="status-dot"
      />
      <span className="text-xs text-secondary" data-testid="status-text">
        {config.label}
      </span>
    </div>
  );
}
