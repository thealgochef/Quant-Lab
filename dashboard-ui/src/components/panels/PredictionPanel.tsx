import { useTradingStore } from "../../stores/tradingStore";

const CLASS_COLORS: Record<string, string> = {
  tradeable_reversal: "text-green",
  reversal: "text-green",
  trap: "text-orange",
  blowthrough: "text-red",
};

export default function PredictionPanel() {
  const prediction = useTradingStore((s) => s.lastPrediction);

  if (!prediction) {
    return (
      <div className="space-y-1" data-testid="prediction-panel">
        <h3 className="text-xs font-semibold uppercase text-secondary">Latest Prediction</h3>
        <div className="text-xs text-secondary">No predictions yet</div>
      </div>
    );
  }

  const predictedClass = String(prediction["predicted_class"] ?? "unknown");
  const isExecutable = prediction["is_executable"] as boolean | undefined;
  const colorClass = CLASS_COLORS[predictedClass] ?? "text-primary";

  return (
    <div className="space-y-1" data-testid="prediction-panel">
      <h3 className="text-xs font-semibold uppercase text-secondary">Latest Prediction</h3>
      <div className="rounded bg-hover p-2">
        <div className="flex items-center justify-between">
          <span className={`text-sm font-semibold ${colorClass}`} data-testid="predicted-class">
            {predictedClass.replace(/_/g, " ")}
          </span>
          {isExecutable !== undefined && (
            <span className={`text-xs ${isExecutable ? "text-green" : "text-secondary"}`}>
              {isExecutable ? "Executable" : "Display only"}
            </span>
          )}
        </div>
        {prediction["event_id"] != null && (
          <div className="mt-1 text-xs text-secondary">
            Event: {String(prediction["event_id"])}
          </div>
        )}
      </div>
    </div>
  );
}
