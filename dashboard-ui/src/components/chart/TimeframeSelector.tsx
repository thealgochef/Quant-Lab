import { ALL_TIMEFRAMES } from "../../utils/constants";
import type { Timeframe } from "../../types/chart";

interface Props {
  active: Timeframe;
  onChange: (tf: Timeframe) => void;
}

export default function TimeframeSelector({ active, onChange }: Props) {
  return (
    <div className="flex gap-1">
      {ALL_TIMEFRAMES.map((tf) => (
        <button
          key={tf}
          onClick={() => onChange(tf)}
          className={`rounded px-2 py-0.5 font-mono text-xs font-medium transition-colors ${
            active === tf
              ? "bg-blue text-white"
              : "text-secondary hover:bg-hover hover:text-primary"
          }`}
        >
          {tf}
        </button>
      ))}
    </div>
  );
}
