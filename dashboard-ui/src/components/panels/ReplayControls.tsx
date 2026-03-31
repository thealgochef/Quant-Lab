import { useState, useCallback } from "react";
import { useTradingStore } from "../../stores/tradingStore";
import { COLORS } from "../../utils/constants";

interface Props {
  sendMessage: (msg: Record<string, unknown>) => void;
}

export default function ReplayControls({ sendMessage }: Props) {
  const [speed, setSpeed] = useState(10);
  const replayDate = useTradingStore((s) => s.replayDate);
  const replayPaused = useTradingStore((s) => s.replayPaused);
  const replayStepMode = useTradingStore((s) => s.replayStepMode);
  const playing = !replayPaused;
  const stepMode = replayStepMode;

  const sendReplayControl = useCallback(
    (action: string, extra?: Record<string, unknown>) => {
      sendMessage({
        type: "replay_control",
        data: { action, ...extra },
      });
    },
    [sendMessage],
  );

  const handlePlayPause = useCallback(() => {
    if (playing) {
      sendReplayControl("pause");
    } else {
      sendReplayControl("play");
    }
  }, [playing, sendReplayControl]);

  const handleStep = useCallback(() => {
    sendReplayControl("step");
  }, [sendReplayControl]);

  const handleSpeedChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = parseFloat(e.target.value);
      // Logarithmic mapping: slider 0-100 -> speed 0.1-100
      const newSpeed = Math.round(0.1 * Math.pow(1000, val / 100) * 10) / 10;
      setSpeed(newSpeed);
      sendReplayControl("set_speed", { speed: newSpeed });
    },
    [sendReplayControl],
  );

  // Reverse: speed -> slider position (0-100)
  const sliderValue = Math.log(speed / 0.1) / Math.log(1000) * 100;

  const handleStepModeToggle = useCallback(() => {
    const newMode = !stepMode;
    sendReplayControl("set_step_mode", { enabled: newMode });
    if (newMode) {
      sendReplayControl("pause");
    }
  }, [stepMode, sendReplayControl]);

  return (
    <div
      className="flex items-center gap-3 border-b px-3 py-1.5 font-mono text-xs"
      style={{ borderColor: COLORS.hover, background: COLORS.panel }}
    >
      {/* Play/Pause */}
      <button
        onClick={handlePlayPause}
        className="rounded px-2 py-0.5 text-white"
        style={{ background: playing ? COLORS.red : COLORS.green }}
        title={playing ? "Pause" : "Play"}
      >
        {playing ? "Pause" : "Play"}
      </button>

      {/* Step */}
      <button
        onClick={handleStep}
        className="rounded px-2 py-0.5 text-white"
        style={{ background: COLORS.blue }}
        title="Step one bar forward"
      >
        Step
      </button>

      {/* Step mode toggle */}
      <label className="flex items-center gap-1 text-secondary">
        <input
          type="checkbox"
          checked={stepMode}
          onChange={handleStepModeToggle}
          className="accent-blue"
        />
        Step Mode
      </label>

      {/* Speed slider */}
      <div className="flex items-center gap-1">
        <span className="text-secondary">Speed:</span>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={sliderValue}
          onChange={handleSpeedChange}
          className="w-20 accent-blue"
        />
        <span className="w-10 text-right text-primary">{speed}x</span>
      </div>

      {/* Replay date */}
      {replayDate && (
        <span className="ml-auto text-secondary">
          Replay: <span className="text-primary">{replayDate}</span>
        </span>
      )}
    </div>
  );
}
