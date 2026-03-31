import { useConfigStore } from "../../stores/configStore";
import { updateOverlays } from "../../hooks/useApi";
import { OVERLAY_LABELS } from "../../utils/constants";

export default function OverlayToggles() {
  const overlays = useConfigStore((s) => s.overlays);
  const setOverlay = useConfigStore((s) => s.setOverlay);

  async function handleToggle(key: string) {
    const newValue = !overlays[key];
    setOverlay(key, newValue);
    await updateOverlays({ ...overlays, [key]: newValue }).catch(() => {
      // Revert on failure
      setOverlay(key, !newValue);
    });
  }

  return (
    <div className="space-y-1" data-testid="overlay-toggles">
      <h3 className="text-xs font-semibold uppercase text-secondary">Overlays</h3>
      <div className="space-y-1">
        {Object.entries(OVERLAY_LABELS).map(([key, label]) => (
          <label
            key={key}
            className="flex cursor-pointer items-center gap-2 text-xs"
          >
            <input
              type="checkbox"
              checked={overlays[key] ?? false}
              onChange={() => handleToggle(key)}
              className="accent-blue"
              data-testid={`overlay-${key}`}
            />
            <span className="text-primary">{label}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
