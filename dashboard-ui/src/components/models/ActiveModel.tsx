import type { ModelVersion } from "../../types/api";

interface Props {
  model: ModelVersion | null;
}

export default function ActiveModel({ model }: Props) {
  if (!model) {
    return (
      <div className="rounded bg-hover p-3 text-xs text-secondary">
        No active model
      </div>
    );
  }

  return (
    <div className="rounded bg-hover p-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-primary">
          Version {model.version}
        </span>
        <span className="rounded bg-green/20 px-2 py-0.5 text-xs font-medium text-green">
          Active
        </span>
      </div>
      <div className="mt-1 text-xs text-secondary">
        {model.activated_at && <span>Activated: {new Date(model.activated_at).toLocaleDateString()}</span>}
      </div>
      {model.metrics && (
        <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
          {Object.entries(model.metrics).map(([k, v]) => (
            <div key={k}>
              <span className="text-secondary">{k}: </span>
              <span className="font-mono text-primary">{String(v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
