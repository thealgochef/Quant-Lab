import type { ModelVersion } from "../../types/api";
import { activateModel, rollbackModel } from "../../hooks/useApi";

interface Props {
  versions: ModelVersion[];
  onRefresh: () => void;
}

export default function ModelHistory({ versions, onRefresh }: Props) {
  async function handleActivate(id: number) {
    await activateModel(id);
    onRefresh();
  }

  async function handleRollback(id: number) {
    await rollbackModel(id);
    onRefresh();
  }

  if (versions.length === 0) {
    return <div className="text-xs text-secondary">No model versions uploaded</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-hover text-left text-secondary">
            <th className="px-2 py-1">Version</th>
            <th className="px-2 py-1">Status</th>
            <th className="px-2 py-1">Uploaded</th>
            <th className="px-2 py-1">Activated</th>
            <th className="px-2 py-1">Actions</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((v) => (
            <tr key={v.id} className="border-b border-hover/50 hover:bg-hover">
              <td className="px-2 py-1 font-mono">{v.version}</td>
              <td className="px-2 py-1">
                {v.is_active ? (
                  <span className="text-green">Active</span>
                ) : (
                  <span className="text-secondary">Inactive</span>
                )}
              </td>
              <td className="px-2 py-1 text-secondary">
                {v.uploaded_at ? new Date(v.uploaded_at).toLocaleDateString() : "—"}
              </td>
              <td className="px-2 py-1 text-secondary">
                {v.activated_at ? new Date(v.activated_at).toLocaleDateString() : "—"}
              </td>
              <td className="px-2 py-1">
                {!v.is_active && (
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleActivate(v.id)}
                      className="rounded bg-blue px-1.5 py-0.5 text-xs text-white hover:opacity-90"
                    >
                      Activate
                    </button>
                    <button
                      onClick={() => handleRollback(v.id)}
                      className="rounded bg-hover px-1.5 py-0.5 text-xs text-secondary hover:text-primary"
                    >
                      Rollback
                    </button>
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
