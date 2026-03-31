import { useCallback, useEffect, useState } from "react";
import { fetchModels } from "../../hooks/useApi";
import type { ModelVersion } from "../../types/api";
import ActiveModel from "../models/ActiveModel";
import ModelUpload from "../models/ModelUpload";
import ModelHistory from "../models/ModelHistory";

export default function ModelManagement() {
  const [active, setActive] = useState<ModelVersion | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);

  const loadModels = useCallback(() => {
    fetchModels()
      .then((data) => {
        setActive(data.active);
        setVersions(data.versions);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  function handleUpload(version: ModelVersion) {
    setVersions((prev) => [version, ...prev]);
  }

  return (
    <div className="h-full overflow-y-auto p-4">
      <div className="space-y-4">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Active Model</h3>
          <ActiveModel model={active} />
        </div>

        <div>
          <h3 className="mb-2 text-sm font-semibold">Upload New Model</h3>
          <ModelUpload onUpload={handleUpload} />
        </div>

        <div>
          <h3 className="mb-2 text-sm font-semibold">Model History</h3>
          <ModelHistory versions={versions} onRefresh={loadModels} />
        </div>
      </div>
    </div>
  );
}
