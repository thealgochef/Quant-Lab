import { useRef, useState } from "react";
import { uploadModel } from "../../hooks/useApi";
import type { ModelVersion } from "../../types/api";

interface Props {
  onUpload: (version: ModelVersion) => void;
}

export default function ModelUpload({ onUpload }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload() {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    try {
      setUploading(true);
      setError(null);
      const result = await uploadModel(file);
      onUpload(result.version);
      if (fileRef.current) fileRef.current.value = "";
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="flex items-end gap-2">
      <div>
        <label className="block text-xs text-secondary">Upload Model (.cbm)</label>
        <input
          ref={fileRef}
          type="file"
          accept=".cbm"
          className="mt-0.5 text-xs text-primary file:mr-2 file:rounded file:border-0 file:bg-blue file:px-2 file:py-1 file:text-xs file:font-medium file:text-white"
        />
      </div>
      <button
        onClick={handleUpload}
        disabled={uploading}
        className="rounded bg-blue px-3 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
      >
        {uploading ? "Uploading..." : "Upload"}
      </button>
      {error && <span className="text-xs text-red">{error}</span>}
    </div>
  );
}
