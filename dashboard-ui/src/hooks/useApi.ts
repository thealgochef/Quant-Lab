import type {
  AccountInfo,
  ClosedTrade,
  ConfigData,
  ModelVersion,
  PerformanceData,
  PortfolioSummary,
} from "../types/api";

const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// --- Trading ---

export async function closeAllPositions(reason: string) {
  return request<{ closed_trades: ClosedTrade[] }>("/trading/close-all", {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}

export async function closeAccountPosition(accountId: string, reason: string) {
  return request<{ closed_trade: ClosedTrade }>(`/trading/close/${accountId}`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}

export async function manualEntry(direction: string) {
  return request<{ positions: Record<string, unknown>[]; count: number }>("/trading/manual-entry", {
    method: "POST",
    body: JSON.stringify({ direction }),
  });
}

// --- Accounts ---

export async function fetchAccounts() {
  return request<{ accounts: AccountInfo[]; summary: PortfolioSummary }>("/accounts");
}

export async function addAccount(label: string, evalCost: number, activationCost: number, group: string) {
  return request<{ account: AccountInfo }>("/accounts", {
    method: "POST",
    body: JSON.stringify({ label, eval_cost: evalCost, activation_cost: activationCost, group }),
  });
}

export async function fetchAccount(accountId: string) {
  return request<{ account: AccountInfo; trade_history: Record<string, unknown>[] }>(
    `/accounts/${accountId}`,
  );
}

export async function requestPayout(accountId: string, amount: number) {
  return request<{ payout: Record<string, unknown> }>(`/accounts/${accountId}/payout`, {
    method: "POST",
    body: JSON.stringify({ amount }),
  });
}

// --- Config ---

export async function fetchConfig() {
  return request<ConfigData>("/config");
}

export async function updateConfig(updates: Partial<ConfigData>) {
  return request<{ config: ConfigData }>("/config", {
    method: "PUT",
    body: JSON.stringify(updates),
  });
}

export async function fetchOverlays() {
  return request<{ overlays: Record<string, boolean> }>("/config/overlays");
}

export async function updateOverlays(overlays: Record<string, boolean>) {
  return request<{ overlays: Record<string, boolean> }>("/config/overlays", {
    method: "PUT",
    body: JSON.stringify({ overlays }),
  });
}

// --- Data ---

export async function fetchTrades() {
  return request<{ trades: Record<string, unknown>[] }>("/data/trades");
}

export async function fetchPredictions() {
  return request<{ predictions: Record<string, unknown>[] }>("/data/predictions");
}

export async function fetchPerformance() {
  return request<PerformanceData>("/data/performance");
}

export async function fetchEquityCurve(accountId?: string) {
  const q = accountId ? `?account_id=${accountId}` : "";
  return request<{ snapshots: Record<string, unknown>[] }>(`/data/equity-curve${q}`);
}

export async function fetchOHLCV(timeframe = "987t", since?: string) {
  const params = new URLSearchParams({ timeframe });
  if (since) params.set("since", since);
  return request<{ bars: Record<string, unknown>[]; timeframe: string }>(
    `/data/ohlcv?${params.toString()}`,
  );
}

// --- Levels ---

export async function fetchLevels() {
  return request<{ zones: Record<string, unknown>[]; manual_levels: Record<string, unknown>[] }>(
    "/levels",
  );
}

export async function addManualLevel(price: number) {
  return request<{ level: Record<string, unknown> }>("/levels/manual", {
    method: "POST",
    body: JSON.stringify({ price }),
  });
}

export async function deleteManualLevel(price: number) {
  return request<{ deleted: boolean }>(`/levels/manual/${price}`, {
    method: "DELETE",
  });
}

// --- Models ---

export async function fetchModels() {
  return request<{ active: ModelVersion | null; versions: ModelVersion[] }>("/models");
}

export async function uploadModel(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE}/models/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`);
  return res.json() as Promise<{ version: ModelVersion }>;
}

export async function activateModel(versionId: number) {
  return request<{ activated: boolean }>(`/models/${versionId}/activate`, {
    method: "POST",
  });
}

export async function rollbackModel(versionId: number) {
  return request<{ activated: boolean }>(`/models/${versionId}/rollback`, {
    method: "POST",
  });
}
