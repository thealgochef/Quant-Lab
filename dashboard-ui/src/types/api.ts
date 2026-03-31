/** Types matching backend API response shapes. */

export interface PriceUpdate {
  price: number;
  bid: number | null;
  ask: number | null;
  timestamp: string;
}

export interface LevelInfo {
  type: string;
  price: number;
  is_manual: boolean;
}

export interface LevelZone {
  zone_id: string;
  price: number;
  side: string;
  is_touched: boolean;
  levels: LevelInfo[];
}

export interface ObservationInfo {
  event_id: string;
  direction: string;
  level_price: number;
  start_time: string;
  end_time: string;
  status: string;
  trades_accumulated: number;
}

export interface PositionInfo {
  account_id: string;
  direction: string;
  entry_price: number;
  contracts: number;
  entry_time: string;
  unrealized_pnl: number;
  tp_price?: number;
  sl_price?: number;
}

export interface AccountInfo {
  account_id: string;
  label: string;
  group: string;
  balance: number;
  profit?: number;
  status: string;
  tier: number;
  max_contracts?: number;
  has_position: boolean;
  daily_pnl: number;
  eval_cost?: number;
  activation_cost?: number;
  daily_loss_limit?: number;
  liquidation_threshold?: number;
  peak_balance?: number;
  safety_net_reached?: boolean;
  payout_number?: number;
  qualifying_days?: number;
}

export interface SessionStats {
  signals_fired: number;
  wins: number;
  losses: number;
  accuracy: number;
}

export interface ConfigData {
  group_a_tp: number;
  group_b_tp: number;
  group_a_sl: number;
  group_b_sl: number;
  second_signal_mode: string;
  overlays: Record<string, boolean>;
}

export interface BackfillData {
  connection_status: string;
  latest_price: number | null;
  latest_bid: number | null;
  latest_ask: number | null;
  active_levels: LevelZone[];
  active_observation: ObservationInfo | null;
  last_prediction: Record<string, unknown> | null;
  open_positions: PositionInfo[];
  todays_trades: Record<string, unknown>[];
  todays_predictions: Record<string, unknown>[];
  session_stats: SessionStats;
  accounts: AccountInfo[];
  config: ConfigData;
  replay_mode?: boolean;
}

export interface PerformanceData {
  total_trades: number;
  wins: number;
  losses: number;
  total_pnl: number;
  win_rate: number;
  prediction_accuracy: number;
}

export interface PortfolioSummary {
  total_accounts: number;
  active_count: number;
  total_invested: number;
  total_payouts: number;
  total_profit: number;
  total_balance: number;
}

export interface ModelVersion {
  id: number;
  version: string;
  is_active: boolean;
  metrics: Record<string, unknown> | null;
  uploaded_at: string | null;
  activated_at: string | null;
}

export interface ClosedTrade {
  account_id: string;
  direction: string;
  entry_price: number;
  exit_price: number;
  contracts: number;
  entry_time: string;
  exit_time: string;
  pnl: number;
  pnl_points: number;
  exit_reason: string;
  group: string;
}
