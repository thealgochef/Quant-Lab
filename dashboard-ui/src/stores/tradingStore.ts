import { create } from "zustand";
import type {
  BackfillData,
  LevelZone,
  ObservationInfo,
  PositionInfo,
  SessionStats,
} from "../types/api";

interface BarUpdate {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface OutcomeData {
  event_id: string;
  predicted_class: string;
  actual_class: string;
  prediction_correct: boolean;
  mfe_points: number;
  mae_points: number;
  resolution_type: string;
}

interface TradingState {
  // Market data
  price: number | null;
  bid: number | null;
  ask: number | null;
  priceTimestamp: string | null;

  // Levels
  activeLevels: LevelZone[];

  // Positions
  openPositions: PositionInfo[];

  // Predictions
  lastPrediction: Record<string, unknown> | null;
  todaysPredictions: Record<string, unknown>[];

  // Trades
  todaysTrades: Record<string, unknown>[];

  // Observation
  activeObservation: ObservationInfo | null;

  // Session stats
  sessionStats: SessionStats;

  // Connection
  connectionStatus: string;

  // Replay
  replayMode: boolean;
  replayDate: string | null;
  replayPaused: boolean;
  replayStepMode: boolean;

  // Streaming bar updates
  latestBar: { timeframe: string; bar: BarUpdate } | null;

  // Actions
  applyBackfill: (data: BackfillData) => void;
  updatePrice: (price: number, bid: number | null, ask: number | null, timestamp: string) => void;
  setLastPrediction: (prediction: Record<string, unknown>) => void;
  addTradeOpened: (trade: Record<string, unknown>) => void;
  addTradeClosed: (trade: Record<string, unknown>) => void;
  setActiveObservation: (obs: Record<string, unknown>) => void;
  updateLevels: (data: Record<string, unknown>) => void;
  setConnectionStatus: (status: string) => void;
  appendBar: (timeframe: string, bar: BarUpdate) => void;
  setReplayDate: (date: string) => void;
  updateSessionStats: (stats: SessionStats) => void;
  updateReplayState: (state: { paused: boolean; step_mode: boolean; speed: number; current_date: string | null }) => void;
  resolveOutcome: (data: OutcomeData) => void;
}

export const useTradingStore = create<TradingState>((set) => ({
  price: null,
  bid: null,
  ask: null,
  priceTimestamp: null,
  activeLevels: [],
  openPositions: [],
  lastPrediction: null,
  todaysPredictions: [],
  todaysTrades: [],
  activeObservation: null,
  sessionStats: { signals_fired: 0, wins: 0, losses: 0, accuracy: 0 },
  connectionStatus: "disconnected",
  replayMode: false,
  replayDate: null,
  replayPaused: true,
  replayStepMode: true,
  latestBar: null,

  applyBackfill: (data) =>
    set({
      price: data.latest_price,
      bid: data.latest_bid,
      ask: data.latest_ask,
      activeLevels: data.active_levels,
      openPositions: data.open_positions,
      lastPrediction: data.last_prediction,
      todaysPredictions: data.todays_predictions,
      todaysTrades: data.todays_trades,
      activeObservation: data.active_observation,
      sessionStats: data.session_stats,
      connectionStatus: data.connection_status,
      replayMode: data.replay_mode ?? false,
    }),

  updatePrice: (price, bid, ask, timestamp) =>
    set({ price, bid, ask, priceTimestamp: timestamp }),

  setLastPrediction: (prediction) =>
    set((state) => ({
      lastPrediction: prediction,
      todaysPredictions: [...state.todaysPredictions, prediction],
    })),

  // Fix 2B: trade_opened → add to todaysTrades AND openPositions
  addTradeOpened: (trade) =>
    set((state) => ({
      todaysTrades: [...state.todaysTrades, trade],
      openPositions: [
        ...state.openPositions,
        {
          account_id: trade.account_id as string,
          direction: trade.direction as string,
          entry_price: trade.entry_price as number,
          contracts: trade.contracts as number,
          entry_time: trade.entry_time as string,
          unrealized_pnl: 0,
          tp_price: trade.tp_price as number | undefined,
          sl_price: trade.sl_price as number | undefined,
        },
      ],
    })),

  // Fix 2B: trade_closed → add to todaysTrades AND remove from openPositions
  addTradeClosed: (trade) =>
    set((state) => ({
      todaysTrades: [...state.todaysTrades, trade],
      openPositions: state.openPositions.filter(
        (p) => p.account_id !== (trade.account_id as string),
      ),
    })),

  setActiveObservation: (obs) =>
    set({ activeObservation: obs as unknown as ObservationInfo }),

  updateLevels: (data) => {
    const levels = data["levels"] as LevelZone[] | undefined;
    if (levels) {
      set({ activeLevels: levels });
    }
  },

  setConnectionStatus: (status) => set({ connectionStatus: status }),

  appendBar: (timeframe, bar) =>
    set({ latestBar: { timeframe, bar } }),

  setReplayDate: (date) => set({ replayDate: date }),

  updateSessionStats: (stats) => set({ sessionStats: stats }),

  updateReplayState: (rs) =>
    set({
      replayPaused: rs.paused,
      replayStepMode: rs.step_mode,
      replayDate: rs.current_date,
    }),

  // Fix 2A: Update matching prediction with outcome data
  resolveOutcome: (data) =>
    set((state) => ({
      todaysPredictions: state.todaysPredictions.map((p) =>
        p.event_id === data.event_id
          ? {
              ...p,
              prediction_correct: data.prediction_correct,
              actual_class: data.actual_class,
              mfe_points: data.mfe_points,
              mae_points: data.mae_points,
              resolution_type: data.resolution_type,
            }
          : p,
      ),
    })),
}));
