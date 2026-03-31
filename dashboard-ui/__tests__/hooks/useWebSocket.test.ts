import { describe, it, expect, beforeEach } from "vitest";
import { useTradingStore } from "../../src/stores/tradingStore";
import { useAccountStore } from "../../src/stores/accountStore";
import { useConfigStore } from "../../src/stores/configStore";
import type { BackfillData } from "../../src/types/api";

/**
 * WebSocket integration tests.
 *
 * These test the store update logic directly (simulating what
 * useWebSocket.handleMessage does) rather than creating real
 * WebSocket connections, since we're testing state management
 * not network connectivity.
 */

const MOCK_BACKFILL: BackfillData = {
  connection_status: "connected",
  latest_price: 21045.75,
  latest_bid: 21045.50,
  latest_ask: 21046.00,
  active_levels: [
    {
      zone_id: "z1",
      price: 21000.0,
      side: "HIGH",
      is_touched: false,
      levels: [{ type: "pdh", price: 21000.0, is_manual: false }],
    },
  ],
  active_observation: null,
  last_prediction: { predicted_class: "tradeable_reversal", is_executable: true },
  open_positions: [
    {
      account_id: "APEX-001",
      direction: "long",
      entry_price: 21040.0,
      contracts: 1,
      entry_time: "2026-03-02T14:30:00Z",
      unrealized_pnl: 100.0,
    },
  ],
  todays_trades: [{ account_id: "APEX-001", pnl: 300 }],
  todays_predictions: [{ predicted_class: "tradeable_reversal", prediction_correct: true }],
  session_stats: { signals_fired: 5, wins: 3, losses: 2, accuracy: 0.6 },
  accounts: [
    {
      account_id: "APEX-001",
      label: "Apex #1",
      group: "A",
      balance: 50300,
      status: "active",
      tier: 1,
      has_position: true,
      daily_pnl: 300,
    },
    {
      account_id: "APEX-002",
      label: "Apex #2",
      group: "B",
      balance: 49800,
      status: "active",
      tier: 1,
      has_position: false,
      daily_pnl: -200,
    },
  ],
  config: {
    group_a_tp: 15,
    group_b_tp: 30,
    group_a_sl: 15,
    group_b_sl: 30,
    second_signal_mode: "ignore",
    overlays: { ema_13: true, ema_48: true, ema_200: false, vwap: false, levels: true },
  },
};

describe("WebSocket integration", () => {
  beforeEach(() => {
    // Reset all stores to defaults
    useTradingStore.setState({
      price: null,
      bid: null,
      ask: null,
      activeLevels: [],
      openPositions: [],
      lastPrediction: null,
      todaysPredictions: [],
      todaysTrades: [],
      activeObservation: null,
      sessionStats: { signals_fired: 0, wins: 0, losses: 0, accuracy: 0 },
      connectionStatus: "disconnected",
    });
    useAccountStore.setState({ accounts: [], summary: null });
    useConfigStore.setState({
      groupATP: 15,
      groupBTP: 30,
      groupASL: 15,
      groupBSL: 30,
      secondSignalMode: "ignore",
      overlays: { ema_13: true, ema_48: true, ema_200: true, vwap: false, levels: true },
    });
  });

  it("backfill message populates all Zustand stores", () => {
    const trading = useTradingStore.getState();
    trading.applyBackfill(MOCK_BACKFILL);
    useAccountStore.getState().setAccounts(MOCK_BACKFILL.accounts);
    useConfigStore.getState().applyBackfill(MOCK_BACKFILL.config);

    // Trading store
    const ts = useTradingStore.getState();
    expect(ts.price).toBe(21045.75);
    expect(ts.bid).toBe(21045.50);
    expect(ts.ask).toBe(21046.00);
    expect(ts.activeLevels).toHaveLength(1);
    expect(ts.openPositions).toHaveLength(1);
    expect(ts.lastPrediction).toEqual({ predicted_class: "tradeable_reversal", is_executable: true });
    expect(ts.todaysTrades).toHaveLength(1);
    expect(ts.todaysPredictions).toHaveLength(1);
    expect(ts.sessionStats.signals_fired).toBe(5);
    expect(ts.connectionStatus).toBe("connected");

    // Account store
    const as = useAccountStore.getState();
    expect(as.accounts).toHaveLength(2);
    expect(as.accounts[0]!.account_id).toBe("APEX-001");

    // Config store
    const cs = useConfigStore.getState();
    expect(cs.groupATP).toBe(15);
    expect(cs.overlays["ema_200"]).toBe(false);
  });

  it("price_update message updates tradingStore price", () => {
    useTradingStore.getState().updatePrice(21050.25, 21050.0, 21050.5, "2026-03-02T14:31:00Z");

    const ts = useTradingStore.getState();
    expect(ts.price).toBe(21050.25);
    expect(ts.bid).toBe(21050.0);
    expect(ts.ask).toBe(21050.5);
    expect(ts.priceTimestamp).toBe("2026-03-02T14:31:00Z");
  });

  it("prediction message updates tradingStore.lastPrediction", () => {
    useTradingStore.getState().setLastPrediction({
      event_id: "evt_99",
      predicted_class: "trap",
      is_executable: false,
    });

    const ts = useTradingStore.getState();
    expect(ts.lastPrediction).toEqual({
      event_id: "evt_99",
      predicted_class: "trap",
      is_executable: false,
    });
    expect(ts.todaysPredictions).toHaveLength(1);
  });

  it("after disconnect/reconnect, stores refresh from new backfill", () => {
    // First backfill
    useTradingStore.getState().applyBackfill(MOCK_BACKFILL);
    expect(useTradingStore.getState().price).toBe(21045.75);
    expect(useTradingStore.getState().openPositions).toHaveLength(1);

    // Simulate reconnect with different data (REPLACE, not merge)
    const reconnectBackfill: BackfillData = {
      ...MOCK_BACKFILL,
      latest_price: 21100.0,
      open_positions: [], // positions closed during disconnect
      todays_trades: [
        { account_id: "APEX-001", pnl: 300 },
        { account_id: "APEX-001", pnl: -50 },
      ],
      session_stats: { signals_fired: 7, wins: 4, losses: 3, accuracy: 0.5714 },
    };

    useTradingStore.getState().applyBackfill(reconnectBackfill);

    const ts = useTradingStore.getState();
    expect(ts.price).toBe(21100.0); // Updated price
    expect(ts.openPositions).toHaveLength(0); // Positions were REPLACED, not merged
    expect(ts.todaysTrades).toHaveLength(2); // New trade list from backfill
    expect(ts.sessionStats.signals_fired).toBe(7);
  });
});
