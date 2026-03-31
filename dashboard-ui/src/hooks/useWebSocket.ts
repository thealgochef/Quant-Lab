import { useEffect, useRef, useCallback } from "react";
import { useTradingStore } from "../stores/tradingStore";
import { useAccountStore } from "../stores/accountStore";
import { useConfigStore } from "../stores/configStore";
import type { BackfillData } from "../types/api";

const DEFAULT_WS_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;
const PING_INTERVAL_MS = 30000;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const pingTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const wsUrlRef = useRef(DEFAULT_WS_URL);
  const mountedRef = useRef(true);
  const connectRef = useRef<() => void>(() => {});

  const sendMessage = useCallback((msg: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }, []);

  const reconnectTo = useCallback((url: string) => {
    wsUrlRef.current = url;
    reconnectAttempt.current = 0;
    // Close current connection — onclose will NOT auto-reconnect
    // because we manually reconnect right after
    const ws = wsRef.current;
    if (ws) {
      ws.onclose = null; // prevent auto-reconnect from old onclose
      if (pingTimer.current) {
        clearInterval(pingTimer.current);
        pingTimer.current = null;
      }
      ws.close();
    }
    useTradingStore.getState().setConnectionStatus("reconnecting");
    connectRef.current();
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    function connect() {
      if (!mountedRef.current) return;

      const ws = new WebSocket(wsUrlRef.current);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectAttempt.current = 0;
        useTradingStore.getState().setConnectionStatus("connected");

        // Subscribe to default tick-bar timeframe
        ws.send(JSON.stringify({
          type: "subscribe_timeframe",
          data: { timeframe: "987t" },
        }));

        // Start ping/pong heartbeat
        pingTimer.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, PING_INTERVAL_MS);
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data as string) as { type: string; data?: unknown };
        handleMessage(msg);
      };

      ws.onclose = () => {
        if (pingTimer.current) {
          clearInterval(pingTimer.current);
          pingTimer.current = null;
        }
        useTradingStore.getState().setConnectionStatus("disconnected");

        if (mountedRef.current) {
          const delay = Math.min(
            RECONNECT_BASE_MS * 2 ** reconnectAttempt.current,
            RECONNECT_MAX_MS,
          );
          reconnectAttempt.current += 1;
          useTradingStore.getState().setConnectionStatus("reconnecting");
          setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connectRef.current = connect;
    connect();

    return () => {
      mountedRef.current = false;
      if (pingTimer.current) clearInterval(pingTimer.current);
      wsRef.current?.close();
    };
  }, []);

  return { sendMessage, reconnectTo };
}

function handleMessage(msg: { type: string; data?: unknown }) {
  const trading = useTradingStore.getState();
  const accounts = useAccountStore.getState();
  const config = useConfigStore.getState();

  switch (msg.type) {
    case "backfill": {
      const data = msg.data as BackfillData;
      // REPLACE all state (not merge) per user requirement
      trading.applyBackfill(data);
      accounts.setAccounts(data.accounts);
      config.applyBackfill(data.config);
      break;
    }
    case "price_update": {
      const data = msg.data as { price: number; bid: number | null; ask: number | null; timestamp: string };
      trading.updatePrice(data.price, data.bid, data.ask, data.timestamp);
      break;
    }
    case "bar_update": {
      const data = msg.data as { timeframe: string; bar: { timestamp: string; open: number; high: number; low: number; close: number; volume: number } };
      trading.appendBar(data.timeframe, data.bar);
      break;
    }
    case "prediction": {
      trading.setLastPrediction(msg.data as Record<string, unknown>);
      break;
    }
    case "trade_opened": {
      trading.addTradeOpened(msg.data as Record<string, unknown>);
      break;
    }
    case "trade_closed": {
      trading.addTradeClosed(msg.data as Record<string, unknown>);
      break;
    }
    case "outcome_resolved": {
      const data = msg.data as {
        event_id: string;
        predicted_class: string;
        actual_class: string;
        prediction_correct: boolean;
        mfe_points: number;
        mae_points: number;
        resolution_type: string;
      };
      trading.resolveOutcome(data);
      break;
    }
    case "account_update": {
      const data = msg.data as { account_id: string; balance: number; profit: number; daily_pnl: number; group: string; status: string; has_position: boolean };
      accounts.updateAccount(data.account_id, {
        balance: data.balance,
        daily_pnl: data.daily_pnl,
        status: data.status,
        has_position: data.has_position,
      });
      break;
    }
    case "observation_started": {
      trading.setActiveObservation(msg.data as Record<string, unknown>);
      break;
    }
    case "level_update": {
      trading.updateLevels(msg.data as Record<string, unknown>);
      break;
    }
    case "session_stats": {
      const data = msg.data as { signals_fired: number; wins: number; losses: number; accuracy: number };
      trading.updateSessionStats(data);
      break;
    }
    case "replay_day": {
      const data = msg.data as { date: string };
      trading.setReplayDate(data.date);
      break;
    }
    case "replay_state": {
      const data = msg.data as { paused: boolean; step_mode: boolean; speed: number; current_date: string | null };
      trading.updateReplayState(data);
      break;
    }
    case "connection_status": {
      const data = msg.data as { status: string };
      trading.setConnectionStatus(data.status);
      break;
    }
    case "pong":
      break;
  }
}
