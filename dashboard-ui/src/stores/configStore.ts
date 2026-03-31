import { create } from "zustand";
import type { ConfigData } from "../types/api";

interface ConfigState {
  groupATP: number;
  groupBTP: number;
  groupASL: number;
  groupBSL: number;
  secondSignalMode: string;
  overlays: Record<string, boolean>;

  applyBackfill: (config: ConfigData) => void;
  updateConfig: (updates: Partial<Pick<ConfigState, "groupATP" | "groupBTP" | "groupASL" | "groupBSL" | "secondSignalMode">>) => void;
  setOverlay: (key: string, value: boolean) => void;
  setOverlays: (overlays: Record<string, boolean>) => void;
}

export const useConfigStore = create<ConfigState>((set) => ({
  groupATP: 15,
  groupBTP: 30,
  groupASL: 15,
  groupBSL: 30,
  secondSignalMode: "ignore",
  overlays: {
    ema_13: true,
    ema_48: true,
    ema_200: true,
    vwap: false,
    levels: true,
    predictions: true,
  },

  applyBackfill: (config) =>
    set({
      groupATP: config.group_a_tp,
      groupBTP: config.group_b_tp,
      groupASL: config.group_a_sl,
      groupBSL: config.group_b_sl,
      secondSignalMode: config.second_signal_mode,
      overlays: config.overlays,
    }),

  updateConfig: (updates) => set(updates),

  setOverlay: (key, value) =>
    set((state) => ({
      overlays: { ...state.overlays, [key]: value },
    })),

  setOverlays: (overlays) => set({ overlays }),
}));
