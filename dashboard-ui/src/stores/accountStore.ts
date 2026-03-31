import { create } from "zustand";
import type { AccountInfo, PortfolioSummary } from "../types/api";

interface AccountState {
  accounts: AccountInfo[];
  summary: PortfolioSummary | null;

  setAccounts: (accounts: AccountInfo[]) => void;
  setSummary: (summary: PortfolioSummary) => void;
  addAccount: (account: AccountInfo) => void;
  updateAccount: (accountId: string, updates: Partial<AccountInfo>) => void;
}

export const useAccountStore = create<AccountState>((set) => ({
  accounts: [],
  summary: null,

  setAccounts: (accounts) => set({ accounts }),

  setSummary: (summary) => set({ summary }),

  addAccount: (account) =>
    set((state) => ({ accounts: [...state.accounts, account] })),

  updateAccount: (accountId, updates) =>
    set((state) => ({
      accounts: state.accounts.map((a) =>
        a.account_id === accountId ? { ...a, ...updates } : a,
      ),
    })),
}));
