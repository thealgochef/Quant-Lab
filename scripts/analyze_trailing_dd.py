"""Trace per-account trailing DD, safety net, and liquidation threshold."""
from __future__ import annotations

import pandas as pd
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

trades = pd.read_csv("data/experiment/backtest_oos_results_trades.csv")

# Use APEX-001 only (all 5 identical)
t1 = trades[trades["account_id"] == "APEX-001"].copy()
t1["entry_dt"] = pd.to_datetime(t1["entry_time"])
t1["exit_dt"] = pd.to_datetime(t1["exit_time"])
t1["date"] = t1["entry_dt"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d")

# Apex 4.0 50K constants
STARTING_BALANCE = 50_000.0
TRAILING_DD = 2_000.0
STARTING_LIQUIDATION = 48_000.0
SAFETY_NET_PEAK = 52_100.0
SAFETY_NET_LIQUIDATION = 50_100.0

balance = STARTING_BALANCE
peak = STARTING_BALANCE
liq_threshold = STARTING_LIQUIDATION
safety_net = False

print("=" * 120)
print("  TRAILING DRAWDOWN TIMELINE — Account A1")
print("=" * 120)
print()
print(f"  {'#':>3s}  {'Date':10s}  {'Dir':5s}  {'Result':>6s}  "
      f"{'P&L':>10s}  {'Balance':>12s}  {'Peak Bal':>12s}  "
      f"{'Liq Thresh':>12s}  {'Safety Net':>10s}  {'Buffer':>10s}")
print(f"  {'---':>3s}  {'----------':10s}  {'-----':5s}  {'------':>6s}  "
      f"{'----------':>10s}  {'------------':>12s}  {'------------':>12s}  "
      f"{'------------':>12s}  {'----------':>10s}  {'----------':>10s}")

# Starting state
print(f"  {'':>3s}  {'START':10s}  {'':5s}  {'':>6s}  "
      f"{'':>10s}  ${balance:>11,.2f}  ${peak:>11,.2f}  "
      f"${liq_threshold:>11,.2f}  {'NO':>10s}  ${balance - liq_threshold:>9,.2f}")

safety_net_date = None
min_buffer = balance - liq_threshold
min_buffer_date = "START"
min_buffer_trade = 0

for i, (_, r) in enumerate(t1.iterrows(), 1):
    pnl = r["pnl"]
    balance += pnl

    # Update peak and trailing DD (mirrors ApexAccount._update_trailing_dd)
    if balance > peak:
        peak = balance
        if not safety_net:
            liq_threshold = peak - TRAILING_DD
            if peak >= SAFETY_NET_PEAK:
                safety_net = True
                liq_threshold = SAFETY_NET_LIQUIDATION
                safety_net_date = r["date"]

    buffer = balance - liq_threshold
    if buffer < min_buffer:
        min_buffer = buffer
        min_buffer_date = r["date"]
        min_buffer_trade = i

    blown = "BLOWN!" if balance <= liq_threshold else ""

    exit_dt = r["exit_dt"].astimezone(ET)
    print(f"  {i:3d}  {exit_dt.strftime('%Y-%m-%d')}  {r['direction']:5s}  "
          f"{r['exit_reason']:>6s}  "
          f"${pnl:>+9,.2f}  ${balance:>11,.2f}  ${peak:>11,.2f}  "
          f"${liq_threshold:>11,.2f}  "
          f"{'LOCKED':>10s}" if safety_net else
          f"  {i:3d}  {exit_dt.strftime('%Y-%m-%d')}  {r['direction']:5s}  "
          f"{r['exit_reason']:>6s}  "
          f"${pnl:>+9,.2f}  ${balance:>11,.2f}  ${peak:>11,.2f}  "
          f"${liq_threshold:>11,.2f}  "
          f"{'trailing':>10s}",
          f"  ${buffer:>9,.2f}  {blown}")

# Summary
print()
print("=" * 120)
print("  SUMMARY")
print("=" * 120)
print()
print(f"  Safety net reached?     {'YES' if safety_net else 'NO'}")
if safety_net_date:
    print(f"  Safety net date:        {safety_net_date}")
    print(f"  Safety net trigger:     Peak balance reached ${SAFETY_NET_PEAK:,.2f}")
    print(f"  Liquidation locked at:  ${SAFETY_NET_LIQUIDATION:,.2f} (permanent)")
print()
print(f"  Final balance:          ${balance:,.2f}")
print(f"  Final peak:             ${peak:,.2f}")
print(f"  Final liq threshold:    ${liq_threshold:,.2f}")
print(f"  Final buffer:           ${balance - liq_threshold:,.2f}")
print()
print(f"  Minimum buffer:         ${min_buffer:,.2f} (trade #{min_buffer_trade} on {min_buffer_date})")
print(f"  Account blown?          {'YES' if balance <= liq_threshold else 'NO'}")

# August analysis
print()
print("=" * 120)
print("  AUGUST DRAWDOWN IN CONTEXT")
print("=" * 120)
print()

# Replay again to find state at start of August
balance2 = STARTING_BALANCE
peak2 = STARTING_BALANCE
liq2 = STARTING_LIQUIDATION
sn2 = False

aug_start_balance = None
aug_start_liq = None
aug_start_sn = None

for _, r in t1.iterrows():
    if r["date"] >= "2025-08-01" and aug_start_balance is None:
        aug_start_balance = balance2
        aug_start_liq = liq2
        aug_start_sn = sn2

    balance2 += r["pnl"]
    if balance2 > peak2:
        peak2 = balance2
        if not sn2:
            liq2 = peak2 - TRAILING_DD
            if peak2 >= SAFETY_NET_PEAK:
                sn2 = True
                liq2 = SAFETY_NET_LIQUIDATION

if aug_start_balance:
    print(f"  At start of August:")
    print(f"    Balance:              ${aug_start_balance:,.2f}")
    print(f"    Liquidation:          ${aug_start_liq:,.2f}")
    print(f"    Safety net:           {'LOCKED' if aug_start_sn else 'trailing'}")
    print(f"    Buffer to blow:       ${aug_start_balance - aug_start_liq:,.2f}")
    print()
    if aug_start_sn:
        print(f"  With safety net LOCKED at ${SAFETY_NET_LIQUIDATION:,.2f}:")
        print(f"    The account could lose ${aug_start_balance - SAFETY_NET_LIQUIDATION:,.2f}")
        print(f"    before being blown — NOT just $2,000!")
        print(f"    That's {(aug_start_balance - SAFETY_NET_LIQUIDATION) / 300:.0f} consecutive")
        print(f"    15-pt SL hits before liquidation.")
    else:
        print(f"  Safety net NOT yet reached at start of August.")
        print(f"  Trailing DD still active — buffer is only ${aug_start_balance - aug_start_liq:,.2f}")

print()
print("=" * 120)
