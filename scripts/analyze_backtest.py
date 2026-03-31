"""Analyze OOS backtest results — answers 6 specific questions."""
from __future__ import annotations

import pandas as pd
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

trades = pd.read_csv("data/experiment/backtest_oos_results_trades.csv")
preds = pd.read_csv("data/experiment/backtest_oos_results_predictions.csv")
daily = pd.read_csv("data/experiment/backtest_oos_results.csv")

# Use only APEX-001 for unique signal analysis (all 5 accounts identical)
t1 = trades[trades["account_id"] == "APEX-001"].copy()
t1["entry_dt"] = pd.to_datetime(t1["entry_time"])
t1["exit_dt"] = pd.to_datetime(t1["exit_time"])
t1["date"] = t1["entry_dt"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d")

# ═══════════════════════════════════════════════════════════════
# Q1: FLATTEN EXIT P&L
# ═══════════════════════════════════════════════════════════════
print("=" * 72)
print("  Q1: FLATTEN EXIT P&L ANALYSIS")
print("=" * 72)

flatten = t1[t1["exit_reason"] == "flatten"].copy()
print(f"\n  Total flatten exits (per account): {len(flatten)}")
print()
print(f"  {'#':>3s}  {'Date':10s}  {'Time(ET)':8s}  {'Dir':5s}  "
      f"{'Entry':>10s}  {'Exit':>10s}  {'P&L':>10s}  {'PnL pts':>8s}")
print(f"  {'---':>3s}  {'----------':10s}  {'--------':8s}  {'-----':5s}  "
      f"{'----------':>10s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}")

for i, (_, r) in enumerate(flatten.iterrows(), 1):
    entry_et = r["entry_dt"].astimezone(ET)
    print(f"  {i:3d}  {entry_et.strftime('%Y-%m-%d')}  "
          f"{entry_et.strftime('%H:%M:%S')}  {r['direction']:5s}  "
          f"{r['entry_price']:>10.2f}  {r['exit_price']:>10.2f}  "
          f"${r['pnl']:>9.2f}  {r['pnl_points']:>+8.1f}")

# Check entry vs exit price
all_zero = all(flatten["pnl"] == 0)
print(f"\n  All $0 P&L? {'YES' if all_zero else 'NO'}")
if all_zero:
    same_price = all(flatten["entry_price"] == flatten["exit_price"])
    print(f"  Entry == Exit price? {'YES' if same_price else 'NO'}")
    print("  These are trades where price returned exactly to entry by flatten time,")
    print("  or the position was opened very close to flatten (3:55 PM CT).")

# ═══════════════════════════════════════════════════════════════
# Q2: PER-ACCOUNT MAX DRAWDOWN
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  Q2: PER-ACCOUNT MAX DRAWDOWN")
print("=" * 72)

acct_trades = trades[trades["account_id"] == "APEX-001"].sort_values("exit_time")
balance = 50000.0
peak = balance
max_dd = 0.0
max_dd_date = ""

print(f"\n  Per-account equity curve (all 5 identical):\n")
print(f"  {'#':>3s}  {'Date':10s}  {'Result':>8s}  "
      f"{'Balance':>12s}  {'Peak':>12s}  {'Drawdown':>10s}")
print(f"  {'---':>3s}  {'----------':10s}  {'--------':>8s}  "
      f"{'------------':>12s}  {'------------':>12s}  {'----------':>10s}")

for i, (_, t) in enumerate(acct_trades.iterrows(), 1):
    balance += t["pnl"]
    peak = max(peak, balance)
    dd = peak - balance
    if dd > max_dd:
        max_dd = dd
        exit_dt = pd.to_datetime(t["exit_time"]).astimezone(ET)
        max_dd_date = exit_dt.strftime("%Y-%m-%d %H:%M")
    exit_dt = pd.to_datetime(t["exit_time"]).astimezone(ET)
    print(f"  {i:3d}  {exit_dt.strftime('%Y-%m-%d')}  {t['exit_reason']:>8s}  "
          f"${balance:>11,.2f}  ${peak:>11,.2f}  ${dd:>9,.2f}")

print(f"\n  Summary:")
print(f"    Starting balance:      $50,000")
print(f"    Final balance:         ${balance:,.2f}")
print(f"    Max drawdown:          ${max_dd:,.2f}")
print(f"    Max DD reached at:     {max_dd_date}")
print(f"    Apex trailing DD:      $2,000")
print(f"    DD margin remaining:   ${2000 - max_dd:,.2f}")
print(f"    Account blown?         {'YES' if max_dd >= 2000 else 'NO'}")

# ═══════════════════════════════════════════════════════════════
# Q3: AUGUST DEEP DIVE
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  Q3: AUGUST DEEP DIVE — EVERY TRADE")
print("=" * 72)

aug_trades = t1[t1["date"].str.startswith("2025-08")].copy()
print(f"\n  August unique signals: {len(aug_trades)}")
print()
print(f"  {'#':>3s}  {'Date':10s}  {'Time(ET)':8s}  {'Dir':5s}  "
      f"{'Entry':>10s}  {'Exit':>10s}  {'P&L':>10s}  {'Exit':>8s}  {'Duration':<12s}")
print(f"  {'---':>3s}  {'----------':10s}  {'--------':8s}  {'-----':5s}  "
      f"{'----------':>10s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}  {'------------':<12s}")

aug_wins = aug_losses = aug_flat = 0
for i, (_, r) in enumerate(aug_trades.iterrows(), 1):
    entry_et = r["entry_dt"].astimezone(ET)
    exit_et = r["exit_dt"].astimezone(ET)
    duration = exit_et - entry_et
    mins, secs = divmod(int(duration.total_seconds()), 60)

    if r["pnl"] > 0:
        aug_wins += 1
    elif r["pnl"] < 0:
        aug_losses += 1
    else:
        aug_flat += 1

    print(f"  {i:3d}  {entry_et.strftime('%Y-%m-%d')}  "
          f"{entry_et.strftime('%H:%M:%S')}  {r['direction']:5s}  "
          f"{r['entry_price']:>10.2f}  {r['exit_price']:>10.2f}  "
          f"${r['pnl']:>9.2f}  {r['exit_reason']:>8s}  {mins}m {secs}s")

wr = aug_wins / (aug_wins + aug_losses) * 100 if (aug_wins + aug_losses) > 0 else 0
print(f"\n  August summary: {aug_wins}W / {aug_losses}L / {aug_flat}F  "
      f"(win rate: {wr:.1f}%)")
print(f"  Net P&L per account: ${aug_trades['pnl'].sum():,.2f}")

# Weekly breakdown
aug_trades["week"] = aug_trades["entry_dt"].dt.isocalendar().week
print(f"\n  Weekly breakdown:")
for week, wdf in aug_trades.groupby("week"):
    w = wdf["pnl"].sum()
    ww = (wdf["pnl"] > 0).sum()
    wl = (wdf["pnl"] < 0).sum()
    wf = (wdf["pnl"] == 0).sum()
    dates = f"{wdf['date'].iloc[0]} to {wdf['date'].iloc[-1]}"
    print(f"    Week {week}: {len(wdf)} trades, {ww}W/{wl}L/{wf}F, "
          f"P&L=${w:,.2f}  ({dates})")

# August prediction analysis
aug_preds = preds[preds["timestamp"].str.startswith("2025-08")].copy()
aug_preds_rth = aug_preds[aug_preds["session"] == "ny_rth"]
print(f"\n  August signals analysis:")
print(f"    Total signals (all sessions): {len(aug_preds)}")
print(f"    RTH signals: {len(aug_preds_rth)}")
for cls in ["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"]:
    n = (aug_preds_rth["predicted_class"] == cls).sum()
    print(f"      RTH {cls}: {n}")

aug_exec = aug_preds[aug_preds["is_executable"] == True]
short_count = (aug_exec["trade_direction"] == "short").sum()
long_count = (aug_exec["trade_direction"] == "long").sum()
print(f"    Direction bias in executable: {long_count} long, {short_count} short")

# Check prediction accuracy for August vs other months
aug_outcomes = aug_preds_rth[aug_preds_rth["prediction_correct"].notna()]
if len(aug_outcomes) > 0:
    aug_acc = aug_outcomes["prediction_correct"].mean()
    print(f"    RTH prediction accuracy: {aug_acc:.1%}")

# ═══════════════════════════════════════════════════════════════
# Q4: NOVEMBER ZERO SIGNALS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  Q4: NOVEMBER ZERO SIGNALS — WHY?")
print("=" * 72)

nov_preds = preds[preds["timestamp"].str.startswith("2025-11")].copy()
nov_daily = daily[daily["date"].str.startswith("2025-11")]
n_trading = len(nov_daily[nov_daily["ticks"] > 0])

print(f"\n  November trading days: {n_trading}")
print(f"  November total signals: {len(nov_preds)}")
print(f"  November executable: {(nov_preds['is_executable'] == True).sum()}")

nov_rth = nov_preds[nov_preds["session"] == "ny_rth"]
print(f"\n  November RTH signals: {len(nov_rth)}")
if len(nov_rth) > 0:
    print("  RTH predicted class breakdown:")
    for cls in ["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"]:
        n = (nov_rth["predicted_class"] == cls).sum()
        print(f"    {cls}: {n}")
else:
    print("  >>> NO RTH signals at all — zones never touched during RTH")

nov_non_rth = nov_preds[nov_preds["session"] != "ny_rth"]
print(f"\n  November non-RTH signals: {len(nov_non_rth)}")
if len(nov_non_rth) > 0:
    for session in sorted(nov_non_rth["session"].unique()):
        s_preds = nov_non_rth[nov_non_rth["session"] == session]
        print(f"    {session}: {len(s_preds)} signals")
        for cls in ["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"]:
            n = (s_preds["predicted_class"] == cls).sum()
            if n > 0:
                print(f"      {cls}: {n}")

print(f"\n  November daily details:")
print(f"  {'Date':10s}  {'Levels':>6s}  {'Zones':>5s}  {'Ticks':>10s}  "
      f"{'Signals':>7s}  {'Exec':>4s}")
print(f"  {'----------':10s}  {'------':>6s}  {'-----':>5s}  {'----------':>10s}  "
      f"{'-------':>7s}  {'----':>4s}")
for _, r in nov_daily.iterrows():
    print(f"  {r['date']:10s}  {r['levels']:>6}  {r['zones']:>5}  "
          f"{r['ticks']:>10}  {r['signals']:>7}  {r['executable']:>4}")

# ═══════════════════════════════════════════════════════════════
# Q5: COMMISSION IMPACT
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  Q5: COMMISSION IMPACT")
print("=" * 72)

n_trades = len(t1)
commission_rt = 4.50
total_comm = n_trades * commission_rt
gross = t1["pnl"].sum()
net = gross - total_comm

print(f"\n  Trades per account:            {n_trades}")
print(f"  Commission per round-trip:     ${commission_rt:.2f}")
print(f"  Total commission per account:  ${total_comm:,.2f}")
print(f"  Gross profit per account:      ${gross:,.2f}")
print(f"  Net profit per account:        ${net:,.2f}")
if gross > 0:
    print(f"  Commission as % of gross:      {total_comm / gross * 100:.1f}%")
print(f"\n  Win rate impact: NONE")
print(f"  Commission does not change TP/SL outcomes. A $300 TP trade still")
print(f"  nets $300 - $4.50 = $295.50. A $300 SL trade costs $300 + $4.50 = $304.50.")
print(f"\n  Adjusted expectancy per trade:")
print(f"    Gross: ${gross / n_trades:+.2f}/trade")
print(f"    Net:   ${net / n_trades:+.2f}/trade")
print(f"\n  True breakeven win rate (with commission):")
loss_per_trade = 300 + commission_rt  # SL + commission
win_per_trade = 300 - commission_rt   # TP - commission
be_wr = loss_per_trade / (win_per_trade + loss_per_trade)
print(f"    Need {be_wr:.2%} win rate to break even (was 50.0% without commissions)")

# ═══════════════════════════════════════════════════════════════
# Q6: CONSECUTIVE LOSSES
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  Q6: CONSECUTIVE LOSSES — LONGEST STREAK")
print("=" * 72)

streaks: list[tuple[int, str, str]] = []
current_streak = 0
current_start = ""

for _, r in t1.iterrows():
    if r["exit_reason"] == "sl":
        current_streak += 1
        if current_streak == 1:
            current_start = r["date"]
    else:
        if current_streak > 0:
            streaks.append((current_streak, current_start, r["date"]))
        current_streak = 0
        current_start = ""

if current_streak > 0:
    streaks.append((current_streak, current_start, t1.iloc[-1]["date"]))

streaks.sort(key=lambda x: -x[0])
max_streak = streaks[0][0] if streaks else 0

print(f"\n  Longest consecutive SL streak: {max_streak} trades")
print(f"\n  All losing streaks (>= 1):")
print(f"  {'Streak':>6s}  {'Start Date':12s}  {'End Date':12s}  {'DD/acct':>10s}")
print(f"  {'------':>6s}  {'------------':12s}  {'------------':12s}  {'----------':>10s}")
for streak, start, end in streaks:
    dd = streak * 300
    print(f"  {streak:>6d}  {start:12s}  {end:12s}  ${dd:>9,.2f}")

print(f"\n  Apex survivability:")
print(f"    Max consecutive losses:    {max_streak}")
print(f"    DD from worst streak:      ${max_streak * 300:,.2f}")
print(f"    Apex trailing DD limit:    $2,000")
print(f"    Survives worst streak?     {'YES' if max_streak * 300 < 2000 else 'NO — BLOWN'}")
print(f"    Max survivable streak:     {2000 // 300} losses (${(2000 // 300) * 300:,.2f})")
if max_streak * 300 >= 2000:
    print(f"\n    WARNING: A streak of {max_streak} consecutive SL hits ($300 each)")
    print(f"    creates a ${max_streak * 300:,.2f} drawdown, exceeding Apex's $2,000 trailing DD.")

print()
print("=" * 72)
