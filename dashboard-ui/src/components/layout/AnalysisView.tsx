import PerformanceCharts from "../analysis/PerformanceCharts";
import TradeHistoryTable from "../analysis/TradeHistoryTable";
import EquityCurves from "../analysis/EquityCurves";
import SessionStats from "../analysis/SessionStats";

export default function AnalysisView() {
  return (
    <div className="h-full overflow-y-auto p-4">
      <div className="space-y-6">
        <PerformanceCharts />
        <SessionStats />
        <EquityCurves />
        <div>
          <h3 className="mb-2 text-sm font-semibold">Trade History</h3>
          <TradeHistoryTable />
        </div>
      </div>
    </div>
  );
}
