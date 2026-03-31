import { useState } from "react";
import TabNavigation from "./components/layout/TabNavigation";
import TradingView from "./components/layout/TradingView";
import AnalysisView from "./components/layout/AnalysisView";
import AccountManagement from "./components/layout/AccountManagement";
import ModelManagement from "./components/layout/ModelManagement";
import { useWebSocket } from "./hooks/useWebSocket";

type Tab = "trading" | "analysis" | "accounts" | "models";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("trading");
  const { sendMessage, reconnectTo } = useWebSocket();

  return (
    <div className="flex h-screen flex-col bg-darkest">
      <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 overflow-hidden">
        {activeTab === "trading" && <TradingView sendMessage={sendMessage} reconnectTo={reconnectTo} />}
        {activeTab === "analysis" && <AnalysisView />}
        {activeTab === "accounts" && <AccountManagement />}
        {activeTab === "models" && <ModelManagement />}
      </main>
    </div>
  );
}
