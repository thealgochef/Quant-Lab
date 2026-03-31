type Tab = "trading" | "analysis" | "accounts" | "models";

const TABS: { id: Tab; label: string }[] = [
  { id: "trading", label: "Trading" },
  { id: "analysis", label: "Analysis" },
  { id: "accounts", label: "Accounts" },
  { id: "models", label: "Models" },
];

interface Props {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

export default function TabNavigation({ activeTab, onTabChange }: Props) {
  return (
    <nav className="flex h-10 shrink-0 items-center gap-1 border-b border-hover bg-panel px-4">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`rounded px-4 py-1.5 text-sm font-medium transition-colors ${
            activeTab === tab.id
              ? "bg-hover text-primary"
              : "text-secondary hover:text-primary"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
