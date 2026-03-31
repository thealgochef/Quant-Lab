import { useState } from "react";
import { addAccount } from "../../hooks/useApi";
import { useAccountStore } from "../../stores/accountStore";

export default function AddAccountForm() {
  const [label, setLabel] = useState("");
  const [evalCost, setEvalCost] = useState("147");
  const [activationCost, setActivationCost] = useState("85");
  const [group, setGroup] = useState<"A" | "B">("A");
  const [error, setError] = useState<string | null>(null);
  const addToStore = useAccountStore((s) => s.addAccount);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    try {
      setError(null);
      const result = await addAccount(label, parseFloat(evalCost), parseFloat(activationCost), group);
      addToStore(result.account);
      setLabel("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add account");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2">
      <div>
        <label className="block text-xs text-secondary">Label</label>
        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          required
          className="mt-0.5 rounded bg-darkest px-2 py-1 text-xs text-primary outline-none focus:ring-1 focus:ring-blue"
          placeholder="Apex #3"
        />
      </div>
      <div>
        <label className="block text-xs text-secondary">Eval Cost</label>
        <input
          type="text"
          value={evalCost}
          onChange={(e) => setEvalCost(e.target.value)}
          className="mt-0.5 w-20 rounded bg-darkest px-2 py-1 font-mono text-xs text-primary outline-none focus:ring-1 focus:ring-blue"
        />
      </div>
      <div>
        <label className="block text-xs text-secondary">Activation</label>
        <input
          type="text"
          value={activationCost}
          onChange={(e) => setActivationCost(e.target.value)}
          className="mt-0.5 w-20 rounded bg-darkest px-2 py-1 font-mono text-xs text-primary outline-none focus:ring-1 focus:ring-blue"
        />
      </div>
      <div>
        <label className="block text-xs text-secondary">Group</label>
        <select
          value={group}
          onChange={(e) => setGroup(e.target.value as "A" | "B")}
          className="mt-0.5 rounded bg-darkest px-2 py-1 text-xs text-primary outline-none focus:ring-1 focus:ring-blue"
        >
          <option value="A">A</option>
          <option value="B">B</option>
        </select>
      </div>
      <button
        type="submit"
        className="rounded bg-blue px-3 py-1 text-xs font-medium text-white hover:opacity-90"
      >
        Add Account
      </button>
      {error && <span className="text-xs text-red">{error}</span>}
    </form>
  );
}
