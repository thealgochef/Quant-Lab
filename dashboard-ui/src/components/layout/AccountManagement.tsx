import { useEffect } from "react";
import { useAccountStore } from "../../stores/accountStore";
import { fetchAccounts } from "../../hooks/useApi";
import AccountSummary from "../accounts/AccountSummary";
import AccountCard from "../accounts/AccountCard";
import AddAccountForm from "../accounts/AddAccountForm";

export default function AccountManagement() {
  const accounts = useAccountStore((s) => s.accounts);
  const summary = useAccountStore((s) => s.summary);
  const setAccounts = useAccountStore((s) => s.setAccounts);
  const setSummary = useAccountStore((s) => s.setSummary);

  useEffect(() => {
    fetchAccounts()
      .then((data) => {
        setAccounts(data.accounts);
        setSummary(data.summary);
      })
      .catch(() => {});
  }, [setAccounts, setSummary]);

  return (
    <div className="h-full overflow-y-auto p-4">
      <div className="space-y-4">
        <AccountSummary summary={summary} />

        <div>
          <h3 className="mb-2 text-sm font-semibold">Add Account</h3>
          <AddAccountForm />
        </div>

        <div className="space-y-1">
          <h3 className="text-sm font-semibold">Accounts ({accounts.length})</h3>
          {accounts.map((acct) => (
            <AccountCard key={acct.account_id} account={acct} />
          ))}
        </div>
      </div>
    </div>
  );
}
