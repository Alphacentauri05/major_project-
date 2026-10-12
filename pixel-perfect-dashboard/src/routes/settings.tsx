import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Menu, RefreshCw } from "lucide-react";
import { API_BASE, fetchHealth } from "@/lib/api";
import { AppShell, BackendStatus, useSidebarToggle } from "@/components/AppShell";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [{ title: "Settings — HeatLens" }],
  }),
  component: SettingsPage,
});

function SettingsPage() {
  const toggleSidebar = useSidebarToggle();
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth, retry: false });

  return (
    <AppShell>
      <header className="page-header">
        <button type="button" className="action-button mobile-menu" aria-label="Toggle navigation" onClick={toggleSidebar}><Menu /></button>
        <div>
          <h1>Settings</h1>
          <p>Connection and system information for this HeatLens instance.</p>
        </div>
      </header>

      <div className="about-grid">
        <section className="ward-panel">
          <div className="panel-heading"><div><h2>Backend connection</h2></div></div>
          <div className="metric-row">
            <div className="metric"><span>API base URL</span><strong className="settings-mono">{API_BASE}</strong></div>
            <div className="metric"><span>Status</span><BackendStatus /></div>
            <div className="metric"><span>Analytics data loaded</span><strong>{health.data?.analytics_data_loaded ? "Yes" : health.isLoading ? "Checking…" : "No"}</strong></div>
          </div>
          <button type="button" className="action-button report-download" onClick={() => health.refetch()}>
            <RefreshCw /> Re-check connection
          </button>
        </section>

        <section className="ward-panel">
          <div className="panel-heading"><div><h2>Preferences</h2></div></div>
          <p className="about-text">
            There are no user-configurable preferences in this version of HeatLens — the platform
            currently ships with one canonical dark theme and one analytics configuration. This
            page will grow as configurable options are added.
          </p>
        </section>
      </div>
    </AppShell>
  );
}
