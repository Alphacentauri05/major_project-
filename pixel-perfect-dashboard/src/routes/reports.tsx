import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Download, FileText, Menu, Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import { fetchWardList, summaryReportUrl, wardReportUrl, type WardSummary } from "@/lib/api";
import { AppShell, useSidebarToggle } from "@/components/AppShell";

function WardCombobox({
  wards,
  selectedWard,
  onSelect,
}: {
  wards: WardSummary[];
  selectedWard: string;
  onSelect: (wardKey: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const selected = wards.find((w) => w.ward_key === selectedWard);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const pool = q
      ? wards.filter((w) => `${w.ward_name} ${w.ward_key}`.toLowerCase().includes(q))
      : wards;
    return pool.slice(0, 50);
  }, [wards, query]);

  return (
    <div className="ward-combobox">
      {selected && !open ? (
        <button type="button" className="ward-combobox-selected" onClick={() => setOpen(true)}>
          <span>{selected.ward_name} <small>({selected.ward_key})</small></span>
          <X onClick={(e) => { e.stopPropagation(); onSelect(""); setQuery(""); }} />
        </button>
      ) : (
        <label className="search ward-combobox-input">
          <Search />
          <input
            value={query}
            placeholder="Search for a ward by name or key…"
            onFocus={() => setOpen(true)}
            onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
          />
        </label>
      )}
      {open && (
        <div className="ward-combobox-list">
          {filtered.length === 0 && <div className="ward-combobox-empty">No wards match.</div>}
          {filtered.map((w) => (
            <button
              key={w.ward_key}
              type="button"
              onClick={() => { onSelect(w.ward_key); setQuery(""); setOpen(false); }}
            >
              {w.ward_name} <small>({w.ward_key})</small>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export const Route = createFileRoute("/reports")({
  head: () => ({
    meta: [
      { title: "Reports — HeatLens" },
      { name: "description", content: "Export ward-level and city-wide Bengaluru urban heat intelligence reports." },
    ],
  }),
  component: ReportsPage,
});

type DownloadState = "idle" | "loading" | "error";

function ReportsPage() {
  const toggleSidebar = useSidebarToggle();
  const wardList = useQuery({ queryKey: ["wardList"], queryFn: fetchWardList });
  const [selectedWard, setSelectedWard] = useState("");
  const [wardState, setWardState] = useState<DownloadState>("idle");
  const [summaryState, setSummaryState] = useState<DownloadState>("idle");
  const [wardErrorDetail, setWardErrorDetail] = useState("");

  const download = async (url: string, filenameFallback: string, onState: (s: DownloadState) => void, onError?: (detail: string) => void) => {
    onState("loading");
    try {
      const response = await fetch(url);
      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: "" }));
        onError?.(body.detail || "Report generation failed.");
        onState("error");
        return;
      }
      const blob = await response.blob();
      const disposition = response.headers.get("content-disposition") ?? "";
      const match = disposition.match(/filename="?([^"]+)"?/);
      const filename = match?.[1] ?? filenameFallback;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
      URL.revokeObjectURL(link.href);
      onState("idle");
    } catch {
      onError?.("Could not reach the backend.");
      onState("error");
    }
  };

  const wards = wardList.data?.wards ?? [];

  return (
    <AppShell>
      <header className="page-header">
        <button type="button" className="action-button mobile-menu" aria-label="Toggle navigation" onClick={toggleSidebar}><Menu /></button>
        <div>
          <h1>Reports</h1>
          <p>Export the analysis. Ward-level and city-wide PDF reports, generated from the connected analytics datasets.</p>
        </div>
      </header>

      <div className="reports-grid">
        <section className="ward-panel report-card">
          <div className="panel-heading"><div><h2><FileText /> Ward Report</h2><p>Current conditions, persistence, trend, XAI, and policy recommendation for one ward</p></div></div>
          <WardCombobox
            wards={wards}
            selectedWard={selectedWard}
            onSelect={(wardKey) => { setSelectedWard(wardKey); setWardState("idle"); }}
          />
          <button
            type="button"
            className="action-button report-download"
            disabled={!selectedWard || wardState === "loading"}
            onClick={() =>
              download(wardReportUrl(selectedWard), `${selectedWard}_report.pdf`, setWardState, setWardErrorDetail)
            }
          >
            <Download /> {wardState === "loading" ? "Generating…" : "Generate Ward Report"}
          </button>
          {wardState === "error" && (
            <p className="ward-note report-error"><AlertTriangle /> {wardErrorDetail || "This report requires AI insight generation, which needs a configured GROQ_API_KEY on the backend."}</p>
          )}
        </section>

        <section className="ward-panel report-card">
          <div className="panel-heading"><div><h2><FileText /> Bengaluru Summary Report</h2><p>City-wide indicators and key findings across all 369 wards</p></div></div>
          <p className="ward-note">Total wards, high-priority counts, and computed insights — the same figures shown on the Overview page.</p>
          <button
            type="button"
            className="action-button report-download"
            disabled={summaryState === "loading"}
            onClick={() => download(summaryReportUrl(), "bengaluru_heat_summary.pdf", setSummaryState)}
          >
            <Download /> {summaryState === "loading" ? "Generating…" : "Generate Bengaluru Summary Report"}
          </button>
          {summaryState === "error" && (
            <p className="ward-note report-error"><AlertTriangle /> Could not generate the summary report. Confirm the backend is running.</p>
          )}
        </section>
      </div>
    </AppShell>
  );
}
