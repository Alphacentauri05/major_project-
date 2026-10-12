import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpDown, Menu, Search } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { fetchWardList, fetchWardsGeo, type WardSummary } from "@/lib/api";
import type { MapMode } from "@/components/WardMap";
import { AppShell, useSidebarToggle } from "@/components/AppShell";

const WardMap = lazy(() =>
  import("@/components/WardMap").then((module) => ({ default: module.WardMap })),
);

export const Route = createFileRoute("/explore")({
  head: () => ({
    meta: [
      { title: "Explore — HeatLens" },
      { name: "description", content: "Search, filter, and locate heat risk across Bengaluru's wards." },
    ],
  }),
  component: ExplorePage,
});

type SortKey = "ward_name" | "mean_lst_celsius" | "future_heat_probability" | "priority_level";

const PRIORITY_ORDER: Record<string, number> = { "Very High": 0, High: 1, Moderate: 2, Low: 3, Unknown: 4 };

function ExplorePage() {
  const [search, setSearch] = useState("");
  const [corporation, setCorporation] = useState("All");
  const [priority, setPriority] = useState("All");
  const [persistence, setPersistence] = useState("All");
  const [trend, setTrend] = useState("All");
  const [sortKey, setSortKey] = useState<SortKey>("future_heat_probability");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [mounted, setMounted] = useState(false);
  const navigate = useNavigate();
  const toggleSidebar = useSidebarToggle();

  useEffect(() => setMounted(true), []);

  const wardList = useQuery({ queryKey: ["wardList"], queryFn: fetchWardList });
  const wardsGeo = useQuery({ queryKey: ["wardsGeo"], queryFn: fetchWardsGeo });

  const wards = wardList.data?.wards ?? [];

  const corporations = useMemo(
    () => Array.from(new Set(wards.map((w) => w.Corporation))).sort(),
    [wards],
  );

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    let rows = wards.filter((ward) => {
      if (corporation !== "All" && ward.Corporation !== corporation) return false;
      if (priority !== "All" && ward.priority_level !== priority) return false;
      if (persistence !== "All" && (ward as WardSummary & { persistence_level?: string }).persistence_level !== persistence) return false;
      if (trend !== "All" && (ward as WardSummary & { combined_adjusted_trend?: string }).combined_adjusted_trend !== trend) return false;

      if (query) {
        const haystack = `${ward.ward_name} ${ward.ward_key} ${ward.Corporation}`.toLowerCase();
        if (!haystack.includes(query)) return false;
      }

      return true;
    });

    rows = [...rows].sort((a, b) => {
      let cmp = 0;
      if (sortKey === "priority_level") {
        cmp = (PRIORITY_ORDER[a.priority_level] ?? 9) - (PRIORITY_ORDER[b.priority_level] ?? 9);
      } else if (sortKey === "ward_name") {
        cmp = a.ward_name.localeCompare(b.ward_name);
      } else {
        cmp = (a[sortKey] ?? -Infinity) - (b[sortKey] ?? -Infinity);
      }
      return sortDir === "asc" ? cmp : -cmp;
    });

    return rows;
  }, [wards, search, corporation, priority, persistence, trend, sortKey, sortDir]);

  const filteredGeo = useMemo(() => {
    if (!wardsGeo.data) return undefined;
    const keep = new Set(filtered.map((w) => w.ward_key));
    return {
      type: "FeatureCollection" as const,
      features: wardsGeo.data.features.filter((f) => keep.has(f.properties.ward_key)),
    };
  }, [wardsGeo.data, filtered]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((dir) => (dir === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const goToWard = (wardKey: string) => navigate({ to: "/wards/$wardKey", params: { wardKey } });

  return (
    <AppShell>
      <header className="page-header">
        <button type="button" className="action-button mobile-menu" aria-label="Toggle navigation" onClick={toggleSidebar}><Menu /></button>
        <div>
          <h1>Explore</h1>
          <p>Where is it happening? Search, filter, and locate heat risk across all 369 wards.</p>
        </div>
        <label className="search"><Search /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search by ward name, ID, or key…" /></label>
      </header>

      <div className="explore-filters">
        <select value={corporation} onChange={(e) => setCorporation(e.target.value)}>
          <option value="All">All corporations</option>
          {corporations.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select value={priority} onChange={(e) => setPriority(e.target.value)}>
          <option value="All">All priorities</option>
          {["High", "Moderate", "Low", "Unknown"].map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <select value={persistence} onChange={(e) => setPersistence(e.target.value)}>
          <option value="All">All persistence levels</option>
          {["High", "Moderate", "Low", "Unknown"].map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <select value={trend} onChange={(e) => setTrend(e.target.value)}>
          <option value="All">All trends</option>
          <option value="Increasing thermal pressure">Increasing thermal pressure</option>
          <option value="Mixed / partial increase">Mixed / partial increase</option>
          <option value="Mixed / partial decrease">Mixed / partial decrease</option>
          <option value="Decreasing thermal pressure">Decreasing thermal pressure</option>
          <option value="Stable / uncertain">Stable / uncertain</option>
        </select>
        <span className="explore-count">{filtered.length} of {wards.length} wards</span>
      </div>

      <div className="explore-grid">
        <section className="map-card explore-map">
          {mounted ? (
            <Suspense fallback={<div className="map-stage" />}>
              <WardMap data={filteredGeo} mode={"Risk" as MapMode} onWardSelect={goToWard} />
            </Suspense>
          ) : (
            <div className="map-stage" />
          )}
        </section>

        <section className="explore-table-card">
          <table className="ward-table">
            <thead>
              <tr>
                <th onClick={() => toggleSort("ward_name")}>Ward <ArrowUpDown /></th>
                <th>Corporation</th>
                <th onClick={() => toggleSort("mean_lst_celsius")}>LST <ArrowUpDown /></th>
                <th onClick={() => toggleSort("future_heat_probability")}>Future Risk <ArrowUpDown /></th>
                <th onClick={() => toggleSort("priority_level")}>Priority <ArrowUpDown /></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((ward) => (
                <tr key={ward.ward_key} onClick={() => goToWard(ward.ward_key)}>
                  <td><strong>{ward.ward_name}</strong><small>{ward.ward_key}</small></td>
                  <td>{ward.Corporation}</td>
                  <td>{ward.mean_lst_celsius !== null ? `${ward.mean_lst_celsius.toFixed(1)}°C` : "—"}</td>
                  <td>{ward.future_heat_probability !== null ? `${(ward.future_heat_probability * 100).toFixed(1)}%` : "—"}</td>
                  <td><span className={`priority-pill priority-${ward.priority_level.toLowerCase().replace(/\s+/g, "-")}`}>{ward.priority_level}</span></td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={5} className="empty-row">No wards match the current filters.</td></tr>
              )}
            </tbody>
          </table>
        </section>
      </div>
    </AppShell>
  );
}
