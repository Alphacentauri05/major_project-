import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight, Expand,
  Flame, Grid2X2, Info, Leaf, Menu,
  Search, Sparkles, TrendingUp, Thermometer,
} from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState, type ReactNode } from "react";
import park from "@/assets/bengaluru-park.jpg";
import skyline from "@/assets/bengaluru-skyline.jpg";
import { fetchOverview, fetchWardList, fetchWardsGeo, type Insight } from "@/lib/api";
import type { MapMode } from "@/components/WardMap";
import { AppShell, useSidebarToggle } from "@/components/AppShell";

// Leaflet touches `window` at module load time, which breaks
// server-side rendering. Load the map only on the client.
const WardMap = lazy(() =>
  import("@/components/WardMap").then((module) => ({ default: module.WardMap })),
);

export const Route = createFileRoute("/overview")({
  head: () => ({
    meta: [
      { title: "Overview — HeatLens" },
      { name: "description", content: "What is happening across Bengaluru's 369 wards right now." },
    ],
  }),
  component: OverviewPage,
});

const INSIGHT_ICONS: Record<Insight["tone"], typeof Flame> = {
  hot: Flame,
  green: Leaf,
  violet: TrendingUp,
  blue: Info,
};

function ActionButton({ children, className = "", onClick, label }: { children: ReactNode; className?: string; onClick?: () => void; label?: string }) {
  return <button type="button" aria-label={label} onClick={onClick} className={`action-button ${className}`}>{children}</button>;
}

function useLiveClock() {
  // Starts null so server and first client render match exactly;
  // the real (and then ticking) time only appears after mount.
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    setNow(new Date());
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  return now;
}

function HeroSearch({ onNavigate }: { onNavigate: (wardKey: string) => void }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const wardList = useQuery({ queryKey: ["wardList"], queryFn: fetchWardList });

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const wards = wardList.data?.wards ?? [];
    return wards
      .filter((w) => `${w.ward_name} ${w.ward_key}`.toLowerCase().includes(q))
      .slice(0, 8);
  }, [query, wardList.data]);

  const select = (wardKey: string) => {
    setQuery("");
    setOpen(false);
    onNavigate(wardKey);
  };

  return (
    <div className="hero-search-wrap">
      <label className="search">
        <Search />
        <input
          value={query}
          placeholder="Search wards or areas..."
          onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && matches[0]) select(matches[0].ward_key);
          }}
        />
      </label>
      {open && query.trim() && (
        <div className="hero-search-results">
          {matches.length === 0 && <div className="hero-search-empty">No wards match "{query}".</div>}
          {matches.map((w) => (
            <button key={w.ward_key} type="button" onMouseDown={() => select(w.ward_key)}>
              {w.ward_name} <small>({w.ward_key})</small>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function OverviewPage() {
  const [mode, setMode] = useState<MapMode>("Risk");
  const [expanded, setExpanded] = useState(false);
  const [mounted, setMounted] = useState(false);
  const now = useLiveClock();
  const navigate = useNavigate();
  const toggleSidebar = useSidebarToggle();

  useEffect(() => setMounted(true), []);

  const overview = useQuery({ queryKey: ["overview"], queryFn: fetchOverview });
  const wardsGeo = useQuery({ queryKey: ["wardsGeo"], queryFn: fetchWardsGeo });

  const stats = overview.data?.stats;
  const insights = overview.data?.insights ?? [];

  const goToWard = (wardKey: string) => navigate({ to: "/wards/$wardKey", params: { wardKey } });

  return (
    <AppShell>
      <header className="hero">
        <img src={skyline} alt="Bengaluru skyline at dusk" width={1920} height={1024} />
        <div className="topbar">
          <ActionButton className="mobile-menu" onClick={toggleSidebar} label="Toggle navigation"><Menu /></ActionButton>
          <HeroSearch onNavigate={goToWard} />
        </div>
        <div className="hero-copy"><h1>Read the city<br />through its <em>heat signal.</em></h1><p>Bengaluru’s microclimates tell a story. Explore, understand, and build a cooler tomorrow.</p></div>
        <div className="hero-script">Bengaluru<br />Breathes<br />Brighter</div>
        <div className="clock">
          <strong>{now ? now.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }) : " "}</strong>
          <i>“Cooler cities.<br />Healthier lives.”</i>
        </div>
      </header>

      <div className="dashboard-body">
        <div className="stats-row">
          <article className="stat hot">
            <div className="stat-icon"><Flame /></div>
            <div>
              <strong>{stats ? `${stats.high_heat_risk_pct}%` : "—"}</strong>
              <span>High Future Heat-Risk Wards</span>
              <small>{stats ? `${stats.high_heat_risk_count} of ${stats.total_wards} wards` : "Loading…"}</small>
            </div>
          </article>
          <article className="stat cool">
            <div className="stat-icon"><Thermometer /></div>
            <div>
              <strong>{stats ? `${stats.highest_observed_lst}°C` : "—"}</strong>
              <span>Highest Observed LST</span>
              <small>{stats ? `${stats.highest_lst_zone} Zone` : "Loading…"}</small>
            </div>
          </article>
          <article className="stat green">
            <div className="stat-icon"><Grid2X2 /></div>
            <div>
              <strong>{stats?.total_wards ?? "—"}</strong>
              <span>Total Wards</span>
              <small>Greater Bengaluru Area boundary</small>
            </div>
          </article>
          <article className="stat violet">
            <div className="stat-icon"><TrendingUp /></div>
            <div>
              <strong>{stats?.high_priority_count ?? "—"}</strong>
              <span>High Decision-Support Priority</span>
              <small>{stats ? `${stats.very_high_policy_count ?? "—"} Very High` : "Loading…"}</small>
            </div>
          </article>
        </div>
        <div className="content-grid">
          <section className={expanded ? "map-card expanded" : "map-card"}>
            <div className="panel-heading"><div><h2>Bengaluru Heat Risk Landscape</h2><p>Explore ward-level heat risk, persistence, and future vulnerability.</p></div><div className="segments">{(["Risk", "Persistence", "Future Risk"] as const).map(item => <ActionButton key={item} onClick={() => setMode(item)} className={mode === item ? "selected" : ""}>{item}</ActionButton>)}</div></div>
            {mounted ? (
              <Suspense fallback={<div className="map-stage" />}>
                <WardMap data={wardsGeo.data} mode={mode} onWardSelect={goToWard} />
              </Suspense>
            ) : (
              <div className="map-stage" />
            )}
            <ActionButton className="fullscreen" onClick={() => setExpanded(!expanded)}><Expand /> {expanded ? "Exit Fullscreen" : "View Fullscreen"}</ActionButton>
          </section>
          <aside className="right-column">
            <section className="insights"><div className="panel-heading"><h2><Sparkles /> Key Insights</h2><a href="/explore">View All <ArrowRight /></a></div>{insights.map(({ title, text, tone }) => { const Icon = INSIGHT_ICONS[tone]; return <article key={title}><div className={`insight-icon ${tone}`}><Icon /></div><div><strong>{title}</strong><p>{text}</p></div></article>; })}</section>
            <section className="trend-photo">
              <img src={park} alt="Tree-lined park in Bengaluru" width={1024} height={768} loading="lazy" />
              <div className="trend-photo-caption"><b>More Green</b><span>Healthier, cooler neighborhoods</span></div>
            </section>
          </aside>
        </div>
      </div>
    </AppShell>
  );
}
