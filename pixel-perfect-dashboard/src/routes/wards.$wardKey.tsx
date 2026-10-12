import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle, ArrowLeft, Bell, Download, Droplets, Info, Leaf, Menu,
  Sparkles, Wind,
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  ApiError, fetchWardAiInsights, fetchWardDetail, fetchWardHistory, fetchWardXaiLocal,
  wardReportUrl, type EnvironmentalIndex, type InterventionDetail, type WardHistoryPoint,
  type XaiContribution,
} from "@/lib/api";
import { AppShell, useSidebarToggle } from "@/components/AppShell";

export const Route = createFileRoute("/wards/$wardKey")({
  head: ({ params }) => ({
    meta: [{ title: `${params.wardKey} — Ward Intelligence — HeatLens` }],
  }),
  component: WardIntelligencePage,
});

function formatDate(value?: string | null) {
  if (!value) return "N/A";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
}

function fmt(value: number | null | undefined, decimals = 1, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return `${value.toFixed(decimals)}${suffix}`;
}

function Gauge({ probability }: { probability: number | null }) {
  const pct = probability !== null ? Math.max(0, Math.min(1, probability)) : 0;
  const angle = pct * 180;
  const radius = 70;
  const cx = 90;
  const cy = 90;
  const toPoint = (deg: number) => {
    const rad = (Math.PI * (180 - deg)) / 180;
    return [cx + radius * Math.cos(rad), cy - radius * Math.sin(rad)];
  };
  const [ex, ey] = toPoint(angle);
  const largeArc = angle > 180 ? 1 : 0;

  return (
    <svg viewBox="0 0 180 100" className="gauge">
      <path d="M20 90 A70 70 0 0 1 160 90" className="gauge-track" />
      {probability !== null && (
        <path d={`M20 90 A70 70 0 ${largeArc} 1 ${ex} ${ey}`} className="gauge-fill" />
      )}
      <text x="90" y="78" textAnchor="middle" className="gauge-value">
        {probability !== null ? `${(probability * 100).toFixed(0)}%` : "N/A"}
      </text>
    </svg>
  );
}

function ShapBars({ contributions }: { contributions: XaiContribution[] }) {
  const max = Math.max(1e-6, ...contributions.map((c) => Math.abs(c.shap_value ?? 0)));

  return (
    <div className="shap-bars">
      {contributions.slice(0, 8).map((c) => {
        const value = c.shap_value ?? 0;
        const widthPct = (Math.abs(value) / max) * 100;
        const positive = value > 0;
        return (
          <div className="shap-row" key={c.feature} title={c.explanation}>
            <span className="shap-label">{c.human_label ?? c.feature.replace(/_/g, " ")}</span>
            <div className="shap-track">
              <div
                className={positive ? "shap-fill positive" : "shap-fill negative"}
                style={{ width: `${widthPct}%` }}
              />
            </div>
            <span className="shap-strength">{c.strength}</span>
            <span className={positive ? "shap-value positive" : "shap-value negative"}>
              {positive ? "+" : ""}{value.toFixed(3)}
            </span>
          </div>
        );
      })}
      <p className="ward-note">
        <Info /> SHAP values show each feature's contribution to the model's output (a log-odds
        scale), not a direct percentage-point change in probability. Association, not causation.
      </p>
    </div>
  );
}

function EnvironmentalProfile({ profile }: { profile: Record<string, EnvironmentalIndex> | undefined }) {
  const icons = { mean_ndvi: Leaf, mean_ndbi: Wind, mean_ndwi: Droplets } as const;
  const tones = { mean_ndvi: "green", mean_ndbi: "blue", mean_ndwi: "cyan" } as const;

  return (
    <div className="env-row">
      {(["mean_ndvi", "mean_ndbi", "mean_ndwi"] as const).map((key) => {
        const entry = profile?.[key];
        const Icon = icons[key];
        return (
          <div className={`env-card ${tones[key]}`} key={key}>
            <Icon />
            <span>{entry?.label ?? key.toUpperCase()}</span>
            <strong>{fmt(entry?.value ?? null, 3)}</strong>
            <small>{entry?.name}</small>
            {entry?.interpretation && <p className="env-interpretation">{entry.interpretation}</p>}
          </div>
        );
      })}
    </div>
  );
}

function InterventionCard({ label, detail }: { label: string; detail: InterventionDetail | undefined }) {
  if (!detail) return null;

  return (
    <div className="intervention-card">
      <div className="intervention-header">
        <span className="intervention-label">{label}</span>
        <strong>{detail.intervention}</strong>
      </div>
      <p><b>Why:</b> {detail.why}</p>
      <p><b>Mechanism:</b> {detail.mechanism}</p>
      {detail.indicators_affected.length > 0 && (
        <p><b>Potential indicators affected:</b> {detail.indicators_affected.join(", ")}</p>
      )}
      <p className="ward-note">{detail.effect_size_note}</p>
    </div>
  );
}

function HistoryCharts({ history }: { history: WardHistoryPoint[] }) {
  const data = history.map((point) => ({
    ...point,
    label: formatDate(point.date),
  }));

  return (
    <>
      <section className="ward-panel">
        <div className="panel-heading"><div><h2>Heat Pattern</h2><p>Heat-pocket coverage across observations</p></div></div>
        <div style={{ width: "100%", height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: -12 }}>
              <defs>
                <linearGradient id="heatArea" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--chart-hot)" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="var(--chart-hot)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} interval="preserveStartEnd" axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={32} unit="%" />
              <Tooltip
                contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }}
                formatter={(value: number) => [`${value.toFixed(2)}%`, "Heat-pocket coverage"]}
              />
              <Area type="monotone" dataKey="heat_pocket_coverage_pct" stroke="var(--chart-hot)" strokeWidth={2} fill="url(#heatArea)" isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="ward-panel">
        <div className="panel-heading"><div><h2>Environmental History</h2><p>Mean land-surface temperature across observations</p></div></div>
        <div style={{ width: "100%", height: 180 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: -12 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} interval="preserveStartEnd" axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 9, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={32} unit="°C" />
              <Tooltip
                contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11 }}
                formatter={(value: number) => [`${value.toFixed(1)}°C`, "Mean LST"]}
              />
              <Line type="monotone" dataKey="mean_lst_celsius" stroke="var(--cyan)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </>
  );
}

function WardIntelligencePage() {
  const { wardKey } = Route.useParams();
  const navigate = useNavigate();
  const toggleSidebar = useSidebarToggle();

  const detail = useQuery({ queryKey: ["wardDetail", wardKey], queryFn: () => fetchWardDetail(wardKey) });
  const history = useQuery({ queryKey: ["wardHistory", wardKey], queryFn: () => fetchWardHistory(wardKey) });
  const xaiLocal = useQuery({ queryKey: ["wardXaiLocal", wardKey], queryFn: () => fetchWardXaiLocal(wardKey) });
  const aiInsights = useQuery({
    queryKey: ["wardAiInsights", wardKey],
    queryFn: () => fetchWardAiInsights(wardKey),
    retry: false,
  });

  if (detail.isError) {
    const notFound = detail.error instanceof ApiError && detail.error.status === 404;
    return (
      <AppShell>
        <div className="ward-empty-state">
          <AlertTriangle />
          <h2>{notFound ? "Ward not found" : "Could not load ward data"}</h2>
          <p>
            {notFound
              ? `No ward matches key "${wardKey}" in the connected boundary dataset.`
              : "The backend could not be reached. Confirm it is running and try again."}
          </p>
          <Link to="/explore" className="landing-cta-secondary"><ArrowLeft /> Back to Explore</Link>
        </div>
      </AppShell>
    );
  }

  const ward = detail.data?.ward;
  const current = detail.data?.current_heat;
  const historical = detail.data?.historical_heat;
  const future = detail.data?.future_prediction;
  const trend = detail.data?.trend;
  const policy = detail.data?.policy;
  const environmental = detail.data?.environmental;
  const insight = aiInsights.data?.ai_insights;

  return (
    <AppShell>
      <header className="page-header">
        <button type="button" className="action-button mobile-menu" aria-label="Toggle navigation" onClick={toggleSidebar}><Menu /></button>
        <div>
          <button type="button" className="ward-back" onClick={() => navigate({ to: "/explore" })}><ArrowLeft /> Explore</button>
          <h1>{detail.isLoading ? "Loading…" : `${ward?.ward_name} · ${ward?.ward_key}`}</h1>
          <p>
            {ward
              ? `${ward.corporation} Corporation · Ward ID ${ward.ward_id} · Latest observation ${formatDate(ward.date)} · Data status: ${ward.data_status}`
              : "Fetching ward context from the connected analytics datasets…"}
          </p>
        </div>
        <a className="action-button icon-button" href={wardReportUrl(wardKey)} target="_blank" rel="noreferrer" aria-label="Download ward report"><Download /></a>
        <button type="button" className="action-button icon-button" aria-label="Notifications"><Bell /></button>
      </header>

      {policy?.priority_level && (
        <div className={`ward-priority-banner priority-${policy.priority_level.toLowerCase().replace(/\s+/g, "-")}`}>
          Decision-support priority: <strong>{policy.priority_level}</strong>
        </div>
      )}

      <div className="ward-grid">
        <div className="ward-main">
          <section className="ward-panel">
            <div className="panel-heading"><div><h2>Current Conditions</h2><p>Observed at the most recent Landsat pass</p></div></div>
            <div className="metric-row">
              <div className="metric"><span>Mean Land Surface Temperature</span><strong>{fmt(current?.mean_lst_celsius, 1, "°C")}</strong></div>
              <div className="metric"><span>Current Heat-Pocket Coverage</span><strong>{fmt(current?.heat_pocket_coverage_pct, 2, "%")}</strong></div>
              <div className="metric" title="Classification based on current mean land-surface temperature."><span>LST Severity <Info /></span><strong>{current?.current_heat_severity ?? "N/A"}</strong></div>
              <div className="metric" title="Classification based on the proportion of the ward currently detected as a heat pocket."><span>Heat-Pocket Status <Info /></span><strong>{current?.current_heat_pocket_level ?? "N/A"}</strong></div>
            </div>
            <p className="ward-note">
              <Info /> LST Severity and Heat-Pocket Status measure different things — a ward can show
              High LST Severity (hot surface temperature) while its Heat-Pocket Status is Low (little
              of the ward currently meets the heat-pocket threshold). Both are shown because they are
              not contradictory.
            </p>
          </section>

          <section className="ward-panel">
            <div className="panel-heading"><div><h2>Environmental Profile</h2><p>Vegetation, built-up, and moisture indices at the latest observation</p></div></div>
            <EnvironmentalProfile profile={environmental} />
          </section>

          {history.data && history.data.history.length > 0 && <HistoryCharts history={history.data.history} />}

          <section className="ward-panel persistence-panel">
            <div className="panel-heading"><div><h2>Persistence</h2><p>Heat-pocket persistence across observations — not a count of consecutive hot months</p></div></div>
            <div className="metric-row">
              <div className="metric"><span>Historical average coverage</span><strong>{fmt(historical?.historical_heat_pocket_coverage_pct, 2, "%")}</strong></div>
              <div className="metric"><span>Historical maximum coverage</span><strong>{fmt(historical?.historical_max_heat_pocket_coverage_pct, 2, "%")}</strong></div>
              <div className="metric"><span>Observations analyzed</span><strong>{historical?.historical_observation_count ?? "N/A"}</strong></div>
              <div className="metric"><span>Persistence level</span><strong>{historical?.persistence_level ?? "N/A"}</strong></div>
            </div>
          </section>

          <section className="ward-panel">
            <div className="panel-heading">
              <div><h2>Historical Trend</h2><p>FDR-corrected, seasonally adjusted heat-pocket trend</p></div>
              {trend?.trend_classification && (
                <span className={`trend-badge ${trend.significant ? `significant-${(trend.trend_direction ?? "").toLowerCase()}` : "stable"}`}>
                  {trend.significant ? `Significant — ${trend.trend_direction}` : "No Significant Trend"}
                </span>
              )}
            </div>
            {trend?.trend_classification ? (
              <>
                <div className="metric-row">
                  <div className="metric"><span>Trend classification</span><strong>{trend.trend_classification}</strong></div>
                  <div className="metric"><span>Direction</span><strong>{trend.trend_direction ?? "N/A"}</strong></div>
                  <div className="metric" title="The estimated rate of change in heat-pocket coverage per year, after adjusting for seasonal (month-of-year) effects. Not a temperature trend."><span>Heat-Pocket Trend Slope <Info /></span><strong>{fmt(trend.trend_slope, 3, " pct pts/yr")}</strong></div>
                  <div className="metric" title="The probability the observed trend is due to chance, after correcting for testing all 369 wards at once. Below 0.05 is typically considered significant."><span>FDR-Adjusted P-Value <Info /></span><strong>{fmt(trend.fdr_adjusted_p_value, 4)}</strong></div>
                  <div className="metric" title="How well the seasonal model fits this ward's observations, from 0 (poor fit) to 1 (perfect fit). This describes model fit, not trend significance."><span>R² <Info /></span><strong>{fmt(trend.r_squared, 3)}</strong></div>
                </div>
                {trend.trend_explanation && <p className="ward-note">{trend.trend_explanation}</p>}
              </>
            ) : (
              <p className="ward-note">Not available in connected datasets.</p>
            )}
          </section>

          <section className="ward-panel xai-panel">
            <div className="panel-heading"><div><h2>Model Explanation (XAI) — association, not causation</h2><p>Feature contributions to the future heat-risk model, from SHAP</p></div></div>
            {xaiLocal.data && xaiLocal.data.contributions.length > 0 ? (
              <ShapBars contributions={xaiLocal.data.contributions} />
            ) : (
              <p className="ward-note">Not available in connected datasets.</p>
            )}
          </section>

          <section className="ward-panel policy-panel">
            <div className="panel-heading"><div><h2>Policy Engine</h2><p>Decision-support recommendation — not a guaranteed or causal outcome</p></div></div>
            {policy?.primary_intervention ? (
              <>
                <div className="metric-row">
                  <div className="metric"><span>Priority</span><strong>{policy.priority_level}</strong></div>
                </div>
                <InterventionCard label="Primary intervention" detail={policy.primary_intervention_detail} />
                <InterventionCard label="Secondary intervention" detail={policy.secondary_intervention_detail} />
              </>
            ) : (
              <p className="ward-note">Not available in connected datasets.</p>
            )}
          </section>
        </div>

        <aside className="ward-side">
          <section className="ward-panel future-panel">
            <div className="panel-heading"><div><h2>Next-Observation Heat Risk</h2></div></div>
            <p className="future-label">Model-estimated probability of high heat at the next usable Landsat observation</p>
            <Gauge probability={future?.future_heat_probability ?? null} />
            <div className="future-meta">
              <span>{future?.future_heat_risk_level ?? "N/A"} risk level</span>
              {future?.future_high_heat_prediction === 1 && <span className="future-flag">High-heat predicted</span>}
            </div>
            <p className="ward-note">
              <Info /> This is a model-estimated probability for the next observed Landsat scene, not
              a long-range weather forecast — the model does not predict a fixed calendar horizon
              such as 1, 5, or 10 years ahead.
            </p>
          </section>

          <section className="ward-panel ai-panel">
            <div className="panel-heading"><div><h2><Sparkles /> AI Insight</h2></div></div>
            {aiInsights.isLoading && <p className="ward-note">Generating grounded AI interpretation…</p>}
            {aiInsights.isError && (
              <p className="ward-note">
                {aiInsights.error instanceof ApiError
                  ? aiInsights.error.detail || "AI insights are not available right now."
                  : "AI insights are not available right now."}
              </p>
            )}
            {insight && (
              <div className="ai-insight-body">
                <strong>Current condition</strong>
                <p>{insight.current_condition}</p>
                <strong>Historical pattern</strong>
                <p>{insight.historical_pattern}</p>
                <strong>Future risk</strong>
                <p>{insight.future_risk}</p>
                {insight.main_drivers?.length > 0 && (
                  <>
                    <strong>Main model drivers</strong>
                    <ul>{insight.main_drivers.map((d, i) => <li key={i}>{d}</li>)}</ul>
                  </>
                )}
                <strong>What this means</strong>
                <p>{insight.what_this_means}</p>

                {insight.key_findings?.length > 0 && (
                  <>
                    <strong>Key findings</strong>
                    <ul>{insight.key_findings.map((f, i) => <li key={i}>{f}</li>)}</ul>
                  </>
                )}

                {insight.recommended_actions?.length > 0 && (
                  <>
                    <strong>Recommended actions</strong>
                    <ul>
                      {insight.recommended_actions.map((a, i) => (
                        <li key={i}><b>{a.action}</b>{a.reason ? ` — ${a.reason}` : ""}</li>
                      ))}
                    </ul>
                  </>
                )}

                {insight.uncertainty_note && (
                  <p className="ward-note"><Info /> {insight.uncertainty_note}</p>
                )}
              </div>
            )}
            <Link to="/heatlens" search={{ ward: wardKey }} className="landing-cta-secondary ai-panel-link">Ask HeatLens AI about this ward</Link>
          </section>
        </aside>
      </div>
    </AppShell>
  );
}
