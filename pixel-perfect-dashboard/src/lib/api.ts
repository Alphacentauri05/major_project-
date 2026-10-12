export const API_BASE = import.meta.env["VITE_API_BASE_URL"] ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(path: string, status: number, detail: string) {
    super(`${path} failed (${status}): ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);

  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = body.detail ?? "";
    } catch {
      detail = await response.text().catch(() => "");
    }
    throw new ApiError(path, response.status, detail);
  }

  return response.json() as Promise<T>;
}

export interface OverviewStats {
  total_wards: number;
  high_heat_risk_pct: number;
  high_heat_risk_count: number;
  highest_observed_lst: number;
  highest_lst_zone: string;
  persistent_heat_pockets: number;
  high_priority_count: number;
  future_high_heat_count: number;
  very_high_policy_count: number | null;
  analysis_date: string | null;
  increasing_trend_wards: number;
}

export interface Insight {
  title: string;
  text: string;
  tone: "hot" | "green" | "violet" | "blue";
}

export interface LstTrendPoint {
  month: string;
  avg_lst_celsius: number | null;
}

export interface DashboardOverview {
  status: string;
  stats: OverviewStats;
  insights: Insight[];
  lst_trend: LstTrendPoint[];
}

export interface WardProperties {
  ward_key: string;
  ward_name: string;
  Corporation: string;
  ward_id: string;
  data_status: string;
  mean_lst_celsius: number | null;
  heat_pocket_coverage_pct: number | null;
  current_heat_severity: string;
  historical_heat_pocket_coverage_pct: number | null;
  persistence_level: string;
  future_heat_probability: number | null;
  future_heat_risk_level: string;
  priority_level: string;
  combined_adjusted_trend?: string;
  policy_priority?: string;
  primary_intervention?: string;
}

export interface WardFeature {
  type: "Feature";
  geometry: { type: string; coordinates: unknown };
  properties: WardProperties;
}

export interface WardGeoJSON {
  type: "FeatureCollection";
  features: WardFeature[];
}

export interface WardSummary {
  ward_key: string;
  ward_name: string;
  Corporation: string;
  mean_lst_celsius: number | null;
  future_heat_probability: number | null;
  future_heat_risk_level: string;
  priority_level: string;
  data_status: string;
}

export interface WardDetail {
  status: string;
  ward: {
    ward_key: string;
    ward_name: string;
    corporation: string;
    ward_id: number;
    date: string;
    data_status: string;
  };
  current_heat: {
    mean_lst_celsius: number | null;
    heat_pocket_coverage_pct: number | null;
    current_heat_severity: string;
    current_heat_pocket_level: string;
  };
  historical_heat: {
    historical_heat_pocket_coverage_pct: number | null;
    historical_max_heat_pocket_coverage_pct: number | null;
    historical_observation_count: number | null;
    persistence_level: string;
  };
  future_prediction: {
    future_heat_probability: number | null;
    future_high_heat_prediction: number | null;
    future_heat_risk_level: string;
    horizon?: string;
  };
  environmental?: Record<string, EnvironmentalIndex>;
  trend: {
    trend_classification?: string;
    trend_direction?: string;
    trend_slope?: number;
    p_value?: number;
    fdr_adjusted_p_value?: number;
    r_squared?: number;
    trend_explanation?: string;
    significant?: boolean;
  };
  xai: { xai_explanation?: string; contributions?: XaiContribution[] };
  policy: {
    priority_level?: string;
    primary_intervention?: string;
    secondary_intervention?: string;
    primary_intervention_detail?: InterventionDetail;
    secondary_intervention_detail?: InterventionDetail;
  };
}

export interface EnvironmentalIndex {
  value: number | null;
  label: string;
  name: string;
  interpretation: string;
}

export interface InterventionDetail {
  intervention: string;
  why: string;
  mechanism: string;
  indicators_affected: string[];
  effect_size_note: string;
}

export interface WardHistoryPoint {
  date: string;
  mean_lst_celsius: number | null;
  heat_pocket_coverage_pct: number | null;
  mean_ndvi: number | null;
  mean_ndbi: number | null;
  mean_ndwi: number | null;
}

export interface XaiContribution {
  feature: string;
  human_label: string;
  feature_value: number | null;
  shap_value: number | null;
  direction: "increases" | "decreases" | "neutral";
  strength: "Strong" | "Moderate" | "Weak";
  explanation: string;
}

export interface RecommendedAction {
  action: string;
  reason: string;
}

export interface AiInsights {
  current_condition: string;
  historical_pattern: string;
  future_risk: string;
  main_drivers: string[];
  what_this_means: string;
  key_findings: string[];
  recommended_actions: RecommendedAction[];
  uncertainty_note: string;
  evidence_basis: string;
}

export interface WardAiInsightsResponse extends WardDetail {
  ai_insights: AiInsights;
}

export interface HeatlensResponse {
  status: string;
  question: string;
  ward_key: string | null;
  answer: string;
}

export function fetchHealth() {
  return getJson<{ status: string; analytics_data_loaded: boolean }>("/health");
}

export function fetchOverview() {
  return getJson<DashboardOverview>("/api/dashboard/overview");
}

export function fetchWardsGeo() {
  return getJson<WardGeoJSON>("/api/wards/geo");
}

export function fetchWardList() {
  return getJson<{ status: string; wards: WardSummary[] }>("/api/wards");
}

export function fetchWardDetail(wardKey: string) {
  return getJson<WardDetail>(`/api/wards/${encodeURIComponent(wardKey)}`);
}

export function fetchWardHistory(wardKey: string) {
  return getJson<{ status: string; ward_key: string; history: WardHistoryPoint[] }>(
    `/api/wards/${encodeURIComponent(wardKey)}/history`,
  );
}

export function fetchWardXaiLocal(wardKey: string) {
  return getJson<{ status: string; ward_key: string; contributions: XaiContribution[] }>(
    `/api/wards/${encodeURIComponent(wardKey)}/xai/local`,
  );
}

export function fetchWardAiInsights(wardKey: string) {
  return getJson<WardAiInsightsResponse>(
    `/api/wards/${encodeURIComponent(wardKey)}/ai-insights`,
  );
}

export function fetchHeatlensAnswer(question: string, wardKey?: string) {
  const wardParam = wardKey ? `&ward_key=${encodeURIComponent(wardKey)}` : "";
  return getJson<HeatlensResponse>(
    `/api/heatlens?question=${encodeURIComponent(question)}${wardParam}`,
  );
}

export function wardReportUrl(wardKey: string) {
  return `${API_BASE}/api/reports/ward/${encodeURIComponent(wardKey)}`;
}

export function summaryReportUrl() {
  return `${API_BASE}/api/reports/summary`;
}
