"""
Aggregation helpers for the HeatLens Overview dashboard.

Everything here derives from the same analytics CSVs and the raw ward
boundary GeoJSON used elsewhere in the API - no separate data source,
no invented numbers.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BOUNDARY_PATH = (
    PROJECT_ROOT / "data" / "raw" / "boundaries" / "gba_wards_369.geojson"
)

ML_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "features" / "ward_temporal_ml_dataset.csv"
)

XAI_LOCAL_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "risk" / "heat_risk_xai_local_20260503.csv"
)


def safe_value(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def load_boundary_geojson():
    with open(BOUNDARY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# WARD GEOJSON ENRICHED WITH RISK / TREND / POLICY PROPERTIES
# ============================================================

def build_ward_geojson(risk_df, trend_df, policy_df):
    geojson = load_boundary_geojson()

    risk_by_key = risk_df.set_index("ward_key")
    trend_by_key = trend_df.set_index("ward_key") if trend_df is not None else None
    policy_by_key = policy_df.set_index("ward_key") if policy_df is not None else None

    features = []

    for feature in geojson["features"]:
        props = feature["properties"]
        ward_key = f"{props.get('Corporation')}_{props.get('ward_id')}"

        merged = {
            "ward_key": ward_key,
            "ward_name": props.get("ward_name"),
            "Corporation": props.get("Corporation"),
            "ward_id": props.get("ward_id"),
        }

        if ward_key in risk_by_key.index:
            risk_row = risk_by_key.loc[ward_key]
            merged.update(
                {
                    "data_status": safe_value(risk_row.get("data_status")),
                    "mean_lst_celsius": safe_value(risk_row.get("mean_lst_celsius")),
                    "heat_pocket_coverage_pct": safe_value(
                        risk_row.get("heat_pocket_coverage_pct")
                    ),
                    "current_heat_severity": safe_value(
                        risk_row.get("current_heat_severity")
                    ),
                    "historical_heat_pocket_coverage_pct": safe_value(
                        risk_row.get("historical_heat_pocket_coverage_pct")
                    ),
                    "persistence_level": safe_value(risk_row.get("persistence_level")),
                    "future_heat_probability": safe_value(
                        risk_row.get("future_heat_probability")
                    ),
                    "future_heat_risk_level": safe_value(
                        risk_row.get("future_heat_risk_level")
                    ),
                    "priority_level": safe_value(risk_row.get("priority_level")),
                }
            )
        else:
            merged.update(
                {
                    "data_status": "Insufficient observation data",
                    "mean_lst_celsius": None,
                    "heat_pocket_coverage_pct": None,
                    "current_heat_severity": "Unknown",
                    "historical_heat_pocket_coverage_pct": None,
                    "persistence_level": "Unknown",
                    "future_heat_probability": None,
                    "future_heat_risk_level": "Unknown",
                    "priority_level": "Unknown",
                }
            )

        if trend_by_key is not None and ward_key in trend_by_key.index:
            trend_row = trend_by_key.loc[ward_key]
            merged["combined_adjusted_trend"] = safe_value(
                trend_row.get("combined_adjusted_trend")
            )

        if policy_by_key is not None and ward_key in policy_by_key.index:
            policy_row = policy_by_key.loc[ward_key]
            merged["policy_priority"] = safe_value(policy_row.get("policy_priority"))
            merged["primary_intervention"] = safe_value(
                policy_row.get("primary_intervention")
            )

        features.append(
            {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": merged,
            }
        )

    return {"type": "FeatureCollection", "features": features}


# ============================================================
# WARD LIST (LIGHTWEIGHT, NO GEOMETRY - FOR SEARCH)
# ============================================================

def build_ward_list(risk_df, trend_df=None, policy_df=None):
    columns = [
        "ward_key",
        "ward_name",
        "Corporation",
        "ward_id",
        "mean_lst_celsius",
        "heat_pocket_coverage_pct",
        "historical_heat_pocket_coverage_pct",
        "persistence_level",
        "future_heat_probability",
        "future_heat_risk_level",
        "priority_level",
        "data_status",
    ]

    df = risk_df[columns].copy()

    if trend_df is not None:
        df = df.merge(
            trend_df[["ward_key", "combined_adjusted_trend"]],
            on="ward_key",
            how="left",
        )

    if policy_df is not None:
        df = df.merge(
            policy_df[["ward_key", "policy_priority"]],
            on="ward_key",
            how="left",
        )

    df = df.replace({np.nan: None})

    return df.to_dict(orient="records")


# ============================================================
# OVERVIEW STATS
# ============================================================

def build_overview_stats(risk_df, trend_df, policy_df=None):
    available = risk_df[risk_df["data_status"] == "Available"].copy()

    total_wards = int(len(risk_df))

    high_risk_count = int((available["future_heat_risk_level"] == "High").sum())
    high_risk_pct = round(high_risk_count / total_wards * 100, 1) if total_wards else 0.0

    highest_lst_row = available.loc[available["mean_lst_celsius"].idxmax()]
    highest_lst = round(float(highest_lst_row["mean_lst_celsius"]), 1)
    highest_lst_zone = str(highest_lst_row["Corporation"])

    persistent_pockets = int((available["persistence_level"] == "High").sum())

    increasing_trend_wards = int(
        (trend_df["adjusted_heat_pocket_trend_direction"] == "Increasing").sum()
    )

    high_priority_count = int((risk_df["priority_level"] == "High").sum())

    future_high_heat_count = int(
        (available["future_high_heat_prediction"] == 1).sum()
    )

    very_high_policy_count = (
        int((policy_df["policy_priority"] == "Very High").sum())
        if policy_df is not None
        else None
    )

    analysis_date = str(risk_df["date"].dropna().iloc[0]) if len(risk_df) else None

    return {
        "total_wards": total_wards,
        "high_heat_risk_pct": high_risk_pct,
        "high_heat_risk_count": high_risk_count,
        "highest_observed_lst": highest_lst,
        "highest_lst_zone": highest_lst_zone,
        "persistent_heat_pockets": persistent_pockets,
        "high_priority_count": high_priority_count,
        "future_high_heat_count": future_high_heat_count,
        "very_high_policy_count": very_high_policy_count,
        "analysis_date": analysis_date,
        "increasing_trend_wards": increasing_trend_wards,
    }


# ============================================================
# INSIGHTS (COMPUTED, NOT SCRIPTED)
# ============================================================

def build_insights(risk_df, trend_df, ml_df):
    available = risk_df[risk_df["data_status"] == "Available"].copy()
    insights = []

    # 1. Corporation(s) with highest mean future heat probability
    by_corp = (
        available.groupby("Corporation")["future_heat_probability"]
        .mean()
        .sort_values(ascending=False)
    )
    if not by_corp.empty:
        top_corp = by_corp.index[0]
        insights.append(
            {
                "title": f"{top_corp} zone shows the highest average future heat risk",
                "text": (
                    f"Mean model-estimated future heat probability of "
                    f"{by_corp.iloc[0] * 100:.1f}% across wards in this zone."
                ),
                "tone": "hot",
            }
        )

    # 2. NDVI vs LST relationship (observed correlation, not causal)
    corr_df = ml_df[["mean_ndvi", "mean_lst_celsius"]].dropna()
    if len(corr_df) > 1:
        correlation = float(corr_df["mean_ndvi"].corr(corr_df["mean_lst_celsius"]))
        slope = float(
            np.polyfit(corr_df["mean_ndvi"], corr_df["mean_lst_celsius"], 1)[0]
        )
        ndvi_range = float(corr_df["mean_ndvi"].max() - corr_df["mean_ndvi"].min())
        spread = abs(slope * ndvi_range)
        insights.append(
            {
                "title": f"Higher vegetation cover tracks with up to {spread:.1f}°C lower LST",
                "text": (
                    f"NDVI and land surface temperature show a correlation of "
                    f"{correlation:.2f} across all observations."
                ),
                "tone": "green",
            }
        )

    # 3. Persistent heat pockets
    persistent = int((available["persistence_level"] == "High").sum())
    insights.append(
        {
            "title": f"{persistent} wards show persistently high heat-pocket coverage",
            "text": "Historical heat-pocket coverage classified High across observations.",
            "tone": "violet",
        }
    )

    # 4. FDR-significant increasing heat-pocket trend wards
    increasing = trend_df[trend_df["adjusted_heat_pocket_trend_direction"] == "Increasing"]
    if len(increasing) > 0:
        top_corp_increasing = increasing["Corporation"].value_counts().idxmax()
        insights.append(
            {
                "title": f"{len(increasing)} wards show a statistically significant rising heat trend",
                "text": (
                    f"Most concentrated in {top_corp_increasing} zone, based on "
                    f"FDR-corrected seasonal trend analysis."
                ),
                "tone": "blue",
            }
        )

    return insights


# ============================================================
# LST SEASONAL TREND (CLIMATOLOGY BY CALENDAR MONTH)
# ============================================================

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def build_lst_trend(ml_df):
    df = ml_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    monthly_avg = df.groupby("month")["mean_lst_celsius"].mean()

    return [
        {
            "month": MONTH_LABELS[month - 1],
            "avg_lst_celsius": round(float(monthly_avg.get(month, np.nan)), 2)
            if month in monthly_avg.index
            else None,
        }
        for month in range(1, 13)
    ]


# ============================================================
# WARD OBSERVATION HISTORY (for Ward Intelligence charts)
# ============================================================

def build_ward_history(ward_key, ml_df):
    df = ml_df[ml_df["ward_key"].astype(str) == str(ward_key)].copy()

    if df.empty:
        return []

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    columns = [
        "mean_lst_celsius",
        "heat_pocket_coverage_pct",
        "mean_ndvi",
        "mean_ndbi",
        "mean_ndwi",
    ]

    records = []
    for _, row in df.iterrows():
        record = {"date": row["date"].strftime("%Y-%m-%d")}
        for column in columns:
            record[column] = safe_value(row.get(column))
        records.append(record)

    return records


# ============================================================
# HUMAN-READABLE FEATURE METADATA (single source of truth for
# XAI display, HeatLens AI context, and PDF reports)
# ============================================================

FEATURE_METADATA = {
    "mean_lst_celsius": {
        "label": "Current Land Surface Temperature",
        "topic": "the current observed land-surface temperature",
    },
    "previous_lst_celsius": {
        "label": "Previous Land Surface Temperature",
        "topic": "the land-surface temperature at the prior observation",
    },
    "mean_ndvi": {
        "label": "Vegetation Index (NDVI)",
        "topic": "the vegetation index (NDVI)",
    },
    "mean_ndbi": {
        "label": "Built-up Index (NDBI)",
        "topic": "the built-up index (NDBI)",
    },
    "mean_ndwi": {
        "label": "Moisture Index (NDWI)",
        "topic": "the moisture index (NDWI)",
    },
    "heat_pocket_coverage_pct": {
        "label": "Current Heat-Pocket Coverage",
        "topic": "the current heat-pocket coverage",
    },
    "previous_heat_pocket_coverage_pct": {
        "label": "Previous Heat-Pocket Coverage",
        "topic": "the heat-pocket coverage at the prior observation",
    },
    "historical_heat_pocket_coverage_pct": {
        "label": "Historical Heat-Pocket Coverage",
        "topic": "the historical average heat-pocket coverage",
    },
    "days_since_previous_observation": {
        "label": "Time Since Previous Observation",
        "topic": "the time elapsed since the prior usable observation",
    },
    "month_sin": {
        "label": "Seasonal Timing (Time of Year)",
        "topic": "the seasonal timing of the observation",
    },
    "month_cos": {
        "label": "Seasonal Timing (Time of Year)",
        "topic": "the seasonal timing of the observation",
    },
}


def describe_feature(feature, direction, rank):
    """Human-readable label, strength tier, and one-line plain-English
    explanation for a single SHAP feature contribution. Shared by the
    Ward Intelligence XAI panel, HeatLens AI context, and PDF reports."""

    meta = FEATURE_METADATA.get(
        feature,
        {"label": feature.replace("_", " ").title(), "topic": f"'{feature}'"},
    )

    if direction == "increases":
        verb = "increased"
    elif direction == "decreases":
        verb = "decreased"
    else:
        verb = "had a neutral effect on"

    strength = "Strong" if rank < 2 else "Moderate" if rank < 5 else "Weak"

    explanation = (
        f"{meta['label']} {verb} the model's predicted probability of high "
        f"heat at the next observation."
    )

    return meta["label"], strength, explanation


# ============================================================
# WARD LOCAL XAI CONTRIBUTIONS (SHAP)
# ============================================================

def build_ward_xai_local(ward_key, xai_local_df):
    if xai_local_df is None:
        return []

    df = xai_local_df[xai_local_df["ward_key"].astype(str) == str(ward_key)].copy()

    if df.empty:
        return []

    df = df.sort_values("absolute_shap_value", ascending=False).reset_index(drop=True)

    contributions = []

    for rank, row in df.iterrows():
        human_label, strength, explanation = describe_feature(
            row["feature"], row["direction"], rank
        )

        contributions.append(
            {
                "feature": row["feature"],
                "human_label": human_label,
                "feature_value": safe_value(row["feature_value"]),
                "shap_value": safe_value(row["shap_value"]),
                "direction": row["direction"],
                "strength": strength,
                "explanation": explanation,
            }
        )

    return contributions


# ============================================================
# ENVIRONMENTAL PROFILE (NDVI / NDBI / NDWI, latest observation)
# ============================================================

ENV_METADATA = {
    "mean_ndvi": {
        "label": "NDVI",
        "name": "Vegetation Index",
        "interpretation": (
            "Higher values indicate more vegetation cover. Typical range is "
            "roughly -1 to 1; built-up or bare surfaces sit near or below 0, "
            "dense vegetation is generally above 0.3."
        ),
    },
    "mean_ndbi": {
        "label": "NDBI",
        "name": "Built-up Index",
        "interpretation": (
            "Higher values indicate more built-up or impervious surface. "
            "Typical range is roughly -1 to 1; vegetation and water bodies "
            "sit below 0."
        ),
    },
    "mean_ndwi": {
        "label": "NDWI",
        "name": "Moisture Index",
        "interpretation": (
            "Higher values indicate more surface water or moisture content. "
            "Typical range is roughly -1 to 1; built-up or dry surfaces sit "
            "below 0."
        ),
    },
}


def build_ward_environmental(ward_key, ml_df):
    """Latest-observation NDVI/NDBI/NDWI for one ward, with shared
    human-readable labels and neutral (non-causal) interpretations."""

    if ml_df is None:
        return {}

    df = ml_df[ml_df["ward_key"].astype(str) == str(ward_key)].copy()

    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").iloc[-1]

    profile = {}

    for column, meta in ENV_METADATA.items():
        profile[column] = {
            "value": safe_value(latest.get(column)),
            "label": meta["label"],
            "name": meta["name"],
            "interpretation": meta["interpretation"],
        }

    return profile


# ============================================================
# HISTORICAL TREND — PLAIN-ENGLISH EXPLANATION
# ============================================================

def describe_trend(trend):
    """Generate a plain-English explanation from actual trend values.
    Shared by Ward Intelligence, HeatLens AI, and PDF reports so the
    wording is always derived from the same numbers."""

    if not trend or trend.get("trend_slope") is None:
        return None

    slope = trend.get("trend_slope")
    fdr_p = trend.get("fdr_adjusted_p_value")
    direction = trend.get("trend_direction") or "Stable/Uncertain"
    classification = trend.get("trend_classification") or "Stable / uncertain"

    significant = fdr_p is not None and fdr_p < 0.05

    direction_word = (
        "an increasing" if direction == "Increasing"
        else "a decreasing" if direction == "Decreasing"
        else "a stable/uncertain"
    )

    sentence = (
        f"Heat-pocket coverage has an estimated slope of {slope:.3f} "
        f"percentage points per year, classified as {direction_word} trend."
    )

    if fdr_p is not None:
        if significant:
            sentence += (
                f" The FDR-adjusted p-value of {fdr_p:.4f} provides "
                f"statistical evidence, after correcting for multiple "
                f"comparisons, that this trend is significant."
            )
        else:
            sentence += (
                f" The FDR-adjusted p-value of {fdr_p:.4f} does not provide "
                f"sufficient statistical evidence, after correcting for "
                f"multiple comparisons, to conclude this trend is "
                f"statistically significant."
            )
    else:
        sentence += " No FDR-adjusted significance test result is available."

    if not significant:
        sentence += (
            " A stable/uncertain historical trend does not mean the ward "
            "has no current or future heat risk - refer to the current "
            "conditions and next-observation prediction for that."
        )

    return {
        "explanation": sentence,
        "significant": significant,
        "classification": classification,
    }


# ============================================================
# POLICY ENGINE — MECHANISM / RATIONALE METADATA
# ============================================================
# Read-time interpretation layer over the Policy Engine's existing
# recommendation strings. Does not change or override the Policy
# Engine's decision logic - only explains it. No effect sizes are
# invented; the current pipeline does not calibrate one.

INTERVENTION_CATEGORIES = [
    (
        "cool roof",
        {
            "mechanism": "Reduces solar heat absorption and surface heating by increasing surface reflectivity.",
            "indicators_affected": ["Land-surface temperature", "Heat-pocket coverage"],
        },
    ),
    (
        "reflective",
        {
            "mechanism": "Reduces solar heat absorption and surface heating by increasing surface reflectivity.",
            "indicators_affected": ["Land-surface temperature", "Heat-pocket coverage"],
        },
    ),
    (
        "cool pavement",
        {
            "mechanism": "Reduces solar heat absorption and surface heating by increasing surface reflectivity.",
            "indicators_affected": ["Land-surface temperature", "Heat-pocket coverage"],
        },
    ),
    (
        "greening",
        {
            "mechanism": "Increases vegetation cover, providing shading and evaporative cooling potential.",
            "indicators_affected": ["Land-surface temperature", "NDVI (vegetation index)", "Heat-pocket coverage"],
        },
    ),
    (
        "vegetation",
        {
            "mechanism": "Increases vegetation cover, providing shading and evaporative cooling potential.",
            "indicators_affected": ["Land-surface temperature", "NDVI (vegetation index)", "Heat-pocket coverage"],
        },
    ),
    (
        "tree canopy",
        {
            "mechanism": "Increases vegetation cover, providing shading and evaporative cooling potential.",
            "indicators_affected": ["Land-surface temperature", "NDVI (vegetation index)", "Heat-pocket coverage"],
        },
    ),
    (
        "monitoring",
        {
            "mechanism": "Decision-support tracking of future satellite observations; not itself a physical heat-mitigation action.",
            "indicators_affected": ["Future heat probability", "Persistence level"],
        },
    ),
    (
        "insufficient data",
        {
            "mechanism": "No intervention is assigned until sufficient observation data becomes available.",
            "indicators_affected": [],
        },
    ),
]


def describe_intervention(intervention_text, priority_level=None, trend_significant=None):
    """Attach a mechanism and affected-indicators explanation to a
    Policy Engine intervention string, using substring matching over
    the Policy Engine's existing vocabulary (no new categories are
    invented, and no effect size is estimated)."""

    if not intervention_text:
        return None

    text_lower = str(intervention_text).lower()

    meta = None
    for keyword, candidate in INTERVENTION_CATEGORIES:
        if keyword in text_lower:
            meta = candidate
            break

    if meta is None:
        meta = {
            "mechanism": "Mechanism not categorized by the current policy metadata.",
            "indicators_affected": [],
        }

    why_parts = []
    if priority_level:
        why_parts.append(f"Policy priority level: {priority_level}.")
    if trend_significant is True:
        why_parts.append("Historical heat-pocket trend is FDR-significant.")
    elif trend_significant is False:
        why_parts.append("Based on current and historical heat conditions.")

    return {
        "intervention": intervention_text,
        "why": " ".join(why_parts) if why_parts else "Based on the Policy Engine's decision rules.",
        "mechanism": meta["mechanism"],
        "indicators_affected": meta["indicators_affected"],
        "effect_size_note": "Effect size not estimated by the current model.",
    }


# ============================================================
# CITY-WIDE CONTEXT (for HeatLens AI when no single ward applies)
# ============================================================
# Reuses the same aggregates shown on Overview, plus named ward lists,
# so city-wide questions ("which wards have X", "how many wards show
# Y") are answered from real data instead of getting no context at
# all.

def _ward_list(df, name_col="ward_name", key_col="ward_key", limit=25):
    rows = df[[key_col, name_col]].head(limit)
    return [f"{row[name_col]} ({row[key_col]})" for _, row in rows.iterrows()]


def build_city_context(risk_df, trend_df, policy_df, ml_df):
    stats = build_overview_stats(risk_df, trend_df, policy_df)
    insights = build_insights(risk_df, trend_df, ml_df)

    available = risk_df[risk_df["data_status"] == "Available"].copy()

    persistent_wards = available[available["persistence_level"] == "High"]
    high_priority_wards = risk_df[risk_df["priority_level"] == "High"].sort_values(
        "future_heat_probability", ascending=False
    )

    increasing_trend = trend_df[trend_df["adjusted_heat_pocket_trend_direction"] == "Increasing"]
    increasing_trend_wards = increasing_trend.merge(
        risk_df[["ward_key", "ward_name"]], on="ward_key", how="left", suffixes=("", "_risk")
    )
    if "ward_name" not in increasing_trend_wards.columns and "ward_name_risk" in increasing_trend_wards.columns:
        increasing_trend_wards["ward_name"] = increasing_trend_wards["ward_name_risk"]

    very_high_policy_wards = (
        policy_df[policy_df["policy_priority"] == "Very High"]
        if policy_df is not None
        else policy_df
    )

    return {
        "scope": "city_wide_bengaluru",
        "note": (
            "No single ward was identified for this question, so this is a "
            "city-wide summary across all 369 wards, computed from the same "
            "connected analytics datasets shown on the Overview page."
        ),
        "overview_stats": stats,
        "computed_insights": [{"title": i["title"], "text": i["text"]} for i in insights],
        "wards_with_high_persistence": _ward_list(persistent_wards),
        "wards_with_high_priority": _ward_list(high_priority_wards),
        "wards_with_significant_increasing_trend": _ward_list(increasing_trend_wards)
        if "ward_name" in increasing_trend_wards.columns
        else [],
        "wards_with_very_high_policy_priority": (
            _ward_list(very_high_policy_wards) if very_high_policy_wards is not None else []
        ),
    }
