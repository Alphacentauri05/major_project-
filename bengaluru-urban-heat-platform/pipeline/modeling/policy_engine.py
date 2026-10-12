from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
    / "ward_heat_risk_20260503.csv"
)

TREND_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "persistence"
    / "ward_historical_trends.csv"
)

XAI_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
    / "heat_risk_xai_explanations_20260503.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "ward_policy_recommendations_20260503.csv"
)


# ============================================================
# EVIDENCE SOURCES
# ============================================================

EPA_HEAT_GUIDE = (
    "U.S. EPA — Guide to Reducing Heat Islands"
)

EPA_TREES = (
    "U.S. EPA — Using Trees and Vegetation to Reduce Heat Islands"
)

EPA_GREEN_ROOFS = (
    "U.S. EPA — Using Green Roofs to Reduce Heat Islands"
)

EPA_COOL_PAVEMENTS = (
    "U.S. EPA — Using Cool Pavements to Reduce Heat Islands"
)

WHO_GREEN_SPACE = (
    "WHO — Urban green space interventions and health: "
    "A review of impacts and effectiveness"
)


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    print("=" * 70)
    print("POLICY ENGINE")
    print("=" * 70)

    print()
    print("Loading risk engine output...")

    if not RISK_PATH.exists():
        raise FileNotFoundError(
            f"Risk file not found:\n{RISK_PATH}"
        )

    risk = pd.read_csv(RISK_PATH)

    print(
        f"Risk rows: {len(risk):,}"
    )

    print()
    print("Loading historical trend output...")

    if not TREND_PATH.exists():
        raise FileNotFoundError(
            f"Trend file not found:\n{TREND_PATH}"
        )

    trends = pd.read_csv(TREND_PATH)

    print(
        f"Trend rows: {len(trends):,}"
    )

    print()
    print("Loading XAI output...")

    if not XAI_PATH.exists():
        raise FileNotFoundError(
            f"XAI file not found:\n{XAI_PATH}"
        )

    xai = pd.read_csv(XAI_PATH)

    print(
        f"XAI rows: {len(xai):,}"
    )

    return risk, trends, xai


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(risk, trends, xai):

    required_risk_columns = [
        "ward_key",
        "ward_name",
        "data_status",
        "mean_lst_celsius",
        "heat_pocket_coverage_pct",
        "current_heat_severity",
        "current_heat_pocket_level",
        "historical_heat_pocket_coverage_pct",
        "persistence_level",
        "future_heat_probability",
        "future_heat_risk_level",
        "priority_level",
    ]

    required_trend_columns = [
        "ward_key",
        "adjusted_heat_pocket_trend_pct_points_per_year",
        "adjusted_heat_pocket_trend_fdr_p_value",
        "adjusted_heat_pocket_fdr_significant",
        "adjusted_heat_pocket_trend_direction",
        "combined_adjusted_trend",
    ]

    required_xai_columns = [
        "ward_key",
        "future_heat_probability",
        "xai_explanation",
    ]

    missing_risk = [
        col
        for col in required_risk_columns
        if col not in risk.columns
    ]

    missing_trend = [
        col
        for col in required_trend_columns
        if col not in trends.columns
    ]

    missing_xai = [
        col
        for col in required_xai_columns
        if col not in xai.columns
    ]

    if missing_risk:
        raise ValueError(
            f"Missing risk columns: {missing_risk}"
        )

    if missing_trend:
        raise ValueError(
            f"Missing trend columns: {missing_trend}"
        )

    if missing_xai:
        raise ValueError(
            f"Missing XAI columns: {missing_xai}"
        )

    print()
    print("Input validation: PASSED")


# ============================================================
# MERGE INPUTS
# ============================================================

def merge_inputs(risk, trends, xai):

    print()
    print("Merging risk, trend and XAI information...")

    trend_columns = [
        "ward_key",
        "adjusted_heat_pocket_trend_pct_points_per_year",
        "adjusted_heat_pocket_trend_fdr_p_value",
        "adjusted_heat_pocket_fdr_significant",
        "adjusted_heat_pocket_trend_direction",
        "combined_adjusted_trend",
    ]

    xai_columns = [
        "ward_key",
        "xai_explanation",
    ]

    df = risk.merge(
        trends[trend_columns],
        on="ward_key",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        xai[xai_columns],
        on="ward_key",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Merged rows: {len(df):,}"
    )

    return df


# ============================================================
# TREND CLASSIFICATION
# ============================================================

def classify_trend(row):

    significant = bool(
        row["adjusted_heat_pocket_fdr_significant"]
    )

    direction = str(
        row["adjusted_heat_pocket_trend_direction"]
    ).strip().lower()

    if significant and direction == "increasing":
        return "FDR-significant increasing"

    if significant and direction == "decreasing":
        return "FDR-significant decreasing"

    if direction == "increasing":
        return "Increasing but not FDR-significant"

    if direction == "decreasing":
        return "Decreasing but not FDR-significant"

    return "Stable/uncertain"


# ============================================================
# POLICY LOGIC
# ============================================================

def generate_policy(row):

    # --------------------------------------------------------
    # Missing / insufficient data
    # --------------------------------------------------------

    if row["data_status"] != "Available":

        return {
            "policy_priority": "Unknown",
            "primary_intervention": "Insufficient data",
            "secondary_intervention": "No intervention assigned",
            "recommendation": (
                "Collect additional observations before "
                "assigning a ward-level heat mitigation priority."
            ),
            "implementation_note": (
                "This ward does not currently have sufficient "
                "observation data for the risk engine."
            ),
        }

    # --------------------------------------------------------
    # Read model outputs
    # --------------------------------------------------------

    risk_level = str(
        row["future_heat_risk_level"]
    ).strip().lower()

    priority = str(
        row["priority_level"]
    ).strip().lower()

    persistence = str(
        row["persistence_level"]
    ).strip().lower()

    trend = row["policy_trend"]

    current_heat = str(
        row["current_heat_severity"]
    ).strip().lower()

    current_pocket = str(
        row["current_heat_pocket_level"]
    ).strip().lower()

    # --------------------------------------------------------
    # FDR-significant increasing heat trend
    # --------------------------------------------------------

    if trend == "FDR-significant increasing":

        if risk_level == "high" or priority == "high":

            return {
                "policy_priority": "Very High",
                "primary_intervention": (
                    "Integrated urban heat mitigation"
                ),
                "secondary_intervention": (
                    "Cool roofs / reflective surfaces "
                    "and urban greening"
                ),
                "recommendation": (
                    "Prioritize this ward for detailed site-level "
                    "assessment and coordinated heat-mitigation "
                    "planning because the model indicates elevated "
                    "future heat risk and a statistically significant "
                    "increasing heat-pocket trend."
                ),
                "implementation_note": (
                    "Use satellite results to prioritize locations; "
                    "confirm physical feasibility, land ownership, "
                    "building suitability and local implementation "
                    "constraints before intervention."
                ),
            }

        return {
            "policy_priority": "High",
            "primary_intervention": (
                "Preventive urban heat mitigation"
            ),
            "secondary_intervention": (
                "Urban greening and cool/reflective surfaces"
            ),
            "recommendation": (
                "Prioritize preventive heat-mitigation planning "
                "because heat-pocket extent shows a statistically "
                "significant increasing trend."
            ),
            "implementation_note": (
                "Conduct site-level assessment before selecting "
                "specific intervention locations."
            ),
        }

    # --------------------------------------------------------
    # High risk + persistent heat
    # --------------------------------------------------------

    if (
        risk_level == "high"
        and persistence == "high"
    ):

        return {
            "policy_priority": "Very High",
            "primary_intervention": (
                "Cool roofs / reflective surfaces"
            ),
            "secondary_intervention": (
                "Urban greening / tree canopy enhancement"
            ),
            "recommendation": (
                "Prioritize persistent hotspot areas for "
                "building- and neighborhood-scale cooling measures."
            ),
            "implementation_note": (
                "Use ward-level results to identify priority zones; "
                "verify building, road and vegetation conditions "
                "before implementation."
            ),
        }

    # --------------------------------------------------------
    # High risk + moderate persistence
    # --------------------------------------------------------

    if (
        risk_level == "high"
        and persistence in ["moderate", "low"]
    ):

        return {
            "policy_priority": "High",
            "primary_intervention": (
                "Cool roofs / reflective surfaces"
            ),
            "secondary_intervention": (
                "Urban greening"
            ),
            "recommendation": (
                "Prioritize heat-mitigation measures while "
                "monitoring whether hotspot conditions persist "
                "across future observations."
            ),
            "implementation_note": (
                "The current/future risk signal is high, but "
                "historical persistence is not classified as high."
            ),
        }

    # --------------------------------------------------------
    # Moderate risk + increasing trend
    # --------------------------------------------------------

    if (
        risk_level == "moderate"
        and trend == "Increasing but not FDR-significant"
    ):

        return {
            "policy_priority": "Moderate-High",
            "primary_intervention": (
                "Preventive urban greening"
            ),
            "secondary_intervention": (
                "Cool or reflective surfaces"
            ),
            "recommendation": (
                "Consider preventive heat-mitigation measures "
                "and continue monitoring because the historical "
                "heat-pocket trend is increasing, although it does "
                "not meet the FDR-significance criterion."
            ),
            "implementation_note": (
                "Treat the trend as an early-warning signal rather "
                "than definitive evidence of long-term change."
            ),
        }

    # --------------------------------------------------------
    # Moderate risk + persistent heat
    # --------------------------------------------------------

    if (
        risk_level == "moderate"
        and persistence == "high"
    ):

        return {
            "policy_priority": "High",
            "primary_intervention": (
                "Urban greening / tree canopy enhancement"
            ),
            "secondary_intervention": (
                "Cool roofs / reflective surfaces"
            ),
            "recommendation": (
                "Prioritize persistent hotspot zones for "
                "preventive cooling interventions even though "
                "future model risk is moderate."
            ),
            "implementation_note": (
                "Use persistent hotspot maps to guide site-level "
                "assessment."
            ),
        }

    # --------------------------------------------------------
    # Moderate risk
    # --------------------------------------------------------

    if risk_level == "moderate":

        return {
            "policy_priority": "Moderate",
            "primary_intervention": (
                "Monitoring and preventive cooling"
            ),
            "secondary_intervention": (
                "Urban greening where feasible"
            ),
            "recommendation": (
                "Continue monitoring and consider low-regret "
                "heat-mitigation measures where site conditions "
                "support them."
            ),
            "implementation_note": (
                "No immediate intervention type is mandated by "
                "the model alone."
            ),
        }

    # --------------------------------------------------------
    # Low risk
    # --------------------------------------------------------

    return {
        "policy_priority": "Low",
        "primary_intervention": (
            "Routine monitoring"
        ),
        "secondary_intervention": (
            "Maintain existing vegetation and cooling measures"
        ),
        "recommendation": (
            "Continue routine monitoring rather than prioritizing "
            "major heat-mitigation intervention based on the "
            "current model outputs."
        ),
        "implementation_note": (
            "Reassess if future observations indicate increasing "
            "heat risk or persistent hotspot development."
        ),
    }


# ============================================================
# APPLY POLICY ENGINE
# ============================================================

def apply_policy_engine(df):

    print()
    print("Generating policy recommendations...")

    df = df.copy()

    # --------------------------------------------------------
    # Convert numeric fields
    # --------------------------------------------------------

    numeric_columns = [
        "mean_lst_celsius",
        "heat_pocket_coverage_pct",
        "historical_heat_pocket_coverage_pct",
        "historical_max_heat_pocket_coverage_pct",
        "future_heat_probability",
        "adjusted_heat_pocket_trend_pct_points_per_year",
        "adjusted_heat_pocket_trend_fdr_p_value",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Trend classification
    # --------------------------------------------------------

    df["policy_trend"] = df.apply(
        classify_trend,
        axis=1,
    )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    policy_results = df.apply(
        generate_policy,
        axis=1,
        result_type="expand",
    )

    df = pd.concat(
        [
            df,
            policy_results,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Evidence basis
    # --------------------------------------------------------

    df["evidence_basis"] = (
        "EPA heat-island mitigation guidance; "
        "WHO urban green-space intervention review"
    )

    # --------------------------------------------------------
    # XAI interpretation
    # --------------------------------------------------------

    df["model_explanation"] = (
        df["xai_explanation"]
    )

    return df


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

def prepare_output(df):

    output_columns = [
        # Identity
        "date",
        "ward_key",
        "Corporation",
        "ward_id",
        "ward_name",
        "data_status",

        # Current heat
        "mean_lst_celsius",
        "heat_pocket_coverage_pct",
        "current_heat_severity",
        "current_heat_pocket_level",

        # Persistence
        "historical_heat_pocket_coverage_pct",
        "historical_max_heat_pocket_coverage_pct",
        "persistence_level",

        # Future model
        "future_heat_probability",
        "future_high_heat_prediction",
        "future_heat_risk_level",
        "priority_level",

        # Historical trend
        "adjusted_heat_pocket_trend_pct_points_per_year",
        "adjusted_heat_pocket_trend_fdr_p_value",
        "adjusted_heat_pocket_fdr_significant",
        "adjusted_heat_pocket_trend_direction",
        "policy_trend",

        # Policy
        "policy_priority",
        "primary_intervention",
        "secondary_intervention",
        "recommendation",
        "implementation_note",

        # Explainability
        "model_explanation",

        # Evidence
        "evidence_basis",
    ]

    existing_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    return df[existing_columns].copy()


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Saved policy recommendations:")
    print(
        f"  {OUTPUT_PATH}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df):

    print()
    print("=" * 70)
    print("POLICY ENGINE COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Wards processed: {len(df):,}"
    )

    print()
    print("Policy priority distribution:")

    print(
        df["policy_priority"]
        .value_counts()
        .to_string()
    )

    print()
    print("Primary intervention distribution:")

    print(
        df["primary_intervention"]
        .value_counts()
        .to_string()
    )

    print()
    print("Trend classification:")

    print(
        df["policy_trend"]
        .value_counts()
        .to_string()
    )

    print()
    print("Example recommendations:")

    example_columns = [
        "ward_name",
        "policy_priority",
        "primary_intervention",
        "secondary_intervention",
    ]

    print(
        df[example_columns]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("DONE.")


# ============================================================
# MAIN
# ============================================================

def run_policy_engine():

    risk, trends, xai = load_inputs()

    validate_inputs(
        risk,
        trends,
        xai,
    )

    df = merge_inputs(
        risk,
        trends,
        xai,
    )

    df = apply_policy_engine(
        df
    )

    df = prepare_output(
        df
    )

    save_output(
        df
    )

    print_summary(
        df
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_policy_engine()