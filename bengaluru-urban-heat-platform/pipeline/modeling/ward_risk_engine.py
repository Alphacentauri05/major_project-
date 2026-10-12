from pathlib import Path
import argparse
import json

import json

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "ward_temporal_ml_dataset.csv"
)

BOUNDARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boundaries"
    / "gba_wards_369.geojson"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "heat_risk_xgboost.json"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "heat_risk_xgboost_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "mean_lst_celsius",
    "previous_lst_celsius",
    "mean_ndvi",
    "mean_ndbi",
    "mean_ndwi",
    "heat_pocket_coverage_pct",
    "previous_heat_pocket_coverage_pct",
    "historical_heat_pocket_coverage_pct",
    "days_since_previous_observation",
    "month_sin",
    "month_cos",
]


# ============================================================
# RISK THRESHOLDS
# ============================================================

# Current heat severity
CURRENT_HEAT_MODERATE = 35.0
CURRENT_HEAT_HIGH = 40.0

# Current heat-pocket coverage
CURRENT_POCKET_MODERATE = 1.0
CURRENT_POCKET_HIGH = 4.187

# Historical persistence
PERSISTENCE_MODERATE = 5.0
PERSISTENCE_HIGH = 20.0

# Future model probability
FUTURE_RISK_MODERATE = 0.39
FUTURE_RISK_HIGH = 0.70


# ============================================================
# CLASSIFICATION HELPERS
# ============================================================

def classify_current_heat(lst):
    if pd.isna(lst):
        return "Unknown"

    if lst >= CURRENT_HEAT_HIGH:
        return "High"
    elif lst >= CURRENT_HEAT_MODERATE:
        return "Moderate"
    else:
        return "Low"


def classify_current_pocket(coverage):
    if pd.isna(coverage):
        return "Unknown"

    if coverage >= CURRENT_POCKET_HIGH:
        return "High"
    elif coverage >= CURRENT_POCKET_MODERATE:
        return "Moderate"
    else:
        return "Low"


def classify_persistence(persistence):
    if pd.isna(persistence):
        return "Unknown"

    if persistence >= PERSISTENCE_HIGH:
        return "High"
    elif persistence >= PERSISTENCE_MODERATE:
        return "Moderate"
    else:
        return "Low"


def classify_future_probability(probability):
    if pd.isna(probability):
        return "Unknown"

    if probability >= FUTURE_RISK_HIGH:
        return "High"
    elif probability >= FUTURE_RISK_MODERATE:
        return "Moderate"
    else:
        return "Low"


def calculate_priority(row):
    """
    Transparent decision-support rule.

    This is NOT a scientifically calibrated numerical risk score.
    It combines three independently visible indicators:

        1. Current heat
        2. Historical persistence
        3. Future heat probability

    Priority is intentionally rule-based and interpretable.
    """

    current_heat = row["current_heat_severity"]
    persistence = row["persistence_level"]
    future_risk = row["future_heat_risk_level"]

    levels = {
        "Low": 0,
        "Moderate": 1,
        "High": 2,
        "Unknown": -1,
    }

    values = [
        levels.get(current_heat, -1),
        levels.get(persistence, -1),
        levels.get(future_risk, -1),
    ]

    known_values = [
        value for value in values
        if value >= 0
    ]

    if not known_values:
        return "Unknown"

    high_count = sum(
        value == 2
        for value in known_values
    )

    moderate_count = sum(
        value == 1
        for value in known_values
    )

    # Two or more high-risk indicators
    if high_count >= 2:
        return "High"

    # One high + one moderate indicator
    if high_count == 1 and moderate_count >= 1:
        return "High"

    # One high indicator
    if high_count == 1:
        return "Moderate"

    # Two or more moderate indicators
    if moderate_count >= 2:
        return "Moderate"

    return "Low"


# ============================================================
# LOAD MODEL + METADATA
# ============================================================

def load_model_and_metadata():

    print("Loading XGBoost model...")

    model = xgb.XGBClassifier()

    model.load_model(MODEL_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print("Model loaded successfully.")

    return model, metadata


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("Loading ward dataset...")

    df = pd.read_csv(DATASET_PATH)

    df["date"] = pd.to_datetime(df["date"])

# Keep ward IDs consistent with the boundary file
    df["ward_id"] = df["ward_id"].astype(str)
    df["Corporation"] = df["Corporation"].astype(str)
    df["ward_key"] = (
    df["Corporation"]
    + "_"
    + df["ward_id"]
)
    print(f"Rows loaded: {len(df):,}")
    print(f"Wards in dataset: {df['ward_key'].nunique()}")

    return df


def load_boundary():

    print("Loading 369-ward boundary...")

    with open(BOUNDARY_PATH, "r", encoding="utf-8") as file:
        geojson = json.load(file)

    boundary = pd.DataFrame(
        [feature["properties"] for feature in geojson["features"]]
    )

    boundary["ward_key"] = (
        boundary["Corporation"].astype(str)
        + "_"
        + boundary["ward_id"].astype(str)
    )

    if len(boundary) != 369:
        raise ValueError(
            f"Expected 369 wards in boundary, "
            f"found {len(boundary)}."
        )

    if boundary["ward_key"].nunique() != 369:
        raise ValueError(
            "Boundary does not contain 369 unique ward keys."
        )

    print("Boundary loaded successfully.")
    print("Official wards:", len(boundary))

    return boundary[
        [
            "ward_key",
            "Corporation",
            "ward_id",
            "ward_name",
        ]
    ].copy()


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    features = df.copy()

    features["month"] = features["date"].dt.month

    features["month_sin"] = np.sin(
        2 * np.pi * features["month"] / 12
    )

    features["month_cos"] = np.cos(
        2 * np.pi * features["month"] / 12
    )

    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    return features


# ============================================================
# RUN RISK ENGINE
# ============================================================

def run_risk_engine(selected_date=None):

    df = load_dataset()

    boundary = load_boundary()

    model, metadata = load_model_and_metadata()

    df = prepare_features(df)

    # --------------------------------------------------------
    # Determine observation date
    # --------------------------------------------------------

    available_dates = sorted(
        df["date"].dropna().unique()
    )

    if not available_dates:
        raise ValueError(
            "No observation dates available."
        )

    if selected_date is None:

        selected_date = available_dates[-1]

    else:

        selected_date = pd.Timestamp(
            selected_date
        )

    print()
    print(
        f"Requested observation date: "
        f"{selected_date.date()}"
    )

    if selected_date not in available_dates:

        raise ValueError(
            f"Date {selected_date.date()} "
            f"is not available in the ML dataset."
        )

    current = df[
        df["date"] == selected_date
    ].copy()

    print(
        f"Wards with observation data: "
        f"{len(current)}"
    )

    # --------------------------------------------------------
    # Merge onto official 369-ward boundary list
    # --------------------------------------------------------

    result = boundary.merge(
        current,
        on=[
            "ward_key",
            "Corporation",
            "ward_id",
            "ward_name",
        ],
        how="left",
        suffixes=("", "_dataset"),
    )

    result["date"] = selected_date

    # --------------------------------------------------------
    # Data availability
    # --------------------------------------------------------

    result["data_status"] = np.where(
        result["mean_lst_celsius"].notna(),
        "Available",
        "Insufficient observation data",
    )

    # --------------------------------------------------------
    # Identify wards with usable model inputs
    # --------------------------------------------------------

    available_mask = (
        result["data_status"] == "Available"
    )

    available_data = result.loc[
        available_mask
    ].copy()

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    X = available_data[
        MODEL_FEATURES
    ].copy()

    if X.isna().any().any():

        missing_counts = X.isna().sum()

        raise ValueError(
            "Missing values detected in model "
            f"features:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    probabilities = model.predict_proba(X)[:, 1]

    available_data[
        "future_heat_probability"
    ] = probabilities

    decision_threshold = metadata[
        "decision_threshold"
    ]

    available_data[
        "future_high_heat_prediction"
    ] = (
        available_data[
            "future_heat_probability"
        ]
        >= decision_threshold
    ).astype(int)

    # --------------------------------------------------------
    # Current heat indicators
    # --------------------------------------------------------

    available_data[
        "current_heat_severity"
    ] = (
        available_data[
            "mean_lst_celsius"
        ]
        .apply(classify_current_heat)
    )

    available_data[
        "current_heat_pocket_level"
    ] = (
        available_data[
            "heat_pocket_coverage_pct"
        ]
        .apply(classify_current_pocket)
    )

    # --------------------------------------------------------
    # Historical persistence
    # --------------------------------------------------------

    available_data[
        "persistence_level"
    ] = (
        available_data[
            "historical_heat_pocket_coverage_pct"
        ]
        .apply(classify_persistence)
    )

    # --------------------------------------------------------
    # Future risk level
    # --------------------------------------------------------

    available_data[
        "future_heat_risk_level"
    ] = (
        available_data[
            "future_heat_probability"
        ]
        .apply(classify_future_probability)
    )

    # --------------------------------------------------------
    # Overall priority
    # --------------------------------------------------------

    available_data[
        "priority_level"
    ] = available_data.apply(
        calculate_priority,
        axis=1,
    )

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

    available_data[
        "high_heat_threshold_pct"
    ] = metadata[
        "target_threshold_heat_pocket_coverage_pct"
    ]

    available_data[
        "decision_threshold"
    ] = decision_threshold

    # --------------------------------------------------------
    # Put calculated data back into full 369-row result
    # --------------------------------------------------------

    calculated_columns = [
        "future_heat_probability",
        "future_high_heat_prediction",
        "current_heat_severity",
        "current_heat_pocket_level",
        "persistence_level",
        "future_heat_risk_level",
        "priority_level",
        "high_heat_threshold_pct",
        "decision_threshold",
    ]

    for column in calculated_columns:

        result.loc[
            available_mask,
            column,
        ] = available_data[column].values

    # --------------------------------------------------------
    # Explicit values for unavailable wards
    # --------------------------------------------------------

    unknown_columns = [
        "current_heat_severity",
        "current_heat_pocket_level",
        "persistence_level",
        "future_heat_risk_level",
        "priority_level",
    ]

    for column in unknown_columns:

        result.loc[
            ~available_mask,
            column,
        ] = "Unknown"

    result.loc[
        ~available_mask,
        "future_high_heat_prediction",
    ] = np.nan

    result.loc[
        ~available_mask,
        "future_heat_probability",
    ] = np.nan

    result.loc[
        ~available_mask,
        "high_heat_threshold_pct",
    ] = metadata[
        "target_threshold_heat_pocket_coverage_pct"
    ]

    result.loc[
        ~available_mask,
        "decision_threshold",
    ] = decision_threshold

    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    output_columns = [
        "date",
        "ward_key",
        "Corporation",
        "ward_id",
        "ward_name",

        # Data availability
        "data_status",

        # Current state
        "mean_lst_celsius",
        "heat_pocket_coverage_pct",
        "current_heat_severity",
        "current_heat_pocket_level",

        # Historical state
        "historical_heat_pocket_coverage_pct",
        "historical_max_heat_pocket_coverage_pct",
        "historical_observation_count",
        "persistence_level",

        # Future risk
        "future_heat_probability",
        "future_high_heat_prediction",
        "future_heat_risk_level",

        # Overall decision support
        "priority_level",

        # Model definitions
        "high_heat_threshold_pct",
        "decision_threshold",
    ]

    result = result[
        output_columns
    ].copy()

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    priority_order = {
        "High": 0,
        "Moderate": 1,
        "Low": 2,
        "Unknown": 3,
    }

    result["_priority_sort"] = (
        result[
            "priority_level"
        ].map(priority_order)
    )

    result = result.sort_values(
        by=[
            "_priority_sort",
            "future_heat_probability",
            "historical_heat_pocket_coverage_pct",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        na_position="last",
    )

    result = result.drop(
        columns=["_priority_sort"]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    date_string = selected_date.strftime(
        "%Y%m%d"
    )

    output_path = (
        OUTPUT_DIR
        / f"ward_heat_risk_{date_string}.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    available_count = int(
        (result["data_status"] == "Available").sum()
    )

    unavailable_count = int(
        (
            result["data_status"]
            == "Insufficient observation data"
        ).sum()
    )

    print()
    print("=" * 60)
    print("WARD HEAT RISK ENGINE COMPLETE")
    print("=" * 60)

    print(
        f"Observation date: "
        f"{selected_date.date()}"
    )

    print(
        f"Official wards: "
        f"{len(result)}"
    )

    print(
        f"Wards with data: "
        f"{available_count}"
    )

    print(
        f"Wards without sufficient data: "
        f"{unavailable_count}"
    )

    print(
        f"Output: "
        f"{output_path}"
    )

    print()
    print("Priority distribution:")

    print(
        result[
            "priority_level"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Future high-heat predictions:")

    print(
        result[
            "future_high_heat_prediction"
        ]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print()
    print("Data status:")

    print(
        result[
            "data_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Top 10 priority wards:")

    print(
        result[
            [
                "ward_name",
                "future_heat_probability",
                "historical_heat_pocket_coverage_pct",
                "mean_lst_celsius",
                "priority_level",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return result


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Bengaluru Ward Heat Risk Engine"
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Observation date in YYYY-MM-DD format. "
            "If omitted, the latest available "
            "observation is used."
        ),
    )

    args = parser.parse_args()

    run_risk_engine(
        args.date
    )