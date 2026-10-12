from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
import shap
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

RISK_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
    / "ward_heat_risk_20260503.csv"
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading XGBoost model...")

    model = xgb.XGBClassifier()

    model.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return metadata


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("Loading ML dataset...")

    df = pd.read_csv(
        DATASET_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    df = df.copy()

    df["month"] = (
        df["date"].dt.month
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing model features: {missing}"
        )

    return df


# ============================================================
# SHAP VALUE EXTRACTION
# ============================================================

def get_shap_values(
    model,
    X
):

    print("Creating SHAP explainer...")

    explainer = shap.TreeExplainer(
        model
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X
    )

    # SHAP can return different structures
    # depending on the installed SHAP/XGBoost versions.

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[-1]

    shap_values = np.asarray(
        shap_values
    )

    # Some newer SHAP/XGBoost combinations
    # can return an extra output dimension.

    if shap_values.ndim == 3:

        shap_values = shap_values[:, :, -1]

    if shap_values.shape != X.shape:

        raise ValueError(
            "Unexpected SHAP value shape. "
            f"X shape: {X.shape}, "
            f"SHAP shape: {shap_values.shape}"
        )

    return explainer, shap_values


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def create_global_importance(
    shap_values,
    X
):

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    total = importance[
        "mean_abs_shap"
    ].sum()

    if total > 0:

        importance[
            "importance_pct"
        ] = (
            importance["mean_abs_shap"]
            / total
            * 100
        )

    else:

        importance[
            "importance_pct"
        ] = 0.0

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False
    )

    return importance.reset_index(
        drop=True
    )


# ============================================================
# LOCAL SHAP DATASET
# ============================================================

def create_local_explanations(
    ward_data,
    X,
    shap_values
):

    explanation_rows = []

    for index in range(
        len(ward_data)
    ):

        row = ward_data.iloc[index]

        values = shap_values[index]

        # Sort features by absolute contribution
        # from strongest to weakest.

        feature_indices = np.argsort(
            np.abs(values)
        )[::-1]

        for rank, feature_index in enumerate(
            feature_indices,
            start=1
        ):

            feature = MODEL_FEATURES[
                feature_index
            ]

            shap_value = float(
                values[feature_index]
            )

            feature_value = float(
                X.iloc[
                    index,
                    feature_index
                ]
            )

            direction = (
                "increases"
                if shap_value > 0
                else "decreases"
                if shap_value < 0
                else "neutral"
            )

            explanation_rows.append(
                {
                    "date": row["date"],
                    "ward_key": row["ward_key"],
                    "Corporation": row["Corporation"],
                    "ward_id": row["ward_id"],
                    "ward_name": row["ward_name"],
                    "future_heat_probability": row[
                        "future_heat_probability"
                    ],
                    "priority_level": row[
                        "priority_level"
                    ],
                    "feature_rank": rank,
                    "feature": feature,
                    "feature_value": feature_value,
                    "shap_value": shap_value,
                    "direction": direction,
                    "absolute_shap_value": abs(
                        shap_value
                    ),
                }
            )

    return pd.DataFrame(
        explanation_rows
    )


# ============================================================
# HUMAN-READABLE EXPLANATION
# ============================================================

def create_text_explanations(
    ward_data,
    shap_values,
    X
):

    rows = []

    for index in range(
        len(ward_data)
    ):

        row = ward_data.iloc[index]

        values = shap_values[index]

        feature_indices = np.argsort(
            np.abs(values)
        )[::-1]

        increasing = []
        decreasing = []

        for feature_index in feature_indices:

            feature = MODEL_FEATURES[
                feature_index
            ]

            shap_value = float(
                values[feature_index]
            )

            if shap_value > 0:

                increasing.append(
                    feature
                )

            elif shap_value < 0:

                decreasing.append(
                    feature
                )

        top_increasing = (
            ", ".join(
                increasing[:3]
            )
            if increasing
            else "None"
        )

        top_decreasing = (
            ", ".join(
                decreasing[:3]
            )
            if decreasing
            else "None"
        )

        explanation = (
            f"Future heat probability is "
            f"{row['future_heat_probability']:.1%}. "
            f"The strongest factors increasing "
            f"the prediction were: "
            f"{top_increasing}. "
            f"The strongest factors decreasing "
            f"the prediction were: "
            f"{top_decreasing}."
        )

        rows.append(
            {
                "date": row["date"],
                "ward_key": row["ward_key"],
                "ward_name": row["ward_name"],
                "future_heat_probability": row[
                    "future_heat_probability"
                ],
                "priority_level": row[
                    "priority_level"
                ],
                "xai_explanation": explanation,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN XAI PIPELINE
# ============================================================

def run_xai():

    print()
    print("=" * 60)
    print("HEAT RISK XAI")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    model = load_model()

    metadata = load_metadata()

    df = prepare_features(
        df
    )

    # --------------------------------------------------------
    # Use the exact observation generated
    # by the Risk Engine
    # --------------------------------------------------------

    if not RISK_OUTPUT_PATH.exists():

        raise FileNotFoundError(
            "Risk Engine output not found:\n"
            f"{RISK_OUTPUT_PATH}\n\n"
            "Run ward_risk_engine.py first."
        )

    risk_df = pd.read_csv(
        RISK_OUTPUT_PATH
    )

    risk_df["date"] = pd.to_datetime(
        risk_df["date"]
    )

    # --------------------------------------------------------
    # Determine observation date
    # --------------------------------------------------------

    observation_dates = sorted(
        risk_df["date"].dropna().unique()
    )

    if not observation_dates:

        raise ValueError(
            "No observation date found "
            "in Risk Engine output."
        )

    selected_date = observation_dates[-1]

    print(
        f"Observation date: "
        f"{selected_date.date()}"
    )

    # --------------------------------------------------------
    # Get usable wards
    # --------------------------------------------------------

    current = df[
        df["date"] == selected_date
    ].copy()

    # Only wards that have actual Risk Engine
    # predictions can receive SHAP explanations.

    predicted_wards = risk_df[
        risk_df["data_status"] == "Available"
    ][
        [
            "ward_key",
            "future_heat_probability",
            "priority_level",
        ]
    ].copy()

    current = current.merge(
        predicted_wards,
        on="ward_key",
        how="inner",
    )

    print(
        f"Wards explained: "
        f"{len(current)}"
    )

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    X = current[
        MODEL_FEATURES
    ].copy()

    if X.isna().any().any():

        missing = X.isna().sum()

        raise ValueError(
            "Missing model features detected:\n"
            f"{missing[missing > 0]}"
        )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explainer, shap_values = get_shap_values(
        model,
        X
    )

    # --------------------------------------------------------
    # Global explanation
    # --------------------------------------------------------

    global_importance = create_global_importance(
        shap_values,
        X
    )

    global_path = (
        OUTPUT_DIR
        / "heat_risk_xai_global_importance.csv"
    )

    global_importance.to_csv(
        global_path,
        index=False
    )

    # --------------------------------------------------------
    # Local explanations
    # --------------------------------------------------------

    local_explanations = create_local_explanations(
        current,
        X,
        shap_values
    )

    local_path = (
        OUTPUT_DIR
        / f"heat_risk_xai_local_{selected_date.strftime('%Y%m%d')}.csv"
    )

    local_explanations.to_csv(
        local_path,
        index=False
    )

    # --------------------------------------------------------
    # Human-readable explanations
    # --------------------------------------------------------

    text_explanations = create_text_explanations(
        current,
        shap_values,
        X
    )

    text_path = (
        OUTPUT_DIR
        / f"heat_risk_xai_explanations_{selected_date.strftime('%Y%m%d')}.csv"
    )

    text_explanations.to_csv(
        text_path,
        index=False
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    xai_metadata = {
        "model_type": metadata.get(
            "model_type"
        ),
        "target": metadata.get(
            "target"
        ),
        "observation_date": str(
            selected_date.date()
        ),
        "wards_explained": int(
            len(current)
        ),
        "explanation_method": (
            "SHAP TreeExplainer"
        ),
        "model_probability_explained": True,
        "features": MODEL_FEATURES,
    }

    metadata_path = (
        OUTPUT_DIR
        / "heat_risk_xai_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            xai_metadata,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("XAI COMPLETE")
    print("=" * 60)

    print(
        f"Wards explained: "
        f"{len(current)}"
    )

    print()
    print("Top global features:")

    print(
        global_importance[
            [
                "feature",
                "importance_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Files saved:")

    print(
        global_path
    )

    print(
        local_path
    )

    print(
        text_path
    )

    print(
        metadata_path
    )

    print()
    print(
        "XAI probability interpretation:"
    )

    print(
        "Positive SHAP values increase the "
        "model's predicted heat-risk probability."
    )

    print(
        "Negative SHAP values decrease the "
        "model's predicted heat-risk probability."
    )

    return (
        global_importance,
        local_explanations,
        text_explanations,
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Explain Bengaluru heat-risk "
            "XGBoost predictions using SHAP."
        )
    )

    parser.parse_args()

    run_xai()