from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "ward_temporal_ml_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FINAL_TEST_DATES = 15

HIGH_HEAT_PERCENTILE = 80


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "mean_lst_celsius",
    "previous_lst_celsius",
    "mean_ndvi",
    "mean_ndbi",
    "mean_ndwi",
    "heat_pocket_coverage_pct",
    "previous_heat_pocket_coverage_pct",
    "historical_heat_pocket_coverage_pct",
    "days_since_previous_observation",
]


TARGET_SOURCE = "next_heat_pocket_coverage_pct"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING ML DATASET")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["next_observation_date"] = pd.to_datetime(
        df["next_observation_date"]
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Wards: {df['ward_key'].nunique()}"
    )

    print(
        f"Observations: {df['date'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # CREATE NEXT-OBSERVATION HEAT-POCKET COVERAGE
    # --------------------------------------------------------
    #
    # For every ward at observation t, find the heat-pocket
    # coverage for the same ward at observation t+1.
    #
    # IMPORTANT:
    # The ML dataset ends at 2026-05-03, while the feature
    # engineering step may still contain 2026-05-03 rows whose
    # next_observation_date points to 2026-06-04. Because
    # 2026-06-04 is not itself present as a row in this ML
    # dataset, an exact self-merge cannot recover that target.
    #
    # Therefore, rows whose future observation is unavailable
    # inside this dataset are removed. This is expected for the
    # final observation and is done BEFORE the final train/test
    # dates are selected.
    # --------------------------------------------------------

    future_lookup = (
        df[
            [
                "ward_key",
                "date",
                "heat_pocket_coverage_pct",
            ]
        ]
        .rename(
            columns={
                "date":
                    "next_observation_date",
                "heat_pocket_coverage_pct":
                    TARGET_SOURCE,
            }
        )
    )

    df = df.merge(
        future_lookup,
        on=[
            "ward_key",
            "next_observation_date",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # CHECK TARGET
    # --------------------------------------------------------

    missing_mask = df[TARGET_SOURCE].isna()
    missing_target = int(missing_mask.sum())

    print(
        f"Next-observation target missing values: "
        f"{missing_target}"
    )

    if missing_target > 0:

        missing_rows = df.loc[
            missing_mask,
            [
                "ward_key",
                "date",
                "next_observation_date",
            ],
        ].copy()

        # Determine whether the exact future WARD + DATE pair
        # exists in the source dataset.
        #
        # Checking only the date is not sufficient: a future date
        # can exist while one specific ward is absent on that date.
        available_future_pairs = set(
            zip(
                future_lookup["ward_key"],
                future_lookup["next_observation_date"],
            )
        )

        missing_rows["_future_pair_exists"] = [
            (
                ward_key,
                next_date,
            ) in available_future_pairs
            for ward_key, next_date
            in zip(
                missing_rows["ward_key"],
                missing_rows["next_observation_date"],
            )
        ]

        unavailable_future_mask = (
            ~missing_rows["_future_pair_exists"]
        )

        unavailable_future_rows = int(
            unavailable_future_mask.sum()
        )

        print(
            f"Rows whose exact future ward/date pair is "
            f"unavailable in this dataset: "
            f"{unavailable_future_rows}"
        )

        # A missing target is valid to exclude when the exact
        # future ward/date pair is unavailable. This includes the
        # final observation and any isolated ward/date gap.
        #
        # If the exact future pair DOES exist but its target is
        # missing, that is a genuine data-integrity problem.
        if unavailable_future_rows != missing_target:

            unexpected_rows = missing_rows[
                ~unavailable_future_mask
            ].copy()

            print(
                "\nUNEXPECTED MISSING TARGET ROW(S):"
            )

            print(
                unexpected_rows[
                    [
                        "ward_key",
                        "date",
                        "next_observation_date",
                    ]
                ].to_string(
                    index=False
                )
            )

            raise ValueError(
                "A future ward/date pair exists but its "
                "heat-pocket coverage target is missing."
            )

        missing_dates = sorted(
            missing_rows.loc[
                unavailable_future_mask,
                "next_observation_date",
            ]
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .unique()
        )

        if missing_dates:
            print(
                "Unavailable future observation date(s): "
                + ", ".join(missing_dates)
            )

        isolated_gaps = missing_rows.loc[
            unavailable_future_mask
            & missing_rows["next_observation_date"].notna(),
            [
                "ward_key",
                "date",
                "next_observation_date",
            ],
        ]

        if len(isolated_gaps) > 0:
            print(
                "\nUnavailable future ward/date pair(s):"
            )

            print(
                isolated_gaps.to_string(
                    index=False
                )
            )

        # Remove only rows for which the exact next target is
        # genuinely unavailable from this ML dataset.
        df = df.loc[
            ~missing_mask
        ].copy()

    print(
        f"Rows remaining after removing rows without "
        f"a future target: {len(df):,}"
    )

    print(
        f"Usable observations after target construction: "
        f"{df['date'].nunique()}"
    )

    print(
        "Next-observation heat-pocket target "
        "created successfully."
    )

    return df

# ============================================================
# TEMPORAL FEATURES
# ============================================================

def add_temporal_features(df):

    df = df.copy()

    month = df["date"].dt.month

    df["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    return df


# ============================================================
# CREATE FUTURE TARGET
# ============================================================

def create_target(
    df,
    threshold,
):

    df = df.copy()

    df["high_heat_next"] = (
        df[TARGET_SOURCE]
        >= threshold
    ).astype(int)

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_classification_metrics(
    y_true,
    predictions,
    probabilities,
):

    metrics = {
        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),
    }

    # ROC-AUC requires both classes.
    if len(np.unique(y_true)) == 2:

        metrics["roc_auc"] = (
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        metrics["pr_auc"] = (
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    else:

        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


# ============================================================
# OPTIMIZE DECISION THRESHOLD
# ============================================================

def optimize_decision_threshold(
    y_true,
    probabilities,
):
    """
    Choose the probability threshold that maximizes F1
    on the validation set.

    The final test set is NOT used here.
    """

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        metrics = calculate_classification_metrics(
            y_true,
            predictions,
            probabilities,
        )

        rows.append(
            {
                "decision_threshold":
                    float(threshold),
                **metrics,
            }
        )

    results = pd.DataFrame(
        rows
    )

    # F1 is the primary optimization criterion.
    # Recall is retained so we can inspect the
    # trade-off between catching high-heat wards
    # and generating false alarms.
    best_row = results.sort_values(
        [
            "f1",
            "recall",
            "precision",
        ],
        ascending=False,
    ).iloc[0]

    best_threshold = float(
        best_row[
            "decision_threshold"
        ]
    )

    return (
        best_threshold,
        results,
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    y_train,
):

    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",
        device="cuda",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. TEMPORAL FEATURES
    # --------------------------------------------------------

    df = add_temporal_features(
        df
    )

    MODEL_FEATURES = FEATURES + [
        "month_sin",
        "month_cos",
    ]

    # --------------------------------------------------------
    # 3. DATA QUALITY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    missing = df[
        MODEL_FEATURES
    ].isna().sum()

    if missing.sum() > 0:

        print(
            missing[
                missing > 0
            ]
        )

        raise ValueError(
            "Missing feature values detected."
        )

    print(
        "Feature missing values: 0"
    )

    # --------------------------------------------------------
    # 4. SORT OBSERVATIONS
    # --------------------------------------------------------

    dates = sorted(
        df["date"].unique()
    )

    # --------------------------------------------------------
    # 5. FINAL TEST HOLDOUT
    # --------------------------------------------------------

    test_dates = dates[
        -FINAL_TEST_DATES:
    ]

    pre_test_dates = dates[
        :-FINAL_TEST_DATES
    ]

    pre_test_df = df[
        df["date"].isin(
            pre_test_dates
        )
    ].copy()

    test_df = df[
        df["date"].isin(
            test_dates
        )
    ].copy()

    print("\n" + "=" * 70)
    print("FINAL HOLDOUT")
    print("=" * 70)

    print(
        f"Training/development observations: "
        f"{len(pre_test_dates)}"
    )

    print(
        f"Development range: "
        f"{pd.Timestamp(pre_test_dates[0]).date()} → "
        f"{pd.Timestamp(pre_test_dates[-1]).date()}"
    )

    print(
        f"\nFinal test observations: "
        f"{len(test_dates)}"
    )

    print(
        f"Final test range: "
        f"{pd.Timestamp(test_dates[0]).date()} → "
        f"{pd.Timestamp(test_dates[-1]).date()}"
    )

    # --------------------------------------------------------
    # 6. DEFINE TARGET THRESHOLD FROM DEVELOPMENT DATA ONLY
    # --------------------------------------------------------
    #
    # The 80th-percentile target threshold is calculated from
    # pre-test data only. The final test period is never used
    # to define what "high heat" means.
    # --------------------------------------------------------

    target_threshold = np.percentile(
        pre_test_df[
            TARGET_SOURCE
        ],
        HIGH_HEAT_PERCENTILE,
    )

    print("\n" + "=" * 70)
    print("HIGH-HEAT TARGET DEFINITION")
    print("=" * 70)

    print(
        f"Target variable: "
        f"{TARGET_SOURCE}"
    )

    print(
        f"High-heat percentile: "
        f"{HIGH_HEAT_PERCENTILE}th"
    )

    print(
        f"High-heat threshold: "
        f"{target_threshold:.4f}% heat-pocket coverage"
    )

    # Create the SAME target definition for validation
    # and final test. This keeps the classification problem
    # identical across evaluation periods.
    pre_test_df = create_target(
        pre_test_df,
        target_threshold,
    )

    test_df = create_target(
        test_df,
        target_threshold,
    )

    # --------------------------------------------------------
    # 7. DEVELOPMENT SPLIT
    # --------------------------------------------------------

    development_dates = sorted(
        pre_test_df["date"].unique()
    )

    validation_start = int(
        len(development_dates)
        * 0.75
    )

    train_dates = development_dates[
        :validation_start
    ]

    validation_dates = development_dates[
        validation_start:
    ]

    train_df = pre_test_df[
        pre_test_df["date"].isin(
            train_dates
        )
    ].copy()

    validation_df = pre_test_df[
        pre_test_df["date"].isin(
            validation_dates
        )
    ].copy()

    print("\n" + "=" * 70)
    print("DEVELOPMENT TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    print(
        f"TRAIN: "
        f"{pd.Timestamp(train_dates[0]).date()} → "
        f"{pd.Timestamp(train_dates[-1]).date()}"
    )

    print(
        f"VALIDATION: "
        f"{pd.Timestamp(validation_dates[0]).date()} → "
        f"{pd.Timestamp(validation_dates[-1]).date()}"
    )

    print(
        "\nTraining high-heat rate: "
        f"{train_df['high_heat_next'].mean() * 100:.2f}%"
    )

    print(
        "Validation high-heat rate: "
        f"{validation_df['high_heat_next'].mean() * 100:.2f}%"
    )

    print(
        "Development high-heat rate: "
        f"{pre_test_df['high_heat_next'].mean() * 100:.2f}%"
    )

    print(
        "Final test high-heat rate: "
        f"{test_df['high_heat_next'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 8. TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST CLASSIFIER")
    print("=" * 70)

    X_train = train_df[
        MODEL_FEATURES
    ]

    y_train = train_df[
        "high_heat_next"
    ]

    X_validation = validation_df[
        MODEL_FEATURES
    ]

    y_validation = validation_df[
        "high_heat_next"
    ]

    model = train_classifier(
        X_train,
        y_train,
    )

    print(
        "GPU XGBoost classification "
        "training completed."
    )

    # --------------------------------------------------------
    # 9. VALIDATION PROBABILITIES
    # --------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # Keep the conventional 0.50 threshold for reference.
    validation_predictions_05 = (
        validation_probabilities
        >= 0.5
    ).astype(int)

    validation_metrics_05 = (
        calculate_classification_metrics(
            y_validation,
            validation_predictions_05,
            validation_probabilities,
        )
    )

    # --------------------------------------------------------
    # 10. OPTIMIZE DECISION THRESHOLD
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 70)

    best_decision_threshold, threshold_results = (
        optimize_decision_threshold(
            y_validation,
            validation_probabilities,
        )
    )

    print(
        f"Default decision threshold: 0.50"
    )

    print(
        f"Optimized decision threshold: "
        f"{best_decision_threshold:.2f}"
    )

    print(
        "\nOptimization criterion: "
        "maximum validation F1"
    )

    optimized_validation_predictions = (
        validation_probabilities
        >= best_decision_threshold
    ).astype(int)

    validation_metrics = (
        calculate_classification_metrics(
            y_validation,
            optimized_validation_predictions,
            validation_probabilities,
        )
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS — OPTIMIZED THRESHOLD")
    print("=" * 70)

    for key, value in validation_metrics.items():

        if np.isnan(value):

            print(
                f"{key}: N/A"
            )

        else:

            print(
                f"{key}: "
                f"{value:.4f}"
            )

    threshold_results_path = (
        OUTPUT_DIR
        / "heat_risk_threshold_optimization.csv"
    )

    threshold_results.to_csv(
        threshold_results_path,
        index=False,
    )

    # --------------------------------------------------------
    # 11. TRAIN FINAL MODEL ON ALL PRE-TEST DATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    X_development = pre_test_df[
        MODEL_FEATURES
    ]

    y_development = pre_test_df[
        "high_heat_next"
    ]

    final_model = train_classifier(
        X_development,
        y_development,
    )

    print(
        "Final model training completed."
    )

    # --------------------------------------------------------
    # 12. FINAL TEST
    # --------------------------------------------------------

    X_test = test_df[
        MODEL_FEATURES
    ]

    y_test = test_df[
        "high_heat_next"
    ]

    test_probabilities = (
        final_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= best_decision_threshold
    ).astype(int)

    test_metrics = (
        calculate_classification_metrics(
            y_test,
            test_predictions,
            test_probabilities,
        )
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS — OPTIMIZED THRESHOLD")
    print("=" * 70)

    for key, value in test_metrics.items():

        if np.isnan(value):

            print(
                f"{key}: N/A"
            )

        else:

            print(
                f"{key}: "
                f"{value:.4f}"
            )

    # --------------------------------------------------------
    # 13. PERSISTENCE BASELINE
    # --------------------------------------------------------

    baseline_predictions = (
        test_df[
            "heat_pocket_coverage_pct"
        ].values
        >= target_threshold
    ).astype(int)

    baseline_metrics = (
        calculate_classification_metrics(
            y_test,
            baseline_predictions,
            baseline_predictions.astype(float),
        )
    )

    print("\n" + "=" * 70)
    print("PERSISTENCE BASELINE")
    print("=" * 70)

    for key, value in baseline_metrics.items():

        if np.isnan(value):

            print(
                f"{key}: N/A"
            )

        else:

            print(
                f"{key}: "
                f"{value:.4f}"
            )

    # --------------------------------------------------------
    # 14. COMPARISON
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model":
                    "Persistence_Baseline",
                **baseline_metrics,
            },
            {
                "model":
                    "XGBoost_Classifier",
                **test_metrics,
            },
        ]
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
        )
    )

    # --------------------------------------------------------
    # 15. CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    print("\n" + "=" * 70)
    print("XGBOOST CONFUSION MATRIX")
    print("=" * 70)

    print(
        "Rows = actual"
    )

    print(
        "Columns = predicted"
    )

    print(
        cm
    )

    # --------------------------------------------------------
    # 16. SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    predictions = test_df[
        [
            "date",
            "next_observation_date",
            "ward_key",
            "Corporation",
            "ward_id",
            "ward_name",
            "heat_pocket_coverage_pct",
            TARGET_SOURCE,
            "high_heat_next",
        ]
    ].copy()

    predictions[
        "predicted_high_heat"
    ] = test_predictions

    predictions[
        "high_heat_probability"
    ] = test_probabilities

    predictions[
        "prediction_correct"
    ] = (
        test_predictions
        == y_test.values
    )

    predictions_path = (
        OUTPUT_DIR
        / "heat_risk_classifier_test_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # 17. SAVE METRICS
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "heat_risk_classifier_metrics.csv"
    )

    comparison.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # 18. SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature":
                MODEL_FEATURES,

            "importance":
                final_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_path = (
        OUTPUT_DIR
        / "heat_risk_classifier_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    # --------------------------------------------------------
    # 19. SAVE TARGET INFORMATION
    # --------------------------------------------------------

    target_info = pd.DataFrame(
        [
            {
                "high_heat_percentile":
                    HIGH_HEAT_PERCENTILE,

                "threshold_heat_pocket_coverage_pct":
                    target_threshold,

                "optimized_decision_threshold":
                    best_decision_threshold,

                "optimization_metric":
                    "validation_f1",

                "training_high_heat_rate":
                    train_df[
                        "high_heat_next"
                    ].mean(),

                "validation_high_heat_rate":
                    validation_df[
                        "high_heat_next"
                    ].mean(),

                "development_high_heat_rate":
                    pre_test_df[
                        "high_heat_next"
                    ].mean(),

                "test_high_heat_rate":
                    test_df[
                        "high_heat_next"
                    ].mean(),
            }
        ]
    )

    target_path = (
        OUTPUT_DIR
        / "heat_risk_target_definition.csv"
    )

    target_info.to_csv(
        target_path,
        index=False,
    )

    # --------------------------------------------------------
    # 20. SAVE FINAL MODEL
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "heat_risk_xgboost.json"
    )

    final_model.save_model(
        model_path
    )

    # --------------------------------------------------------
    # 21. SAVE MODEL METADATA
    # --------------------------------------------------------

    metadata = {
        "model_type":
            "XGBoost binary classifier",

        "target":
            "high_heat_next",

        "target_source":
            TARGET_SOURCE,

        "high_heat_percentile":
            HIGH_HEAT_PERCENTILE,

        "target_threshold_heat_pocket_coverage_pct":
            float(target_threshold),

        "decision_threshold":
            float(best_decision_threshold),

        "decision_threshold_optimization":
            "Validation-set F1 maximization",

        "features":
            MODEL_FEATURES,

        "training_start":
            str(
                pd.Timestamp(
                    train_dates[0]
                ).date()
            ),

        "training_end":
            str(
                pd.Timestamp(
                    train_dates[-1]
                ).date()
            ),

        "development_start":
            str(
                pd.Timestamp(
                    pre_test_dates[0]
                ).date()
            ),

        "development_end":
            str(
                pd.Timestamp(
                    pre_test_dates[-1]
                ).date()
            ),

        "final_test_start":
            str(
                pd.Timestamp(
                    test_dates[0]
                ).date()
            ),

        "final_test_end":
            str(
                pd.Timestamp(
                    test_dates[-1]
                ).date()
            ),

        "xgboost_parameters": {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda",
            "random_state": RANDOM_STATE,
        },

        "final_test_metrics": {
            key:
                None if np.isnan(value)
                else float(value)
            for key, value
            in test_metrics.items()
        },
    }

    metadata_path = (
        MODEL_DIR
        / "heat_risk_xgboost_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 22. FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Test predictions:\n"
        f"{predictions_path}"
    )

    print(
        f"\nMetrics:\n"
        f"{metrics_path}"
    )

    print(
        f"\nFeature importance:\n"
        f"{importance_path}"
    )

    print(
        f"\nTarget definition:\n"
        f"{target_path}"
    )

    print(
        f"\nThreshold optimization:\n"
        f"{threshold_results_path}"
    )

    print(
        f"\nFINAL MODEL:\n"
        f"{model_path}"
    )

    print(
        f"\nMODEL METADATA:\n"
        f"{metadata_path}"
    )

    print("\n" + "=" * 70)
    print("HEAT-RISK CLASSIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()