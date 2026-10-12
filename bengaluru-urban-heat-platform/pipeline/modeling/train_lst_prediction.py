from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Keep the final 15 observations completely untouched.
FINAL_TEST_DATES = 15

# Walk-forward validation settings.
INITIAL_TRAIN_DATES = 30
VALIDATION_BLOCK_SIZE = 3


# ============================================================
# FEATURES
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
    "years_since_start",
]

TARGET_COLUMN = "next_observation_lst_celsius"


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

# This is the best configuration from our previous
# validation search, so we keep it fixed during
# walk-forward validation.
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
}


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
        f"Columns: {len(df.columns)}"
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

    first_year = df["date"].dt.year.min()

    df["years_since_start"] = (
        df["date"].dt.year
        - first_year
    )

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "MAE_C": mae,
        "RMSE_C": rmse,
        "R2": r2,
    }


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

def persistence_predictions(df):

    return df[
        "mean_lst_celsius"
    ].values


# ============================================================
# SEASONAL BASELINE
# ============================================================

def seasonal_predictions(
    train_df,
    validation_df,
):

    # Calculate the average LST for each ward/month
    # using ONLY the training observations.
    #
    # This prevents future information from entering
    # the baseline.

    lookup = (
        train_df
        .groupby(
            [
                "ward_key",
                train_df["date"].dt.month,
            ]
        )["mean_lst_celsius"]
        .mean()
    )

    ward_mean = (
        train_df
        .groupby(
            "ward_key"
        )["mean_lst_celsius"]
        .mean()
    )

    global_mean = (
        train_df[
            "mean_lst_celsius"
        ].mean()
    )

    predictions = []

    for _, row in validation_df.iterrows():

        key = (
            row["ward_key"],
            row["date"].month,
        )

        if key in lookup.index:

            prediction = lookup.loc[key]

        elif row["ward_key"] in ward_mean.index:

            prediction = ward_mean.loc[
                row["ward_key"]
            ]

        else:

            prediction = global_mean

        predictions.append(
            prediction
        )

    return np.array(
        predictions
    )


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_xgboost(
    train_df,
):

    X_train = train_df[
        MODEL_FEATURES
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    model = xgb.XGBRegressor(
        n_estimators=XGBOOST_PARAMS[
            "n_estimators"
        ],
        learning_rate=XGBOOST_PARAMS[
            "learning_rate"
        ],
        max_depth=XGBOOST_PARAMS[
            "max_depth"
        ],
        min_child_weight=XGBOOST_PARAMS[
            "min_child_weight"
        ],
        subsample=XGBOOST_PARAMS[
            "subsample"
        ],
        colsample_bytree=XGBOOST_PARAMS[
            "colsample_bytree"
        ],
        reg_alpha=XGBOOST_PARAMS[
            "reg_alpha"
        ],
        reg_lambda=XGBOOST_PARAMS[
            "reg_lambda"
        ],
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


# ============================================================
# CREATE WALK-FORWARD FOLDS
# ============================================================

def create_walk_forward_folds(
    dates,
):

    folds = []

    validation_start = INITIAL_TRAIN_DATES

    fold_number = 1

    while (
        validation_start
        < len(dates)
    ):

        validation_end = min(
            validation_start
            + VALIDATION_BLOCK_SIZE,
            len(dates),
        )

        train_dates = dates[
            :validation_start
        ]

        validation_dates = dates[
            validation_start:validation_end
        ]

        if len(validation_dates) == 0:
            break

        folds.append(
            {
                "fold": fold_number,
                "train_dates": train_dates,
                "validation_dates": validation_dates,
            }
        )

        fold_number += 1

        validation_start = validation_end

    return folds


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def run_walk_forward_validation(
    pre_test_df,
    dates,
):

    print("\n" + "=" * 70)
    print("EXPANDING WALK-FORWARD VALIDATION")
    print("=" * 70)

    folds = create_walk_forward_folds(
        dates
    )

    print(
        f"Initial training dates: "
        f"{INITIAL_TRAIN_DATES}"
    )

    print(
        f"Validation block size: "
        f"{VALIDATION_BLOCK_SIZE}"
    )

    print(
        f"Total folds: "
        f"{len(folds)}"
    )

    fold_results = []

    for fold_info in folds:

        fold_number = fold_info[
            "fold"
        ]

        train_dates = fold_info[
            "train_dates"
        ]

        validation_dates = fold_info[
            "validation_dates"
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

        print("\n" + "-" * 70)

        print(
            f"FOLD {fold_number}"
        )

        print(
            f"Train: "
            f"{pd.Timestamp(train_dates[0]).date()} → "
            f"{pd.Timestamp(train_dates[-1]).date()}"
        )

        print(
            f"Validate: "
            f"{pd.Timestamp(validation_dates[0]).date()} → "
            f"{pd.Timestamp(validation_dates[-1]).date()}"
        )

        print(
            f"Train rows: "
            f"{len(train_df):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation_df):,}"
        )

        # ----------------------------------------------------
        # Actual values
        # ----------------------------------------------------

        y_true = validation_df[
            TARGET_COLUMN
        ].values

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        persistence_pred = (
            persistence_predictions(
                validation_df
            )
        )

        persistence_metrics = (
            calculate_metrics(
                y_true,
                persistence_pred,
            )
        )

        # ----------------------------------------------------
        # Seasonal baseline
        # ----------------------------------------------------

        seasonal_pred = (
            seasonal_predictions(
                train_df,
                validation_df,
            )
        )

        seasonal_metrics = (
            calculate_metrics(
                y_true,
                seasonal_pred,
            )
        )

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        model = train_xgboost(
            train_df
        )

        xgb_pred = model.predict(
            validation_df[
                MODEL_FEATURES
            ]
        )

        xgb_metrics = (
            calculate_metrics(
                y_true,
                xgb_pred,
            )
        )

        print(
            f"\nPersistence:"
            f" MAE={persistence_metrics['MAE_C']:.3f}"
            f" RMSE={persistence_metrics['RMSE_C']:.3f}"
            f" R²={persistence_metrics['R2']:.3f}"
        )

        print(
            f"Seasonal:"
            f" MAE={seasonal_metrics['MAE_C']:.3f}"
            f" RMSE={seasonal_metrics['RMSE_C']:.3f}"
            f" R²={seasonal_metrics['R2']:.3f}"
        )

        print(
            f"XGBoost:"
            f" MAE={xgb_metrics['MAE_C']:.3f}"
            f" RMSE={xgb_metrics['RMSE_C']:.3f}"
            f" R²={xgb_metrics['R2']:.3f}"
        )

        fold_results.append(
            {
                "fold": fold_number,
                "train_start": pd.Timestamp(
                    train_dates[0]
                ).date(),

                "train_end": pd.Timestamp(
                    train_dates[-1]
                ).date(),

                "validation_start": pd.Timestamp(
                    validation_dates[0]
                ).date(),

                "validation_end": pd.Timestamp(
                    validation_dates[-1]
                ).date(),

                "persistence_MAE_C":
                    persistence_metrics[
                        "MAE_C"
                    ],

                "persistence_RMSE_C":
                    persistence_metrics[
                        "RMSE_C"
                    ],

                "persistence_R2":
                    persistence_metrics[
                        "R2"
                    ],

                "seasonal_MAE_C":
                    seasonal_metrics[
                        "MAE_C"
                    ],

                "seasonal_RMSE_C":
                    seasonal_metrics[
                        "RMSE_C"
                    ],

                "seasonal_R2":
                    seasonal_metrics[
                        "R2"
                    ],

                "xgboost_MAE_C":
                    xgb_metrics[
                        "MAE_C"
                    ],

                "xgboost_RMSE_C":
                    xgb_metrics[
                        "RMSE_C"
                    ],

                "xgboost_R2":
                    xgb_metrics[
                        "R2"
                    ],
            }
        )

    return pd.DataFrame(
        fold_results
    )


# ============================================================
# FINAL TEST
# ============================================================

def evaluate_final_test(
    pre_test_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    print(
        "The final test period has not been used "
        "during walk-forward validation."
    )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    y_test = test_df[
        TARGET_COLUMN
    ].values

    persistence_pred = (
        persistence_predictions(
            test_df
        )
    )

    persistence_metrics = (
        calculate_metrics(
            y_test,
            persistence_pred,
        )
    )

    # --------------------------------------------------------
    # Seasonal baseline
    # --------------------------------------------------------

    seasonal_pred = (
        seasonal_predictions(
            pre_test_df,
            test_df,
        )
    )

    seasonal_metrics = (
        calculate_metrics(
            y_test,
            seasonal_pred,
        )
    )

    # --------------------------------------------------------
    # Final XGBoost
    # --------------------------------------------------------

    print(
        "\nTraining final XGBoost "
        "on all pre-test observations..."
    )

    final_model = train_xgboost(
        pre_test_df
    )

    xgb_pred = final_model.predict(
        test_df[
            MODEL_FEATURES
        ]
    )

    xgb_metrics = (
        calculate_metrics(
            y_test,
            xgb_pred,
        )
    )

    print(
        "\nPersistence:"
    )

    print(
        f"  MAE  : "
        f"{persistence_metrics['MAE_C']:.4f} °C"
    )

    print(
        f"  RMSE : "
        f"{persistence_metrics['RMSE_C']:.4f} °C"
    )

    print(
        f"  R²   : "
        f"{persistence_metrics['R2']:.4f}"
    )

    print(
        "\nSeasonal:"
    )

    print(
        f"  MAE  : "
        f"{seasonal_metrics['MAE_C']:.4f} °C"
    )

    print(
        f"  RMSE : "
        f"{seasonal_metrics['RMSE_C']:.4f} °C"
    )

    print(
        f"  R²   : "
        f"{seasonal_metrics['R2']:.4f}"
    )

    print(
        "\nXGBoost:"
    )

    print(
        f"  MAE  : "
        f"{xgb_metrics['MAE_C']:.4f} °C"
    )

    print(
        f"  RMSE : "
        f"{xgb_metrics['RMSE_C']:.4f} °C"
    )

    print(
        f"  R²   : "
        f"{xgb_metrics['R2']:.4f}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions = test_df[
        [
            "date",
            "next_observation_date",
            "ward_key",
            "Corporation",
            "ward_id",
            "ward_name",
            "mean_lst_celsius",
            TARGET_COLUMN,
        ]
    ].copy()

    predictions[
        "persistence_prediction"
    ] = persistence_pred

    predictions[
        "seasonal_prediction"
    ] = seasonal_pred

    predictions[
        "xgboost_prediction"
    ] = xgb_pred

    predictions[
        "xgboost_error_celsius"
    ] = (
        xgb_pred
        - y_test
    )

    predictions_path = (
        OUTPUT_DIR
        / "walk_forward_final_test_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    final_metrics = pd.DataFrame(
        [
            {
                "model":
                    "Persistence_Baseline",
                "MAE_C":
                    persistence_metrics[
                        "MAE_C"
                    ],
                "RMSE_C":
                    persistence_metrics[
                        "RMSE_C"
                    ],
                "R2":
                    persistence_metrics[
                        "R2"
                    ],
            },
            {
                "model":
                    "Seasonal_Baseline",
                "MAE_C":
                    seasonal_metrics[
                        "MAE_C"
                    ],
                "RMSE_C":
                    seasonal_metrics[
                        "RMSE_C"
                    ],
                "R2":
                    seasonal_metrics[
                        "R2"
                    ],
            },
            {
                "model":
                    "Final_XGBoost",
                "MAE_C":
                    xgb_metrics[
                        "MAE_C"
                    ],
                "RMSE_C":
                    xgb_metrics[
                        "RMSE_C"
                    ],
                "R2":
                    xgb_metrics[
                        "R2"
                    ],
            },
        ]
    )

    metrics_path = (
        OUTPUT_DIR
        / "walk_forward_final_test_metrics.csv"
    )

    final_metrics.to_csv(
        metrics_path,
        index=False,
    )

    return (
        final_metrics,
        final_model,
        predictions_path,
        metrics_path,
    )


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

    # --------------------------------------------------------
    # 3. QUALITY CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    missing = df[
        MODEL_FEATURES
        + [TARGET_COLUMN]
    ].isna().sum()

    if missing.sum() > 0:

        print(
            "Missing values:"
        )

        print(
            missing[
                missing > 0
            ]
        )

        raise ValueError(
            "Missing values detected."
        )

    print(
        "Missing values: 0"
    )

    duplicates = df.duplicated(
        subset=[
            "ward_key",
            "date",
        ]
    ).sum()

    print(
        f"Duplicate ward/date rows: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate ward/date rows detected."
        )

    # --------------------------------------------------------
    # 4. SORT DATES
    # --------------------------------------------------------

    dates = sorted(
        df["date"].unique()
    )

    # --------------------------------------------------------
    # 5. FINAL TEST SPLIT
    # --------------------------------------------------------

    pre_test_dates = dates[
        :-FINAL_TEST_DATES
    ]

    final_test_dates = dates[
        -FINAL_TEST_DATES:
    ]

    pre_test_df = df[
        df["date"].isin(
            pre_test_dates
        )
    ].copy()

    test_df = df[
        df["date"].isin(
            final_test_dates
        )
    ].copy()

    print("\n" + "=" * 70)
    print("FINAL HOLDOUT SPLIT")
    print("=" * 70)

    print(
        f"Pre-test observations: "
        f"{len(pre_test_dates)}"
    )

    print(
        f"Pre-test range: "
        f"{pd.Timestamp(pre_test_dates[0]).date()} → "
        f"{pd.Timestamp(pre_test_dates[-1]).date()}"
    )

    print(
        f"\nFinal test observations: "
        f"{len(final_test_dates)}"
    )

    print(
        f"Final test range: "
        f"{pd.Timestamp(final_test_dates[0]).date()} → "
        f"{pd.Timestamp(final_test_dates[-1]).date()}"
    )

    print(
        "\nFINAL TEST IS COMPLETELY HELD OUT."
    )

    # --------------------------------------------------------
    # 6. WALK-FORWARD VALIDATION
    # --------------------------------------------------------

    walk_forward_results = (
        run_walk_forward_validation(
            pre_test_df,
            pre_test_dates,
        )
    )

    # --------------------------------------------------------
    # 7. SAVE FOLD RESULTS
    # --------------------------------------------------------

    fold_results_path = (
        OUTPUT_DIR
        / "walk_forward_fold_results.csv"
    )

    walk_forward_results.to_csv(
        fold_results_path,
        index=False,
    )

    # --------------------------------------------------------
    # 8. WALK-FORWARD SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 70)

    summary_rows = []

    for model_name in [
        "persistence",
        "seasonal",
        "xgboost",
    ]:

        mae_column = (
            f"{model_name}_MAE_C"
        )

        rmse_column = (
            f"{model_name}_RMSE_C"
        )

        r2_column = (
            f"{model_name}_R2"
        )

        summary_rows.append(
            {
                "model":
                    model_name,
                "mean_MAE_C":
                    walk_forward_results[
                        mae_column
                    ].mean(),
                "std_MAE_C":
                    walk_forward_results[
                        mae_column
                    ].std(),
                "mean_RMSE_C":
                    walk_forward_results[
                        rmse_column
                    ].mean(),
                "std_RMSE_C":
                    walk_forward_results[
                        rmse_column
                    ].std(),
                "mean_R2":
                    walk_forward_results[
                        r2_column
                    ].mean(),
                "std_R2":
                    walk_forward_results[
                        r2_column
                    ].std(),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "walk_forward_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # 9. FINAL TEST
    # --------------------------------------------------------

    (
        final_metrics,
        final_model,
        predictions_path,
        metrics_path,
    ) = evaluate_final_test(
        pre_test_df,
        test_df,
    )

    # --------------------------------------------------------
    # 10. FEATURE IMPORTANCE
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
        / "walk_forward_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 11. FILES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Walk-forward folds:\n"
        f"{fold_results_path}"
    )

    print(
        f"\nWalk-forward summary:\n"
        f"{summary_path}"
    )

    print(
        f"\nFinal test metrics:\n"
        f"{metrics_path}"
    )

    print(
        f"\nFinal test predictions:\n"
        f"{predictions_path}"
    )

    print(
        f"\nFeature importance:\n"
        f"{importance_path}"
    )

    print("\n" + "=" * 70)
    print("WALK-FORWARD EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()