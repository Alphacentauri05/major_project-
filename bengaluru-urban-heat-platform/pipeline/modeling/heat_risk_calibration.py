from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "predictions"
    / "heat_risk_classifier_test_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

N_BINS = 10


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    print("Loading final test predictions...")

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print()
    print("Columns:")

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# IDENTIFY TARGET / PROBABILITY COLUMNS
# ============================================================

def identify_columns(df):

    target_candidates = [
        "actual",
        "target",
        "y_true",
        "high_heat_next",
        "actual_high_heat",
    ]

    probability_candidates = [
    "high_heat_probability",
    "probability",
    "predicted_probability",
    "future_heat_probability",
    "predicted_prob",
    "heat_probability",
    "y_probability",
]

    target_column = None

    for column in target_candidates:

        if column in df.columns:

            target_column = column
            break

    probability_column = None

    for column in probability_candidates:

        if column in df.columns:

            probability_column = column
            break

    if target_column is None:

        raise ValueError(
            "Could not identify the actual target column."
        )

    if probability_column is None:

        raise ValueError(
            "Could not identify the predicted probability column."
        )

    print()
    print(
        f"Target column: {target_column}"
    )

    print(
        f"Probability column: {probability_column}"
    )

    return (
        target_column,
        probability_column,
    )


# ============================================================
# CREATE CALIBRATION BINS
# ============================================================

def create_calibration_bins(
    y_true,
    probabilities,
):

    bins = np.linspace(
        0,
        1,
        N_BINS + 1
    )

    bin_numbers = pd.cut(
        probabilities,
        bins=bins,
        labels=False,
        include_lowest=True
    )

    calibration_rows = []

    for bin_index in range(
        N_BINS
    ):

        mask = (
            bin_numbers
            == bin_index
        )

        count = int(
            mask.sum()
        )

        if count == 0:

            continue

        mean_probability = float(
            probabilities[mask].mean()
        )

        observed_frequency = float(
            y_true[mask].mean()
        )

        calibration_gap = (
            observed_frequency
            - mean_probability
        )

        calibration_rows.append(
            {
                "bin": bin_index + 1,

                "probability_range": (
                    f"{bins[bin_index]:.1f}"
                    f"-"
                    f"{bins[bin_index + 1]:.1f}"
                ),

                "sample_count": count,

                "mean_predicted_probability": (
                    mean_probability
                ),

                "observed_positive_frequency": (
                    observed_frequency
                ),

                "calibration_gap": (
                    calibration_gap
                ),

                "absolute_calibration_gap": (
                    abs(calibration_gap)
                ),
            }
        )

    return pd.DataFrame(
        calibration_rows
    )


# ============================================================
# CALIBRATION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    logloss = log_loss(
        y_true,
        probabilities
    )

    calibration_bins = (
        create_calibration_bins(
            y_true,
            probabilities
        )
    )

    # Weighted mean absolute calibration gap.
    total = calibration_bins[
        "sample_count"
    ].sum()

    if total > 0:

        weighted_gap = np.sum(
            calibration_bins[
                "absolute_calibration_gap"
            ]
            * calibration_bins[
                "sample_count"
            ]
        ) / total

    else:

        weighted_gap = np.nan

    metrics = pd.DataFrame(
        {
            "metric": [
                "Brier score",
                "Log loss",
                "Weighted mean absolute calibration gap",
            ],

            "value": [
                brier,
                logloss,
                weighted_gap,
            ],
        }
    )

    return (
        metrics,
        calibration_bins
    )


# ============================================================
# CALIBRATION PLOT
# ============================================================

def create_calibration_plot(
    y_true,
    probabilities,
):

    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=N_BINS,
            strategy="uniform"
        )
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="XGBoost"
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed frequency of high heat"
    )

    plt.title(
        "Heat-Risk Probability Calibration"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "heat_risk_calibration_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# INTERPRETATION
# ============================================================

def print_interpretation(
    metrics,
    calibration_bins
):

    brier = float(
        metrics.loc[
            metrics["metric"]
            == "Brier score",
            "value"
        ].iloc[0]
    )

    logloss = float(
        metrics.loc[
            metrics["metric"]
            == "Log loss",
            "value"
        ].iloc[0]
    )

    weighted_gap = float(
        metrics.loc[
            metrics["metric"]
            == "Weighted mean absolute calibration gap",
            "value"
        ].iloc[0]
    )

    print()
    print("=" * 60)
    print("CALIBRATION METRICS")
    print("=" * 60)

    print(
        f"Brier score: "
        f"{brier:.4f}"
    )

    print(
        f"Log loss: "
        f"{logloss:.4f}"
    )

    print(
        f"Weighted mean absolute "
        f"calibration gap: "
        f"{weighted_gap:.4f}"
    )

    print()
    print(
        "Calibration bins:"
    )

    print(
        calibration_bins.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "A well-calibrated model has predicted "
        "probabilities that broadly match observed "
        "event frequencies."
    )

    print(
        "For example, predictions around 0.70 should "
        "correspond to roughly 70% observed high-heat "
        "events when enough samples are available."
    )

    print(
        "Brier score and calibration gaps should be "
        "interpreted comparatively rather than using "
        "a universal pass/fail threshold."
    )


# ============================================================
# MAIN
# ============================================================

def run_calibration():

    print()
    print("=" * 60)
    print("HEAT-RISK PROBABILITY CALIBRATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_predictions()

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    (
        target_column,
        probability_column,
    ) = identify_columns(
        df
    )

    # --------------------------------------------------------
    # Prepare arrays
    # --------------------------------------------------------

    y_true = (
        pd.to_numeric(
            df[target_column],
            errors="coerce"
        )
        .values
    )

    probabilities = (
        pd.to_numeric(
            df[probability_column],
            errors="coerce"
        )
        .values
    )

    valid = (
        np.isfinite(y_true)
        &
        np.isfinite(probabilities)
    )

    y_true = y_true[
        valid
    ]

    probabilities = probabilities[
        valid
    ]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if len(y_true) == 0:

        raise ValueError(
            "No valid prediction rows found."
        )

    if not np.all(
        np.isin(
            y_true,
            [0, 1]
        )
    ):

        raise ValueError(
            "Target must contain binary 0/1 values."
        )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7
    )

    print()
    print(
        f"Valid test predictions: "
        f"{len(y_true):,}"
    )

    print(
        f"Positive events: "
        f"{int(y_true.sum()):,}"
    )

    print(
        f"Negative events: "
        f"{int((1 - y_true).sum()):,}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    (
        metrics,
        calibration_bins,
    ) = calculate_metrics(
        y_true,
        probabilities
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "heat_risk_calibration_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save bins
    # --------------------------------------------------------

    bins_path = (
        OUTPUT_DIR
        / "heat_risk_calibration_bins.csv"
    )

    calibration_bins.to_csv(
        bins_path,
        index=False
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_path = create_calibration_plot(
        y_true,
        probabilities
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_interpretation(
        metrics,
        calibration_bins
    )

    print()
    print("Files saved:")

    print(
        metrics_path
    )

    print(
        bins_path
    )

    print(
        plot_path
    )

    print()
    print(
        "Calibration analysis complete."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_calibration()