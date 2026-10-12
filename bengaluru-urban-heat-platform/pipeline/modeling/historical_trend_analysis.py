from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, t
from scipy.stats import false_discovery_control


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "persistence"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_OBSERVATIONS = 5
P_VALUE_THRESHOLD = 0.05


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("Loading temporal ML dataset...")

    df = pd.read_csv(
        DATASET_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["month"] = df["date"].dt.month

    first_date = df["date"].min()

    df["time_years"] = (
        (
            df["date"] - first_date
        ).dt.total_seconds()
        / (
            365.25 * 24 * 60 * 60
        )
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Observation dates: "
        f"{df['date'].nunique()}"
    )

    print(
        f"Wards: "
        f"{df['ward_key'].nunique()}"
    )

    return df


# ============================================================
# SEASONALLY ADJUSTED TREND
# ============================================================

def seasonal_adjusted_trend(
    time_years,
    values,
    months
):

    """
    Estimate:

        value = intercept
               + time trend
               + monthly seasonal effects

    January is the reference month.

    The coefficient of time represents the estimated
    long-term trend after accounting for month-of-year.
    """

    time_years = np.asarray(
        time_years,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    months = np.asarray(
        months,
        dtype=int
    )

    # --------------------------------------------------------
    # Remove invalid observations
    # --------------------------------------------------------

    valid = (
        np.isfinite(time_years)
        & np.isfinite(values)
        & np.isfinite(months)
    )

    time_years = time_years[valid]
    values = values[valid]
    months = months[valid]

    if len(values) < MIN_OBSERVATIONS:

        return {
            "slope": np.nan,
            "p_value": np.nan,
            "r2": np.nan,
        }

    # --------------------------------------------------------
    # Design matrix
    # --------------------------------------------------------

    columns = [
        np.ones(len(time_years)),
        time_years,
    ]

    for month in range(2, 13):

        columns.append(
            (months == month).astype(float)
        )

    X = np.column_stack(
        columns
    )

    # --------------------------------------------------------
    # Check matrix rank
    # --------------------------------------------------------

    rank = np.linalg.matrix_rank(
        X
    )

    if rank < X.shape[1]:

        return {
            "slope": np.nan,
            "p_value": np.nan,
            "r2": np.nan,
        }

    # --------------------------------------------------------
    # Least squares
    # --------------------------------------------------------

    coefficients, _, _, _ = np.linalg.lstsq(
        X,
        values,
        rcond=None
    )

    predictions = X @ coefficients

    residuals = (
        values - predictions
    )

    n = len(values)
    p = X.shape[1]

    degrees_of_freedom = n - p

    if degrees_of_freedom <= 1:

        return {
            "slope": np.nan,
            "p_value": np.nan,
            "r2": np.nan,
        }

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    residual_sum_squares = np.sum(
        residuals ** 2
    )

    total_sum_squares = np.sum(
        (
            values - values.mean()
        ) ** 2
    )

    if total_sum_squares > 0:

        r2 = (
            1
            - residual_sum_squares
            / total_sum_squares
        )

    else:

        r2 = np.nan

    # --------------------------------------------------------
    # Standard error of time coefficient
    # --------------------------------------------------------

    sigma_squared = (
        residual_sum_squares
        / degrees_of_freedom
    )

    try:

        XtX_inv = np.linalg.inv(
            X.T @ X
        )

        coefficient_variance = (
            sigma_squared
            * XtX_inv[1, 1]
        )

        if (
            coefficient_variance <= 0
            or not np.isfinite(
                coefficient_variance
            )
        ):

            return {
                "slope": float(
                    coefficients[1]
                ),
                "p_value": np.nan,
                "r2": float(r2),
            }

        standard_error = np.sqrt(
            coefficient_variance
        )

        if standard_error == 0:

            return {
                "slope": float(
                    coefficients[1]
                ),
                "p_value": np.nan,
                "r2": float(r2),
            }

        t_statistic = (
            coefficients[1]
            / standard_error
        )

        p_value = (
            2
            * t.sf(
                abs(t_statistic),
                degrees_of_freedom
            )
        )

    except (
        np.linalg.LinAlgError,
        ValueError,
        FloatingPointError,
    ):

        p_value = np.nan

    return {
        "slope": float(
            coefficients[1]
        ),
        "p_value": float(
            p_value
        ) if np.isfinite(p_value) else np.nan,
        "r2": float(
            r2
        ) if np.isfinite(r2) else np.nan,
    }


# ============================================================
# TREND CLASSIFICATION
# ============================================================

def classify_trend(
    slope,
    adjusted_p_value
):

    if (
        pd.notna(adjusted_p_value)
        and adjusted_p_value < P_VALUE_THRESHOLD
        and slope > 0
    ):

        return "Increasing"

    elif (
        pd.notna(adjusted_p_value)
        and adjusted_p_value < P_VALUE_THRESHOLD
        and slope < 0
    ):

        return "Decreasing"

    else:

        return "Stable/Uncertain"


# ============================================================
# CALCULATE WARD TREND
# ============================================================

def calculate_ward_trend(
    ward_df
):

    ward_df = ward_df.sort_values(
        "date"
    ).copy()

    if len(ward_df) < MIN_OBSERVATIONS:

        return None

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    first_date = ward_df[
        "date"
    ].min()

    last_date = ward_df[
        "date"
    ].max()

    first_lst = float(
        ward_df[
            "mean_lst_celsius"
        ].iloc[0]
    )

    last_lst = float(
        ward_df[
            "mean_lst_celsius"
        ].iloc[-1]
    )

    first_heat = float(
        ward_df[
            "heat_pocket_coverage_pct"
        ].iloc[0]
    )

    last_heat = float(
        ward_df[
            "heat_pocket_coverage_pct"
        ].iloc[-1]
    )

    # --------------------------------------------------------
    # Raw LST trend
    # --------------------------------------------------------

    raw_lst = linregress(
        ward_df["time_years"],
        ward_df["mean_lst_celsius"]
    )

    # --------------------------------------------------------
    # Adjusted LST trend
    # --------------------------------------------------------

    adjusted_lst = seasonal_adjusted_trend(
        ward_df["time_years"],
        ward_df["mean_lst_celsius"],
        ward_df["month"]
    )

    # --------------------------------------------------------
    # Raw heat-pocket trend
    # --------------------------------------------------------

    raw_heat = linregress(
        ward_df["time_years"],
        ward_df[
            "heat_pocket_coverage_pct"
        ]
    )

    # --------------------------------------------------------
    # Adjusted heat-pocket trend
    # --------------------------------------------------------

    adjusted_heat = seasonal_adjusted_trend(
        ward_df["time_years"],
        ward_df[
            "heat_pocket_coverage_pct"
        ],
        ward_df["month"]
    )

    return {

        "ward_key": ward_df[
            "ward_key"
        ].iloc[0],

        "Corporation": ward_df[
            "Corporation"
        ].iloc[0],

        "ward_id": ward_df[
            "ward_id"
        ].iloc[0],

        "ward_name": ward_df[
            "ward_name"
        ].iloc[0],

        "observation_count": len(
            ward_df
        ),

        "first_observation": (
            first_date.date()
        ),

        "last_observation": (
            last_date.date()
        ),

        # ----------------------------------------------------
        # Endpoint changes
        # ----------------------------------------------------

        "first_mean_lst_celsius": first_lst,

        "last_mean_lst_celsius": last_lst,

        "lst_endpoint_change_celsius": (
            last_lst - first_lst
        ),

        "first_heat_pocket_coverage_pct": (
            first_heat
        ),

        "last_heat_pocket_coverage_pct": (
            last_heat
        ),

        "heat_pocket_endpoint_change_percentage_points": (
            last_heat - first_heat
        ),

        # ----------------------------------------------------
        # Raw LST
        # ----------------------------------------------------

        "raw_lst_trend_celsius_per_year": (
            raw_lst.slope
        ),

        "raw_lst_trend_p_value": (
            raw_lst.pvalue
        ),

        "raw_lst_trend_r2": (
            raw_lst.rvalue ** 2
        ),

        # ----------------------------------------------------
        # Seasonally adjusted LST
        # ----------------------------------------------------

        "adjusted_lst_trend_celsius_per_year": (
            adjusted_lst["slope"]
        ),

        "adjusted_lst_trend_p_value": (
            adjusted_lst["p_value"]
        ),

        "adjusted_lst_trend_r2": (
            adjusted_lst["r2"]
        ),

        # ----------------------------------------------------
        # Raw heat-pocket trend
        # ----------------------------------------------------

        "raw_heat_pocket_trend_pct_points_per_year": (
            raw_heat.slope
        ),

        "raw_heat_pocket_trend_p_value": (
            raw_heat.pvalue
        ),

        "raw_heat_pocket_trend_r2": (
            raw_heat.rvalue ** 2
        ),

        # ----------------------------------------------------
        # Seasonally adjusted heat-pocket trend
        # ----------------------------------------------------

        "adjusted_heat_pocket_trend_pct_points_per_year": (
            adjusted_heat["slope"]
        ),

        "adjusted_heat_pocket_trend_p_value": (
            adjusted_heat["p_value"]
        ),

        "adjusted_heat_pocket_trend_r2": (
            adjusted_heat["r2"]
        ),
    }


# ============================================================
# BENJAMINI-HOCHBERG FDR CORRECTION
# ============================================================

def apply_fdr_correction(
    trend_df,
    p_value_column,
    output_column
):

    """
    Apply Benjamini-Hochberg false discovery rate
    correction across all wards with valid p-values.
    """

    valid_mask = trend_df[
        p_value_column
    ].notna()

    adjusted_values = np.full(
        len(trend_df),
        np.nan
    )

    if valid_mask.sum() == 0:

        trend_df[
            output_column
        ] = adjusted_values

        return trend_df

    raw_p_values = trend_df.loc[
        valid_mask,
        p_value_column
    ].values

    adjusted_p_values = (
        false_discovery_control(
            raw_p_values,
            method="bh"
        )
    )

    adjusted_values[
        np.where(valid_mask)[0]
    ] = adjusted_p_values

    trend_df[
        output_column
    ] = adjusted_values

    return trend_df


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_trend_analysis():

    print()
    print("=" * 60)
    print("FDR-CORRECTED SEASONALLY ADJUSTED TREND ANALYSIS")
    print("=" * 60)

    df = load_dataset()

    # --------------------------------------------------------
    # Calculate ward-level trends
    # --------------------------------------------------------

    results = []

    for ward_key, ward_df in df.groupby(
        "ward_key"
    ):

        result = calculate_ward_trend(
            ward_df
        )

        if result is not None:

            results.append(
                result
            )

    trend_df = pd.DataFrame(
        results
    )

    print()
    print(
        f"Wards with sufficient data: "
        f"{len(trend_df)}"
    )

    # ========================================================
    # FDR CORRECTION
    # ========================================================

    trend_df = apply_fdr_correction(
        trend_df,
        "adjusted_lst_trend_p_value",
        "adjusted_lst_trend_fdr_p_value"
    )

    trend_df = apply_fdr_correction(
        trend_df,
        "adjusted_heat_pocket_trend_p_value",
        "adjusted_heat_pocket_trend_fdr_p_value"
    )

    # --------------------------------------------------------
    # FDR significance flags
    # --------------------------------------------------------

    trend_df[
        "adjusted_lst_fdr_significant"
    ] = (
        trend_df[
            "adjusted_lst_trend_fdr_p_value"
        ]
        < P_VALUE_THRESHOLD
    )

    trend_df[
        "adjusted_heat_pocket_fdr_significant"
    ] = (
        trend_df[
            "adjusted_heat_pocket_trend_fdr_p_value"
        ]
        < P_VALUE_THRESHOLD
    )

    # --------------------------------------------------------
    # FDR-based classifications
    # --------------------------------------------------------

    trend_df[
        "adjusted_lst_trend_direction"
    ] = trend_df.apply(
        lambda row: classify_trend(
            row[
                "adjusted_lst_trend_celsius_per_year"
            ],
            row[
                "adjusted_lst_trend_fdr_p_value"
            ]
        ),
        axis=1
    )

    trend_df[
        "adjusted_heat_pocket_trend_direction"
    ] = trend_df.apply(
        lambda row: classify_trend(
            row[
                "adjusted_heat_pocket_trend_pct_points_per_year"
            ],
            row[
                "adjusted_heat_pocket_trend_fdr_p_value"
            ]
        ),
        axis=1
    )

    # ========================================================
    # COMBINED TREND
    # ========================================================

    def combined_trend(row):

        lst = row[
            "adjusted_lst_trend_direction"
        ]

        heat = row[
            "adjusted_heat_pocket_trend_direction"
        ]

        if (
            lst == "Increasing"
            and heat == "Increasing"
        ):

            return (
                "Increasing thermal pressure"
            )

        elif (
            lst == "Decreasing"
            and heat == "Decreasing"
        ):

            return (
                "Decreasing thermal pressure"
            )

        elif (
            lst == "Increasing"
            or heat == "Increasing"
        ):

            return (
                "Mixed / partial increase"
            )

        elif (
            lst == "Decreasing"
            or heat == "Decreasing"
        ):

            return (
                "Mixed / partial decrease"
            )

        else:

            return (
                "Stable / uncertain"
            )

    trend_df[
        "combined_adjusted_trend"
    ] = trend_df.apply(
        combined_trend,
        axis=1
    )

    # ========================================================
    # SORT
    # ========================================================

    trend_df = trend_df.sort_values(
        "adjusted_lst_trend_celsius_per_year",
        ascending=False
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE COMPLETE RESULTS
    # ========================================================

    trend_path = (
        OUTPUT_DIR
        / "ward_historical_trends.csv"
    )

    trend_df.to_csv(
        trend_path,
        index=False
    )

    # ========================================================
    # FILTER SIGNIFICANT LST TRENDS
    # ========================================================

    increasing_lst = trend_df[
        trend_df[
            "adjusted_lst_trend_direction"
        ] == "Increasing"
    ].sort_values(
        "adjusted_lst_trend_celsius_per_year",
        ascending=False
    )

    decreasing_lst = trend_df[
        trend_df[
            "adjusted_lst_trend_direction"
        ] == "Decreasing"
    ].sort_values(
        "adjusted_lst_trend_celsius_per_year"
    )

    # ========================================================
    # FILTER SIGNIFICANT HEAT-POCKET TRENDS
    # ========================================================

    increasing_heat = trend_df[
        trend_df[
            "adjusted_heat_pocket_trend_direction"
        ] == "Increasing"
    ].sort_values(
        "adjusted_heat_pocket_trend_pct_points_per_year",
        ascending=False
    )

    decreasing_heat = trend_df[
        trend_df[
            "adjusted_heat_pocket_trend_direction"
        ] == "Decreasing"
    ].sort_values(
        "adjusted_heat_pocket_trend_pct_points_per_year"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        {
            "metric": [

                "wards_analyzed",

                "fdr_significant_increasing_lst_wards",

                "fdr_significant_decreasing_lst_wards",

                "stable_uncertain_lst_wards",

                "fdr_significant_increasing_heat_pocket_wards",

                "fdr_significant_decreasing_heat_pocket_wards",

                "stable_uncertain_heat_pocket_wards",

                "mean_adjusted_lst_trend_celsius_per_year",

                "median_adjusted_lst_trend_celsius_per_year",

                "mean_adjusted_heat_pocket_trend_pct_points_per_year",

                "median_adjusted_heat_pocket_trend_pct_points_per_year",
            ],

            "value": [

                len(trend_df),

                len(increasing_lst),

                len(decreasing_lst),

                int(
                    (
                        trend_df[
                            "adjusted_lst_trend_direction"
                        ]
                        == "Stable/Uncertain"
                    ).sum()
                ),

                len(increasing_heat),

                len(decreasing_heat),

                int(
                    (
                        trend_df[
                            "adjusted_heat_pocket_trend_direction"
                        ]
                        == "Stable/Uncertain"
                    ).sum()
                ),

                trend_df[
                    "adjusted_lst_trend_celsius_per_year"
                ].mean(),

                trend_df[
                    "adjusted_lst_trend_celsius_per_year"
                ].median(),

                trend_df[
                    "adjusted_heat_pocket_trend_pct_points_per_year"
                ].mean(),

                trend_df[
                    "adjusted_heat_pocket_trend_pct_points_per_year"
                ].median(),
            ],
        }
    )

    summary_path = (
        OUTPUT_DIR
        / "historical_trend_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # TOP INCREASING TRENDS
    # ========================================================

    top_lst_path = (
        OUTPUT_DIR
        / "top_increasing_adjusted_lst_trends.csv"
    )

    increasing_lst.head(
        25
    ).to_csv(
        top_lst_path,
        index=False
    )

    top_heat_path = (
        OUTPUT_DIR
        / "top_increasing_adjusted_heat_pocket_trends.csv"
    )

    increasing_heat.head(
        25
    ).to_csv(
        top_heat_path,
        index=False
    )

    # ========================================================
    # TOP DECREASING TRENDS
    # ========================================================

    decreasing_lst_path = (
        OUTPUT_DIR
        / "top_decreasing_adjusted_lst_trends.csv"
    )

    decreasing_lst.head(
        25
    ).to_csv(
        decreasing_lst_path,
        index=False
    )

    decreasing_heat_path = (
        OUTPUT_DIR
        / "top_decreasing_adjusted_heat_pocket_trends.csv"
    )

    decreasing_heat.head(
        25
    ).to_csv(
        decreasing_heat_path,
        index=False
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print("FDR-CORRECTED TREND ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    print(
        "FDR-corrected seasonally adjusted LST:"
    )

    print(
        trend_df[
            "adjusted_lst_trend_direction"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "FDR-corrected seasonally adjusted "
        "heat-pocket trend:"
    )

    print(
        trend_df[
            "adjusted_heat_pocket_trend_direction"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # LST
    # --------------------------------------------------------

    print()
    print(
        "Top 10 wards by FDR-significant "
        "increasing LST trend:"
    )

    if len(increasing_lst) > 0:

        print(
            increasing_lst[
                [
                    "ward_name",
                    "ward_key",
                    "adjusted_lst_trend_celsius_per_year",
                    "adjusted_lst_trend_p_value",
                    "adjusted_lst_trend_fdr_p_value",
                    "adjusted_lst_trend_r2",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No FDR-significant increasing "
            "LST trends found."
        )

    # --------------------------------------------------------
    # Heat pockets
    # --------------------------------------------------------

    print()
    print(
        "Top 10 wards by FDR-significant "
        "increasing heat-pocket trend:"
    )

    if len(increasing_heat) > 0:

        print(
            increasing_heat[
                [
                    "ward_name",
                    "ward_key",
                    "adjusted_heat_pocket_trend_pct_points_per_year",
                    "adjusted_heat_pocket_trend_p_value",
                    "adjusted_heat_pocket_trend_fdr_p_value",
                    "adjusted_heat_pocket_trend_r2",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No FDR-significant increasing "
            "heat-pocket trends found."
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print(
        "Raw vs seasonally adjusted vs FDR:"
    )

    raw_increasing = (
        trend_df[
            "raw_lst_trend_p_value"
        ]
        < P_VALUE_THRESHOLD
    ).sum()

    adjusted_increasing = (
        (
            trend_df[
                "adjusted_lst_trend_p_value"
            ]
            < P_VALUE_THRESHOLD
        )
        &
        (
            trend_df[
                "adjusted_lst_trend_celsius_per_year"
            ]
            > 0
        )
    ).sum()

    fdr_increasing = len(
        increasing_lst
    )

    print(
        f"Raw significant increasing LST wards: "
        f"{raw_increasing}"
    )

    print(
        f"Seasonally adjusted nominal "
        f"significant increasing LST wards: "
        f"{adjusted_increasing}"
    )

    print(
        f"FDR-significant increasing LST wards: "
        f"{fdr_increasing}"
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print()
    print("Files saved:")

    print(trend_path)
    print(summary_path)
    print(top_lst_path)
    print(top_heat_path)
    print(decreasing_lst_path)
    print(decreasing_heat_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_trend_analysis()