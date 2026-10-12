import os
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from ai.services.dashboard import (
    ML_DATASET_PATH,
    XAI_LOCAL_PATH,
    build_ward_environmental,
    build_ward_xai_local,
    describe_intervention,
    describe_trend,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
    / "ward_heat_risk_20260503.csv"
)

TREND_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "persistence"
    / "ward_historical_trends.csv"
)

XAI_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
    / "heat_risk_xai_explanations_20260503.csv"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
    / "ward_policy_recommendations_20260503.csv"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# GROQ CLIENT (created lazily so the API can start and serve
# non-AI analytics endpoints even without a key configured)
# ============================================================

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load all ward-level inputs required by the AI layer."""

    risk_df = pd.read_csv(RISK_FILE)
    trend_df = pd.read_csv(TREND_FILE)
    xai_df = pd.read_csv(XAI_FILE)
    policy_df = pd.read_csv(POLICY_FILE)

    return risk_df, trend_df, xai_df, policy_df


# ============================================================
# HELPERS
# ============================================================

def safe_value(value):
    """Convert pandas NaN/NumPy values into JSON-safe values."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def get_ward_context(
    ward_key,
    risk_df,
    trend_df,
    xai_df,
    policy_df,
    ml_df=None,
    xai_local_df=None,
):
    """
    Build a grounded context object for one ward.

    This is the single canonical context builder used by the ward
    detail endpoint, the AI insights endpoint, HeatLens AI, and the
    PDF report - so every surface shows the same values. `ml_df` and
    `xai_local_df` are optional so existing callers that only need
    the core risk/trend/xai/policy fields keep working unchanged;
    passing them additionally unlocks the environmental profile and
    human-readable XAI contributions.
    """

    risk_rows = risk_df[
        risk_df["ward_key"].astype(str) == str(ward_key)
    ]

    if risk_rows.empty:
        raise ValueError(f"Ward '{ward_key}' not found in risk dataset.")

    risk_row = risk_rows.iloc[0]

    trend_rows = trend_df[
        trend_df["ward_key"].astype(str) == str(ward_key)
    ]

    xai_rows = xai_df[
        xai_df["ward_key"].astype(str) == str(ward_key)
    ]

    policy_rows = policy_df[
        policy_df["ward_key"].astype(str) == str(ward_key)
    ]

    trend_row = trend_rows.iloc[0] if not trend_rows.empty else None
    xai_row = xai_rows.iloc[0] if not xai_rows.empty else None
    policy_row = policy_rows.iloc[0] if not policy_rows.empty else None

    context = {
        "ward": {
            "ward_key": safe_value(risk_row.get("ward_key")),
            "ward_name": safe_value(risk_row.get("ward_name")),
            "corporation": safe_value(risk_row.get("Corporation")),
            "ward_id": safe_value(risk_row.get("ward_id")),
            "date": safe_value(risk_row.get("date")),
            "data_status": safe_value(risk_row.get("data_status")),
        },

        "current_heat": {
            "mean_lst_celsius": safe_value(
                risk_row.get("mean_lst_celsius")
            ),
            "heat_pocket_coverage_pct": safe_value(
                risk_row.get("heat_pocket_coverage_pct")
            ),
            "current_heat_severity": safe_value(
                risk_row.get("current_heat_severity")
            ),
            "current_heat_pocket_level": safe_value(
                risk_row.get("current_heat_pocket_level")
            ),
        },

        "historical_heat": {
            "historical_heat_pocket_coverage_pct": safe_value(
                risk_row.get("historical_heat_pocket_coverage_pct")
            ),
            "historical_max_heat_pocket_coverage_pct": safe_value(
                risk_row.get("historical_max_heat_pocket_coverage_pct")
            ),
            "historical_observation_count": safe_value(
                risk_row.get("historical_observation_count")
            ),
            "persistence_level": safe_value(
                risk_row.get("persistence_level")
            ),
        },

        "future_prediction": {
            "future_heat_probability": safe_value(
                risk_row.get("future_heat_probability")
            ),
            "future_high_heat_prediction": safe_value(
                risk_row.get("future_high_heat_prediction")
            ),
            "future_heat_risk_level": safe_value(
                risk_row.get("future_heat_risk_level")
            ),
            "horizon": (
                "Model-estimated probability for the next observed/usable "
                "Landsat scene - not a fixed calendar horizon such as "
                "1, 5, or 10 years."
            ),
        },

        "environmental": (
            build_ward_environmental(ward_key, ml_df) if ml_df is not None else {}
        ),

        "trend": {},

        "xai": {},

        "policy": {},
    }

    if trend_row is not None:

        trend_mapping = {
            "trend_classification": "combined_adjusted_trend",
            "trend_direction": "adjusted_heat_pocket_trend_direction",
            "trend_slope": "adjusted_heat_pocket_trend_pct_points_per_year",
            "p_value": "adjusted_heat_pocket_trend_p_value",
            "fdr_adjusted_p_value": "adjusted_heat_pocket_trend_fdr_p_value",
            "r_squared": "adjusted_heat_pocket_trend_r2",
        }

        for output_field, source_field in trend_mapping.items():
            if source_field in trend_row.index:
                context["trend"][output_field] = safe_value(
                    trend_row[source_field]
                )

        trend_summary = describe_trend(context["trend"])

        if trend_summary is not None:
            context["trend"]["trend_explanation"] = trend_summary["explanation"]
            context["trend"]["significant"] = trend_summary["significant"]

    if xai_row is not None:

        xai_fields = [
            "xai_explanation",
        ]

        for field in xai_fields:
            if field in xai_row.index:
                context["xai"][field] = safe_value(
                    xai_row[field]
                )

    if xai_local_df is not None:
        context["xai"]["contributions"] = build_ward_xai_local(ward_key, xai_local_df)

    if policy_row is not None:

        policy_fields = [
            "priority_level",
            "primary_intervention",
            "secondary_intervention",
            "policy_rationale",
            "trend_classification",
        ]

        for field in policy_fields:
            if field in policy_row.index:
                context["policy"][field] = safe_value(
                    policy_row[field]
                )

        context["policy"]["primary_intervention_detail"] = describe_intervention(
            context["policy"].get("primary_intervention"),
            priority_level=context["policy"].get("priority_level"),
            trend_significant=context["trend"].get("significant"),
        )

        context["policy"]["secondary_intervention_detail"] = describe_intervention(
            context["policy"].get("secondary_intervention"),
            priority_level=context["policy"].get("priority_level"),
            trend_significant=context["trend"].get("significant"),
        )

    return context


# ============================================================
# AI PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI interpretation layer for a Bengaluru satellite-based
urban heat intelligence system.

Your task is to explain supplied ward-level analytical results in a
clear, cautious, research-grade manner.

IMPORTANT:

1. Use ONLY the information supplied in the ward context.

2. Do NOT invent:
   - measurements
   - causes
   - trends
   - statistics
   - intervention effects
   - temperature reductions
   - population characteristics
   - health impacts
   - demographic information
   - infrastructure conditions
   - land-use characteristics not present in the context.

3. Do NOT make causal claims.

For example, do not say:
"Low vegetation caused the high heat."

Instead say:
"The available model inputs include vegetation-related indicators,
but the supplied analysis does not establish causality."

4. The future_heat_probability is a MODEL-ESTIMATED PROBABILITY.
Never call it:
- confidence
- certainty
- likelihood with certainty
- guaranteed prediction.

5. The future prediction refers to the NEXT OBSERVED LANDSAT
observation represented in the supplied dataset. Do not describe it
as a prediction for an arbitrary future date unless that date is
explicitly provided.

6. Historical heat-pocket coverage represents the spatial extent of
the detected heat-pocket class across satellite observations.
Do not describe persistence as the number of consecutive months a
ward was hot.

7. Trend results must be interpreted exactly as supplied.

If a trend is stable/uncertain, say that no statistically significant
trend was identified by the supplied analysis.

If a trend is FDR-significant, describe the direction and magnitude
only when those values are supplied.

8. The Policy Engine is the authoritative source for intervention
recommendations.

Do NOT replace or override its recommendation.

9. Distinguish between:
   A. analytical findings,
   B. model interpretation,
   C. policy recommendations.

10. Monitoring is a SYSTEM/DECISION-SUPPORT ACTION.
Do not describe monitoring itself as a heat-mitigation intervention.

11. For mitigation interventions, use only the evidence basis supplied
in the context.

The system may use established evidence regarding interventions such
as:
- cool roofs
- cool pavements
- trees and vegetation
- green infrastructure

However, do NOT claim that an intervention will reduce temperature
by a particular amount unless an effect size is explicitly supplied.

12. Do not infer that an intervention is effective specifically in
this ward. Instead use wording such as:
"The Policy Engine recommends..."
or
"This intervention is consistent with established heat-island
mitigation guidance."

13. If the supplied data are insufficient, explicitly say so.

14. Avoid dramatic or alarmist wording.

15. Do not mention "heat exposure", "health risk", or human
vulnerability unless those quantities are explicitly present in the
supplied data.

16. Keep explanations concise but informative.

17. Output ONLY valid JSON matching the requested schema.

18. Do NOT restate the same figure (e.g. the same LST value or the
same probability) in more than one field. Each field should add new
information, not repeat a previous field in different words.

19. NDVI, NDBI, and NDWI are standard remote-sensing indices supplied
in the context when available. Describe them only in general,
non-causal terms (what the index measures, and its supplied value);
do not claim they explain or cause the ward's heat conditions.
"""


USER_PROMPT_TEMPLATE = """
Generate an AI interpretation for the following Bengaluru ward.

WARD CONTEXT
============

{ward_context}

Return ONLY valid JSON using exactly this structure. Each field must
add NEW information - do not restate the same figure in more than
one field:

{{
  "current_condition": "ONE concise sentence on current mean LST, current heat-pocket coverage, and LST severity/heat-pocket status - using the supplied values once.",
  "historical_pattern": "ONE concise sentence on historical average/maximum heat-pocket coverage, observation count, and persistence level.",
  "future_risk": "ONE concise sentence on the model-estimated future heat probability and future risk level for the next observed/usable Landsat scene (not a calendar-year forecast).",
  "main_drivers": [
    "Top model driver 1, from the supplied XAI contributions, in plain English (2-6 words).",
    "Top model driver 2, in plain English.",
    "Up to 4 total, only if supplied."
  ],
  "what_this_means": "ONE concise interpretation connecting the above without repeating the same numbers already stated.",
  "key_findings": [
    "Finding 1 - must add information not already covered above.",
    "Finding 2.",
    "Up to 5 maximum."
  ],
  "recommended_actions": [
    {{"action": "The Policy Engine's primary intervention, verbatim.", "reason": "One short reason grounded in the supplied risk/persistence/trend/policy fields."}},
    {{"action": "The Policy Engine's secondary intervention, verbatim, if supplied.", "reason": "One short reason."}}
  ],
  "uncertainty_note": "State that the future value is a model-estimated probability for the next observed Landsat scene (not a long-range forecast) and that recommendations are decision-support, not guaranteed or causal outcomes.",
  "evidence_basis": "Briefly state the evidence basis for any mitigation category mentioned by the Policy Engine. Do not invent effect sizes or ward-specific intervention outcomes - if none is calibrated in the supplied context, say effect size is not estimated by the current model."
}}
"""


# ============================================================
# AI GENERATION
# ============================================================

def generate_ai_insights(ward_context):
    """Generate grounded AI insights for one ward."""

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY not configured. Add it to the .env file "
            "to enable AI-generated insights."
        )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        ward_context=json.dumps(
            ward_context,
            indent=2,
            ensure_ascii=False
        )
    )

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
        )

    except Exception as exc:

        if getattr(exc, "status_code", None) == 429 or "rate_limit" in str(exc).lower():
            raise RuntimeError(
                "AI insight generation has hit the upstream model provider's "
                "rate limit for today. Please try again in a few minutes."
            ) from exc

        raise

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Groq returned invalid JSON:\n{content}"
        ) from exc

    return result


# ============================================================
# BATCH GENERATION
# ============================================================

AI_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
    / "ward_ai_insights_20260503.csv"
)

AI_CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "risk"
    / "ward_ai_insights_checkpoint_20260503.csv"
)


def result_to_row(ward_context, result):
    """Flatten one AI JSON response into one CSV row."""

    ward = ward_context.get("ward", {})
    policy = ward_context.get("policy", {})

    def join_points(value):
        if isinstance(value, list):
            return " | ".join(
                str(item).strip()
                for item in value
                if str(item).strip()
            )
        return str(value).strip() if value is not None else ""

    return {
        "date": ward.get("date"),
        "ward_key": ward.get("ward_key"),
        "ward_name": ward.get("ward_name"),
        "Corporation": ward.get("corporation"),
        "ward_id": ward.get("ward_id"),
        "data_status": ward.get("data_status"),

        "policy_priority_level": result.get(
            "policy_priority_level"
        ),

        "current_condition": result.get("current_condition", ""),
        "historical_pattern": result.get("historical_pattern", ""),
        "future_risk": result.get("future_risk", ""),
        "main_drivers": join_points(result.get("main_drivers", [])),
        "what_this_means": result.get("what_this_means", ""),

        "key_findings": join_points(
            result.get("key_findings", [])
        ),

        "recommended_actions": join_points(
            [
                f"{item.get('action', '')}: {item.get('reason', '')}"
                if isinstance(item, dict) else str(item)
                for item in result.get("recommended_actions", [])
            ]
        ),

        "uncertainty_note": join_points(
            result.get("uncertainty_note", [])
        ),

        "evidence_basis": join_points(
            result.get("evidence_basis", [])
        ),

        "primary_intervention": policy.get(
            "primary_intervention"
        ),

        "secondary_intervention": policy.get(
            "secondary_intervention"
        ),
    }


def generate_all_ward_insights():
    """
    Generate AI insights for all wards.

    Uses a checkpoint so an interrupted run can resume without
    regenerating already completed wards.
    """

    print("=" * 70)
    print("AI INSIGHTS - ALL WARDS")
    print("=" * 70)

    print("\nLoading datasets...")

    risk_df, trend_df, xai_df, policy_df = load_data()
    ml_df = pd.read_csv(ML_DATASET_PATH)
    xai_local_df = pd.read_csv(XAI_LOCAL_PATH)

    print(f"Risk rows:   {len(risk_df)}")
    print(f"Trend rows:  {len(trend_df)}")
    print(f"XAI rows:    {len(xai_df)}")
    print(f"Policy rows: {len(policy_df)}")

    # --------------------------------------------------------
    # Authoritative ward list comes from the risk dataset.
    # It contains all 369 wards.
    # --------------------------------------------------------

    ward_keys = (
        risk_df["ward_key"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(f"\nWards to process: {len(ward_keys)}")

    # --------------------------------------------------------
    # Resume from checkpoint if available.
    # --------------------------------------------------------

    completed = {}

    if AI_CHECKPOINT_FILE.exists():

        checkpoint_df = pd.read_csv(
            AI_CHECKPOINT_FILE
        )

        if "ward_key" in checkpoint_df.columns:

            for _, row in checkpoint_df.iterrows():

                completed[
                    str(row["ward_key"])
                ] = row.to_dict()

            print(
                f"Existing checkpoint found: "
                f"{len(completed)} wards already completed."
            )

    else:

        print(
            "No checkpoint found. Starting from the first ward."
        )

    rows = list(completed.values())

    total = len(ward_keys)

    for index, ward_key in enumerate(
        ward_keys,
        start=1
    ):

        if ward_key in completed:

            print(
                f"[{index}/{total}] "
                f"{ward_key} - already completed"
            )

            continue

        print(
            f"\n[{index}/{total}] "
            f"Generating: {ward_key}"
        )

        try:

            context = get_ward_context(
                ward_key,
                risk_df,
                trend_df,
                xai_df,
                policy_df,
                ml_df,
                xai_local_df,
            )

            result = generate_ai_insights(
                context
            )

            row = result_to_row(
                context,
                result
            )

            rows.append(row)

            completed[ward_key] = row

            # ------------------------------------------------
            # Save checkpoint after EVERY successful ward.
            # ------------------------------------------------

            checkpoint_df = pd.DataFrame(
                rows
            )

            checkpoint_df.to_csv(
                AI_CHECKPOINT_FILE,
                index=False
            )

            print(
                f"    SUCCESS | "
                f"Priority: "
                f"{row['policy_priority_level']}"
            )

            # Small delay to reduce pressure on API limits.
            time.sleep(0.5)

        except Exception as exc:

            print(
                f"    ERROR | {ward_key} | {exc}"
            )

            # Do not create a fake AI result.
            # The ward can be retried on the next run.
            continue

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    final_df = pd.DataFrame(rows)

    if not final_df.empty:

        # Keep the same authoritative ward ordering.
        order_map = {
            ward_key: position
            for position, ward_key
            in enumerate(ward_keys)
        }

        final_df["_order"] = (
            final_df["ward_key"]
            .map(order_map)
        )

        final_df = (
            final_df
            .sort_values("_order")
            .drop(columns=["_order"])
        )

        final_df.to_csv(
            AI_OUTPUT_FILE,
            index=False
        )

    successful = len(final_df)

    failed = total - successful

    print("\n" + "=" * 70)
    print("AI INSIGHTS BATCH COMPLETE")
    print("=" * 70)

    print(f"Total wards:      {total}")
    print(f"Successful:       {successful}")
    print(f"Not generated:    {failed}")

    print(
        f"\nFinal output:\n"
        f"{AI_OUTPUT_FILE}"
    )

    if failed > 0:

        print(
            "\nSome wards were not generated. "
            "Run the same command again to retry them."
        )

    else:

        print(
            "\nALL 369 WARDS GENERATED SUCCESSFULLY."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    generate_all_ward_insights()


if __name__ == "__main__":
    main()
