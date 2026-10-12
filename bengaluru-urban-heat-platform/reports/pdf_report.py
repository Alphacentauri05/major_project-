from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from ai.services.ai_insights import (
    load_data,
    get_ward_context,
    generate_ai_insights,
)

from ai.services.dashboard import (
    ML_DATASET_PATH,
    XAI_LOCAL_PATH,
    build_overview_stats,
    build_insights,
)

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    """
    Normalize Unicode characters that may not render correctly
    with ReportLab's default Helvetica font.
    """

    if value is None:
        return ""

    text = str(value)

    replacements = {
        "\u2010": "-",   # hyphen
        "\u2011": "-",   # non-breaking hyphen
        "\u2012": "-",   # figure dash
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2212": "-",   # minus sign
        "\u00a0": " ",   # non-breaking space
        "\u2009": " ",   # thin space
        "\u200b": "",    # zero-width space
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ============================================================
# HELPERS
# ============================================================

def format_value(value, decimals=2):
    """Format numerical values for the report."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return clean_text(value)


def format_percentage(value):
    """Format a percentage value."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return clean_text(value)


def format_probability(value):
    """Convert probability from 0-1 to percentage."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return clean_text(value)


def format_prediction(value):
    """Format binary prediction as Yes/No."""

    if value is None:
        return "N/A"

    try:

        if int(float(value)) == 1:
            return "Yes"

        return "No"

    except (TypeError, ValueError):
        return clean_text(value)


# ============================================================
# TABLE HELPER
# ============================================================

def create_two_column_table(data):
    """Create a consistent two-column report table."""

    table = Table(
        data,
        colWidths=[3.2 * inch, 2.8 * inch],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


# ============================================================
# PDF GENERATION
# ============================================================

def generate_ward_report(ward_key):
    """
    Generate a Bengaluru Urban Heat Intelligence PDF
    for one ward.
    """

    # --------------------------------------------------------
    # Load analytical datasets
    # --------------------------------------------------------

    (
        risk_df,
        trend_df,
        xai_df,
        policy_df,
    ) = load_data()

    ml_df = pd.read_csv(ML_DATASET_PATH)
    xai_local_df = pd.read_csv(XAI_LOCAL_PATH)

    # --------------------------------------------------------
    # Build ward context - the same canonical context builder
    # used by the ward detail endpoint, the AI insights endpoint,
    # and HeatLens AI, so the PDF can never disagree with the
    # dashboard.
    # --------------------------------------------------------

    context = get_ward_context(
        ward_key,
        risk_df,
        trend_df,
        xai_df,
        policy_df,
        ml_df,
        xai_local_df,
    )

    ward = context["ward"]
    current = context["current_heat"]
    historical = context["historical_heat"]
    future = context["future_prediction"]
    trend = context["trend"]
    xai = context["xai"]
    policy = context["policy"]
    environmental = context.get("environmental", {})

    # --------------------------------------------------------
    # Generate AI insights
    # --------------------------------------------------------

    ai_insights = generate_ai_insights(context)

    # --------------------------------------------------------
    # File name
    # --------------------------------------------------------

    safe_ward_key = clean_text(
        str(ward_key).replace("/", "_")
    )

    date_value = ward.get(
        "date",
        datetime.now().strftime("%Y-%m-%d"),
    )

    output_file = (
        OUTPUT_DIR
        / f"bengaluru_heat_report_{safe_ward_key}_{date_value}.pdf"
    )

    # --------------------------------------------------------
    # PDF document
    # --------------------------------------------------------

    document = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Bengaluru Urban Heat Intelligence Report",
        author="Bengaluru Urban Heat Intelligence Platform",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=8,
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        spaceAfter=5,
    )

    # Table cells need Paragraph-wrapped text (not plain strings) for
    # reportlab to wrap long sentences within the column width instead
    # of overflowing past the page edge.
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10.5,
        spaceAfter=0,
    )

    def cell(text):
        return Paragraph(clean_text(text), table_cell_style)

    # --------------------------------------------------------
    # Story
    # --------------------------------------------------------

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "Bengaluru Urban Heat Intelligence Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            clean_text(
                f"{ward.get('ward_name', 'Unknown Ward')} "
                f"({ward.get('ward_key', ward_key)})"
            ),
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            clean_text(
                f"Corporation: {ward.get('corporation', 'N/A')} "
                f"| Ward ID: {ward.get('ward_id', 'N/A')} "
                f"| Latest satellite observation: {date_value}"
            ),
            subtitle_style,
        )
    )

    # ========================================================
    # 1. CURRENT HEAT CONDITIONS
    # ========================================================

    story.append(
        Paragraph(
            "1. Current Heat Conditions",
            heading_style,
        )
    )

    current_table_data = [
        ["Indicator", "Result"],

        [
            "Mean land-surface temperature",
            f"{format_value(current.get('mean_lst_celsius'))} °C",
        ],

        [
            "Current heat severity",
            clean_text(
                current.get(
                    "current_heat_severity",
                    "N/A",
                )
            ),
        ],

        [
            "Current heat-pocket coverage",
            format_percentage(
                current.get(
                    "heat_pocket_coverage_pct"
                )
            ),
        ],

        [
            "Current heat-pocket level",
            clean_text(
                current.get(
                    "current_heat_pocket_level",
                    "N/A",
                )
            ),
        ],
    ]

    story.append(
        create_two_column_table(
            current_table_data
        )
    )

    story.append(Spacer(1, 10))

    # ========================================================
    # 2. ENVIRONMENTAL PROFILE
    # ========================================================

    story.append(
        Paragraph(
            "2. Environmental Profile",
            heading_style,
        )
    )

    if environmental:

        env_table_data = [["Index", "Value", "Interpretation"]]

        for column in ("mean_ndvi", "mean_ndbi", "mean_ndwi"):

            entry = environmental.get(column)

            if not entry:
                continue

            env_table_data.append(
                [
                    clean_text(entry.get("name")),
                    format_value(entry.get("value"), 3),
                    cell(entry.get("interpretation")),
                ]
            )

        env_table = Table(
            env_table_data,
            colWidths=[1.3 * inch, 0.8 * inch, 3.9 * inch],
            repeatRows=1,
        )

        env_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(env_table)

    else:

        story.append(
            Paragraph(
                "Environmental index values are not available for this ward.",
                body_style,
            )
        )

    story.append(Spacer(1, 10))

    # ========================================================
    # 3. HISTORICAL HEAT
    # ========================================================

    story.append(
        Paragraph(
            "3. Historical Heat-Pocket Behaviour",
            heading_style,
        )
    )

    historical_table_data = [
        ["Indicator", "Result"],

        [
            "Historical average heat-pocket coverage",
            format_percentage(
                historical.get(
                    "historical_heat_pocket_coverage_pct"
                )
            ),
        ],

        [
            "Historical maximum coverage",
            format_percentage(
                historical.get(
                    "historical_max_heat_pocket_coverage_pct"
                )
            ),
        ],

        [
            "Historical observations",
            clean_text(
                historical.get(
                    "historical_observation_count",
                    "N/A",
                )
            ),
        ],

        [
            "Persistence level",
            clean_text(
                historical.get(
                    "persistence_level",
                    "N/A",
                )
            ),
        ],
    ]

    story.append(
        create_two_column_table(
            historical_table_data
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Persistence represents the recurrence of heat pockets "
            "across available satellite observations. It does not "
            "represent a continuous number of hot months.",
            small_style,
        )
    )

    # ========================================================
    # 4. HISTORICAL TREND
    # ========================================================

    story.append(
        Paragraph(
            "4. Historical Trend Assessment",
            heading_style,
        )
    )

    trend_data = [
        ["Indicator", "Result"],

        [
            "Trend classification",
            clean_text(
                trend.get(
                    "trend_classification",
                    "N/A",
                )
            ),
        ],

        [
            "Trend direction",
            clean_text(
                trend.get(
                    "trend_direction",
                    "N/A",
                )
            ),
        ],

        [
            "Heat-pocket trend slope",
            (
                f"{format_value(trend.get('trend_slope'), 3)} "
                "percentage points/year"
            ),
        ],

        [
            "Raw p-value",
            format_value(
                trend.get("p_value"),
                4,
            ),
        ],

        [
            "FDR-adjusted p-value",
            format_value(
                trend.get(
                    "fdr_adjusted_p_value"
                ),
                4,
            ),
        ],

        [
            "R²",
            format_value(
                trend.get("r_squared"),
                3,
            ),
        ],
    ]

    story.append(
        create_two_column_table(
            trend_data
        )
    )

    story.append(Spacer(1, 10))

    trend_explanation = clean_text(trend.get("trend_explanation", ""))

    if trend_explanation:

        story.append(
            Paragraph(
                trend_explanation,
                body_style,
            )
        )

    story.append(
        Paragraph(
            "Trend slopes represent changes in scene-relative "
            "heat-pocket spatial extent, expressed in percentage "
            "points per year - not a temperature trend.",
            small_style,
        )
    )

    # ========================================================
    # 5. NEXT-OBSERVATION HEAT PREDICTION
    # ========================================================

    story.append(
        Paragraph(
            "5. Next-Observation Heat Prediction",
            heading_style,
        )
    )

    future_table_data = [
        ["Indicator", "Result"],

        [
            "Model-estimated heat probability",
            format_probability(
                future.get(
                    "future_heat_probability"
                )
            ),
        ],

        [
            "Future high-heat prediction",
            format_prediction(
                future.get(
                    "future_high_heat_prediction"
                )
            ),
        ],

        [
            "Future heat-risk level",
            clean_text(
                future.get(
                    "future_heat_risk_level",
                    "N/A",
                )
            ),
        ],
    ]

    story.append(
        create_two_column_table(
            future_table_data
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "This is a model-estimated probability for the next "
            "observed/usable Landsat scene, not a fixed calendar "
            "horizon (such as 1, 5, or 10 years) and not a long-range "
            "weather forecast.",
            small_style,
        )
    )

    # ========================================================
    # 6. EXPLAINABLE AI
    # ========================================================

    story.append(
        Paragraph(
            "6. Explainable AI Assessment",
            heading_style,
        )
    )

    contributions = xai.get("contributions", [])

    if contributions:

        xai_table_data = [["Feature", "Contribution", "Strength", "Explanation"]]

        for item in contributions[:6]:

            direction_label = (
                "Positive" if item.get("direction") == "increases"
                else "Negative" if item.get("direction") == "decreases"
                else "Neutral"
            )

            xai_table_data.append(
                [
                    cell(item.get("human_label")),
                    direction_label,
                    clean_text(item.get("strength")),
                    cell(item.get("explanation")),
                ]
            )

        xai_table = Table(
            xai_table_data,
            colWidths=[1.4 * inch, 0.8 * inch, 0.7 * inch, 3.1 * inch],
            repeatRows=1,
        )

        xai_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        story.append(xai_table)
        story.append(Spacer(1, 8))

    xai_explanation = clean_text(
        xai.get(
            "xai_explanation",
            "No XAI explanation available.",
        )
    )

    story.append(
        Paragraph(
            xai_explanation,
            body_style,
        )
    )

    story.append(
        Paragraph(
            "Model contribution / association, not causation.",
            small_style,
        )
    )

    # ========================================================
    # 7. POLICY RECOMMENDATION
    # ========================================================

    story.append(
        Paragraph(
            "7. Policy Recommendation",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Priority level:</b> "
            f"{clean_text(policy.get('priority_level', 'N/A'))}",
            body_style,
        )
    )

    for detail_key, fallback_label in (
        ("primary_intervention_detail", "Primary intervention"),
        ("secondary_intervention_detail", "Secondary intervention"),
    ):

        detail = policy.get(detail_key)

        if not detail:
            continue

        story.append(
            Paragraph(
                f"<b>{clean_text(fallback_label)}:</b> "
                f"{clean_text(detail.get('intervention'))}",
                body_style,
            )
        )

        story.append(
            Paragraph(
                f"<i>Why:</i> {clean_text(detail.get('why'))}",
                small_style,
            )
        )

        story.append(
            Paragraph(
                f"<i>Mechanism:</i> {clean_text(detail.get('mechanism'))}",
                small_style,
            )
        )

        indicators = detail.get("indicators_affected") or []

        if indicators:
            story.append(
                Paragraph(
                    f"<i>Potential indicators affected:</i> "
                    f"{clean_text(', '.join(indicators))}",
                    small_style,
                )
            )

        story.append(
            Paragraph(
                f"<i>{clean_text(detail.get('effect_size_note'))}</i>",
                small_style,
            )
        )

        story.append(Spacer(1, 6))

    story.append(Spacer(1, 6))

    # ========================================================
    # 8. AI INSIGHTS
    # ========================================================

    story.append(
        Paragraph(
            "8. AI Insights",
            heading_style,
        )
    )

    summary = " ".join(
        clean_text(ai_insights.get(field, ""))
        for field in ("current_condition", "historical_pattern", "future_risk")
        if ai_insights.get(field)
    )

    if summary:

        story.append(
            Paragraph(
                f"<b>Summary:</b> {summary}",
                body_style,
            )
        )

    key_findings = ai_insights.get(
        "key_findings",
        [],
    )

    if key_findings:

        story.append(
            Paragraph(
                "<b>Key findings:</b>",
                body_style,
            )
        )

        for finding in key_findings:

            finding = clean_text(finding)

            story.append(
                Paragraph(
                    f"• {finding}",
                    body_style,
                )
            )

    main_drivers = ai_insights.get("main_drivers", [])
    what_this_means = clean_text(ai_insights.get("what_this_means", ""))

    if main_drivers or what_this_means:

        story.append(
            Paragraph(
                "<b>Model interpretation:</b>",
                body_style,
            )
        )

        for driver in main_drivers:

            story.append(
                Paragraph(
                    f"• {clean_text(driver)}",
                    body_style,
                )
            )

        if what_this_means:

            story.append(
                Paragraph(
                    what_this_means,
                    body_style,
                )
            )

    trend_interpretation = clean_text(trend.get("trend_explanation", ""))

    if trend_interpretation:

        story.append(
            Paragraph(
                f"<b>Trend interpretation:</b> "
                f"{trend_interpretation}",
                body_style,
            )
        )

    # ========================================================
    # 9. RECOMMENDED ACTIONS
    # ========================================================

    story.append(
        Paragraph(
            "9. Recommended Actions",
            heading_style,
        )
    )

    recommended_actions = ai_insights.get(
        "recommended_actions",
        [],
    )

    if recommended_actions:

        for action in recommended_actions:

            if isinstance(action, dict):

                action_text = clean_text(action.get("action", ""))
                reason_text = clean_text(action.get("reason", ""))

                story.append(
                    Paragraph(
                        f"• <b>{action_text}</b>"
                        + (f" — {reason_text}" if reason_text else ""),
                        body_style,
                    )
                )

            else:

                story.append(
                    Paragraph(
                        f"• {clean_text(action)}",
                        body_style,
                    )
                )

    # ========================================================
    # 10. UNCERTAINTY & EVIDENCE
    # ========================================================

    story.append(
        Paragraph(
            "10. Uncertainty and Evidence Basis",
            heading_style,
        )
    )

    uncertainty_note = clean_text(
        ai_insights.get(
            "uncertainty_note",
            "",
        )
    )

    if uncertainty_note:

        story.append(
            Paragraph(
                f"<b>Uncertainty:</b> "
                f"{uncertainty_note}",
                body_style,
            )
        )

    evidence_basis = clean_text(
        ai_insights.get(
            "evidence_basis",
            "",
        )
    )

    if evidence_basis:

        story.append(
            Paragraph(
                f"<b>Evidence basis:</b> "
                f"{evidence_basis}",
                body_style,
            )
        )

    # ========================================================
    # 11. METHODOLOGY NOTE
    # ========================================================

    story.append(
        Paragraph(
            "11. Methodology Note",
            heading_style,
        )
    )

    methodology_text = (
        "This report summarizes satellite-derived urban heat "
        "analytics for Bengaluru using Landsat observations, "
        "ward-level heat-pocket detection, multi-temporal "
        "persistence analysis, machine-learning prediction, "
        "explainable AI, historical trend analysis, and a "
        "rule-based policy decision-support engine. "
        "AI-generated narrative content is grounded in the "
        "analytical outputs supplied to the system."
    )

    story.append(
        Paragraph(
            clean_text(methodology_text),
            body_style,
        )
    )

    # ========================================================
    # FOOTER
    # ========================================================

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Bengaluru Urban Heat Intelligence Platform",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            small_style,
        )
    )

    # --------------------------------------------------------
    # Build PDF
    # --------------------------------------------------------

    document.build(story)

    return output_file


# ============================================================
# BENGALURU SUMMARY REPORT
# ============================================================

def generate_summary_report():
    """
    Generate a city-wide Bengaluru Urban Heat Intelligence
    summary PDF, using the same aggregates served by the
    Overview dashboard - no separate data source.
    """

    risk_df, trend_df, xai_df, policy_df = load_data()
    ml_df = pd.read_csv(ML_DATASET_PATH)

    stats = build_overview_stats(risk_df, trend_df, policy_df)
    insights = build_insights(risk_df, trend_df, ml_df)

    analysis_date = stats.get("analysis_date") or datetime.now().strftime("%Y-%m-%d")

    output_file = (
        OUTPUT_DIR / f"bengaluru_heat_summary_{analysis_date}.pdf"
    )

    document = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="Bengaluru Urban Heat Intelligence Summary",
        author="Bengaluru Urban Heat Intelligence Platform",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=20, leading=24, spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER,
        fontSize=10, leading=14, spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=14,
        leading=18, spaceBefore=14, spaceAfter=8, keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "BodyText", parent=styles["BodyText"], fontSize=9.5,
        leading=14, spaceAfter=8,
    )
    small_style = ParagraphStyle(
        "SmallText", parent=styles["BodyText"], fontSize=8,
        leading=11, spaceAfter=5,
    )

    story = []

    story.append(Paragraph("Bengaluru Urban Heat Intelligence Summary", title_style))
    story.append(Paragraph(f"Analysis date: {clean_text(analysis_date)}", subtitle_style))

    story.append(Paragraph("1. City-Wide Indicators", heading_style))

    indicators_table = [
        ["Indicator", "Value"],
        ["Total wards (GBA boundary)", str(stats["total_wards"])],
        [
            "High future heat-risk wards",
            f"{stats['high_heat_risk_count']} ({stats['high_heat_risk_pct']}%)",
        ],
        ["High decision-support priority wards", str(stats["high_priority_count"])],
        [
            "Very High policy-priority wards",
            str(stats["very_high_policy_count"])
            if stats["very_high_policy_count"] is not None
            else "N/A",
        ],
        [
            "Wards with future high-heat prediction",
            str(stats["future_high_heat_count"]),
        ],
        [
            "Highest observed land-surface temperature",
            f"{stats['highest_observed_lst']} °C ({clean_text(stats['highest_lst_zone'])} zone)",
        ],
        [
            "Wards with persistently high heat-pocket coverage",
            str(stats["persistent_heat_pockets"]),
        ],
        [
            "Wards with a statistically significant rising heat-pocket trend",
            str(stats["increasing_trend_wards"]),
        ],
    ]

    story.append(create_two_column_table(indicators_table))
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "All indicators are computed directly from the connected risk, "
            "persistence, and policy datasets for the analysis date shown above. "
            "None of these values are estimated or hardcoded.",
            small_style,
        )
    )

    story.append(Paragraph("2. Key Findings", heading_style))

    for insight in insights:
        story.append(
            Paragraph(
                f"<b>{clean_text(insight['title'])}.</b> {clean_text(insight['text'])}",
                body_style,
            )
        )

    story.append(Paragraph("3. Methodology Note", heading_style))

    story.append(
        Paragraph(
            clean_text(
                "This summary uses satellite-derived urban heat analytics for "
                "Bengaluru based on Landsat observations, ward-level heat-pocket "
                "detection, multi-temporal persistence analysis, machine-learning "
                "prediction of next-observation heat risk, FDR-corrected historical "
                "trend analysis, and a rule-based policy decision-support engine. "
                "Figures reflect the most recent processed observation date shown above."
            ),
            body_style,
        )
    )

    story.append(Spacer(1, 20))
    story.append(Paragraph("Bengaluru Urban Heat Intelligence Platform", subtitle_style))

    document.build(story)

    return output_file


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BENGALURU URBAN HEAT PDF REPORT")
    print("=" * 70)

    ward_key = "Central_57"

    print(f"\nGenerating report for: {ward_key}")

    output_file = generate_ward_report(
        ward_key
    )

    print("\nPDF generated successfully!")
    print(f"Location: {output_file}")

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)