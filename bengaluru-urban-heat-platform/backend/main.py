import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ai.services.ai_insights import (
    load_data,
    get_ward_context,
    generate_ai_insights,
)

from ai.services.heatlens import (
    find_ward_from_question,
    ask_heatlens,
)

from ai.services.dashboard import (
    ML_DATASET_PATH,
    XAI_LOCAL_PATH,
    build_ward_geojson,
    build_ward_list,
    build_overview_stats,
    build_insights,
    build_lst_trend,
    build_ward_history,
    build_ward_xai_local,
    build_city_context,
)

from reports.pdf_report import (
    generate_ward_report,
    generate_summary_report,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Bengaluru Urban Heat Intelligence API",
    description="Backend API for the Bengaluru Urban Heat Intelligence Platform.",
    version="1.0.0",
)


# ============================================================
# CORS
#
# Allowed origins come from the ALLOWED_ORIGINS env var (comma-
# separated), so the deployed frontend's real domain can be added
# without touching code. Defaults to the local dev server only -
# never wildcard "*", since this API is credentialed.
# ============================================================

ALLOWED_ORIGINS_RAW = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in ALLOWED_ORIGINS_RAW.split(",")
    if origin.strip()
]
ALLOW_ALL = "*" in ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ALL else ALLOWED_ORIGINS,
    allow_origin_regex=None if ALLOW_ALL else r"https://.*\.vercel\.app",
    allow_credentials=not ALLOW_ALL,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD ANALYTICS DATA
# ============================================================

try:

    risk_df, trend_df, xai_df, policy_df = load_data()

    print("=" * 70)
    print("BENGALURU URBAN HEAT INTELLIGENCE API")
    print("=" * 70)

    print("Analytics datasets loaded successfully.")
    print(f"Risk rows:   {len(risk_df)}")
    print(f"Trend rows:  {len(trend_df)}")
    print(f"XAI rows:    {len(xai_df)}")
    print(f"Policy rows: {len(policy_df)}")

    print("=" * 70)

except Exception as exc:

    print("=" * 70)
    print("ERROR LOADING ANALYTICS DATA")
    print("=" * 70)
    print(exc)
    print("=" * 70)

    risk_df = None
    trend_df = None
    xai_df = None
    policy_df = None


try:

    ml_df = pd.read_csv(ML_DATASET_PATH)

except Exception as exc:

    print("ERROR LOADING ML DATASET FOR DASHBOARD AGGREGATES")
    print(exc)

    ml_df = None


try:

    xai_local_df = pd.read_csv(XAI_LOCAL_PATH)

except Exception as exc:

    print("ERROR LOADING LOCAL XAI DATASET")
    print(exc)

    xai_local_df = None


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.api_route("/", methods=["GET", "HEAD"])
def root():

    return {
        "name": "Bengaluru Urban Heat Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():

    data_loaded = all(
        df is not None
        for df in [
            risk_df,
            trend_df,
            xai_df,
            policy_df,
        ]
    )

    return {
        "status": "healthy" if data_loaded else "degraded",
        "analytics_data_loaded": data_loaded,
    }


# ============================================================
# DASHBOARD OVERVIEW (stats, insights, LST trend)
# ============================================================

@app.get("/api/dashboard/overview")
def dashboard_overview():

    if risk_df is None or trend_df is None or ml_df is None:

        raise HTTPException(
            status_code=503,
            detail="Analytics datasets are not available.",
        )

    stats = build_overview_stats(risk_df, trend_df, policy_df)
    insights = build_insights(risk_df, trend_df, ml_df)
    lst_trend = build_lst_trend(ml_df)

    return {
        "status": "success",
        "stats": stats,
        "insights": insights,
        "lst_trend": lst_trend,
    }


# ============================================================
# WARD MAP (GeoJSON enriched with risk / trend / policy)
# ============================================================

@app.get("/api/wards/geo")
def wards_geo():

    if risk_df is None:

        raise HTTPException(
            status_code=503,
            detail="Analytics datasets are not available.",
        )

    return build_ward_geojson(risk_df, trend_df, policy_df)


# ============================================================
# WARD LIST (lightweight, for search)
# ============================================================

@app.get("/api/wards")
def wards_list():

    if risk_df is None:

        raise HTTPException(
            status_code=503,
            detail="Analytics datasets are not available.",
        )

    return {
        "status": "success",
        "wards": build_ward_list(risk_df, trend_df, policy_df),
    }


# ============================================================
# WARD DETAIL (non-AI, grounded analytics only)
# ============================================================

@app.get("/api/wards/{ward_key}")
def ward_detail(ward_key: str):

    if not all(
        df is not None
        for df in [risk_df, trend_df, xai_df, policy_df]
    ):

        raise HTTPException(
            status_code=503,
            detail="Analytics datasets are not available.",
        )

    try:

        ward_context = get_ward_context(
            ward_key,
            risk_df,
            trend_df,
            xai_df,
            policy_df,
            ml_df,
            xai_local_df,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    return {
        "status": "success",
        **ward_context,
    }


# ============================================================
# WARD OBSERVATION HISTORY (for charts)
# ============================================================

@app.get("/api/wards/{ward_key}/history")
def ward_history(ward_key: str):

    if ml_df is None:

        raise HTTPException(
            status_code=503,
            detail="ML dataset is not available.",
        )

    history = build_ward_history(ward_key, ml_df)

    if not history:

        raise HTTPException(
            status_code=404,
            detail=f"No observation history found for ward '{ward_key}'.",
        )

    return {
        "status": "success",
        "ward_key": ward_key,
        "history": history,
    }


# ============================================================
# WARD LOCAL XAI (SHAP CONTRIBUTIONS)
# ============================================================

@app.get("/api/wards/{ward_key}/xai/local")
def ward_xai_local(ward_key: str):

    if xai_local_df is None:

        raise HTTPException(
            status_code=503,
            detail="Local XAI dataset is not available.",
        )

    contributions = build_ward_xai_local(ward_key, xai_local_df)

    return {
        "status": "success",
        "ward_key": ward_key,
        "contributions": contributions,
    }


# ============================================================
# REPORTS (PDF)
# ============================================================

@app.get("/api/reports/ward/{ward_key}")
def report_ward(ward_key: str):

    if not all(
        df is not None
        for df in [risk_df, trend_df, xai_df, policy_df]
    ):

        raise HTTPException(
            status_code=503,
            detail="Analytics datasets are not available.",
        )

    try:

        output_file = generate_ward_report(ward_key)

    except ValueError as exc:

        raise HTTPException(status_code=404, detail=str(exc))

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Report generation failed: {exc}",
        )

    return FileResponse(
        path=output_file,
        media_type="application/pdf",
        filename=output_file.name,
    )


@app.get("/api/reports/summary")
def report_summary():

    if not all(
        df is not None
        for df in [risk_df, trend_df, xai_df, policy_df]
    ) or ml_df is None:

        raise HTTPException(
            status_code=503,
            detail="Analytics datasets are not available.",
        )

    try:

        output_file = generate_summary_report()

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Summary report generation failed: {exc}",
        )

    return FileResponse(
        path=output_file,
        media_type="application/pdf",
        filename=output_file.name,
    )


# ============================================================
# WARD AI INSIGHTS
# ============================================================

@app.get("/api/wards/{ward_key}/ai-insights")
def get_ward_ai_insights(ward_key: str):

    if not all(
        df is not None
        for df in [
            risk_df,
            trend_df,
            xai_df,
            policy_df,
        ]
    ):

        raise HTTPException(
            status_code=503,
            detail="Analytics datasets are not available.",
        )

    # --------------------------------------------------------
    # Build ward context
    # --------------------------------------------------------

    try:

        ward_context = get_ward_context(
            ward_key,
            risk_df,
            trend_df,
            xai_df,
            policy_df,
            ml_df,
            xai_local_df,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to build ward context: {exc}",
        )

    # --------------------------------------------------------
    # Generate AI insights
    # --------------------------------------------------------

    try:

        ai_result = generate_ai_insights(
            ward_context
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"AI insight generation failed: {exc}",
        )

    return {
        "status": "success",
        **ward_context,
        "ai_insights": ai_result,
    }


# ============================================================
# HEATLENS
# ============================================================

@app.get("/api/heatlens")
def heatlens(question: str, ward_key: str | None = None):

    """
    Natural-language question answering through HeatLens.

    `ward_key` is optional and should be passed by the frontend when
    the user has an active ward context (e.g. opened HeatLens AI from
    a ward page) - this keeps follow-up questions ("what about NDBI?")
    grounded in that ward even though the question text alone doesn't
    name it. If omitted, the ward is detected from the question text;
    if that also fails, a city-wide context is used instead of no
    context at all.

    Example:

    /api/heatlens?question=Why is Cottonpete high priority?
    """

    # --------------------------------------------------------
    # Validate analytics data
    # --------------------------------------------------------

    if risk_df is None:

        raise HTTPException(
            status_code=503,
            detail="Analytics datasets are not available.",
        )

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    question = question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # --------------------------------------------------------
    # Resolve ward: explicit frontend context first, then detect
    # from the question text.
    # --------------------------------------------------------

    resolved_ward_key = (ward_key or "").strip() or None

    if resolved_ward_key and not (risk_df["ward_key"].astype(str) == resolved_ward_key).any():
        resolved_ward_key = None

    if not resolved_ward_key:

        try:

            resolved_ward_key = find_ward_from_question(
                question,
                risk_df,
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=f"Failed to identify ward: {exc}",
            )

    # --------------------------------------------------------
    # Build context: ward-specific if resolved, otherwise a real
    # city-wide summary (never an empty placeholder).
    # --------------------------------------------------------

    if resolved_ward_key:

        try:

            context = get_ward_context(
                resolved_ward_key,
                risk_df,
                trend_df,
                xai_df,
                policy_df,
                ml_df,
                xai_local_df,
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=f"Failed to build HeatLens context: {exc}",
            )

    else:

        try:

            context = build_city_context(risk_df, trend_df, policy_df, ml_df)

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=f"Failed to build city-wide HeatLens context: {exc}",
            )

    ward_key = resolved_ward_key

    # --------------------------------------------------------
    # Generate HeatLens answer
    # --------------------------------------------------------

    try:

        answer = ask_heatlens(
            question,
            context,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"HeatLens generation failed: {exc}",
        )

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {
        "status": "success",

        "question": question,

        "ward_key": ward_key,

        "answer": answer,
    }