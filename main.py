"""Main FastAPI application for the Medical AI Analyst."""

import logging
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from async_lru import alru_cache

from services.anomaly_service import AnomalyDetectionService
from services.agent_service import AgentService
from services.german_alerts_service import GermanPharmacovigilanceService
from services.openfda_service import MedicalDataService
from config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle and load services."""
    app.state.anomaly_service = AnomalyDetectionService()
    app.state.agent_service = AgentService()
    app.state.alerts_service = GermanPharmacovigilanceService()
    app.state.data_service = MedicalDataService()
    logger.info("Services loaded successfully")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Medical AI Analyst API", lifespan=lifespan)

# Restrict CORS in production (set via frontend URL env var or similar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MedicalRequest(BaseModel):
    """Request schema for drug analysis."""
    drug_name: str = Field(..., min_length=2, max_length=50, examples=["ibuprofen"])


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@alru_cache(maxsize=100)
async def cached_analyze(drug: str, app_state: dict):
    """Cached internal method for analysis to prevent repeated expensive calls."""
    try:
        anomaly_result = await app_state['anomaly_service'].detect_anomalies(drug)
    except Exception as e:
        logger.exception(f"Error in anomaly detection for {drug}")
        raise HTTPException(status_code=500, detail=f"Datenanalyse fehlgeschlagen: {str(e)}")

    try:
        agent_summary, german_alerts = await asyncio.gather(
            app_state['agent_service'].analyze_with_agent(
                drug_name=drug,
                current_count=anomaly_result.get("current_count", 0),
                z_score=anomaly_result.get("z_score", 0.0)
            ),
            app_state['alerts_service'].fetch_german_alerts(drug),
            return_exceptions=True
        )

        if isinstance(agent_summary, Exception):
            logger.exception(f"Error in agent service for {drug}")
            raise HTTPException(status_code=500, detail=f"KI-Analyse fehlgeschlagen: {str(agent_summary)}")
        if isinstance(german_alerts, Exception):
            logger.exception(f"Error fetching german alerts for {drug}")
            german_alerts = []

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error for {drug}")
        raise HTTPException(status_code=500, detail=f"Unerwarteter Fehler: {str(e)}")

    return {
        "drug_name": drug,
        "quantitative_analysis": anomaly_result,
        "qualitative_analysis": agent_summary,
        "german_regulatory_alerts": german_alerts
    }

_analysis_cache = {}

@app.post("/analyze")
async def analyze_drug(request: MedicalRequest, api_request: Request):
    """Analyze a drug for reporting anomalies and fetch clinical literature."""
    drug = request.drug_name.lower()

    if drug in _analysis_cache:
        return _analysis_cache[drug]

    try:
        anomaly_result = await api_request.app.state.anomaly_service.detect_anomalies(drug)
    except Exception as e:
        logger.exception(f"Error in anomaly detection for {drug}")
        raise HTTPException(status_code=500, detail=f"Datenanalyse fehlgeschlagen: {str(e)}")

    try:
        # Run these two independent tasks concurrently
        agent_task = asyncio.create_task(
            api_request.app.state.agent_service.analyze_with_agent(
                drug_name=drug,
                current_count=anomaly_result.get("current_count", 0),
                z_score=anomaly_result.get("z_score", 0.0)
            )
        )
        alerts_task = asyncio.create_task(
            api_request.app.state.alerts_service.fetch_german_alerts(drug)
        )

        agent_summary, german_alerts = await asyncio.gather(agent_task, alerts_task, return_exceptions=True)

        if isinstance(agent_summary, Exception):
            logger.exception(f"Error in agent service for {drug}")
            raise HTTPException(status_code=500, detail=f"KI-Analyse fehlgeschlagen: {str(agent_summary)}")
        if isinstance(german_alerts, Exception):
            logger.exception(f"Error fetching german alerts for {drug}")
            german_alerts = []

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error for {drug}")
        raise HTTPException(status_code=500, detail=f"Unerwarteter Fehler: {str(e)}")

    response_data = {
        "drug_name": drug,
        "quantitative_analysis": anomaly_result,
        "qualitative_analysis": agent_summary,
        "german_regulatory_alerts": german_alerts
    }

    _analysis_cache[drug] = response_data
    return response_data

_trend_cache = {}

@app.get("/trend")
async def get_drug_trend(drug_name: str, api_request: Request):
    """Microservice endpoint to get 4-week trend for a drug."""
    drug = drug_name.lower()

    if drug in _trend_cache:
        return _trend_cache[drug]

    try:
        df = await api_request.app.state.data_service.fetch_adverse_events_timeseries(drug)
        if not df.empty and len(df) >= 4:
            total_cases = int(df['Adverse_Events'].tail(4).sum())
        else:
            total_cases = 0

        result = {"drug_name": drug, "cases_4_weeks": total_cases}
        _trend_cache[drug] = result
        return result
    except Exception as e:
        logger.exception(f"Error fetching trend data for {drug}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trendanalyse fehlgeschlagen: {str(e)}")
