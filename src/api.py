"""
FastAPI REST API and Web Server for SIH26162 Review-1 Prototype.

Provides endpoints to query thermal events, industrial facilities, summary statistics,
and serves the interactive GIS dashboard.
"""

import logging
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import STATIC_DIR
from src.pipeline import PipelineRunner
from src.risk_engine import calculate_event_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SIH26162 - Industrial Fire & Thermal Source Monitoring (Review-1)",
    description="College Theme-Based Project Review-1 Prototype: NASA FIRMS + Geospatial + Baseline ML + Risk GIS",
    version="1.0.0-review1"
)

# Initialize pipeline
pipeline_runner = PipelineRunner()
# Run pipeline once at startup
enriched_events_df, facilities_df, summary_stats = pipeline_runner.run()

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class HotspotPredictionRequest(BaseModel):
    brightness: float = Field(..., description="Brightness temperature (Kelvin)", json_schema_extra={"example": 385.0})
    bright_t31: float = Field(..., description="Channel T31 brightness (Kelvin)", json_schema_extra={"example": 308.0})
    frp: float = Field(..., description="Fire Radiative Power (MW)", json_schema_extra={"example": 75.0})
    distance_to_facility_km: float = Field(..., description="Distance to nearest industrial facility (km)", json_schema_extra={"example": 0.35})
    daynight_flag: int = Field(1, description="1 for Day, 0 for Night", json_schema_extra={"example": 1})
    confidence_score: float = Field(90.0, description="Detection confidence 0-100", json_schema_extra={"example": 90.0})
    persistence_level: Optional[str] = Field("Low", description="Persistence level (High, Medium, Low)")

@app.get("/")
def serve_dashboard():
    """Serves the main interactive GIS dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(index_file)

@app.get("/api/status")
def get_system_status():
    """Returns system status, active modules, and prototype disclaimers."""
    return {
        "project": "SIH26162",
        "title": "AI-Based Detection & Classification of Industrial Fires & Persistent Thermal Sources",
        "phase": "Review-1 Working Prototype",
        "status": "OPERATIONAL",
        "features_active": [
            "1. Thermal Event Detection & Visualization",
            "2. Geospatial Context & Proximity",
            "3. Baseline ML Classification (Random Forest)",
            "4. Persistence & Historical Analysis",
            "5. Risk/Priority Assessment & GIS Dashboard"
        ],
        "disclaimer": (
            "PROTOTYPE NOTICE: This is a college theme-based demonstrator for Review-1. "
            "Data is sample/synthetic; ML model is an exploratory baseline; "
            "results do NOT represent confirmed industrial fires."
        )
    }

@app.get("/api/events")
def get_all_events(
    classification: Optional[str] = None,
    risk_priority: Optional[str] = None,
    persistence_level: Optional[str] = None
):
    """
    Returns all enriched thermal anomaly events with optional filters.
    """
    global enriched_events_df
    df = enriched_events_df.copy()

    if classification and classification != "ALL":
        df = df[df["predicted_class"] == classification]
    if risk_priority and risk_priority != "ALL":
        df = df[df["risk_priority"] == risk_priority]
    if persistence_level and persistence_level != "ALL":
        df = df[df["persistence_level"] == persistence_level]

    return df.to_dict(orient="records")

@app.get("/api/events/{event_id}")
def get_event_detail(event_id: str):
    """Returns detailed records and spatial context for a specific thermal event."""
    global enriched_events_df
    match = enriched_events_df[enriched_events_df["event_id"] == event_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Thermal event '{event_id}' not found.")
    return match.iloc[0].to_dict()

@app.get("/api/facilities")
def get_facilities():
    """Returns the list of industrial infrastructure facilities."""
    global facilities_df
    return facilities_df.to_dict(orient="records")

@app.get("/api/statistics")
def get_statistics():
    """Returns real-time aggregated summary metrics for the dashboard."""
    global summary_stats
    return summary_stats

@app.post("/api/predict")
def predict_adhoc_hotspot(req: HotspotPredictionRequest):
    """
    On-demand prediction endpoint: classifies a custom thermal anomaly and computes risk score.
    """
    import pandas as pd
    features = pd.DataFrame([{
        "brightness": req.brightness,
        "bright_t31": req.bright_t31,
        "frp": req.frp,
        "distance_to_facility_km": req.distance_to_facility_km,
        "daynight_flag": req.daynight_flag,
        "confidence_score": req.confidence_score
    }])
    
    preds, confs = pipeline_runner.classifier.predict(features)
    pred_class = str(preds[0])
    conf = float(confs[0])

    score, priority, expl = calculate_event_risk(
        predicted_class=pred_class,
        distance_km=req.distance_to_facility_km,
        frp=req.frp,
        ml_confidence=conf,
        persistence_level=req.persistence_level or "Low"
    )

    return {
        "predicted_class": pred_class,
        "ml_confidence_percent": conf,
        "risk_score": score,
        "risk_priority": priority,
        "risk_explanation": expl,
        "model_type": "RandomForestClassifier (Baseline)",
        "disclaimer": "PROTOTYPE ESTIMATE - NOT CONFIRMED FIRE"
    }

@app.post("/api/reload")
def reload_pipeline():
    """Reruns the end-to-end pipeline and updates in-memory datasets."""
    global enriched_events_df, facilities_df, summary_stats
    enriched_events_df, facilities_df, summary_stats = pipeline_runner.run()
    return {"status": "SUCCESS", "message": "Pipeline refreshed successfully.", "events_count": len(enriched_events_df)}
