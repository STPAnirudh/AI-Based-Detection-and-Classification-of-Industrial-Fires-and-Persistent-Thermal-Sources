"""
Unit and Integration Test Suite for SIH26162 Review-1 Prototype.

Validates:
1. Data loading and required columns.
2. Coordinate validity checking.
3. Great-circle (Haversine) distance calculation.
4. Nearest facility assignment.
5. Persistence grouping and metrics.
6. Baseline ML classifier inference and confidence scores.
7. Multi-factor risk calculation and bounds.
8. End-to-end pipeline execution.
9. FastAPI REST API endpoints.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from src.config import (
    DEFAULT_HOTSPOTS_CSV,
    DEFAULT_FACILITIES_CSV,
    DEFAULT_TRAINING_CSV,
    PROTOTYPE_CLASSES
)
from src.cleaning import validate_coordinates, clean_hotspot_dataframe
from src.ingestion import load_hotspot_data, load_facility_data
from src.geospatial import haversine_distance, find_nearest_facility
from src.persistence import assign_spatial_clusters
from src.ml_classifier import BaselineClassifier
from src.risk_engine import calculate_event_risk, evaluate_hotspot_risks
from src.pipeline import PipelineRunner
from src.api import app

# -------------------------------------------------------------
# 1. DATA LOADING & REQUIRED COLUMNS
# -------------------------------------------------------------
def test_hotspot_data_loading():
    """Verify FIRMS-compatible thermal hotspots load with required fields."""
    df = load_hotspot_data(DEFAULT_HOTSPOTS_CSV)
    assert not df.empty, "Hotspots dataset should not be empty"
    required = ["event_id", "latitude", "longitude", "brightness", "frp", "confidence_score"]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"
    assert len(df) >= 20, "Demo hotspot dataset should have at least 20 records"

def test_facility_data_loading():
    """Verify industrial facility context dataset loads properly."""
    df = load_facility_data(DEFAULT_FACILITIES_CSV)
    assert not df.empty, "Facility dataset should not be empty"
    required = ["facility_id", "name", "latitude", "longitude", "facility_type"]
    for col in required:
        assert col in df.columns, f"Missing facility column: {col}"
    assert len(df) >= 5, "Facility dataset should have multiple industrial sites"

# -------------------------------------------------------------
# 2. COORDINATE VALIDITY
# -------------------------------------------------------------
@pytest.mark.parametrize("lat,lon,expected", [
    (17.68, 83.25, True),
    (0.0, 0.0, True),
    (-89.9, 179.9, True),
    (91.5, 50.0, False),     # Lat > 90
    (-95.0, 20.0, False),    # Lat < -90
    (20.0, 185.0, False),    # Lon > 180
    (20.0, -195.0, False),   # Lon < -180
    (None, 80.0, False),
    ("bad_lat", 80.0, False)
])
def test_coordinate_validity(lat, lon, expected):
    """Verify latitude and longitude boundary checks."""
    assert validate_coordinates(lat, lon) == expected

# -------------------------------------------------------------
# 3. DISTANCE CALCULATION
# -------------------------------------------------------------
def test_haversine_distance():
    """Verify great-circle distance computation."""
    # Distance to identical point must be 0
    assert haversine_distance(17.63, 83.18, 17.63, 83.18) == 0.0

    # Symmetry: dist(A, B) == dist(B, A)
    d_ab = haversine_distance(17.6325, 83.1842, 17.6952, 83.2558)
    d_ba = haversine_distance(17.6952, 83.2558, 17.6325, 83.1842)
    assert d_ab == d_ba

    # Known distance between Vizag Steel Plant and HPCL Refinery is ~10.2 km
    assert 9.0 < d_ab < 11.5, f"Unexpected distance: {d_ab} km"

# -------------------------------------------------------------
# 4. NEAREST FACILITY LOOKUP
# -------------------------------------------------------------
def test_nearest_facility_lookup():
    """Verify that a hotspot adjacent to a facility is matched correctly."""
    facilities = load_facility_data(DEFAULT_FACILITIES_CSV)
    
    # Point located 50 meters from Vizag Steel Plant
    near_vizag = find_nearest_facility(17.6326, 83.1843, facilities)
    assert near_vizag["nearest_facility_id"] == "FAC_001"
    assert near_vizag["distance_to_facility_km"] < 0.1
    assert "Steel Plant" in near_vizag["nearest_facility_type"]

# -------------------------------------------------------------
# 5. PERSISTENCE & HISTORICAL ANALYSIS
# -------------------------------------------------------------
def test_persistence_calculation():
    """Verify spatial grouping and persistence level calculation."""
    df = load_hotspot_data(DEFAULT_HOTSPOTS_CSV)
    clustered = assign_spatial_clusters(df, radius_km=1.0)

    assert "cluster_id" in clustered.columns
    assert "occurrence_count" in clustered.columns
    assert "persistence_level" in clustered.columns
    assert "first_observed_date" in clustered.columns

    # Verify occurrence count logic
    high_persist = clustered[clustered["persistence_level"] == "High"]
    assert not high_persist.empty, "There should be recurring events with High persistence"
    assert (high_persist["occurrence_count"] >= 6).all()

# -------------------------------------------------------------
# 6. BASELINE ML CLASSIFIER
# -------------------------------------------------------------
def test_baseline_ml_classifier():
    """Verify training, prediction, and probability output from Random Forest."""
    clf = BaselineClassifier()
    summary = clf.train(DEFAULT_TRAINING_CSV)
    
    assert summary["model_type"] == "RandomForestClassifier"
    assert summary["test_accuracy"] >= 0.85
    assert "frp" in summary["feature_importances"]

    # Test sample feature input
    sample_features = pd.DataFrame([{
        "brightness": 390.0,
        "bright_t31": 310.0,
        "frp": 95.0,
        "distance_to_facility_km": 0.2,
        "daynight_flag": 1,
        "confidence_score": 95.0
    }])

    preds, confs = clf.predict(sample_features)
    assert len(preds) == 1
    assert preds[0] in PROTOTYPE_CLASSES
    assert 50.0 <= confs[0] <= 100.0

# -------------------------------------------------------------
# 7. RISK CALCULATION
# -------------------------------------------------------------
def test_risk_scoring():
    """Verify multi-factor risk score calculation and priority boundaries."""
    # Critical industrial fire event: close to plant, very high FRP, high confidence
    score_high, prio_high, expl_high = calculate_event_risk(
        predicted_class="Potential Industrial Fire",
        distance_km=0.2,
        frp=90.0,
        ml_confidence=95.0,
        persistence_level="High"
    )
    assert 70.0 <= score_high <= 100.0
    assert prio_high == "HIGH"
    assert "Score" in expl_high

    # Low risk event: distant from facility, low FRP, low confidence
    score_low, prio_low, expl_low = calculate_event_risk(
        predicted_class="Other/Uncertain",
        distance_km=15.0,
        frp=2.0,
        ml_confidence=30.0,
        persistence_level="Low"
    )
    assert 0.0 <= score_low < 40.0
    assert prio_low == "LOW"

# -------------------------------------------------------------
# 8. END-TO-END PIPELINE INTEGRATION
# -------------------------------------------------------------
def test_full_pipeline_execution():
    """Verify end-to-end execution of all 6 prototype stages."""
    runner = PipelineRunner()
    enriched_df, facilities_df, stats = runner.run()

    assert not enriched_df.empty
    assert not facilities_df.empty
    assert stats["total_events"] == len(enriched_df)
    assert stats["total_facilities"] == len(facilities_df)

    # Check key enriched columns
    expected_cols = [
        "event_id", "latitude", "longitude", "nearest_facility_name",
        "distance_to_facility_km", "cluster_id", "occurrence_count",
        "predicted_class", "ml_confidence", "risk_score", "risk_priority"
    ]
    for col in expected_cols:
        assert col in enriched_df.columns, f"Missing output column in enriched DataFrame: {col}"

# -------------------------------------------------------------
# 9. REST API ENDPOINTS
# -------------------------------------------------------------
def test_api_endpoints():
    """Verify REST API endpoints using FastAPI TestClient."""
    client = TestClient(app)

    # Status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "OPERATIONAL"

    # Statistics
    res_stats = client.get("/api/statistics")
    assert res_stats.status_code == 200
    assert res_stats.json()["total_events"] > 0

    # Events list
    res_events = client.get("/api/events")
    assert res_events.status_code == 200
    events = res_events.json()
    assert isinstance(events, list)
    assert len(events) > 0

    # Single event detail
    first_id = events[0]["event_id"]
    res_single = client.get(f"/api/events/{first_id}")
    assert res_single.status_code == 200
    assert res_single.json()["event_id"] == first_id

    # Facilities
    res_fac = client.get("/api/facilities")
    assert res_fac.status_code == 200
    assert len(res_fac.json()) > 0

    # Ad-hoc prediction
    pred_payload = {
        "brightness": 380.0,
        "bright_t31": 305.0,
        "frp": 80.0,
        "distance_to_facility_km": 0.4,
        "daynight_flag": 1,
        "confidence_score": 90.0
    }
    res_pred = client.post("/api/predict", json=pred_payload)
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "predicted_class" in data
    assert "risk_score" in data
