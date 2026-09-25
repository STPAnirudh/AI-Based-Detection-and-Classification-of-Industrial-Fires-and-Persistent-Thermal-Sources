"""
Configuration and constants for SIH26162 Review-1 Prototype.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

# Dataset Paths
DEFAULT_HOTSPOTS_CSV = DATA_DIR / "demo_firms_hotspots.csv"
DEFAULT_FACILITIES_CSV = DATA_DIR / "demo_industrial_facilities.csv"
DEFAULT_TRAINING_CSV = DATA_DIR / "demo_ml_training_data.csv"
MODEL_CACHE_PATH = DATA_DIR / "baseline_random_forest.joblib"

# Spatial & Validation Constraints
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0
SPATIAL_CLUSTER_DEGREE_THRESHOLD = 0.01  # Approximately ~1.1 km at the equator

# Classification Prototype Categories
PROTOTYPE_CLASSES = [
    "Potential Industrial Fire",
    "Persistent Thermal Source",
    "Natural/Vegetation Fire",
    "Other/Uncertain"
]

# ML Feature Columns
ML_FEATURE_COLS = [
    "brightness",
    "bright_t31",
    "frp",
    "distance_to_facility_km",
    "daynight_flag",
    "confidence_score"
]

# Risk Scoring Weight Constants (Prototype heuristic - not scientifically validated)
RISK_WEIGHTS = {
    "class_weights": {
        "Potential Industrial Fire": 35.0,
        "Persistent Thermal Source": 20.0,
        "Natural/Vegetation Fire": 15.0,
        "Other/Uncertain": 5.0
    },
    "proximity_weights": {
        "very_close": 25.0,   # < 0.5 km
        "close": 20.0,        # 0.5 km - 1.5 km
        "moderate": 10.0,     # 1.5 km - 3.0 km
        "far": 0.0            # > 3.0 km
    },
    "frp_weights": {
        "extreme": 20.0,      # > 50 MW
        "high": 15.0,         # 20 - 50 MW
        "moderate": 10.0,     # 5 - 20 MW
        "low": 5.0            # < 5 MW
    },
    "persistence_weights": {
        "High": 10.0,
        "Medium": 6.0,
        "Low": 2.0
    }
}
