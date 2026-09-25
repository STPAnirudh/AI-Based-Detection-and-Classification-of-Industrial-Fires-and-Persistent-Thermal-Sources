"""
Data Cleaning and Validation Module for SIH26162 Review-1 Prototype.

Validates NASA FIRMS-compatible thermal anomaly records and normalizes fields
for downstream geospatial and machine learning processing.
"""

import logging
from typing import Tuple
import pandas as pd
import numpy as np

from src.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

logger = logging.getLogger(__name__)

REQUIRED_FIRMS_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "acq_date",
    "acq_time"
]

def validate_coordinates(lat: float, lon: float) -> bool:
    """Check whether latitude and longitude are within standard geographical bounds."""
    try:
        if pd.isna(lat) or pd.isna(lon):
            return False
        lat_f, lon_f = float(lat), float(lon)
        return (LAT_MIN <= lat_f <= LAT_MAX) and (LON_MIN <= lon_f <= LON_MAX)
    except (ValueError, TypeError):
        return False

def normalize_confidence(conf_val) -> Tuple[float, str]:
    """
    Normalizes FIRMS confidence values (which can be string categories or integer percentages)
    into a numeric score [0-100] and standard categorical label.
    """
    if pd.isna(conf_val):
        return 50.0, "nominal"
    
    val_str = str(conf_val).strip().lower()
    
    # Textual confidence categories used by VIIRS
    if val_str == "high" or val_str == "h":
        return 90.0, "high"
    elif val_str == "nominal" or val_str == "n":
        return 65.0, "nominal"
    elif val_str == "low" or val_str == "l":
        return 35.0, "low"
    
    # Numeric confidence used by MODIS (0-100)
    try:
        num = float(val_str)
        num = max(0.0, min(100.0, num))
        cat = "high" if num >= 80 else ("nominal" if num >= 50 else "low")
        return num, cat
    except ValueError:
        return 50.0, "nominal"

def normalize_daynight(val) -> int:
    """Converts day/night indicator ('D'/'N') to binary flag (1 for Day, 0 for Night)."""
    if pd.isna(val):
        return 1
    val_str = str(val).strip().upper()
    return 1 if val_str == "D" else 0

def clean_hotspot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, validates, and normalizes a thermal anomaly DataFrame.
    
    Operations:
    1. Checks required columns.
    2. Drops rows with null coordinates or invalid geographical ranges.
    3. Coerces thermal attributes (brightness, bright_t31, frp) to float and validates non-negativity.
    4. Normalizes confidence into both numeric and categorical forms.
    5. Normalizes daynight flag.
    6. Ensures unique event_id.
    """
    initial_count = len(df)
    
    # Verify required columns exist
    missing = [c for c in REQUIRED_FIRMS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Hotspot dataset is missing required FIRMS columns: {missing}")

    cleaned = df.copy()

    # Drop null coordinates
    cleaned = cleaned.dropna(subset=["latitude", "longitude"])
    
    # Validate coordinate bounds
    valid_coords = cleaned.apply(
        lambda row: validate_coordinates(row["latitude"], row["longitude"]), axis=1
    )
    cleaned = cleaned[valid_coords]

    # Convert numeric fields
    numeric_cols = ["latitude", "longitude", "brightness"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    
    # Optional FIRMS thermal fields with safe defaults if missing
    if "bright_t31" in cleaned.columns:
        cleaned["bright_t31"] = pd.to_numeric(cleaned["bright_t31"], errors="coerce").fillna(cleaned["brightness"] - 50.0)
    else:
        cleaned["bright_t31"] = cleaned["brightness"] - 50.0
        
    if "frp" in cleaned.columns:
        cleaned["frp"] = pd.to_numeric(cleaned["frp"], errors="coerce").fillna(10.0)
    else:
        cleaned["frp"] = 10.0

    # Ensure non-negative thermal values
    cleaned = cleaned[(cleaned["brightness"] > 0) & (cleaned["bright_t31"] > 0) & (cleaned["frp"] >= 0)]

    # Normalize confidence
    conf_tuples = cleaned["confidence"].apply(normalize_confidence) if "confidence" in cleaned.columns else [(50.0, "nominal")] * len(cleaned)
    cleaned["confidence_score"] = [t[0] for t in conf_tuples]
    cleaned["confidence_cat"] = [t[1] for t in conf_tuples]

    # Normalize day/night
    if "daynight" in cleaned.columns:
        cleaned["daynight_flag"] = cleaned["daynight"].apply(normalize_daynight)
    else:
        cleaned["daynight_flag"] = 1

    # Ensure event_id exists
    if "event_id" not in cleaned.columns:
        cleaned["event_id"] = [f"EVT_{i+1:04d}" for i in range(len(cleaned))]

    # Format acq_time if needed (e.g. 815 -> "08:15", "1930" -> "19:30")
    def format_time(t):
        if pd.isna(t):
            return "00:00"
        t_str = str(t).split(".")[0].zfill(4)
        return f"{t_str[:2]}:{t_str[2:]}"
    
    cleaned["acq_time_fmt"] = cleaned["acq_time"].apply(format_time)

    cleaned = cleaned.reset_index(drop=True)
    dropped_count = initial_count - len(cleaned)
    if dropped_count > 0:
        logger.info(f"Cleaned hotspots: dropped {dropped_count} invalid rows out of {initial_count}.")
    
    return cleaned
