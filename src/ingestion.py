"""
Data Ingestion Module for SIH26162 Review-1 Prototype.

Loads and parses:
1. NASA FIRMS-compatible thermal anomaly CSV files.
2. Industrial facility context datasets.
"""

import logging
from pathlib import Path
from typing import Union
import pandas as pd

from src.config import DEFAULT_HOTSPOTS_CSV, DEFAULT_FACILITIES_CSV
from src.cleaning import clean_hotspot_dataframe, validate_coordinates

logger = logging.getLogger(__name__)

FACILITY_REQUIRED_COLUMNS = ["facility_id", "name", "latitude", "longitude", "facility_type"]

def load_hotspot_data(csv_path: Union[str, Path] = DEFAULT_HOTSPOTS_CSV) -> pd.DataFrame:
    """
    Ingests and cleans thermal anomaly records from a NASA FIRMS-compatible CSV.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Thermal hotspot dataset not found at: {path}")

    logger.info(f"Ingesting thermal hotspots from {path.name}...")
    raw_df = pd.read_csv(path)
    cleaned_df = clean_hotspot_dataframe(raw_df)
    logger.info(f"Successfully loaded and cleaned {len(cleaned_df)} thermal anomaly records.")
    return cleaned_df

def load_facility_data(csv_path: Union[str, Path] = DEFAULT_FACILITIES_CSV) -> pd.DataFrame:
    """
    Ingests industrial infrastructure context dataset.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Facility dataset not found at: {path}")

    logger.info(f"Ingesting industrial facilities from {path.name}...")
    df = pd.read_csv(path)
    
    missing = [c for c in FACILITY_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Facility dataset is missing required columns: {missing}")

    # Validate coordinate bounds
    valid = df.apply(lambda r: validate_coordinates(r["latitude"], r["longitude"]), axis=1)
    df = df[valid].copy()
    
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    logger.info(f"Successfully loaded {len(df)} industrial facilities.")
    return df
