"""
Geospatial Context Module for SIH26162 Review-1 Prototype.

Calculates great-circle (Haversine) distances between thermal anomaly hotspots
and known industrial facilities, provides geographic proximity context, and
includes an optional OpenStreetMap / Overpass API connector.
"""

import math
import logging
from typing import Dict, Any, Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance between two geographic coordinates in kilometers.
    
    Formula:
        a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        c = 2 * atan2(√a, √(1-a))
        d = R * c
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return round(EARTH_RADIUS_KM * c, 3)

def find_nearest_facility(lat: float, lon: float, facilities_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Finds the nearest industrial facility to a given coordinate from the facility DataFrame.
    """
    if facilities_df.empty:
        return {
            "nearest_facility_id": "NONE",
            "nearest_facility_name": "No Facility Data Available",
            "nearest_facility_type": "Unknown",
            "distance_to_facility_km": 999.0,
            "data_source": "NONE"
        }

    min_dist = float("inf")
    best_match = None

    for _, fac in facilities_df.iterrows():
        dist = haversine_distance(lat, lon, fac["latitude"], fac["longitude"])
        if dist < min_dist:
            min_dist = dist
            best_match = fac

    return {
        "nearest_facility_id": best_match["facility_id"],
        "nearest_facility_name": best_match["name"],
        "nearest_facility_type": best_match["facility_type"],
        "distance_to_facility_km": min_dist,
        "data_source": "DEMO_FACILITY_DATASET"
    }

def enrich_hotspots_with_geospatial_context(
    hotspots_df: pd.DataFrame, facilities_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes proximity to the nearest industrial facility for each thermal hotspot.
    """
    df = hotspots_df.copy()

    nearest_ids = []
    nearest_names = []
    nearest_types = []
    distances = []
    data_sources = []

    for _, row in df.iterrows():
        context = find_nearest_facility(row["latitude"], row["longitude"], facilities_df)
        nearest_ids.append(context["nearest_facility_id"])
        nearest_names.append(context["nearest_facility_name"])
        nearest_types.append(context["nearest_facility_type"])
        distances.append(context["distance_to_facility_km"])
        data_sources.append(context["data_source"])

    df["nearest_facility_id"] = nearest_ids
    df["nearest_facility_name"] = nearest_names
    df["nearest_facility_type"] = nearest_types
    df["distance_to_facility_km"] = distances
    df["geospatial_data_source"] = data_sources

    return df

def query_osm_overpass_facilities(
    lat: float, lon: float, radius_km: float = 5.0, timeout_sec: int = 3
) -> Optional[pd.DataFrame]:
    """
    Attempts to query OpenStreetMap via the Overpass API for industrial landuse/man_made tags.
    Falls back gracefully if network is unavailable or times out.
    
    NOTE: Real OSM API access depends on external network connectivity and rate-limits.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    radius_meters = int(radius_km * 1000)
    
    query = f"""
    [out:json][timeout:{timeout_sec}];
    (
      node["industrial"](around:{radius_meters},{lat},{lon});
      node["man_made"="works"](around:{radius_meters},{lat},{lon});
      node["landuse"="industrial"](around:{radius_meters},{lat},{lon});
    );
    out center 10;
    """
    
    try:
        response = requests.post(overpass_url, data={"data": query}, timeout=timeout_sec)
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            if elements:
                records = []
                for el in elements:
                    tags = el.get("tags", {})
                    records.append({
                        "facility_id": f"OSM_{el.get('id')}",
                        "name": tags.get("name", "Industrial Facility (OSM)"),
                        "latitude": el.get("lat") or el.get("center", {}).get("lat"),
                        "longitude": el.get("lon") or el.get("center", {}).get("lon"),
                        "facility_type": tags.get("industrial", tags.get("man_made", "Industrial")),
                        "description": "Live OpenStreetMap record"
                    })
                df = pd.DataFrame(records)
                logger.info(f"Retrieved {len(df)} live facilities from OpenStreetMap.")
                return df
    except Exception as e:
        logger.warning(f"OSM Overpass API lookup failed ({e}). Utilizing reliable demo facility dataset.")
    
    return None
