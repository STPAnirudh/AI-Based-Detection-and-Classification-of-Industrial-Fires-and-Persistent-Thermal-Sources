"""
Persistence and Historical Analysis Module for SIH26162 Review-1 Prototype.

Groups historical thermal observations that occur at approximately the same location
and computes explainable persistence metrics:
- occurrence_count
- first_observed_date
- latest_observed_date
- days_span
- persistence_level ('High', 'Medium', 'Low')
"""

import logging
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from src.geospatial import haversine_distance

logger = logging.getLogger(__name__)

DEFAULT_CLUSTER_RADIUS_KM = 1.0

def assign_spatial_clusters(hotspots_df: pd.DataFrame, radius_km: float = DEFAULT_CLUSTER_RADIUS_KM) -> pd.DataFrame:
    """
    Groups thermal events into spatial clusters based on proximity (<= radius_km).
    Uses a greedy spatial grouping algorithm that is simple, deterministic, and explainable.
    """
    df = hotspots_df.copy()
    if df.empty:
        return df

    clusters: List[Dict[str, Any]] = []
    cluster_ids = []

    for _, row in df.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        matched_cluster = None

        # Check proximity to known cluster centroids
        for c in clusters:
            dist = haversine_distance(lat, lon, c["centroid_lat"], c["centroid_lon"])
            if dist <= radius_km:
                matched_cluster = c
                break

        if matched_cluster is not None:
            # Add to existing cluster
            cluster_id = matched_cluster["cluster_id"]
            matched_cluster["events"].append(row["event_id"])
            matched_cluster["dates"].append(str(row["acq_date"]))
            # Update centroid
            n = len(matched_cluster["events"])
            matched_cluster["centroid_lat"] = (matched_cluster["centroid_lat"] * (n - 1) + lat) / n
            matched_cluster["centroid_lon"] = (matched_cluster["centroid_lon"] * (n - 1) + lon) / n
        else:
            # Create new cluster
            cluster_id = f"CLUS_{len(clusters)+1:03d}"
            clusters.append({
                "cluster_id": cluster_id,
                "centroid_lat": lat,
                "centroid_lon": lon,
                "events": [row["event_id"]],
                "dates": [str(row["acq_date"])]
            })

        cluster_ids.append(cluster_id)

    df["cluster_id"] = cluster_ids

    # Compute persistence summary for each cluster
    cluster_metrics = {}
    for c in clusters:
        dates = sorted(c["dates"])
        first_date = dates[0]
        latest_date = dates[-1]
        try:
            d1 = datetime.strptime(first_date, "%Y-%m-%d")
            d2 = datetime.strptime(latest_date, "%Y-%m-%d")
            days_span = (d2 - d1).days
        except Exception:
            days_span = 0

        occ_count = len(c["events"])

        # Determine explainable persistence level
        if occ_count >= 6:
            level = "High"
        elif occ_count >= 3:
            level = "Medium"
        else:
            level = "Low"

        cluster_metrics[c["cluster_id"]] = {
            "occurrence_count": occ_count,
            "first_observed_date": first_date,
            "latest_observed_date": latest_date,
            "days_span": days_span,
            "persistence_level": level
        }

    # Map cluster metrics back to individual rows
    df["occurrence_count"] = df["cluster_id"].apply(lambda cid: cluster_metrics[cid]["occurrence_count"])
    df["first_observed_date"] = df["cluster_id"].apply(lambda cid: cluster_metrics[cid]["first_observed_date"])
    df["latest_observed_date"] = df["cluster_id"].apply(lambda cid: cluster_metrics[cid]["latest_observed_date"])
    df["days_span"] = df["cluster_id"].apply(lambda cid: cluster_metrics[cid]["days_span"])
    df["persistence_level"] = df["cluster_id"].apply(lambda cid: cluster_metrics[cid]["persistence_level"])

    logger.info(f"Spatial persistence analysis complete: identified {len(clusters)} spatial clusters across {len(df)} events.")
    return df
