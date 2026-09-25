"""
Unified End-to-End Processing Pipeline for SIH26162 Review-1 Prototype.

Orchestrates:
Data Ingestion -> Cleaning -> Geospatial Context -> ML Classification -> Persistence Analysis -> Risk Engine.
"""

import logging
from typing import Dict, Any, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.config import (
    DEFAULT_HOTSPOTS_CSV,
    DEFAULT_FACILITIES_CSV,
    DEFAULT_TRAINING_CSV
)
from src.ingestion import load_hotspot_data, load_facility_data
from src.geospatial import enrich_hotspots_with_geospatial_context
from src.ml_classifier import BaselineClassifier
from src.persistence import assign_spatial_clusters
from src.risk_engine import evaluate_hotspot_risks

logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(
        self,
        hotspots_csv: Path = DEFAULT_HOTSPOTS_CSV,
        facilities_csv: Path = DEFAULT_FACILITIES_CSV,
        training_csv: Path = DEFAULT_TRAINING_CSV
    ):
        self.hotspots_csv = Path(hotspots_csv)
        self.facilities_csv = Path(facilities_csv)
        self.training_csv = Path(training_csv)
        self.classifier = BaselineClassifier()
        self.facilities_df: pd.DataFrame = pd.DataFrame()
        self.enriched_df: pd.DataFrame = pd.DataFrame()

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes the 6-stage Review-1 pipeline and returns:
        (enriched_hotspots_df, facilities_df, summary_statistics)
        """
        logger.info("==================================================")
        logger.info("Executing Review-1 End-to-End Prototype Pipeline")
        logger.info("==================================================")

        # 1. Ingestion & Cleaning
        hotspots_df = load_hotspot_data(self.hotspots_csv)
        self.facilities_df = load_facility_data(self.facilities_csv)

        # 2. Geospatial Context & Distance Calculation
        geo_df = enrich_hotspots_with_geospatial_context(hotspots_df, self.facilities_df)

        # 3. Persistence & Historical Analysis
        persist_df = assign_spatial_clusters(geo_df)

        # 4. Baseline Machine Learning Classification
        self.classifier.load_or_train(self.training_csv)
        ml_df = self.classifier.classify_hotspots(persist_df)

        # 5. Risk / Priority Engine
        self.enriched_df = evaluate_hotspot_risks(ml_df)

        # 6. Generate Summary Statistics
        stats = self.compute_summary_statistics()

        logger.info(f"Pipeline executed successfully for {len(self.enriched_df)} events.")
        return self.enriched_df, self.facilities_df, stats

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """Calculates dashboard summary metrics."""
        df = self.enriched_df
        if df.empty:
            return {}

        class_counts = df["predicted_class"].value_counts().to_dict()
        risk_counts = df["risk_priority"].value_counts().to_dict()
        persist_counts = df["persistence_level"].value_counts().to_dict()

        return {
            "total_events": len(df),
            "total_facilities": len(self.facilities_df),
            "total_clusters": df["cluster_id"].nunique() if "cluster_id" in df.columns else 0,
            "counts_by_class": class_counts,
            "counts_by_risk": risk_counts,
            "counts_by_persistence": persist_counts,
            "high_risk_count": int(risk_counts.get("HIGH", 0)),
            "potential_industrial_fires": int(class_counts.get("Potential Industrial Fire", 0)),
            "persistent_sources": int(class_counts.get("Persistent Thermal Source", 0)),
            "model_accuracy_demo": self.classifier.training_summary.get("test_accuracy", 0.0),
            "timestamp": datetime.now().isoformat(),
            "demo_notice": "DEMO PROTOTYPE DATA AND BASELINE MODEL (REVIEW-1)"
        }
