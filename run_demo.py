"""
SIH26162 — Review-1 Prototype Demonstration Script.

Allows running:
1. Full end-to-end pipeline in terminal with formatted reports.
2. Automated test suite execution.
3. Interactive GIS Web Dashboard server (FastAPI + Uvicorn).
"""

import sys
import argparse
import subprocess
import uvicorn
import pandas as pd
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from src.pipeline import PipelineRunner

def print_banner():
    print("=" * 75)
    print("  SIH26162 - AI-Based Detection & Classification of Industrial Fires")
    print("             and Persistent Thermal Sources")
    print("             Theme-Based Project | Review-1 Working Prototype")
    print("=" * 75)
    print(" [!] NOTICE: Demonstrator using NASA FIRMS-compatible sample data")
    print("     and a baseline Random Forest classifier. Not a certified model.")
    print("=" * 75 + "\n")

def run_cli_pipeline():
    print_banner()
    runner = PipelineRunner()
    enriched_df, facilities_df, stats = runner.run()

    print("\n--- 1. INGESTION & GEOSPATIAL CONTEXT ---")
    print(f"Total Hotspots Ingested: {len(enriched_df)}")
    print(f"Industrial Facilities Ingested: {len(facilities_df)}")
    print("\nSample Facilities Context:")
    for _, fac in facilities_df.head(4).iterrows():
        print(f"  * [{fac['facility_id']}] {fac['name']} ({fac['facility_type']}) at ({fac['latitude']}, {fac['longitude']})")

    print("\n--- 2. PERSISTENCE & HISTORICAL ANALYSIS ---")
    clusters_count = enriched_df["cluster_id"].nunique()
    print(f"Identified {clusters_count} spatial clusters.")
    persist_summary = enriched_df.groupby("persistence_level")["event_id"].count()
    for level, count in persist_summary.items():
        print(f"  * Persistence Level '{level}': {count} events")

    print("\n--- 3. BASELINE ML CLASSIFICATION ---")
    print(f"Model Type: RandomForestClassifier (scikit-learn)")
    print(f"Synthetic Test Set Accuracy: {stats.get('model_accuracy_demo', 0.0)*100:.1f}%")
    print("Predicted Class Breakdown:")
    for cls, count in stats.get("counts_by_class", {}).items():
        print(f"  * {cls}: {count} events")

    print("\n--- 4. RISK & PRIORITY ASSESSMENT ---")
    print("Risk Priority Breakdown:")
    for prio, count in stats.get("counts_by_risk", {}).items():
        print(f"  * {prio} PRIORITY: {count} events")

    print("\n--- 5. SAMPLE ENRICHED EVENTS TABLE ---")
    cols = [
        "event_id", "acq_date", "predicted_class", "ml_confidence",
        "nearest_facility_name", "distance_to_facility_km", "persistence_level",
        "risk_score", "risk_priority"
    ]
    sample_view = enriched_df[cols].head(8)
    print(sample_view.to_string(index=False))

    print("\n" + "=" * 75)
    print(" Prototype pipeline execution completed successfully.")
    print("=" * 75)

def run_tests():
    print_banner()
    print("Running automated test suite with pytest...\n")
    import pytest
    sys.exit(pytest.main(["-v", "tests"]))

def start_server(host="127.0.0.1", port=8000):
    print_banner()
    print(f"Starting Interactive GIS Web Dashboard at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server.\n")
    uvicorn.run("src.api:app", host=host, port=port, reload=False, log_level="info")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIH26162 Review-1 Prototype Runner")
    parser.add_argument("--cli", action="store_true", help="Run full pipeline and print summary report in terminal")
    parser.add_argument("--test", action="store_true", help="Execute automated test suite")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for GIS dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port for GIS dashboard")

    args = parser.parse_args()

    if args.cli:
        run_cli_pipeline()
    elif args.test:
        run_tests()
    else:
        # Default: run pipeline summary check, then start server
        run_cli_pipeline()
        print("\nLaunching GIS Dashboard server...")
        start_server(host=args.host, port=args.port)
