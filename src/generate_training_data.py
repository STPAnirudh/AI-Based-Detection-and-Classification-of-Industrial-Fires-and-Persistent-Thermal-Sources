"""
Script to generate clearly labelled synthetic training data for Review-1 prototype.

DISCLAIMER:
This data is purely SYNTHETIC / DEMO data created specifically to demonstrate the
scikit-learn baseline machine learning classification pipeline.
It is NOT scientifically validated final training data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_training_dataset(output_path: str = "data/demo_ml_training_data.csv", random_seed: int = 42):
    np.random.seed(random_seed)
    records = []

    # 1. Potential Industrial Fire (n=40)
    # Characterized by high thermal intensity (FRP), high brightness, close to facilities (< 1.5 km)
    for _ in range(40):
        records.append({
            "brightness": np.round(np.random.uniform(365.0, 415.0), 2),
            "bright_t31": np.round(np.random.uniform(302.0, 325.0), 2),
            "frp": np.round(np.random.uniform(45.0, 160.0), 2),
            "distance_to_facility_km": np.round(np.random.uniform(0.05, 1.4), 3),
            "daynight_flag": int(np.random.choice([0, 1])),
            "confidence_score": np.round(np.random.uniform(80.0, 100.0), 1),
            "label": "Potential Industrial Fire"
        })

    # 2. Persistent Thermal Source (n=40)
    # Characterized by moderate-to-high steady heat, regular night & day detection, inside/near facilities (< 1.0 km)
    for _ in range(40):
        records.append({
            "brightness": np.round(np.random.uniform(342.0, 368.0), 2),
            "bright_t31": np.round(np.random.uniform(294.0, 303.0), 2),
            "frp": np.round(np.random.uniform(12.0, 35.0), 2),
            "distance_to_facility_km": np.round(np.random.uniform(0.01, 0.9), 3),
            "daynight_flag": int(np.random.choice([0, 1], p=[0.55, 0.45])),
            "confidence_score": np.round(np.random.uniform(75.0, 100.0), 1),
            "label": "Persistent Thermal Source"
        })

    # 3. Natural/Vegetation Fire (n=40)
    # Characterized by location distant from industrial zones (> 3.5 km), moderate FRP, often day
    for _ in range(40):
        records.append({
            "brightness": np.round(np.random.uniform(325.0, 355.0), 2),
            "bright_t31": np.round(np.random.uniform(290.0, 302.0), 2),
            "frp": np.round(np.random.uniform(8.0, 35.0), 2),
            "distance_to_facility_km": np.round(np.random.uniform(3.5, 25.0), 3),
            "daynight_flag": int(np.random.choice([0, 1], p=[0.25, 0.75])),
            "confidence_score": np.round(np.random.uniform(50.0, 90.0), 1),
            "label": "Natural/Vegetation Fire"
        })

    # 4. Other/Uncertain (n=40)
    # Low FRP, low brightness, marginal confidence or ambiguous context
    for _ in range(40):
        records.append({
            "brightness": np.round(np.random.uniform(305.0, 328.0), 2),
            "bright_t31": np.round(np.random.uniform(285.0, 296.0), 2),
            "frp": np.round(np.random.uniform(1.0, 11.0), 2),
            "distance_to_facility_km": np.round(np.random.uniform(0.8, 20.0), 3),
            "daynight_flag": int(np.random.choice([0, 1])),
            "confidence_score": np.round(np.random.uniform(20.0, 55.0), 1),
            "label": "Other/Uncertain"
        })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Generated synthetic training dataset: {out_file} with {len(df)} samples.")
    return df

if __name__ == "__main__":
    generate_synthetic_training_dataset()
