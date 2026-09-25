"""
Risk and Priority Assessment Engine for SIH26162 Review-1 Prototype.

Calculates a multi-factor prototype risk score [0-100] and maps it to priority levels:
'HIGH', 'MEDIUM', 'LOW'.

PROTOTYPE NOTICE:
The risk scoring logic implemented here is a demonstrator heuristic combining
spatial proximity, thermal intensity, ML category, confidence, and persistence.
It is NOT a scientifically validated or certified disaster risk model.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
from src.config import RISK_WEIGHTS

logger = logging.getLogger(__name__)

def calculate_event_risk(
    predicted_class: str,
    distance_km: float,
    frp: float,
    ml_confidence: float,
    persistence_level: str
) -> Tuple[float, str, str]:
    """
    Computes a prototype risk score (0 to 100) and priority level for a single thermal anomaly.
    
    Formula:
        Score = Class_Weight + Proximity_Weight + Thermal_Weight + Confidence_Weight + Persistence_Weight
    """
    # 1. Classification Weight (0 - 35 pts)
    class_weights = RISK_WEIGHTS["class_weights"]
    w_class = class_weights.get(predicted_class, 5.0)

    # 2. Proximity Weight (0 - 25 pts)
    prox_weights = RISK_WEIGHTS["proximity_weights"]
    if distance_km < 0.5:
        w_prox = prox_weights["very_close"]
    elif distance_km < 1.5:
        w_prox = prox_weights["close"]
    elif distance_km < 3.0:
        w_prox = prox_weights["moderate"]
    else:
        w_prox = prox_weights["far"]

    # 3. FRP Thermal Intensity Weight (0 - 20 pts)
    frp_weights = RISK_WEIGHTS["frp_weights"]
    if frp > 50.0:
        w_frp = frp_weights["extreme"]
    elif frp > 20.0:
        w_frp = frp_weights["high"]
    elif frp > 5.0:
        w_frp = frp_weights["moderate"]
    else:
        w_frp = frp_weights["low"]

    # 4. ML Confidence Weight (0 - 10 pts)
    w_conf = round(min(10.0, max(0.0, (ml_confidence / 100.0) * 10.0)), 1)

    # 5. Persistence Weight (0 - 10 pts)
    persist_weights = RISK_WEIGHTS["persistence_weights"]
    w_persist = persist_weights.get(persistence_level, 2.0)

    # Total Score capped at 100.0
    total_score = round(min(100.0, max(0.0, w_class + w_prox + w_frp + w_conf + w_persist)), 1)

    # Priority Classification
    if total_score >= 70.0:
        priority = "HIGH"
    elif total_score >= 40.0:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    explanation = (
        f"Score {total_score} [Class: {w_class} pts | Proximity: {w_prox} pts | "
        f"FRP: {w_frp} pts | Conf: {w_conf} pts | Persistence: {w_persist} pts]"
    )

    return total_score, priority, explanation

def evaluate_hotspot_risks(hotspots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates risk and priority across all thermal hotspots in a DataFrame.
    """
    df = hotspots_df.copy()

    risk_scores = []
    priorities = []
    explanations = []

    for _, row in df.iterrows():
        score, prio, expl = calculate_event_risk(
            predicted_class=str(row.get("predicted_class", "Other/Uncertain")),
            distance_km=float(row.get("distance_to_facility_km", 999.0)),
            frp=float(row.get("frp", 10.0)),
            ml_confidence=float(row.get("ml_confidence", 50.0)),
            persistence_level=str(row.get("persistence_level", "Low"))
        )
        risk_scores.append(score)
        priorities.append(prio)
        explanations.append(expl)

    df["risk_score"] = risk_scores
    df["risk_priority"] = priorities
    df["risk_explanation"] = explanations

    logger.info(
        f"Risk evaluation complete. High: {(df['risk_priority'] == 'HIGH').sum()}, "
        f"Medium: {(df['risk_priority'] == 'MEDIUM').sum()}, Low: {(df['risk_priority'] == 'LOW').sum()}."
    )
    return df
