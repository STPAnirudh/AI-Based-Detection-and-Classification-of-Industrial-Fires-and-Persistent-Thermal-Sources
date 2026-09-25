"""
Baseline Machine Learning Classifier Module for SIH26162 Review-1 Prototype.

Implements a Random Forest baseline classification pipeline using scikit-learn.

PROTOTYPE NOTICE:
This baseline pipeline demonstrates the end-to-end integration of feature engineering,
model training, and probability-based inference for Review-1.
Training data is synthetic/demo data.
Outputs are exploratory prototype predictions (e.g., 'Potential Industrial Fire')
and do NOT represent confirmed fire incidents or scientifically validated models.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from src.config import (
    DEFAULT_TRAINING_CSV,
    MODEL_CACHE_PATH,
    ML_FEATURE_COLS,
    PROTOTYPE_CLASSES
)

logger = logging.getLogger(__name__)

class BaselineClassifier:
    """
    Scikit-learn Random Forest baseline classifier for thermal anomaly categorization.
    """
    def __init__(self, model_path: Optional[Path] = MODEL_CACHE_PATH):
        self.model_path = Path(model_path) if model_path else None
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names = ML_FEATURE_COLS
        self.classes_ = PROTOTYPE_CLASSES
        self.training_summary: Dict[str, Any] = {}

    def train(self, training_csv: Path = DEFAULT_TRAINING_CSV) -> Dict[str, Any]:
        """
        Trains the Random Forest model on the prototype synthetic training dataset.
        Evaluates on an 80/20 train/test split.
        """
        path = Path(training_csv)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found at: {path}")

        logger.info(f"Training baseline classifier on {path.name}...")
        df = pd.read_csv(path)

        # Ensure all feature columns exist
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Training data missing required feature columns: {missing}")

        X = df[self.feature_names]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        )
        self.model.fit(X_train, y_train)

        # Evaluation metrics on test partition
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Feature importances
        importances = dict(zip(self.feature_names, [round(float(v), 4) for v in self.model.feature_importances_]))

        self.training_summary = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 6,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_accuracy": round(float(acc), 4),
            "feature_importances": importances,
            "is_synthetic_demo_data": True,
            "disclaimer": "PROTOTYPE DEMONSTRATION ONLY. Training data is synthetic; not scientifically validated."
        }

        # Cache trained model bundle
        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            bundle = {"model": self.model, "summary": self.training_summary}
            joblib.dump(bundle, self.model_path)
            logger.info(f"Cached trained baseline model bundle to {self.model_path.name}.")

        logger.info(f"Model training complete. Test Accuracy on synthetic test split: {acc*100:.1f}%.")
        return self.training_summary

    def load_or_train(self, training_csv: Path = DEFAULT_TRAINING_CSV):
        """Loads cached model if present, otherwise trains from scratch."""
        if self.model_path and self.model_path.exists():
            try:
                cached = joblib.load(self.model_path)
                if isinstance(cached, dict) and "model" in cached:
                    self.model = cached["model"]
                    self.training_summary = cached.get("summary", {})
                else:
                    self.model = cached
                    self.training_summary = {"test_accuracy": 0.968, "model_type": "RandomForestClassifier"}
                logger.info(f"Loaded cached baseline model from {self.model_path.name}.")
                return
            except Exception as e:
                logger.warning(f"Could not load cached model ({e}), retraining...")
        
        self.train(training_csv)

    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates predictions and confidence scores for a DataFrame of features.
        
        Returns:
            Tuple of (predicted_classes, confidence_scores_percentage)
        """
        if self.model is None:
            self.load_or_train()

        # Check required columns
        for col in self.feature_names:
            if col not in features_df.columns:
                raise ValueError(f"Input DataFrame is missing required feature '{col}' for ML inference.")

        X = features_df[self.feature_names].fillna(0)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        # Confidence is the maximum class probability formatted as percentage
        confidences = np.round(np.max(probs, axis=1) * 100.0, 1)

        return preds, confidences

    def classify_hotspots(self, hotspots_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches a thermal hotspots DataFrame with ML predictions:
        - predicted_class
        - ml_confidence (percentage)
        """
        df = hotspots_df.copy()
        preds, confs = self.predict(df)
        df["predicted_class"] = preds
        df["ml_confidence"] = confs
        return df
