"""
Baseline Machine Learning Model Class for Forecast Bust Prediction.
Uses XGBoost Classifier configured with class imbalance weighting (scale_pos_weight)
and early stopping.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class BustBaselineModel:
    """
    Supervised learning baseline classifier for predicting forecast bust probability.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 9.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self.feature_names = []
        self.is_trained = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ):
        """Train XGBoost model on feature matrix."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        elif not self.feature_names and hasattr(X_train, "shape"):
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        self.is_trained = True

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary bust classification (0 or 1)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call fit() or load() first.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict probability distribution.
        Returns array of shape (N, 2) where column 1 is P(bust=1).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call fit() or load() first.")
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importance dictionary sorted descending."""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        importance_map = dict(zip(self.feature_names, [float(v) for v in importances]))
        sorted_map = dict(sorted(importance_map.items(), key=lambda item: item[1], reverse=True))
        return sorted_map

    def save(self, filepath: Path | str):
        """Serialize trained model state to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "scale_pos_weight": self.scale_pos_weight,
            },
        }
        joblib.dump(state, filepath)
        print(f"Model successfully saved to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path | str) -> "BustBaselineModel":
        """Load trained model state from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        state = joblib.load(filepath)
        instance = cls(**state.get("params", {}))
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        instance.is_trained = state["is_trained"]
        return instance
