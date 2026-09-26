"""
Unit tests for baseline ML model training, evaluation, data leakage checks, and prediction.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from baseline.model import BustBaselineModel
from baseline.evaluate import evaluate_bust_predictions
from baseline.predict import prepare_features


@pytest.fixture
def mock_tabular_dataset():
    """Create mock tabular dataset for baseline model testing."""
    np.random.seed(42)
    n = 500

    lead_h = np.random.choice([24, 48, 72, 96, 120], size=n).astype(float)
    lats = np.random.uniform(0, 40, size=n).astype(float)
    lons = np.random.uniform(60, 100, size=n).astype(float)
    fcst_p = np.random.uniform(0, 50, size=n).astype(float)

    X = pd.DataFrame({
        "lead_time_hours": lead_h,
        "latitude": lats,
        "longitude": lons,
        "forecast_precipitation": fcst_p,
        "fcst_precip_sq": fcst_p ** 2,
        "fcst_precip_sqrt": np.sqrt(fcst_p),
        "fcst_precip_is_zero": (fcst_p == 0).astype(float),
        "fcst_precip_is_heavy": (fcst_p > 10).astype(float),
        "spatial_distance_to_center": np.sqrt((lats - 20)**2 + (lons - 80)**2),
        "lead_time_precip_interaction": fcst_p * lead_h,
    })

    # Bust target depends partly on heavy rain + lead time
    p_bust = 1.0 / (1.0 + np.exp(-(fcst_p * 0.05 + lead_h * 0.01 - 2.0)))
    y = (np.random.uniform(0, 1, size=n) < p_bust).astype(int)

    return X, pd.Series(y)


def test_model_training_and_predict(mock_tabular_dataset):
    X, y = mock_tabular_dataset
    model = BustBaselineModel(n_estimators=10, max_depth=3)
    model.fit(X, y)

    assert model.is_trained is True

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert preds.shape == (len(X),)
    assert probs.shape == (len(X), 2)
    assert ((probs[:, 1] >= 0.0) & (probs[:, 1] <= 1.0)).all()


def test_evaluation_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.2, 0.6])

    metrics = evaluate_bust_predictions(y_true, y_probs, threshold=0.5)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert metrics["roc_auc"] > 0.8
    assert metrics["precision"] > 0.5


def test_prepare_features_auto_engineering():
    sample_raw = {
        "lead_time_hours": 48.0,
        "latitude": 15.0,
        "longitude": 75.0,
        "forecast_precipitation": 20.0,
    }

    df_prepared = prepare_features(sample_raw)

    assert "fcst_precip_sq" in df_prepared.columns
    assert "lead_time_precip_interaction" in df_prepared.columns
    assert df_prepared["fcst_precip_sq"].iloc[0] == 400.0
    assert df_prepared["lead_time_precip_interaction"].iloc[0] == 960.0
