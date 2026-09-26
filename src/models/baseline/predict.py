"""
Standalone Inference API for Forecast Bust Probability Prediction.
Exposes predict_bust_probability(features) function returning probability and confidence score.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "baseline"))

from model import BustBaselineModel

MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "baseline_xgboost_model.joblib"
_cached_model = None


def load_inference_model() -> BustBaselineModel:
    """Load or return cached trained baseline model instance."""
    global _cached_model
    if _cached_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model file not found at {MODEL_PATH}. "
                "Please run 'python models/baseline/train.py' first."
            )
        _cached_model = BustBaselineModel.load(MODEL_PATH)
    return _cached_model


def prepare_features(features: dict | pd.DataFrame | list[dict]) -> pd.DataFrame:
    """
    Format and engineer input features to match the exact schema used during training.
    """
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, list):
        df = pd.DataFrame(features)
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise ValueError("Input features must be a dict, list of dicts, or pandas DataFrame.")

    # Ensure required base columns exist
    req_base = ["lead_time_hours", "latitude", "longitude", "forecast_precipitation"]
    for col in req_base:
        if col not in df.columns:
            raise KeyError(f"Missing required base feature column: '{col}'")

    fcst_p = df["forecast_precipitation"].values.astype(np.float32)
    lats = df["latitude"].values.astype(np.float32)
    lons = df["longitude"].values.astype(np.float32)
    lead_h = df["lead_time_hours"].values.astype(np.float32)

    # Auto-engineer derived features if missing
    if "fcst_precip_sq" not in df.columns:
        df["fcst_precip_sq"] = fcst_p ** 2
    if "fcst_precip_sqrt" not in df.columns:
        df["fcst_precip_sqrt"] = np.sqrt(np.maximum(0.0, fcst_p))
    if "fcst_precip_is_zero" not in df.columns:
        df["fcst_precip_is_zero"] = (fcst_p == 0.0).astype(np.float32)
    if "fcst_precip_is_heavy" not in df.columns:
        df["fcst_precip_is_heavy"] = (fcst_p > 10.0).astype(np.float32)
    if "spatial_distance_to_center" not in df.columns:
        df["spatial_distance_to_center"] = np.sqrt((lats - 20.0)**2 + (lons - 80.0)**2)
    if "lead_time_precip_interaction" not in df.columns:
        df["lead_time_precip_interaction"] = fcst_p * lead_h

    # Ensure exact column ordering
    expected_order = [
        "lead_time_hours",
        "latitude",
        "longitude",
        "forecast_precipitation",
        "fcst_precip_sq",
        "fcst_precip_sqrt",
        "fcst_precip_is_zero",
        "fcst_precip_is_heavy",
        "spatial_distance_to_center",
        "lead_time_precip_interaction",
    ]
    return df[expected_order]


def predict_bust_probability(features: dict | pd.DataFrame | list[dict]) -> dict | list[dict]:
    """
    Predict forecast bust probability and model confidence score.
    Returns dict (if single sample) or list of dicts (if multiple samples).
    """
    model = load_inference_model()
    df_feat = prepare_features(features)

    probs = model.predict_proba(df_feat)[:, 1]

    results = []
    for prob in probs:
        prob_val = float(np.round(prob, 4))
        # Confidence score: rescaled distance from decision threshold 0.5
        confidence = float(np.round(abs(prob_val - 0.5) * 2.0, 4))
        is_bust = prob_val >= 0.5

        if prob_val < 0.3:
            risk = "Low Risk"
        elif prob_val < 0.6:
            risk = "Moderate Risk"
        else:
            risk = "High Bust Risk"

        results.append({
            "bust_probability": prob_val,
            "confidence": confidence,
            "is_bust_predicted": is_bust,
            "risk_category": risk,
        })

    return results[0] if isinstance(features, dict) else results


def main():
    # Sample test inference
    sample_feature = {
        "lead_time_hours": 48.0,
        "latitude": 19.0,
        "longitude": 73.0,
        "forecast_precipitation": 35.5,
    }

    print("Running sample prediction with features:", sample_feature)
    res = predict_bust_probability(sample_feature)
    print("Prediction Result:")
    print(res)


if __name__ == "__main__":
    main()
