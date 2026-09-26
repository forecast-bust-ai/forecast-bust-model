"""
Baseline ML Training Pipeline for Forecast Bust Probability Prediction.
Loads processed training dataset, extracts strictly forecast-time features,
applies chronological time-based train/val/test split (Day 1-3 train, Day 4 val, Day 5 test),
trains XGBoost baseline model with class imbalance weighting, and evaluates metrics.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "baseline"))

from config import PROCESSED_DATA_DIR, METADATA_DIR
from model import BustBaselineModel
from evaluate import evaluate_bust_predictions, print_evaluation_summary

TRAINED_MODELS_DIR = PROJECT_ROOT / "models" / "trained"


def extract_features_and_target(ds: xr.Dataset) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature DataFrame and target Series strictly from forecast-time variables.
    DOES NOT USE verification precipitation or error metrics to prevent data leakage.
    """
    # 1. Flatten dataset dimensions into tabular DataFrame
    df = ds.to_dataframe().reset_index()

    # Convert timedelta lead_time to integer hours
    if "lead_time" in df.columns:
        if pd.api.types.is_timedelta64_dtype(df["lead_time"]):
            df["lead_time_hours"] = df["lead_time"].dt.total_seconds() / 3600.0
        else:
            df["lead_time_hours"] = df["lead_time"].astype(float)

    # 2. Extract Base Forecast Features
    fcst_p = df["forecast_precipitation"].values.astype(np.float32)
    lats = df["latitude"].values.astype(np.float32)
    lons = df["longitude"].values.astype(np.float32)
    lead_h = df["lead_time_hours"].values.astype(np.float32)

    # 3. Engineer Forecast Features (Strictly from forecast inputs)
    feature_df = pd.DataFrame({
        "lead_time_hours": lead_h,
        "latitude": lats,
        "longitude": lons,
        "forecast_precipitation": fcst_p,
        "fcst_precip_sq": fcst_p ** 2,
        "fcst_precip_sqrt": np.sqrt(np.maximum(0.0, fcst_p)),
        "fcst_precip_is_zero": (fcst_p == 0.0).astype(np.float32),
        "fcst_precip_is_heavy": (fcst_p > 10.0).astype(np.float32),
        "spatial_distance_to_center": np.sqrt((lats - 20.0)**2 + (lons - 80.0)**2),
        "lead_time_precip_interaction": fcst_p * lead_h,
    })

    target_series = df["bust"].astype(int)

    return feature_df, target_series


def train_baseline_pipeline(input_nc_path: Path):
    """
    Execute full baseline model training pipeline.
    """
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training dataset from {input_nc_path.name}...")
    ds = xr.open_dataset(input_nc_path)

    # 1. Feature Extraction & Leakage Check
    X, y = extract_features_and_target(ds)
    print(f"Extracted {X.shape[1]} features across {len(X):,} grid-time samples.")
    print(f"Feature List: {list(X.columns)}")

    # Data Leakage Verification
    forbidden_terms = ["verification", "observed", "error", "raw_error", "absolute_error"]
    for col in X.columns:
        for term in forbidden_terms:
            if term in col.lower():
                raise RuntimeError(f"DATA LEAKAGE DETECTED! Feature '{col}' contains forbidden input term '{term}'.")

    print("[VERIFIED] Zero data leakage: All features derived strictly from forecast inputs.")

    # 2. Time-Based Train / Validation / Test Split (Chronological along lead_time_hours)
    train_mask = X["lead_time_hours"] <= 72.0      # Day 1, Day 2, Day 3 (24h, 48h, 72h)
    val_mask = X["lead_time_hours"] == 96.0        # Day 4 (96h)
    test_mask = X["lead_time_hours"] == 120.0      # Day 5 (120h)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\n--- TIME-BASED DATASET SPLIT ---")
    print(f"  Train Set (Day 1-3 / 24-72h):  {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%) | Busts: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"  Val Set   (Day 4 / 96h):       {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%) | Busts: {y_val.sum():,} ({y_val.mean()*100:.1f}%)")
    print(f"  Test Set  (Day 5 / 120h):      {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%) | Busts: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

    # Calculate scale_pos_weight for XGBoost class weighting
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = float(num_neg / max(1, num_pos))
    print(f"  Configured XGBoost scale_pos_weight: {scale_pos_weight:.2f}x")

    # 3. Model Training
    print("\nTraining XGBoost Baseline Classifier...")
    model = BustBaselineModel(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # 4. Model Evaluation
    val_probs = model.predict_proba(X_val)[:, 1]
    val_metrics = evaluate_bust_predictions(y_val, val_probs)
    print_evaluation_summary(val_metrics, dataset_name="Validation (Day 4 / 96h)")

    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_bust_predictions(y_test, test_probs)
    print_evaluation_summary(test_metrics, dataset_name="Test (Day 5 / 120h)")

    # 5. Feature Importance Analysis
    importances = model.get_feature_importances()
    print(f"\n--- FEATURE IMPORTANCES ---")
    for feat, imp in importances.items():
        print(f"  {feat:<30}: {imp*100:.2f}%")

    # 6. Save Model and Metadata Artifacts
    model_path = TRAINED_MODELS_DIR / "baseline_xgboost_model.joblib"
    model.save(model_path)

    # Save feature configuration
    feature_config = {
        "feature_list": list(X.columns),
        "target_variable": "bust",
        "leakage_prevention": "Strict forecast-time features only",
        "scale_pos_weight_applied": scale_pos_weight,
    }
    feature_config_path = TRAINED_MODELS_DIR / "feature_config.json"
    with open(feature_config_path, "w") as f:
        json.dump(feature_config, f, indent=2)

    # Save evaluation metrics
    full_metrics_summary = {
        "model_type": "XGBClassifier Baseline",
        "split_strategy": "Time-based (Day 1-3 Train, Day 4 Val, Day 5 Test)",
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importances": importances,
    }
    metrics_path = TRAINED_MODELS_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(full_metrics_summary, f, indent=2)

    print(f"\nSaved training artifacts to {TRAINED_MODELS_DIR.resolve()}")
    return model, full_metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Train baseline XGBoost forecast bust classifier.")
    parser.add_argument("--input-nc", type=str, help="Path to input training NetCDF file")
    args = parser.parse_args()

    nc_files = list(PROCESSED_DATA_DIR.glob("training_dataset_*.nc"))
    if not nc_files:
        print("No training dataset found in data/processed/. Run create_training_dataset.py first.")
        return

    input_path = Path(args.input_nc) if args.input_nc else nc_files[0]
    train_baseline_pipeline(input_path)


if __name__ == "__main__":
    main()
