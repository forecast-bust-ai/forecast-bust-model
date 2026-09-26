"""
Training Pipeline for Spatial PyTorch CNN Model.
Applies chronological temporal split (Day 1-3 train, Day 4 validation, Day 5 test),
trains BustSpatialCNN with class-imbalance weighted BCE loss, and compares performance with XGBoost.
"""

import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "spatial"))

from config import PROCESSED_DATA_DIR
from dataset import ForecastBustSpatialDataset
from cnn import BustSpatialCNN
from evaluate import evaluate_spatial_predictions, print_model_comparison_table

CNN_TRAINED_DIR = PROJECT_ROOT / "models" / "trained" / "cnn"
XGB_TRAINED_DIR = PROJECT_ROOT / "models" / "trained" / "xgboost"
MODEL_TRAINED_ROOT = PROJECT_ROOT / "models" / "trained"


def train_spatial_cnn_pipeline(input_nc_path: Path):
    """
    Train PyTorch BustSpatialCNN model and evaluate against XGBoost baseline.
    """
    CNN_TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    XGB_TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {input_nc_path.name} for spatial CNN training...")
    ds = xr.open_dataset(input_nc_path, decode_timedelta=False)

    # 1. Chronological Temporal Split (Lead Times: Day 1-3 train, Day 4 val, Day 5 test)
    train_indices = [0, 1, 2]  # 24h, 48h, 72h (Day 1, 2, 3)
    val_indices = [3]         # 96h (Day 4)
    test_indices = [4]        # 120h (Day 5)

    print(f"\n--- CHRONOLOGICAL TEMPORAL SPLIT ---")
    print(f"  Training Period:   Lead times 24h, 48h, 72h (Day 1–3)")
    print(f"  Validation Period: Lead time 96h (Day 4)")
    print(f"  Testing Period:    Lead time 120h (Day 5)")

    train_dataset = ForecastBustSpatialDataset(ds, train_indices)
    val_dataset = ForecastBustSpatialDataset(ds, val_indices)
    test_dataset = ForecastBustSpatialDataset(ds, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 2. Model, Loss, Optimizer Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BustSpatialCNN(in_channels=5, hidden_dim=32).to(device)

    # Calculate class imbalance weighting ratio (pos_weight)
    train_targets_flat = np.concatenate([t.numpy().flatten() for t in train_dataset.targets])
    num_pos = np.sum(train_targets_flat == 1)
    num_neg = np.sum(train_targets_flat == 0)
    pos_weight_val = float(num_neg / max(1, num_pos))

    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

    print(f"\nInitialized BustSpatialCNN on device: {device}")
    print(f"Class Imbalance pos_weight: {pos_weight_val:.2f}x")

    # 3. Training Loop
    epochs = 35
    best_val_loss = float("inf")

    print("\nStarting PyTorch Spatial CNN Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(inputs)

        train_loss /= len(train_dataset)

        # Validation per epoch
        model.eval()
        with torch.no_grad():
            val_input, val_target = val_dataset[0]
            val_input = val_input.unsqueeze(0).to(device)
            val_target = val_target.unsqueeze(0).to(device)

            val_logits = model(val_input)
            val_loss = criterion(val_logits, val_target).item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 4. Out-of-Time Test Evaluation (Day 5 / 120h)
    model.eval()
    with torch.no_grad():
        test_input, test_target = test_dataset[0]
        test_input_tensor = test_input.unsqueeze(0).to(device)
        test_probs = model.predict_proba(test_input_tensor).squeeze().cpu().numpy()
        test_target_np = test_target.squeeze().numpy()

    cnn_test_metrics = evaluate_spatial_predictions(test_target_np, test_probs, threshold=0.5)

    # 5. Load Existing XGBoost Metrics for Comparison Table
    xgb_metrics_path = XGB_TRAINED_DIR / "baseline_metrics.json"
    if not xgb_metrics_path.exists():
        xgb_metrics_path = MODEL_TRAINED_ROOT / "baseline_metrics.json"

    if xgb_metrics_path.exists():
        with open(xgb_metrics_path, "r") as f:
            xgb_full = json.load(f)
            xgb_test_metrics = xgb_full.get("test_metrics", {})
    else:
        xgb_test_metrics = {}

    print_model_comparison_table(xgb_test_metrics, cnn_test_metrics)

    # 6. Save Spatial CNN Artifacts
    cnn_model_path = CNN_TRAINED_DIR / "spatial_cnn_model.pt"
    torch.save(model.state_dict(), cnn_model_path)
    print(f"\nSaved PyTorch Spatial CNN state dict to {cnn_model_path.resolve()}")

    # Save model configuration
    cnn_config = {
        "model_architecture": "BustSpatialCNN",
        "in_channels": 5,
        "channel_list": [
            "normalized_forecast_precipitation",
            "normalized_lead_time_hours",
            "normalized_latitude_grid",
            "normalized_longitude_grid",
            "spatial_precipitation_gradient",
        ],
        "training_period": "Lead times 24h, 48h, 72h (Day 1-3)",
        "validation_period": "Lead time 96h (Day 4)",
        "testing_period": "Lead time 120h (Day 5)",
        "pos_weight_applied": pos_weight_val,
        "max_precip_norm_factor": train_dataset.max_precip,
    }

    cnn_config_path = CNN_TRAINED_DIR / "cnn_config.json"
    with open(cnn_config_path, "w") as f:
        json.dump(cnn_config, f, indent=2)

    # Save metrics
    cnn_metrics_summary = {
        "model_type": "BustSpatialCNN (PyTorch)",
        "split_strategy": "Chronological (Day 1-3 Train, Day 4 Val, Day 5 Test)",
        "test_metrics": cnn_test_metrics,
        "comparison_with_xgboost": {
            "xgboost_roc_auc": xgb_test_metrics.get("roc_auc"),
            "cnn_roc_auc": cnn_test_metrics.get("roc_auc"),
            "xgboost_pr_auc": xgb_test_metrics.get("pr_auc"),
            "cnn_pr_auc": cnn_test_metrics.get("pr_auc"),
        }
    }
    cnn_metrics_path = CNN_TRAINED_DIR / "cnn_metrics.json"
    with open(cnn_metrics_path, "w") as f:
        json.dump(cnn_metrics_summary, f, indent=2)

    # Re-organize baseline XGBoost artifacts into models/trained/xgboost/ if needed
    baseline_joblib = MODEL_TRAINED_ROOT / "baseline_xgboost_model.joblib"
    if baseline_joblib.exists() and not (XGB_TRAINED_DIR / "baseline_xgboost_model.joblib").exists():
        import shutil
        shutil.copy(baseline_joblib, XGB_TRAINED_DIR / "baseline_xgboost_model.joblib")
        if (MODEL_TRAINED_ROOT / "feature_config.json").exists():
            shutil.copy(MODEL_TRAINED_ROOT / "feature_config.json", XGB_TRAINED_DIR / "feature_config.json")
        if (MODEL_TRAINED_ROOT / "baseline_metrics.json").exists():
            shutil.copy(MODEL_TRAINED_ROOT / "baseline_metrics.json", XGB_TRAINED_DIR / "baseline_metrics.json")

    print(f"Saved Spatial CNN artifacts to {CNN_TRAINED_DIR.resolve()}")
    return model, cnn_metrics_summary


def main():
    nc_files = list(PROCESSED_DATA_DIR.glob("training_dataset_*.nc"))
    if not nc_files:
        print("No training dataset found in data/processed/. Run create_training_dataset.py first.")
        return

    train_spatial_cnn_pipeline(nc_files[0])


if __name__ == "__main__":
    main()
