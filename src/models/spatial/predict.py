"""
Standalone Spatial Inference Module for PyTorch BustSpatialCNN Model.
Generates 2D spatial bust probability maps and confidence grids.
"""

import json
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "spatial"))

from cnn import BustSpatialCNN
from dataset import prepare_spatial_tensor_sample

CNN_MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "cnn" / "spatial_cnn_model.pt"
_cached_cnn = None


def load_spatial_cnn_model() -> BustSpatialCNN:
    """Load or return cached PyTorch spatial CNN model instance."""
    global _cached_cnn
    if _cached_cnn is None:
        if not CNN_MODEL_PATH.exists():
            raise FileNotFoundError(f"PyTorch CNN model state dict not found at {CNN_MODEL_PATH}. Run train.py first.")

        model = BustSpatialCNN(in_channels=5, hidden_dim=32)
        state_dict = torch.load(CNN_MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        _cached_cnn = model

    return _cached_cnn


def predict_spatial_bust_map(
    fcst_precip_2d: np.ndarray,
    lead_time_hours: float,
    lats_1d: np.ndarray,
    lons_1d: np.ndarray,
    max_precip: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D spatial bust probability map and 2D confidence grid using PyTorch Spatial CNN.
    Returns (prob_map_2d, confidence_map_2d).
    """
    model = load_spatial_cnn_model()

    tensor_5d = prepare_spatial_tensor_sample(
        fcst_precip_2d, lead_time_hours, lats_1d, lons_1d, max_precip=max_precip
    )
    input_batch = tensor_5d.unsqueeze(0)  # Shape: (1, 5, H, W)

    with torch.no_grad():
        probs_tensor = model.predict_proba(input_batch)  # Shape: (1, 1, H, W)

    prob_map = probs_tensor.squeeze().cpu().numpy().astype(np.float32)
    conf_map = (np.abs(prob_map - 0.5) * 2.0).astype(np.float32)

    return prob_map, conf_map


def predict_single_point_cnn(
    lead_time_hours: float,
    latitude: float,
    longitude: float,
    forecast_precipitation: float,
    lats_grid: np.ndarray | None = None,
    lons_grid: np.ndarray | None = None,
) -> dict:
    """
    Run spatial CNN inference for a single scalar coordinate point.
    Constructs a local spatial neighborhood context for the point.
    """
    if lats_grid is None:
        lats_grid = np.linspace(0, 40, 161)
    if lons_grid is None:
        lons_grid = np.linspace(60, 100, 161)

    # Find closest grid index
    lat_idx = int(np.argmin(np.abs(lats_grid - latitude)))
    lon_idx = int(np.argmin(np.abs(lons_grid - longitude)))

    # Construct spatial background field
    fcst_2d = np.zeros((len(lats_grid), len(lons_grid)), dtype=np.float32)
    fcst_2d[lat_idx, lon_idx] = forecast_precipitation

    prob_map, conf_map = predict_spatial_bust_map(
        fcst_2d, lead_time_hours, lats_grid, lons_grid
    )

    prob_val = float(np.round(prob_map[lat_idx, lon_idx], 4))
    conf_val = float(np.round(conf_map[lat_idx, lon_idx], 4))
    is_bust = prob_val >= 0.5

    if prob_val < 0.3:
        risk = "Low Risk"
    elif prob_val < 0.6:
        risk = "Moderate Risk"
    else:
        risk = "High Bust Risk"

    return {
        "bust_probability": prob_val,
        "confidence": conf_val,
        "is_bust_predicted": is_bust,
        "risk_category": risk,
    }


def main():
    lats = np.linspace(0, 40, 161)
    lons = np.linspace(60, 100, 161)
    fcst_dummy = np.random.uniform(0, 40, size=(161, 161)).astype(np.float32)

    print("Testing PyTorch Spatial CNN prediction on 161x161 grid...")
    probs, confs = predict_spatial_bust_map(fcst_dummy, 48.0, lats, lons)
    print(f"Output probability map shape: {probs.shape} | Range: [{probs.min():.4f}, {probs.max():.4f}]")


if __name__ == "__main__":
    main()
