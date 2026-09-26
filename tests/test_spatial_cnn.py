"""
Unit tests for PyTorch Spatial CNN model, tensor dataset construction, and spatial prediction.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.spatial.dataset import prepare_spatial_tensor_sample, ForecastBustSpatialDataset
from models.spatial.cnn import BustSpatialCNN
from models.spatial.evaluate import evaluate_spatial_predictions


@pytest.fixture
def mock_spatial_dataset():
    """Create mock 3D spatial xarray dataset."""
    lats = np.linspace(0, 40, 21)
    lons = np.linspace(60, 100, 21)
    lead_times = np.array([24, 48, 72, 96, 120])
    valid_times = np.array([np.datetime64(f"2024-06-0{i+2}T00:00:00") for i in range(5)])

    np.random.seed(42)
    fcst = np.random.uniform(0, 50, size=(5, 21, 21)).astype(np.float32)
    bust_mask = (fcst > 30.0).astype(np.int32)

    ds = xr.Dataset(
        data_vars={
            "forecast_precipitation": (["lead_time", "latitude", "longitude"], fcst, {"units": "mm"}),
            "bust": (["lead_time", "latitude", "longitude"], bust_mask, {"units": "binary_indicator"}),
        },
        coords={
            "lead_time": (["lead_time"], lead_times),
            "valid_time": (["lead_time"], valid_times),
            "latitude": (["latitude"], lats),
            "longitude": (["longitude"], lons),
        },
        attrs={"initialization_time": "2024-06-01T00:00:00"}
    )
    return ds


def test_prepare_spatial_tensor_sample():
    fcst_2d = np.random.uniform(0, 50, size=(21, 21)).astype(np.float32)
    lats = np.linspace(0, 40, 21)
    lons = np.linspace(60, 100, 21)

    tensor = prepare_spatial_tensor_sample(fcst_2d, 48.0, lats, lons)

    assert tensor.shape == (5, 21, 21)
    assert tensor.dtype == torch.float32
    # Check lead time channel normalization (48 / 120 = 0.4)
    assert pytest.approx(float(tensor[1, 0, 0]), rel=1e-3) == 0.4


def test_cnn_architecture_forward():
    model = BustSpatialCNN(in_channels=5, hidden_dim=16)
    dummy_input = torch.randn(2, 5, 21, 21)

    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)

    assert logits.shape == (2, 1, 21, 21)
    assert probs.shape == (2, 1, 21, 21)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_spatial_dataset(mock_spatial_dataset):
    dataset = ForecastBustSpatialDataset(mock_spatial_dataset, lead_time_indices=[0, 1, 2])

    assert len(dataset) == 3
    sample, target = dataset[0]

    assert sample.shape == (5, 21, 21)
    assert target.shape == (1, 21, 21)


def test_spatial_evaluation():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = evaluate_spatial_predictions(y_true, y_probs)

    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
