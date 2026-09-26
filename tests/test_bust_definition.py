"""
Unit tests for forecast error calculation and bust target labeling logic.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from calculate_error import calculate_forecast_errors
from define_bust import determine_bust_threshold, apply_bust_labeling


@pytest.fixture
def mock_aligned_dataset():
    """Create mock aligned forecast and verification dataset."""
    lats = np.linspace(0, 40, 41)
    lons = np.linspace(60, 100, 41)
    lead_times = np.array([24, 48, 72, 96, 120])
    valid_times = np.array([np.datetime64(f"2024-06-0{i+2}T00:00:00") for i in range(5)])

    np.random.seed(123)
    fcst = np.random.uniform(0, 50, size=(5, 41, 41)).astype(np.float32)
    verif = fcst + np.random.normal(0, 10, size=(5, 41, 41)).astype(np.float32)
    verif = np.maximum(0.0, verif)

    ds = xr.Dataset(
        data_vars={
            "forecast_precipitation": (["lead_time", "latitude", "longitude"], fcst, {"units": "mm"}),
            "verification_precipitation": (["lead_time", "latitude", "longitude"], verif, {"units": "mm"}),
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


def test_calculate_forecast_errors(mock_aligned_dataset):
    ds_err, stats = calculate_forecast_errors(mock_aligned_dataset)

    assert "absolute_error" in ds_err.data_vars
    assert "raw_error" in ds_err.data_vars

    # Verify absolute_error is >= 0
    assert (ds_err["absolute_error"].values >= 0).all()

    # Verify overall MAE matches mean of absolute error array
    expected_mae = float(np.mean(ds_err["absolute_error"].values))
    assert pytest.approx(stats["overall_metrics"]["mae"], rel=1e-4) == expected_mae


def test_determine_bust_threshold_percentile():
    abs_errors = np.linspace(0, 100, 1001)
    threshold, desc = determine_bust_threshold(abs_errors, strategy="percentile", percentile_threshold=90.0)

    assert pytest.approx(threshold, rel=1e-2) == 90.0
    assert "90.0th percentile" in desc


def test_determine_bust_threshold_absolute():
    abs_errors = np.linspace(0, 100, 1001)
    threshold, desc = determine_bust_threshold(abs_errors, strategy="absolute", abs_threshold_mm=15.0)

    assert threshold == 15.0
    assert "15.0 mm" in desc


def test_apply_bust_labeling(mock_aligned_dataset):
    ds_err, _ = calculate_forecast_errors(mock_aligned_dataset)
    ds_labeled, summary = apply_bust_labeling(ds_err, threshold=15.0)

    assert "bust" in ds_labeled.data_vars
    bust_vals = ds_labeled["bust"].values

    # Check binary labels 0 and 1 only
    assert set(np.unique(bust_vals)).issubset({0, 1})

    # Check that bust=1 corresponds to absolute_error >= 15.0
    abs_err_vals = ds_err["absolute_error"].values
    expected_busts = (abs_err_vals >= 15.0).astype(int)
    np.testing.assert_array_equal(bust_vals, expected_busts)

    # Check summary statistics match
    assert summary["threshold_applied_mm"] == 15.0
    assert summary["class_distribution"]["total_samples"] == mock_aligned_dataset["forecast_precipitation"].size
