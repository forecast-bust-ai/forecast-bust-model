"""
Unit tests for weather data preprocessing and alignment modules.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess_gfs import standardize_gfs_dataset, detect_coordinate_names, detect_precip_variable
from preprocess_era5 import preprocess_era5_dataset
from align_data import validate_aligned_dataset, align_gfs_and_era5
from config import REGION_BBOX


@pytest.fixture
def mock_raw_gfs_dataset():
    """Create mock raw GFS dataset with 0..360 longitude and descending latitude."""
    lats = np.linspace(90, -90, 721)
    lons = np.linspace(0, 359.75, 1440)
    init_time = np.datetime64("2024-06-01T00:00:00")
    valid_time = np.datetime64("2024-06-02T00:00:00")
    step = np.timedelta64(24, "h")

    # Generate dummy precipitation field
    data = np.random.uniform(0, 50, size=(721, 1440)).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "tp": (["latitude", "longitude"], data, {"units": "kg m**-2", "long_name": "Total Precipitation"})
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "time": init_time,
            "valid_time": valid_time,
            "step": step,
        }
    )
    return ds


@pytest.fixture
def mock_raw_era5_dataset():
    """Create mock raw ERA5 dataset with meters (m) precipitation units."""
    lats = np.linspace(40, 0, 161)
    lons = np.linspace(60, 100, 161)
    time = np.datetime64("2024-06-02T00:00:00")

    # Generate dummy ERA5 precipitation in meters
    data_m = np.random.uniform(0, 0.05, size=(161, 161)).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "total_precipitation": (["latitude", "longitude"], data_m, {"units": "m", "long_name": "Total Precipitation"})
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "time": (["time"], [time]),
        }
    )
    return ds


def test_detect_coordinate_names(mock_raw_gfs_dataset):
    coords = detect_coordinate_names(mock_raw_gfs_dataset)
    assert coords["latitude"] == "latitude"
    assert coords["longitude"] == "longitude"
    assert coords["time"] == "time"
    assert coords["valid_time"] == "valid_time"
    assert coords["step"] == "step"


def test_detect_precip_variable(mock_raw_gfs_dataset):
    var = detect_precip_variable(mock_raw_gfs_dataset)
    assert var == "tp"


def test_standardize_gfs_dataset(mock_raw_gfs_dataset):
    clean_ds = standardize_gfs_dataset(mock_raw_gfs_dataset, bbox=REGION_BBOX)

    # Check variable name
    assert "forecast_precipitation" in clean_ds.data_vars

    # Check units
    assert clean_ds["forecast_precipitation"].attrs["units"] == "mm"

    # Check bounding box limits
    lats = clean_ds["latitude"].values
    lons = clean_ds["longitude"].values

    assert lats.min() >= REGION_BBOX["south"]
    assert lats.max() <= REGION_BBOX["north"]
    assert lons.min() >= REGION_BBOX["west"]
    assert lons.max() <= REGION_BBOX["east"]

    # Check latitude is ascending
    assert lats[0] < lats[-1]

    # Check attributes preserved
    assert clean_ds.attrs["lead_time_hours"] == 24
    assert clean_ds.attrs["valid_time"].startswith("2024-06-02T00:00:00")


def test_preprocess_era5_dataset(mock_raw_era5_dataset):
    clean_ds = preprocess_era5_dataset(mock_raw_era5_dataset, bbox=REGION_BBOX)

    assert "verification_precipitation" in clean_ds.data_vars
    assert clean_ds["verification_precipitation"].attrs["units"] == "mm"

    # Verify unit conversion from m to mm (max value > 1.0 mm)
    assert clean_ds["verification_precipitation"].values.max() > 1.0


def test_validate_aligned_dataset_pass(mock_raw_gfs_dataset):
    clean_gfs = standardize_gfs_dataset(mock_raw_gfs_dataset)
    aligned_ds = align_gfs_and_era5([clean_gfs])

    report = validate_aligned_dataset(aligned_ds)
    assert report["passed_all_checks"] is True
    assert report["checks"]["no_duplicate_timestamps"] is True
    assert report["checks"]["valid_lat_lon_range"] is True
    assert report["checks"]["no_impossible_values"] is True


def test_validate_aligned_dataset_fail_negative_values(mock_raw_gfs_dataset):
    clean_gfs = standardize_gfs_dataset(mock_raw_gfs_dataset)
    aligned_ds = align_gfs_and_era5([clean_gfs])

    # Inject invalid negative precipitation value
    aligned_ds["forecast_precipitation"].values[0, 0, 0] = -10.0

    report = validate_aligned_dataset(aligned_ds)
    assert report["passed_all_checks"] is False
    assert report["checks"]["no_impossible_values"] is False
