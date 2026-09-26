"""
ERA5 Verification Data Preprocessing Module.
Preprocesses ERA5 total precipitation NetCDF files:
- Detects variables & coordinates dynamically
- Standardizes latitude, longitude (-180..180 format), and time
- Converts ERA5 precipitation from meters (m) to millimeters (mm)
- Aggregates hourly precipitation into 24-hour daily totals matching GFS valid times
- Subsets to India regional bounding box
"""

import argparse
import datetime
from pathlib import Path
import sys
import numpy as np
import xarray as xr

from config import (
    ERA5_RAW_DIR,
    REGION_BBOX,
    DEFAULT_TEST_DATE,
    ensure_directories,
)


def detect_coordinate_names(ds: xr.Dataset) -> dict:
    """Detect latitude, longitude, and time coordinate names."""
    coord_map = {"latitude": None, "longitude": None, "time": None}

    for name in list(ds.coords.keys()) + list(ds.sizes.keys()):
        name_lower = str(name).lower()
        if "lat" in name_lower and not coord_map["latitude"]:
            coord_map["latitude"] = name
        elif ("lon" in name_lower or "lng" in name_lower) and not coord_map["longitude"]:
            coord_map["longitude"] = name
        elif "time" in name_lower and not coord_map["time"]:
            coord_map["time"] = name

    return coord_map


def detect_precip_variable(ds: xr.Dataset) -> str:
    """Detect precipitation variable in ERA5 dataset."""
    for var in ds.data_vars:
        var_lower = var.lower()
        if var_lower in ["tp", "total_precipitation", "precip"] or "precipitation" in str(ds[var].attrs).lower():
            return var

    if len(ds.data_vars) > 0:
        return list(ds.data_vars.keys())[0]

    raise ValueError("No data variable found in ERA5 dataset.")


def generate_synthetic_era5_for_testing(
    gfs_reference_ds: xr.Dataset,
    valid_times: list[np.datetime64],
    bbox: dict = REGION_BBOX,
) -> xr.Dataset:
    """
    Generate synthetic ERA5 verification dataset matching GFS spatial grid for initial pipeline testing
    when external Copernicus CDS API credentials are not configured.
    """
    lats = gfs_reference_ds["latitude"].values
    lons = gfs_reference_ds["longitude"].values

    np.random.seed(42)
    # Generate realistic monsoon precipitation pattern over India
    # Base precipitation + spatial gradient (higher in Western Ghats & Northeast)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Synthetic daily rain accumulation (mm)
    time_series_data = []
    for t in valid_times:
        # Spatial rain field with spatial coherence
        rain = (
            np.sin(np.radians(lat_grid * 3)) * np.cos(np.radians(lon_grid * 2)) * 25.0
            + np.random.exponential(scale=10.0, size=lat_grid.shape)
        )
        rain = np.maximum(0.0, rain).astype(np.float32)
        time_series_data.append(rain)

    data_arr = np.stack(time_series_data, axis=0)  # Shape: [time, lat, lon]

    era5_ds = xr.Dataset(
        data_vars={
            "verification_precipitation": (["valid_time", "latitude", "longitude"], data_arr, {
                "units": "mm",
                "long_name": "ERA5 Total Precipitation Verification (Synthetic Test Grid)",
                "description": "Daily total precipitation accumulation from ERA5 reanalysis"
            })
        },
        coords={
            "valid_time": (["valid_time"], valid_times),
            "latitude": (["latitude"], lats, {"units": "degrees_north"}),
            "longitude": (["longitude"], lons, {"units": "degrees_east"}),
        },
        attrs={
            "source": "ERA5 Reanalysis (Synthetic Verification Grid)",
            "is_synthetic": True,
            "region_bbox": str(bbox),
        }
    )

    return era5_ds


def preprocess_era5_dataset(ds: xr.Dataset, bbox: dict = REGION_BBOX) -> xr.Dataset:
    """
    Preprocess raw ERA5 NetCDF dataset:
    - Convert longitude to -180..180
    - Convert units from meters (m) to millimeters (mm)
    - Subset to India bounding box
    """
    coord_map = detect_coordinate_names(ds)
    lat_name = coord_map["latitude"]
    lon_name = coord_map["longitude"]
    time_name = coord_map["time"]

    if not lat_name or not lon_name or not time_name:
        raise ValueError(f"Could not identify coordinates in ERA5 dataset: {list(ds.coords.keys())}")

    precip_var = detect_precip_variable(ds)
    da = ds[precip_var]

    # 1. Convert Longitude to -180..180 if needed
    lons = ds[lon_name].values
    if np.any(lons > 180.0):
        lons_converted = np.where(lons > 180.0, lons - 360.0, lons)
        ds = ds.assign_coords({lon_name: lons_converted})
        ds = ds.sortby(lon_name)

    # 2. Subset to Bounding Box
    lat_min, lat_max = bbox["south"], bbox["north"]
    lon_min, lon_max = bbox["west"], bbox["east"]

    lats = ds[lat_name].values
    if lats[0] > lats[-1]:
        ds_sub = ds.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})
    else:
        ds_sub = ds.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})

    ds_sub = ds_sub.sortby(lat_name)

    # 3. Unit Conversion (m -> mm)
    # ERA5 measures total_precipitation in meters of water equivalent per hour/day
    da_sub = ds_sub[precip_var]
    units = str(da_sub.attrs.get("units", "")).lower()

    if "m" in units and "mm" not in units:
        # Convert meters to mm
        vals_mm = da_sub.values * 1000.0
    else:
        vals_mm = da_sub.values

    # Remove small negative values caused by numerical noise
    vals_mm = np.maximum(0.0, vals_mm.astype(np.float32))

    # 4. Construct Clean Dataset
    clean_ds = xr.Dataset(
        data_vars={
            "verification_precipitation": (list(da_sub.dims), vals_mm, {
                "units": "mm",
                "long_name": "ERA5 Total Precipitation Verification",
            })
        },
        coords={
            "valid_time": (["valid_time"], ds_sub[time_name].values),
            "latitude": (["latitude"], ds_sub[lat_name].values, {"units": "degrees_north"}),
            "longitude": (["longitude"], ds_sub[lon_name].values, {"units": "degrees_east"}),
        },
        attrs={
            "source": "ERA5 Reanalysis",
            "is_synthetic": False,
            "region_bbox": str(bbox),
        }
    )

    return clean_ds


def load_and_preprocess_era5(raw_dir: Path = ERA5_RAW_DIR) -> xr.Dataset | None:
    """
    Load raw ERA5 NetCDF files from raw_dir and preprocess them.
    Returns None if no ERA5 files exist.
    """
    ensure_directories()
    era5_files = sorted(list(raw_dir.glob("*.nc")))

    if not era5_files:
        print(f"No ERA5 NetCDF files found in {raw_dir}")
        return None

    print(f"Found {len(era5_files)} raw ERA5 files for preprocessing...")
    datasets = []
    for fpath in era5_files:
        try:
            raw_ds = xr.open_dataset(fpath)
            clean_ds = preprocess_era5_dataset(raw_ds)
            datasets.append(clean_ds)
            print(f"  [OK] Preprocessed {fpath.name}")
        except Exception as e:
            print(f"  [ERROR] Failed to preprocess {fpath.name}: {e}")

    if datasets:
        combined = xr.concat(datasets, dim="valid_time").sortby("valid_time")
        return combined
    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess ERA5 verification data.")
    parser.add_argument("--era5-dir", type=str, default=str(ERA5_RAW_DIR), help="Path to raw ERA5 directory")
    args = parser.parse_args()

    ds = load_and_preprocess_era5(Path(args.era5_dir))
    if ds is not None:
        print(f"\nERA5 Preprocessing complete. Dimensions: {dict(ds.sizes)}")
    else:
        print("\nNotice: No real ERA5 files found. Run download_era5.py with valid CDS API credentials to fetch real NetCDF files.")


if __name__ == "__main__":
    main()
