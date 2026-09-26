"""
GFS Forecast Data Preprocessing Module.
Loads raw GFS GRIB2 files, standardizes coordinates (latitude, longitude, time),
converts units to mm, clips to the India regional bounding box, and formats
forecast initialization and lead times.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import xarray as xr

from config import (
    GFS_RAW_DIR,
    PROCESSED_DATA_DIR,
    REGION_BBOX,
    ensure_directories,
)


def detect_coordinate_names(ds: xr.Dataset) -> dict:
    """
    Detect coordinate names for latitude, longitude, initialization time, valid time, and step.
    Does NOT assume hardcoded coordinate names.
    """
    coord_map = {
        "latitude": None,
        "longitude": None,
        "time": None,
        "valid_time": None,
        "step": None,
    }

    for name in list(ds.coords.keys()) + list(ds.sizes.keys()):
        name_lower = str(name).lower()
        if "lat" in name_lower and not coord_map["latitude"]:
            coord_map["latitude"] = name
        elif ("lon" in name_lower or "lng" in name_lower) and not coord_map["longitude"]:
            coord_map["longitude"] = name
        elif name_lower in ["time", "init_time", "forecast_time"] and not coord_map["time"]:
            coord_map["time"] = name
        elif "valid" in name_lower and not coord_map["valid_time"]:
            coord_map["valid_time"] = name
        elif "step" in name_lower or "lead" in name_lower:
            coord_map["step"] = name

    return coord_map


def detect_precip_variable(ds: xr.Dataset) -> str:
    """
    Detect precipitation variable name in dataset.
    """
    for var in ds.data_vars:
        var_lower = var.lower()
        attrs = str(ds[var].attrs).lower()
        if var_lower in ["tp", "apcp", "precip", "total_precipitation"] or "precipitation" in attrs:
            return var

    # Fallback to first data variable
    if len(ds.data_vars) > 0:
        return list(ds.data_vars.keys())[0]

    raise ValueError("No data variable found in dataset.")


def standardize_gfs_dataset(ds: xr.Dataset, bbox: dict = REGION_BBOX) -> xr.Dataset:
    """
    Preprocess and standardize a single GFS forecast Dataset:
    - Standardize lat/lon coordinates
    - Convert longitude to -180..180 format if needed
    - Subset to India bounding box
    - Standardize precipitation units to mm
    - Validate values (remove small negative values caused by numerical noise)
    """
    coord_map = detect_coordinate_names(ds)
    lat_name = coord_map["latitude"]
    lon_name = coord_map["longitude"]
    time_name = coord_map["time"]
    valid_time_name = coord_map["valid_time"]
    step_name = coord_map["step"]

    if not lat_name or not lon_name:
        raise ValueError(f"Could not identify lat/lon coordinates in dataset: {list(ds.coords.keys())}")

    precip_var = detect_precip_variable(ds)
    da = ds[precip_var]

    # 1. Handle Longitude convention (convert 0..360 to -180..180 if max lon > 180)
    lons = ds[lon_name].values
    if np.any(lons > 180.0):
        # Convert coordinates to -180..180
        lons_converted = np.where(lons > 180.0, lons - 360.0, lons)
        ds = ds.assign_coords({lon_name: lons_converted})

        # Sort along longitude if coordinates are no longer monotonic
        ds = ds.sortby(lon_name)

    # 2. Subset to Bounding Box
    lat_min, lat_max = bbox["south"], bbox["north"]
    lon_min, lon_max = bbox["west"], bbox["east"]

    # Handle latitude sorting (ascending or descending)
    lats = ds[lat_name].values
    if lats[0] > lats[-1]:
        ds_sub = ds.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})
    else:
        ds_sub = ds.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})

    # Sort latitude to always be ascending (0 to 40)
    ds_sub = ds_sub.sortby(lat_name)

    # 3. Extract & Standardize Variable Data
    da_sub = ds_sub[precip_var]

    # Convert units to mm (kg m**-2 is equal to mm for water)
    # Ensure no negative numerical noise (e.g. -1e-9 -> 0.0)
    vals = np.maximum(0.0, da_sub.values.astype(np.float32))

    # 4. Standardize Timestamps
    init_time = ds_sub[time_name].values if time_name and time_name in ds_sub.coords else np.datetime64("NaT")
    valid_time = ds_sub[valid_time_name].values if valid_time_name and valid_time_name in ds_sub.coords else np.datetime64("NaT")

    # Extract lead time in hours
    if step_name and step_name in ds_sub.coords:
        step_val = ds_sub[step_name].values
        if isinstance(step_val, np.timedelta64):
            lead_time_hours = int(step_val / np.timedelta64(1, "h"))
        else:
            lead_time_hours = int(step_val)
    else:
        lead_time_hours = int((valid_time - init_time) / np.timedelta64(1, "h")) if (valid_time != np.datetime64("NaT") and init_time != np.datetime64("NaT")) else 0

    # 5. Reconstruct Clean Dataset with Standardized Coordinates
    clean_ds = xr.Dataset(
        data_vars={
            "forecast_precipitation": (["latitude", "longitude"], vals, {
                "units": "mm",
                "long_name": "GFS Total Accumulated Precipitation",
                "description": "Precipitation accumulation over lead time window"
            })
        },
        coords={
            "latitude": (["latitude"], ds_sub[lat_name].values, {"units": "degrees_north"}),
            "longitude": (["longitude"], ds_sub[lon_name].values, {"units": "degrees_east"}),
        },
        attrs={
            "initialization_time": str(init_time),
            "valid_time": str(valid_time),
            "lead_time_hours": lead_time_hours,
            "source": "GFS 0.25deg",
            "region_bbox": str(bbox),
        }
    )

    return clean_ds


def preprocess_all_gfs_files(raw_dir: Path = GFS_RAW_DIR) -> list[xr.Dataset]:
    """
    Process all raw GFS files in directory and return list of standardized datasets.
    """
    ensure_directories()
    gfs_files = sorted([p for p in raw_dir.glob("*.grib2") if not p.name.endswith(".idx")])

    if not gfs_files:
        print(f"No GFS GRIB2 files found in {raw_dir}")
        return []

    processed_list = []
    print(f"Found {len(gfs_files)} raw GFS files for preprocessing...")

    for fpath in gfs_files:
        try:
            raw_ds = xr.open_dataset(fpath, engine="cfgrib")
            clean_ds = standardize_gfs_dataset(raw_ds)
            processed_list.append(clean_ds)
            print(f"  [OK] Preprocessed {fpath.name} | Lead Time: {clean_ds.attrs['lead_time_hours']}h | Valid Time: {clean_ds.attrs['valid_time']}")
        except Exception as e:
            print(f"  [ERROR] Failed to preprocess {fpath.name}: {e}")

    return processed_list


def main():
    parser = argparse.ArgumentParser(description="Preprocess GFS forecast data.")
    parser.add_argument("--gfs-dir", type=str, default=str(GFS_RAW_DIR), help="Path to raw GFS directory")
    args = parser.parse_args()

    datasets = preprocess_all_gfs_files(Path(args.gfs_dir))
    print(f"\nPreprocessing complete. Successfully standardized {len(datasets)} GFS forecast frames.")


if __name__ == "__main__":
    main()
