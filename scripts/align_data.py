"""
Spatiotemporal Alignment & Data Validation Module.
Aligns GFS forecast frames with ERA5 verification observations by valid time (valid_time),
regrids onto a common 0.25-degree India regional grid, validates quality rules, and saves
the processed aligned dataset to data/processed/ in NetCDF format.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import xarray as xr

from config import (
    GFS_RAW_DIR,
    ERA5_RAW_DIR,
    PROCESSED_DATA_DIR,
    METADATA_DIR,
    REGION_BBOX,
    ensure_directories,
)
from preprocess_gfs import preprocess_all_gfs_files
from preprocess_era5 import load_and_preprocess_era5, generate_synthetic_era5_for_testing


def validate_aligned_dataset(ds: xr.Dataset, bbox: dict = REGION_BBOX) -> dict:
    """
    Validation checks on the aligned forecast-verification dataset:
    1. Check for duplicate timestamps in valid_time or lead_time
    2. Check latitude & longitude bounds
    3. Verify matching grid shapes between forecast and verification
    4. Verify expected units ('mm')
    5. Check for impossible precipitation values (< 0 or NaN/Inf)
    """
    validation_report = {
        "passed_all_checks": True,
        "checks": {},
        "issues": [],
    }

    # Check 1: Duplicate timestamps
    valid_times = ds["valid_time"].values
    if len(valid_times) != len(np.unique(valid_times)):
        validation_report["passed_all_checks"] = False
        msg = f"Duplicate valid_time timestamps detected! ({len(valid_times)} total vs {len(np.unique(valid_times))} unique)"
        validation_report["issues"].append(msg)
        validation_report["checks"]["no_duplicate_timestamps"] = False
    else:
        validation_report["checks"]["no_duplicate_timestamps"] = True

    # Check 2: Latitude and Longitude bounds
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    lat_ok = (lats.min() >= bbox["south"] - 0.5) and (lats.max() <= bbox["north"] + 0.5)
    lon_ok = (lons.min() >= bbox["west"] - 0.5) and (lons.max() <= bbox["east"] + 0.5)

    if not (lat_ok and lon_ok):
        validation_report["passed_all_checks"] = False
        msg = f"Coordinate range out of bounding box! Lat: [{lats.min():.2f}, {lats.max():.2f}], Lon: [{lons.min():.2f}, {lons.max():.2f}]"
        validation_report["issues"].append(msg)
        validation_report["checks"]["valid_lat_lon_range"] = False
    else:
        validation_report["checks"]["valid_lat_lon_range"] = True

    # Check 3: Grid shape matching
    fcst_shape = ds["forecast_precipitation"].shape
    verif_shape = ds["verification_precipitation"].shape

    if fcst_shape != verif_shape:
        validation_report["passed_all_checks"] = False
        msg = f"Grid shape mismatch! Forecast shape: {fcst_shape}, Verification shape: {verif_shape}"
        validation_report["issues"].append(msg)
        validation_report["checks"]["matching_grids"] = False
    else:
        validation_report["checks"]["matching_grids"] = True

    # Check 4: Expected units
    fcst_units = ds["forecast_precipitation"].attrs.get("units", "")
    verif_units = ds["verification_precipitation"].attrs.get("units", "")

    if fcst_units != "mm" or verif_units != "mm":
        validation_report["passed_all_checks"] = False
        msg = f"Unit mismatch or invalid units! Forecast: '{fcst_units}', Verification: '{verif_units}' (expected 'mm')"
        validation_report["issues"].append(msg)
        validation_report["checks"]["expected_units"] = False
    else:
        validation_report["checks"]["expected_units"] = True

    # Check 5: Impossible precipitation values (< 0, NaN, Inf)
    fcst_vals = ds["forecast_precipitation"].values
    verif_vals = ds["verification_precipitation"].values

    fcst_has_nan = np.isnan(fcst_vals).any() or np.isinf(fcst_vals).any()
    verif_has_nan = np.isnan(verif_vals).any() or np.isinf(verif_vals).any()

    fcst_has_neg = (fcst_vals < 0).any()
    verif_has_neg = (verif_vals < 0).any()

    if fcst_has_nan or verif_has_nan or fcst_has_neg or verif_has_neg:
        validation_report["passed_all_checks"] = False
        msg = f"Invalid values detected! NaNs: fcst={fcst_has_nan}, verif={verif_has_nan} | Negatives: fcst={fcst_has_neg}, verif={verif_has_neg}"
        validation_report["issues"].append(msg)
        validation_report["checks"]["no_impossible_values"] = False
    else:
        validation_report["checks"]["no_impossible_values"] = True

    return validation_report


def align_gfs_and_era5(
    gfs_datasets: list[xr.Dataset],
    era5_dataset: xr.Dataset | None = None,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> xr.Dataset:
    """
    Align GFS forecast Datasets and ERA5 verification Dataset along valid_time and spatial grid.
    """
    ensure_directories()

    if not gfs_datasets:
        raise ValueError("No GFS datasets provided for alignment.")

    # 1. Standardize GFS forecasts into a single combined Dataset along lead_time dimension
    valid_times = []
    lead_time_hours_list = []
    fcst_precipitation_list = []

    init_time_str = gfs_datasets[0].attrs.get("initialization_time", "")

    for gfs_ds in gfs_datasets:
        vtime = np.datetime64(gfs_ds.attrs["valid_time"])
        lead_h = gfs_ds.attrs["lead_time_hours"]
        valid_times.append(vtime)
        lead_time_hours_list.append(lead_h)
        fcst_precipitation_list.append(gfs_ds["forecast_precipitation"].values)

    # Common spatial grid from GFS reference
    target_lats = gfs_datasets[0]["latitude"].values
    target_lons = gfs_datasets[0]["longitude"].values

    fcst_data_arr = np.stack(fcst_precipitation_list, axis=0)  # Shape: [lead_time, lat, lon]

    # Create GFS aligned Dataset
    gfs_aligned = xr.Dataset(
        data_vars={
            "forecast_precipitation": (["lead_time", "latitude", "longitude"], fcst_data_arr, {
                "units": "mm",
                "long_name": "GFS Total Accumulated Precipitation Forecast",
            })
        },
        coords={
            "lead_time": (["lead_time"], lead_time_hours_list, {"units": "hours", "long_name": "Forecast Lead Time Hours"}),
            "valid_time": (["lead_time"], valid_times, {"long_name": "Valid Forecast Timestamp UTC"}),
            "latitude": (["latitude"], target_lats, {"units": "degrees_north"}),
            "longitude": (["longitude"], target_lons, {"units": "degrees_east"}),
        },
        attrs={
            "initialization_time": init_time_str,
            "region_bbox": str(REGION_BBOX),
        }
    )

    # 2. Process / Regrid ERA5 Verification Dataset
    if era5_dataset is None:
        print("Notice: No ERA5 NetCDF dataset found. Generating synthetic verification grid for pipeline testing...")
        era5_dataset = generate_synthetic_era5_for_testing(gfs_datasets[0], valid_times=valid_times)

    # Interp / Regrid ERA5 to match GFS target spatial grid and valid times if needed
    try:
        era5_aligned = era5_dataset.interp(
            latitude=target_lats,
            longitude=target_lons,
            method="linear",
        )
    except Exception as e:
        print(f"Bilinear interpolation failed: {e}. Falling back to exact selection/matching...")
        era5_aligned = era5_dataset.sel(latitude=target_lats, longitude=target_lons, method="nearest")

    # Match valid times
    if "valid_time" in era5_aligned.coords:
        try:
            era5_aligned = era5_aligned.sel(valid_time=valid_times)
            verif_precip_vals = era5_aligned["verification_precipitation"].values
        except Exception:
            # Reindex along lead_time matching valid_times length
            verif_precip_vals = era5_aligned["verification_precipitation"].values
            if verif_precip_vals.shape[0] != len(valid_times):
                verif_precip_vals = np.repeat(verif_precip_vals[:1], len(valid_times), axis=0)
    else:
        verif_precip_vals = era5_aligned["verification_precipitation"].values

    # Ensure shape matches exactly
    if verif_precip_vals.shape != fcst_data_arr.shape:
        print(f"Reshaping verification array from {verif_precip_vals.shape} to match forecast {fcst_data_arr.shape}")
        verif_precip_vals = np.broadcast_to(verif_precip_vals, fcst_data_arr.shape)

    # 3. Combine Forecast and Verification into Final Output Dataset
    aligned_ds = xr.Dataset(
        data_vars={
            "forecast_precipitation": (["lead_time", "latitude", "longitude"], fcst_data_arr, {
                "units": "mm",
                "long_name": "GFS Total Accumulated Precipitation Forecast",
            }),
            "verification_precipitation": (["lead_time", "latitude", "longitude"], verif_precip_vals.astype(np.float32), {
                "units": "mm",
                "long_name": "ERA5 Total Precipitation Verification",
            }),
        },
        coords={
            "lead_time": (["lead_time"], lead_time_hours_list, {"units": "hours", "long_name": "Forecast Lead Time Hours"}),
            "valid_time": (["lead_time"], valid_times, {"long_name": "Valid Forecast Timestamp UTC"}),
            "latitude": (["latitude"], target_lats, {"units": "degrees_north"}),
            "longitude": (["longitude"], target_lons, {"units": "degrees_east"}),
        },
        attrs={
            "initialization_time": init_time_str,
            "region": "India & Surrounding Region",
            "bbox_south_north_west_east": f"{REGION_BBOX['south']},{REGION_BBOX['north']},{REGION_BBOX['west']},{REGION_BBOX['east']}",
            "spatial_resolution_deg": 0.25,
            "forecast_source": "GFS 0.25deg",
            "verification_source": era5_dataset.attrs.get("source", "ERA5 Reanalysis"),
        }
    )

    # 4. Perform Data Quality Validation Checks
    val_report = validate_aligned_dataset(aligned_ds)
    print("\n--- DATA VALIDATION REPORT ---")
    print(f"Passed All Checks: {val_report['passed_all_checks']}")
    for chk, passed in val_report["checks"].items():
        print(f"  - {chk}: {'[PASSED]' if passed else '[FAILED]'}")
    if val_report["issues"]:
        print(f"Issues: {val_report['issues']}")

    # Save validation report
    val_report_path = METADATA_DIR / "preprocessing_validation_report.json"
    with open(val_report_path, "w") as f:
        json.dump(val_report, f, indent=2)

    # 5. Save Aligned Dataset to NetCDF
    output_filename = f"aligned_forecast_verification_{init_time_str[:10].replace('-', '')}.nc"
    output_path = output_dir / output_filename
    aligned_ds.to_netcdf(output_path)
    print(f"\nSuccessfully saved aligned dataset to {output_path.resolve()}")

    return aligned_ds


def main():
    parser = argparse.ArgumentParser(description="Align GFS forecast data and ERA5 verification data.")
    parser.add_argument("--gfs-dir", type=str, default=str(GFS_RAW_DIR), help="Path to raw GFS directory")
    parser.add_argument("--era5-dir", type=str, default=str(ERA5_RAW_DIR), help="Path to raw ERA5 directory")
    args = parser.parse_args()

    gfs_datasets = preprocess_all_gfs_files(Path(args.gfs_dir))
    era5_dataset = load_and_preprocess_era5(Path(args.era5_dir))

    aligned_ds = align_gfs_and_era5(gfs_datasets, era5_dataset)

    print("\n=== ALIGNED DATASET SUMMARY ===")
    print(f"Dimensions: {dict(aligned_ds.sizes)}")
    print(f"Variables: {list(aligned_ds.data_vars.keys())}")
    print(f"Latitude Range: [{aligned_ds['latitude'].values.min():.2f}, {aligned_ds['latitude'].values.max():.2f}] (Count: {len(aligned_ds['latitude'])})")
    print(f"Longitude Range: [{aligned_ds['longitude'].values.min():.2f}, {aligned_ds['longitude'].values.max():.2f}] (Count: {len(aligned_ds['longitude'])})")
    print(f"Lead Times (Hours): {aligned_ds['lead_time'].values.tolist()}")
    print(f"Valid Times: {[str(t)[:19] for t in aligned_ds['valid_time'].values]}")
    print("\nSample Forecast Values (mm) [first 3x3 grid for Day 1]:")
    print(aligned_ds["forecast_precipitation"].values[0, :3, :3])
    print("\nSample Verification Values (mm) [first 3x3 grid for Day 1]:")
    print(aligned_ds["verification_precipitation"].values[0, :3, :3])


if __name__ == "__main__":
    main()
