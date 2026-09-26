"""
Forecast Error Calculation & Statistical Analysis Module.
Calculates absolute error, raw error, MAE, RMSE, percentiles, spatial error distribution,
and error distribution by forecast lead time.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr

from config import PROCESSED_DATA_DIR, METADATA_DIR


def calculate_forecast_errors(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """
    Compute absolute error and error metrics from aligned forecast and verification dataset.
    """
    fcst = ds["forecast_precipitation"]
    verif = ds["verification_precipitation"]

    # 1. Compute Element-wise Errors
    raw_error = fcst - verif
    abs_error = np.abs(raw_error)

    # Add error variables to dataset
    ds_err = ds.copy()
    ds_err["raw_error"] = (fcst.dims, raw_error.values.astype(np.float32), {
        "units": "mm",
        "long_name": "Forecast Error (Forecast - Verification)",
    })
    ds_err["absolute_error"] = (fcst.dims, abs_error.values.astype(np.float32), {
        "units": "mm",
        "long_name": "Absolute Forecast Error |Forecast - Verification|",
    })

    # 2. Flatten values for statistical analysis
    abs_err_flat = abs_error.values.flatten()
    raw_err_flat = raw_error.values.flatten()

    mae = float(np.mean(abs_err_flat))
    rmse = float(np.sqrt(np.mean(raw_err_flat ** 2)))
    bias = float(np.mean(raw_err_flat))

    percentiles = {
        "p50": float(np.percentile(abs_err_flat, 50)),
        "p75": float(np.percentile(abs_err_flat, 75)),
        "p90": float(np.percentile(abs_err_flat, 90)),
        "p95": float(np.percentile(abs_err_flat, 95)),
        "p99": float(np.percentile(abs_err_flat, 99)),
    }

    # 3. Analyze Error Distribution by Lead Time
    error_by_lead_time = {}
    lead_times = ds["lead_time"].values

    for idx, lt in enumerate(lead_times):
        # Convert timedelta64 or int to lead time hours integer
        if isinstance(lt, np.timedelta64):
            lt_h = int(lt / np.timedelta64(1, "h"))
        else:
            lt_h = int(lt)

        abs_slice = abs_error.values[idx].flatten()
        raw_slice = raw_error.values[idx].flatten()

        error_by_lead_time[f"Day_{lt_h // 24} ({lt_h}h)"] = {
            "lead_time_hours": lt_h,
            "mae": float(np.mean(abs_slice)),
            "rmse": float(np.sqrt(np.mean(raw_slice ** 2))),
            "p50": float(np.percentile(abs_slice, 50)),
            "p90": float(np.percentile(abs_slice, 90)),
            "p95": float(np.percentile(abs_slice, 95)),
            "p99": float(np.percentile(abs_slice, 99)),
            "max_error": float(np.max(abs_slice)),
        }

    # 4. Spatial Error Summary (Mean error across lead times per grid cell)
    spatial_mae = abs_error.mean(dim="lead_time").values

    stats_summary = {
        "total_samples": len(abs_err_flat),
        "overall_metrics": {
            "mae": mae,
            "rmse": rmse,
            "mean_bias": bias,
            "min_error": float(np.min(abs_err_flat)),
            "max_error": float(np.max(abs_err_flat)),
        },
        "percentile_distribution": percentiles,
        "error_by_lead_time": error_by_lead_time,
        "spatial_error_summary": {
            "min_grid_mae": float(np.min(spatial_mae)),
            "max_grid_mae": float(np.max(spatial_mae)),
            "mean_grid_mae": float(np.mean(spatial_mae)),
        },
    }

    return ds_err, stats_summary


def main():
    parser = argparse.ArgumentParser(description="Calculate forecast errors and statistical metrics.")
    parser.add_argument("--input-nc", type=str, help="Path to input aligned NetCDF dataset.")
    args = parser.parse_args()

    if args.input_nc:
        nc_path = Path(args.input_nc)
    else:
        nc_files = list(PROCESSED_DATA_DIR.glob("aligned_forecast_verification_*.nc"))
        if not nc_files:
            print("No aligned NetCDF file found in data/processed/. Run align_data.py first.")
            return
        nc_path = nc_files[0]

    print(f"Loading aligned dataset: {nc_path.name}...")
    ds = xr.open_dataset(nc_path)

    ds_err, stats = calculate_forecast_errors(ds)

    print("\n==================================================")
    print("HISTORICAL FORECAST ERROR STATISTICAL ANALYSIS")
    print("==================================================")
    print(f"Total Grid-Time Samples: {stats['total_samples']:,}")
    print(f"Overall MAE:  {stats['overall_metrics']['mae']:.3f} mm")
    print(f"Overall RMSE: {stats['overall_metrics']['rmse']:.3f} mm")
    print(f"Mean Bias:    {stats['overall_metrics']['mean_bias']:.3f} mm")
    print(f"Max Error:    {stats['overall_metrics']['max_error']:.3f} mm")

    print("\n--- PERCENTILE DISTRIBUTION ---")
    for p, val in stats["percentile_distribution"].items():
        print(f"  {p.upper()}: {val:.3f} mm")

    print("\n--- ERROR DISTRIBUTION BY LEAD TIME ---")
    for lt_name, metrics in stats["error_by_lead_time"].items():
        print(f"  {lt_name}: MAE = {metrics['mae']:.3f} mm | RMSE = {metrics['rmse']:.3f} mm | p90 = {metrics['p90']:.3f} mm")

    # Save error report
    report_path = METADATA_DIR / "error_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved error analysis report to {report_path.resolve()}")


if __name__ == "__main__":
    main()
