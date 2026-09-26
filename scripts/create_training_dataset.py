"""
Training Dataset Generation Pipeline.
Integrates forecast error calculation and bust labeling to construct the final
supervised learning training dataset (containing forecast, verification, error metrics, and bust labels).
Saves output to data/processed/training_dataset_YYYYMMDD.nc.
"""

import argparse
import json
from pathlib import Path
import xarray as xr

from config import (
    PROCESSED_DATA_DIR,
    METADATA_DIR,
    ensure_directories,
)
from calculate_error import calculate_forecast_errors
from define_bust import apply_bust_labeling


def generate_training_dataset(input_nc_path: Path, output_nc_path: Path | None = None) -> xr.Dataset:
    """
    Generate supervised learning training dataset from aligned NetCDF file.
    """
    ensure_directories()
    print(f"Loading aligned dataset from {input_nc_path.name}...")
    ds_aligned = xr.open_dataset(input_nc_path)

    # Step 1: Calculate Errors
    print("Calculating forecast errors (absolute error, raw error)...")
    ds_err, err_stats = calculate_forecast_errors(ds_aligned)

    # Step 2: Apply Bust Threshold Labeling
    print("Applying configurable bust threshold labeling...")
    ds_training, bust_summary = apply_bust_labeling(ds_err)

    # Clean up dimensions and attrs for final training dataset
    final_attrs = dict(ds_aligned.attrs)
    final_attrs.update({
        "dataset_type": "Supervised ML Training Dataset for Forecast Bust Detection",
        "bust_threshold_applied_mm": float(bust_summary["threshold_applied_mm"]),
        "bust_threshold_justification": bust_summary["threshold_justification"],
        "class_imbalance_ratio": bust_summary["class_distribution"]["imbalance_ratio"],
        "total_training_samples": bust_summary["class_distribution"]["total_samples"],
        "num_bust_samples": bust_summary["class_distribution"]["num_busts"],
        "num_non_bust_samples": bust_summary["class_distribution"]["num_non_busts"],
    })
    ds_training.attrs = final_attrs

    # Determine Output Path
    if output_nc_path is None:
        init_time_str = ds_aligned.attrs.get("initialization_time", "20240601")[:10].replace("-", "")
        output_nc_path = PROCESSED_DATA_DIR / f"training_dataset_{init_time_str}.nc"

    print(f"Saving final training dataset to {output_nc_path.resolve()}...")
    ds_training.to_netcdf(output_nc_path)

    # Save complete summary metadata
    full_summary = {
        "dataset_file": output_nc_path.name,
        "error_statistics": err_stats,
        "bust_distribution": bust_summary,
    }

    summary_json_path = METADATA_DIR / "training_dataset_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"Saved training dataset summary metadata to {summary_json_path.resolve()}")
    return ds_training


def main():
    parser = argparse.ArgumentParser(description="Create final ML training dataset with error & bust target labels.")
    parser.add_argument("--input-nc", type=str, help="Path to input aligned NetCDF file")
    parser.add_argument("--output-nc", type=str, help="Path to output training NetCDF file")
    args = parser.parse_args()

    nc_files = list(PROCESSED_DATA_DIR.glob("aligned_forecast_verification_*.nc"))
    if not nc_files:
        print("No aligned NetCDF dataset found in data/processed/. Run align_data.py first.")
        return

    input_path = Path(args.input_nc) if args.input_nc else nc_files[0]
    output_path = Path(args.output_nc) if args.output_nc else None

    ds_train = generate_training_dataset(input_path, output_path)

    print("\n==================================================")
    print("FINAL TRAINING DATASET SPECIFICATIONS")
    print("==================================================")
    print(f"Dimensions: {dict(ds_train.sizes)}")
    print(f"Variables: {list(ds_train.data_vars.keys())}")
    print(f"Coordinates: {list(ds_train.coords.keys())}")
    print("\nAttributes:")
    for k, v in ds_train.attrs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
