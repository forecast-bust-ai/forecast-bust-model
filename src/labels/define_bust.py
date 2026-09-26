"""
Configurable Forecast Bust Labeling Module.
Derives forecast bust binary labels (bust=1, non-bust=0) based on historical error statistics.
Supports configurable percentile, absolute, and hybrid threshold strategies.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr

from config import (
    BUST_THRESHOLD_STRATEGY,
    BUST_ABSOLUTE_THRESHOLD_MM,
    BUST_PERCENTILE_THRESHOLD,
    METADATA_DIR,
)


def determine_bust_threshold(
    abs_errors: np.ndarray,
    strategy: str = BUST_THRESHOLD_STRATEGY,
    abs_threshold_mm: float = BUST_ABSOLUTE_THRESHOLD_MM,
    percentile_threshold: float = BUST_PERCENTILE_THRESHOLD,
) -> tuple[float, str]:
    """
    Determine threshold based on statistical analysis and strategy configuration.
    Returns (threshold_value, justification_text).
    """
    abs_flat = abs_errors.flatten()
    p_val = float(np.percentile(abs_flat, percentile_threshold))

    if strategy == "percentile":
        threshold = p_val
        justification = (
            f"Percentile-based strategy ({percentile_threshold}th percentile). "
            f"Threshold set to {threshold:.3f} mm absolute error. "
            f"Isolates top {100 - percentile_threshold:.1f}% most extreme forecast errors."
        )
    elif strategy == "absolute":
        threshold = abs_threshold_mm
        justification = (
            f"Fixed absolute error threshold strategy ({abs_threshold_mm:.1f} mm). "
            f"Classifies any forecast error exceeding {abs_threshold_mm:.1f} mm as a forecast bust."
        )
    elif strategy == "hybrid":
        threshold = max(abs_threshold_mm, p_val)
        justification = (
            f"Hybrid strategy combining absolute minimum ({abs_threshold_mm:.1f} mm) "
            f"and {percentile_threshold}th percentile ({p_val:.3f} mm). Effective threshold: {threshold:.3f} mm."
        )
    else:
        threshold = p_val
        justification = f"Default fallback percentile strategy. Threshold: {threshold:.3f} mm."

    return threshold, justification


def apply_bust_labeling(ds_err: xr.Dataset, threshold: float | None = None) -> tuple[xr.Dataset, dict]:
    """
    Apply bust threshold to dataset containing 'absolute_error'.
    """
    abs_err = ds_err["absolute_error"].values

    if threshold is None:
        threshold, justification = determine_bust_threshold(abs_err)
    else:
        justification = f"Explicitly provided threshold: {threshold:.3f} mm."

    # Generate binary bust target (1 = Bust, 0 = Non-bust)
    bust_mask = (abs_err >= threshold).astype(np.int32)

    ds_out = ds_err.copy()
    ds_out["bust"] = (ds_err["forecast_precipitation"].dims, bust_mask, {
        "units": "binary_indicator",
        "long_name": "Forecast Bust Indicator (1=Bust, 0=Normal)",
        "threshold_mm": threshold,
    })

    # Class balance stats
    total_cells = int(bust_mask.size)
    num_busts = int(np.sum(bust_mask == 1))
    num_non_busts = int(np.sum(bust_mask == 0))

    bust_pct = (num_busts / total_cells) * 100.0
    non_bust_pct = (num_non_busts / total_cells) * 100.0

    imbalance_ratio = f"{num_non_busts / max(1, num_busts):.2f}:1"

    # Evaluate ML sufficiency
    sample_sufficiency_evaluation = {
        "total_samples": total_cells,
        "is_sufficient_for_ml": total_cells >= 10000,
        "reasoning": (
            f"Dataset contains {total_cells:,} spatial-temporal grid points with {num_busts:,} bust samples. "
            "While sufficient for initial exploratory ML prototyping, expanding to 30-90 historical forecast dates "
            "is recommended for robust seasonal generalization across Indian monsoon regimes."
        ),
        "class_imbalance_identified": True,
        "imbalance_description": (
            f"Class imbalance present: {non_bust_pct:.1f}% Non-Bust vs {bust_pct:.1f}% Bust "
            f"(Imbalance Ratio: {imbalance_ratio}). Focal loss, class weighting, or SMOTE/oversampling "
            "should be used during ML model training."
        ),
    }

    summary = {
        "threshold_applied_mm": threshold,
        "threshold_justification": justification,
        "class_distribution": {
            "total_samples": total_cells,
            "num_non_busts": num_non_busts,
            "num_busts": num_busts,
            "non_bust_percentage": non_bust_pct,
            "bust_percentage": bust_pct,
            "imbalance_ratio": imbalance_ratio,
        },
        "ml_sufficiency_evaluation": sample_sufficiency_evaluation,
    }

    return ds_out, summary


def main():
    parser = argparse.ArgumentParser(description="Define forecast bust threshold and compute class distribution.")
    parser.add_argument("--input-nc", type=str, help="Path to input NetCDF dataset.")
    parser.add_argument("--threshold", type=float, help="Explicit bust threshold in mm (optional)")
    args = parser.parse_args()

    from calculate_error import calculate_forecast_errors
    from config import PROCESSED_DATA_DIR

    nc_files = list(PROCESSED_DATA_DIR.glob("aligned_forecast_verification_*.nc"))
    if not nc_files:
        print("No aligned NetCDF dataset found.")
        return

    nc_path = Path(args.input_nc) if args.input_nc else nc_files[0]
    ds = xr.open_dataset(nc_path)
    ds_err, _ = calculate_forecast_errors(ds)

    ds_labeled, summary = apply_bust_labeling(ds_err, threshold=args.threshold)

    print("\n==================================================")
    print("FORECAST BUST TARGET CREATION SUMMARY")
    print("==================================================")
    print(f"Applied Threshold:     {summary['threshold_applied_mm']:.3f} mm absolute error")
    print(f"Threshold Rationale:   {summary['threshold_justification']}")
    print(f"\n--- CLASS DISTRIBUTION ---")
    print(f"  Non-Bust (0): {summary['class_distribution']['num_non_busts']:,} ({summary['class_distribution']['non_bust_percentage']:.2f}%)")
    print(f"  Bust (1):     {summary['class_distribution']['num_busts']:,} ({summary['class_distribution']['bust_percentage']:.2f}%)")
    print(f"  Imbalance Ratio: {summary['class_distribution']['imbalance_ratio']}")

    print(f"\n--- ML SAMPLE SUFFICIENCY EVALUATION ---")
    print(f"  Sufficient for ML Prototype: {summary['ml_sufficiency_evaluation']['is_sufficient_for_ml']}")
    print(f"  Reasoning: {summary['ml_sufficiency_evaluation']['reasoning']}")
    print(f"  Class Imbalance: {summary['ml_sufficiency_evaluation']['imbalance_description']}")

    # Save summary
    out_json = METADATA_DIR / "bust_distribution_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary metadata to {out_json.resolve()}")


if __name__ == "__main__":
    main()
