"""
Utility script to dynamically inspect raw GFS (GRIB2) and ERA5 (NetCDF) weather files.
Detects variable names, dimensions, coordinate systems, units, timestamps, and min/max ranges.
"""

import argparse
from pathlib import Path
import sys
import xarray as xr

from config import GFS_RAW_DIR, ERA5_RAW_DIR


def inspect_file(filepath: Path) -> dict:
    """
    Inspect a GRIB2 or NetCDF file without hardcoding variable names or coordinates.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: File {filepath} does not exist.")
        return {}

    print(f"\n==================================================")
    print(f"INSPECTING FILE: {filepath.name}")
    print(f"Path: {filepath.resolve()}")
    print(f"Size: {filepath.stat().st_size} bytes")
    print(f"==================================================")

    # Determine xarray engine
    engine = "cfgrib" if filepath.suffix in [".grib", ".grib2", ".grb"] else "netcdf4"

    try:
        ds = xr.open_dataset(filepath, engine=engine)
    except Exception as e:
        print(f"Failed to open dataset with engine='{engine}': {e}")
        try:
            ds = xr.open_dataset(filepath)
        except Exception as e2:
            print(f"Fallback open_dataset failed: {e2}")
            return {}

    inspection_results = {
        "filename": filepath.name,
        "engine": engine,
        "data_vars": {},
        "coords": {},
        "dims": dict(ds.sizes),
        "attrs": dict(ds.attrs),
    }

    print("\n--- DATA VARIABLES ---")
    for var_name in ds.data_vars:
        da = ds[var_name]
        var_info = {
            "dims": list(da.dims),
            "shape": list(da.shape),
            "dtype": str(da.dtype),
            "units": da.attrs.get("units", "N/A"),
            "long_name": da.attrs.get("long_name", "N/A"),
            "min": float(da.min().values) if da.size > 0 else None,
            "max": float(da.max().values) if da.size > 0 else None,
        }
        inspection_results["data_vars"][var_name] = var_info
        print(f"Variable: '{var_name}'")
        print(f"  Long Name: {var_info['long_name']}")
        print(f"  Units:     {var_info['units']}")
        print(f"  Shape:     {var_info['shape']} (dims: {var_info['dims']})")
        print(f"  Min Value: {var_info['min']}")
        print(f"  Max Value: {var_info['max']}")

    print("\n--- COORDINATES ---")
    for coord_name in ds.coords:
        c = ds[coord_name]
        c_vals = c.values
        coord_info = {
            "dims": list(c.dims),
            "shape": list(c.shape),
            "dtype": str(c.dtype),
            "min": str(c_vals.min()) if c_vals.size > 0 else "N/A",
            "max": str(c_vals.max()) if c_vals.size > 0 else "N/A",
        }
        inspection_results["coords"][coord_name] = coord_info
        print(f"Coord: '{coord_name}'")
        print(f"  Min: {coord_info['min']} | Max: {coord_info['max']} | Shape: {coord_info['shape']}")

    return inspection_results


def main():
    parser = argparse.ArgumentParser(description="Inspect GFS and ERA5 raw weather datasets.")
    parser.add_argument("--file", type=str, help="Path to specific dataset file to inspect.")
    args = parser.parse_args()

    if args.file:
        inspect_file(Path(args.file))
    else:
        # Inspect sample files in raw directory
        gfs_files = list(GFS_RAW_DIR.glob("*.grib2"))
        era5_files = list(ERA5_RAW_DIR.glob("*.nc"))

        if gfs_files:
            print("Found GFS files. Inspecting first file...")
            inspect_file(gfs_files[0])
        else:
            print(f"No GFS files found in {GFS_RAW_DIR}")

        if era5_files:
            print("Found ERA5 files. Inspecting first file...")
            inspect_file(era5_files[0])
        else:
            print(f"No ERA5 files found in {ERA5_RAW_DIR}")


if __name__ == "__main__":
    main()
