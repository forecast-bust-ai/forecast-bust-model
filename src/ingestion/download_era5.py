"""
ERA5 Reanalysis Data Ingestion Script.
Downloads ERA5 total precipitation verification data from ECMWF Copernicus Climate Data Store (CDS).
Requires CDS API credentials (CDSAPI_KEY environment variable or ~/.cdsapirc file).
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys

# Import project configuration
from config import (
    ERA5_RAW_DIR,
    METADATA_DIR,
    DEFAULT_TEST_DATE,
    REGION_BBOX,
    CDSAPI_URL,
    CDSAPI_KEY,
    ensure_directories,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("download_era5")


def check_cds_credentials() -> tuple[bool, str]:
    """
    Verify if CDS API credentials are configured via environment variables or ~/.cdsapirc.
    Returns (has_credentials, status_message).
    """
    cdsapirc_path = Path.home() / ".cdsapirc"

    if CDSAPI_KEY:
        return True, "Found CDSAPI_KEY in environment variables."

    if cdsapirc_path.exists():
        try:
            content = cdsapirc_path.read_text()
            if "key:" in content or "key =" in content:
                return True, f"Found active credentials file at {cdsapirc_path}."
        except Exception as e:
            return False, f"Error reading {cdsapirc_path}: {e}"

    instruction = (
        "MISSING CDS API CREDENTIALS.\n"
        "To download ERA5 verification data from ECMWF Copernicus:\n"
        "  1. Register for an account at https://cds.climate.copernicus.eu\n"
        "  2. Copy your API key / Personal Access Token from your CDS profile page.\n"
        "  3. Set environment variable: export CDSAPI_KEY='your_api_key'\n"
        "     or add `CDSAPI_KEY=your_api_key` to project .env file\n"
        "     or create ~/.cdsapirc file containing:\n"
        "        url: https://cds.climate.copernicus.eu/api\n"
        "        key: your_api_key\n"
    )
    return False, instruction


def validate_netcdf_file(filepath: Path) -> dict:
    """
    Validate a downloaded ERA5 NetCDF file using netCDF4 / xarray.
    """
    validation = {
        "exists": filepath.exists(),
        "size_bytes": 0,
        "valid_netcdf": False,
        "is_valid": False,
        "error": None,
    }

    if not filepath.exists():
        validation["error"] = "File does not exist."
        return validation

    size = filepath.stat().st_size
    validation["size_bytes"] = size

    if size == 0:
        validation["error"] = "File is 0 bytes."
        return validation

    # Test netCDF4 or header magic bytes
    try:
        import netCDF4
        with netCDF4.Dataset(filepath, "r") as nc:
            validation["valid_netcdf"] = True
            validation["dimensions"] = list(nc.dimensions.keys())
            validation["variables"] = list(nc.variables.keys())
            validation["is_valid"] = True
    except ImportError:
        # Fallback to magic byte check (CDF or HDF5 signature)
        with open(filepath, "rb") as f:
            header = f.read(4)
            if header.startswith(b"CDF") or header.startswith(b"\x89HDF"):
                validation["valid_netcdf"] = True
                validation["is_valid"] = True
            else:
                validation["error"] = f"Unrecognized header magic bytes: {header}"
    except Exception as e:
        validation["error"] = f"Failed to open NetCDF file: {str(e)}"

    return validation


def download_era5_verification(
    date_str: str = DEFAULT_TEST_DATE,
    output_dir: Path = ERA5_RAW_DIR,
) -> dict:
    """
    Download ERA5 precipitation verification data for specified single date.
    """
    ensure_directories()

    has_creds, status_msg = check_cds_credentials()

    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")

    target_filename = f"era5_{dt.strftime('%Y%m%d')}_total_precipitation.nc"
    target_path = output_dir / target_filename

    manifest_path = METADATA_DIR / "era5_manifest.json"

    if not has_creds:
        logger.warning(status_msg)
        record = {
            "dataset": "ERA5",
            "date": date_str,
            "variable": "total_precipitation",
            "status": "SKIPPED_MISSING_CREDENTIALS",
            "message": status_msg,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(manifest_path, "w") as f:
            json.dump([record], f, indent=2)
        return record

    logger.info(f"CDS API Credentials verified: {status_msg}")
    logger.info(f"Initiating ERA5 download for date: {date_str}, Region BBOX: {REGION_BBOX}")

    try:
        import cdsapi

        # Create CDS Client
        if CDSAPI_KEY:
            client = cdsapi.Client(url=CDSAPI_URL, key=CDSAPI_KEY)
        else:
            client = cdsapi.Client()

        # ERA5 single-levels request parameters
        request_params = {
            "product_type": "reanalysis",
            "variable": "total_precipitation",
            "year": year,
            "month": month,
            "day": day,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [
                REGION_BBOX["north"],
                REGION_BBOX["west"],
                REGION_BBOX["south"],
                REGION_BBOX["east"],
            ],
            "format": "netcdf",
        }

        logger.info(f"Submitting CDS API request to save -> {target_path}...")
        client.retrieve("reanalysis-era5-single-levels", request_params, str(target_path))

        val = validate_netcdf_file(target_path)

        record = {
            "dataset": "ERA5",
            "date": date_str,
            "variable": "total_precipitation",
            "bbox": REGION_BBOX,
            "local_path": str(target_path.resolve()),
            "file_size_bytes": val["size_bytes"],
            "validation": val,
            "status": "SUCCESS" if val["is_valid"] else "VALIDATION_FAILED",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        if val["is_valid"]:
            logger.info(f"Successfully downloaded & validated ERA5 dataset ({val['size_bytes']} bytes).")
        else:
            logger.error(f"Validation FAILED for ERA5 file: {val['error']}")

    except Exception as e:
        logger.error(f"ERA5 Download failed: {str(e)}")
        record = {
            "dataset": "ERA5",
            "date": date_str,
            "variable": "total_precipitation",
            "status": "ERROR",
            "error": str(e),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    with open(manifest_path, "w") as f:
        json.dump([record], f, indent=2)

    return record


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 precipitation verification data from ECMWF CDS API.")
    parser.add_argument("--date", type=str, default=DEFAULT_TEST_DATE, help="Target date YYYY-MM-DD")
    parser.add_argument("--check-credentials", action="store_true", help="Check CDS API credential status only")
    args = parser.parse_args()

    if args.check_credentials:
        has_creds, msg = check_cds_credentials()
        print(f"CDS API Status: {'OK' if has_creds else 'MISSING'}")
        print(msg)
        sys.exit(0 if has_creds else 1)

    download_era5_verification(date_str=args.date)


if __name__ == "__main__":
    main()
