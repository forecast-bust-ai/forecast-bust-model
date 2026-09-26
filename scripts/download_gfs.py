"""
GFS Weather Forecast Data Ingestion Script.
Downloads GFS (Global Forecast System) 0.25-degree forecast files from NOAA AWS Open Data S3 bucket.
Supports variable-specific byte-range downloading (APCP - Accumulated Precipitation) to minimize download bandwidth.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Import project configuration
from config import GFS_RAW_DIR, METADATA_DIR, DEFAULT_TEST_DATE, DEFAULT_CYCLE, DEFAULT_LEAD_TIMES, ensure_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("download_gfs")

S3_BUCKET = "noaa-gfs-bdp-pds"
S3_REGION = "us-east-1"


def get_unsigned_s3_client():
    """Create an unsigned boto3 S3 client with explicit regional endpoint for public NOAA data access."""
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        endpoint_url=f"https://s3.{S3_REGION}.amazonaws.com",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5, "mode": "standard"}),
    )


def parse_idx_for_variable(idx_text: str, variable_name: str = "APCP") -> list[tuple[int, int | str, str]]:
    """
    Parse NOAA GFS index file to extract byte offsets for a specific variable.
    Returns list of tuples: (start_byte, end_byte, line_desc)
    """
    lines = idx_text.splitlines()
    matches = []

    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) >= 5 and variable_name in parts[3]:
            start_byte = int(parts[1])
            if i + 1 < len(lines):
                end_byte = int(lines[i + 1].split(":")[1]) - 1
            else:
                end_byte = ""  # Read till EOF
            matches.append((start_byte, end_byte, line))

    return matches


def validate_grib2_file(filepath: Path) -> dict:
    """
    Validate a downloaded GRIB2 file.
    Checks existence, non-zero size, and magic bytes header ('GRIB').
    """
    validation = {
        "exists": filepath.exists(),
        "size_bytes": 0,
        "valid_header": False,
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

    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
            if header == b"GRIB":
                validation["valid_header"] = True
                validation["is_valid"] = True
            else:
                validation["error"] = f"Invalid magic header: {header} (expected b'GRIB')"
    except Exception as e:
        validation["error"] = f"Failed to read header: {str(e)}"

    return validation


def download_gfs_forecast(
    date_str: str = DEFAULT_TEST_DATE,
    cycle: str = DEFAULT_CYCLE,
    lead_times: list[int] = DEFAULT_LEAD_TIMES,
    variable: str = "APCP",
    output_dir: Path = GFS_RAW_DIR,
) -> list[dict]:
    """
    Download GFS precipitation forecast files for specified date, cycle, and lead times.
    """
    ensure_directories()

    # Format date: YYYYMMDD
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    ymd = dt.strftime("%Y%m%d")
    cycle_formatted = f"{int(cycle):02d}"

    s3_client = get_unsigned_s3_client()
    manifest_records = []

    logger.info(f"Starting GFS download for date: {date_str}, cycle: {cycle_formatted}Z, lead times: {lead_times}")

    for fhr in lead_times:
        fhr_str = f"{fhr:03d}"
        s3_key_grib = f"gfs.{ymd}/{cycle_formatted}/atmos/gfs.t{cycle_formatted}z.pgrb2.0p25.f{fhr_str}"
        s3_key_idx = f"{s3_key_grib}.idx"

        target_filename = f"gfs_{ymd}_t{cycle_formatted}z_f{fhr_str}_{variable}.grib2"
        target_path = output_dir / target_filename

        logger.info(f"Fetching index file for f{fhr_str} (Key: {s3_key_idx})...")

        try:
            # 1. Fetch index file
            idx_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key_idx)
            idx_text = idx_obj["Body"].read().decode("utf-8")

            # 2. Parse byte ranges for requested variable
            var_matches = parse_idx_for_variable(idx_text, variable_name=variable)
            if not var_matches:
                logger.warning(f"Variable '{variable}' not found in index file for f{fhr_str}. Downloading entire file...")
                range_header = None
            else:
                start_b, end_b, line_desc = var_matches[0]
                range_header = f"bytes={start_b}-{end_b}"
                logger.info(f"Found variable '{variable}': {line_desc} -> Range: {range_header}")

            # 3. Download GRIB data chunk
            logger.info(f"Downloading GFS data to {target_path}...")
            if range_header:
                grib_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key_grib, Range=range_header)
            else:
                grib_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key_grib)

            data = grib_obj["Body"].read()

            with open(target_path, "wb") as f:
                f.write(data)

            # 4. Validate downloaded file
            val = validate_grib2_file(target_path)

            record = {
                "dataset": "GFS",
                "date": date_str,
                "cycle": cycle_formatted,
                "lead_time_hours": fhr,
                "variable": variable,
                "s3_bucket": S3_BUCKET,
                "s3_key": s3_key_grib,
                "local_path": str(target_path.resolve()),
                "file_size_bytes": val["size_bytes"],
                "validation": val,
                "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

            if val["is_valid"]:
                logger.info(f"Successfully downloaded & validated {target_filename} ({val['size_bytes']} bytes).")
            else:
                logger.error(f"Validation FAILED for {target_filename}: {val['error']}")

            manifest_records.append(record)

        except Exception as e:
            logger.error(f"Failed to download GFS forecast f{fhr_str}: {str(e)}")
            manifest_records.append({
                "dataset": "GFS",
                "date": date_str,
                "cycle": cycle_formatted,
                "lead_time_hours": fhr,
                "variable": variable,
                "error": str(e),
                "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

    # Save manifest log
    manifest_path = METADATA_DIR / "gfs_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_records, f, indent=2)

    logger.info(f"GFS Ingestion complete. Manifest updated at {manifest_path}")
    return manifest_records


def main():
    parser = argparse.ArgumentParser(description="Download GFS forecast data from NOAA AWS S3.")
    parser.add_argument("--date", type=str, default=DEFAULT_TEST_DATE, help="Forecast date YYYY-MM-DD")
    parser.add_argument("--cycle", type=str, default=DEFAULT_CYCLE, help="Forecast cycle (00, 06, 12, 18)")
    parser.add_argument("--lead-times", nargs="+", type=int, default=DEFAULT_LEAD_TIMES, help="Lead times in hours (e.g. 24 48 72 96 120)")
    parser.add_argument("--variable", type=str, default="APCP", help="Target variable (default: APCP)")
    args = parser.parse_args()

    download_gfs_forecast(
        date_str=args.date,
        cycle=args.cycle,
        lead_times=args.lead_times,
        variable=args.variable,
    )


if __name__ == "__main__":
    main()
