"""
Centralized Configuration for Forecast Bust AI Ingestion & Processing.
Loads environment variables, defines spatial bounding boxes, dataset parameters,
and directory paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load optional .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GFS_RAW_DIR = RAW_DATA_DIR / "gfs"
ERA5_RAW_DIR = RAW_DATA_DIR / "era5"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# -----------------------------------------------------------------------------
# Region Bounding Box (India & Surrounding Region)
# North, West, South, East (degrees)
# -----------------------------------------------------------------------------
REGION_BBOX = {
    "north": 40.0,
    "south": 0.0,
    "west": 60.0,
    "east": 100.0,
}

# -----------------------------------------------------------------------------
# Target Variable Definitions
# -----------------------------------------------------------------------------
TARGET_VARIABLE = "precipitation"
GFS_PRECIP_VAR = "APCP"  # Total Accumulated Precipitation in GRIB2
ERA5_PRECIP_VAR = "total_precipitation"  # ERA5 single-level variable name

# -----------------------------------------------------------------------------
# Default Single Date Test Ingestion Configuration
# -----------------------------------------------------------------------------
DEFAULT_TEST_DATE = "2024-06-01"
DEFAULT_CYCLE = "00"  # 00Z Forecast run
DEFAULT_LEAD_TIMES = [24, 48, 72, 96, 120]  # Forecast lead hours (Day 1 to 5)

# -----------------------------------------------------------------------------
# Forecast Bust Threshold Configuration (Configurable)
# -----------------------------------------------------------------------------
# Strategy options: "absolute", "percentile", "hybrid"
BUST_THRESHOLD_STRATEGY = os.getenv("BUST_THRESHOLD_STRATEGY", "percentile")
BUST_ABSOLUTE_THRESHOLD_MM = float(os.getenv("BUST_ABSOLUTE_THRESHOLD_MM", 15.0))  # mm/day error
BUST_PERCENTILE_THRESHOLD = float(os.getenv("BUST_PERCENTILE_THRESHOLD", 90.0))    # 90th percentile error

# -----------------------------------------------------------------------------
# Remote Data Sources & Credentials
# -----------------------------------------------------------------------------
# GFS Public AWS Open Data S3 Bucket URL
GFS_AWS_BASE_URL = os.getenv("GFS_AWS_BASE_URL", "https://noaa-gfs-bdp-pds.s3.amazonaws.com")
GFS_NOMADS_BASE_URL = os.getenv("GFS_NOMADS_BASE_URL", "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl")

# Copernicus Climate Data Store (CDS) API Credentials
CDSAPI_URL = os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api/v2")
CDSAPI_KEY = os.getenv("CDSAPI_KEY", None)


def ensure_directories():
    """Ensure all required project directories exist."""
    for d in [DATA_DIR, RAW_DATA_DIR, GFS_RAW_DIR, ERA5_RAW_DIR, PROCESSED_DATA_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {BASE_DIR}")
    print(f"GFS Raw Directory: {GFS_RAW_DIR}")
    print(f"ERA5 Raw Directory: {ERA5_RAW_DIR}")
    print(f"Region Bounding Box: {REGION_BBOX}")
