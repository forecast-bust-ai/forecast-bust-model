# Forecast Bust AI — ML Pipeline & Data Science (`forecast-bust-model`)

**Organization:** `forecast-bust-ai`  
**Problem Statement ID:** 26079 (NCMRWF / Ministry of Earth Sciences)

## Overview
This repository contains the complete weather data ingestion, GFS/ERA5 spatial preprocessing, forecast error calculation, bust labeling, XGBoost baseline, and PyTorch 2D Spatial CNN model training pipeline.

## Repository Layout
- `src/ingestion/`: GFS and ERA5 data downloader scripts.
- `src/preprocessing/`: Spatial regridding, coordinate normalization, and dataset temporal alignment.
- `src/features/` & `src/labels/`: Forecast error calculation and historical 90th-percentile bust labeling.
- `src/models/baseline/`: XGBoost Classifier baseline training & evaluation.
- `src/models/spatial/`: 2D Spatial PyTorch CNN (`BustSpatialCNN`) model architecture, dataset generator, and trainer.
- `artifacts/`: Exported model artifacts (`.joblib`, `.pt`) and performance metric JSON files.
- `tests/`: Comprehensive unit test suite.

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train XGBoost Baseline
python3 src/models/baseline/train.py

# Train PyTorch Spatial CNN
python3 src/models/spatial/train.py

# Run ML Unit Tests
OMP_NUM_THREADS=1 python3 -m pytest tests/
```
