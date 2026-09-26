"""
Pytest configuration for forecast-bust-model repository.
Adds src/ and src/models/ to Python sys.path so modules resolve cleanly.
"""

import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
SRC_DIR = MODEL_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))
sys.path.insert(0, str(SRC_DIR / "preprocessing"))
sys.path.insert(0, str(SRC_DIR / "labels"))
sys.path.insert(0, str(SRC_DIR / "features"))
