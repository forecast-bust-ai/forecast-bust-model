"""
PyTorch Spatial Dataset Module for Forecast Bust Detection.
Constructs 2D spatial feature tensors (channels, height, width) and binary target maps
strictly from forecast-time variables.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


def compute_spatial_gradient(array_2d: np.ndarray) -> np.ndarray:
    """Compute spatial gradient magnitude (Sobel-like 2D gradient) of forecast precipitation."""
    gy, gx = np.gradient(array_2d)
    grad_mag = np.sqrt(gx**2 + gy**2).astype(np.float32)
    return grad_mag


def prepare_spatial_tensor_sample(
    fcst_precip_2d: np.ndarray,
    lead_time_hours: float,
    lats_1d: np.ndarray,
    lons_1d: np.ndarray,
    max_precip: float = 100.0,
) -> torch.Tensor:
    """
    Construct 5-channel 2D spatial feature tensor of shape (5, H, W):
    - Channel 0: Normalized Forecast Precipitation
    - Channel 1: Normalized Lead Time
    - Channel 2: Normalized Latitude grid
    - Channel 3: Normalized Longitude grid
    - Channel 4: Spatial Precipitation Gradient
    """
    H, W = fcst_precip_2d.shape

    # Normalize precipitation (0 to ~1)
    ch0 = np.clip(fcst_precip_2d / max_precip, 0.0, 1.0).astype(np.float32)

    # Lead time channel
    ch1 = np.full((H, W), lead_time_hours / 120.0, dtype=np.float32)

    # Latitude & Longitude coordinate grid channels
    lat_grid, lon_grid = np.meshgrid(lats_1d, lons_1d, indexing="ij")
    ch2 = ((lat_grid - 20.0) / 20.0).astype(np.float32)
    ch3 = ((lon_grid - 80.0) / 20.0).astype(np.float32)

    # Spatial gradient magnitude channel
    ch4 = compute_spatial_gradient(ch0)

    # Stack channels -> (5, H, W)
    tensor_5d = np.stack([ch0, ch1, ch2, ch3, ch4], axis=0)
    return torch.from_numpy(tensor_5d)


class ForecastBustSpatialDataset(Dataset):
    """
    PyTorch Dataset serving 2D spatial grids for CNN training.
    """

    def __init__(self, ds_nc: xr.Dataset, lead_time_indices: list[int]):
        self.ds = ds_nc
        self.lead_time_indices = lead_time_indices

        self.lats = ds_nc["latitude"].values
        self.lons = ds_nc["longitude"].values

        # Maximum precip for normalization
        self.max_precip = float(np.max(ds_nc["forecast_precipitation"].values))
        if self.max_precip <= 0:
            self.max_precip = 100.0

        self.samples = []
        self.targets = []
        self.metadata = []

        for idx in self.lead_time_indices:
            ds_lt = ds_nc.isel(lead_time=idx)

            lt_val = ds_nc["lead_time"].values[idx]
            if isinstance(lt_val, np.timedelta64):
                lt_hours = float(lt_val / np.timedelta64(1, "h"))
            else:
                lt_hours = float(lt_val)

            fcst_2d = ds_lt["forecast_precipitation"].values.astype(np.float32)
            target_2d = ds_lt["bust"].values.astype(np.float32)

            feature_tensor = prepare_spatial_tensor_sample(
                fcst_2d, lt_hours, self.lats, self.lons, max_precip=self.max_precip
            )
            target_tensor = torch.from_numpy(target_2d).unsqueeze(0)  # Shape: (1, H, W)

            self.samples.append(feature_tensor)
            self.targets.append(target_tensor)
            self.metadata.append({
                "lead_time_index": idx,
                "lead_time_hours": lt_hours,
                "valid_time": str(ds_lt["valid_time"].values) if "valid_time" in ds_lt.coords else "",
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.targets[idx]
