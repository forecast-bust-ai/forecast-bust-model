"""
PyTorch Spatial CNN Model Architecture for Forecast Bust Detection.
Uses fully convolutional 2D architecture to process atmospheric grid fields and output
spatial bust probability maps.
"""

import torch
import torch.nn as nn


class BustSpatialCNN(nn.Module):
    """
    Fully Convolutional Neural Network (FCN) for 2D spatial forecast bust prediction.
    Preserves spatial height & width dimensions (H, W) from input to output.
    """

    def __init__(self, in_channels: int = 5, hidden_dim: int = 32):
        super(BustSpatialCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 1x1 Convolution to reduce channels to 1 logit per grid cell
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch, 5, H, W)
        returns: (batch, 1, H, W) logits
        """
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        logits = self.head(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities.
        returns: (batch, 1, H, W) probabilities in range [0, 1]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
