"""
Discriminator — PatchGAN Architecture
=======================================
Instead of judging the full image as real/fake, PatchGAN judges
overlapping 70x70 patches. This encourages sharper local texture
and more realistic colorization.

Input : L channel (1, H, W) + AB channels (2, H, W) → concatenated to (3, H, W)
Output: Patch map of real/fake predictions
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Receives concatenated [L | AB] as input (3 channels total).
    """

    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # No BN on first layer (standard practice)
            *block(3,   64,  stride=2, norm=False),   # 64  x 128x128
            *block(64,  128, stride=2),                # 128 x 64x64
            *block(128, 256, stride=2),                # 256 x 32x32
            *block(256, 512, stride=1),                # 512 x 31x31
            nn.Conv2d(512, 1, 4, 1, 1)                # 1   x 30x30 (patch output)
        )

    def forward(self, L, AB):
        """
        Args:
            L  : grayscale L channel  (B, 1, H, W)
            AB : color AB channels    (B, 2, H, W)
        Returns:
            Patch prediction map      (B, 1, 30, 30)
        """
        x = torch.cat([L, AB], dim=1)   # (B, 3, H, W)
        return self.model(x)
