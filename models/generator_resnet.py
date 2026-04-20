"""
models/generator_resnet.py — ResNet34 Pretrained Encoder + U-Net Decoder
=========================================================================
Drop-in replacement for generator.py with much better colorization quality.
Uses pretrained ResNet34 encoder that already knows colors from ImageNet.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Decoder block: ConvTranspose2d + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    ResNet34 Encoder + U-Net Decoder for image colorization.

    Input  : L channel (1 × 256 × 256) — grayscale lightness
    Output : AB channels (2 × 256 × 256) — color prediction

    Architecture:
        Encoder: Pretrained ResNet34 (frozen first 2 layers)
        Decoder: U-Net skip connections for detail preservation
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # ── Encoder: Pretrained ResNet34 ─────────────────────────────────────
        resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)

        # Modify first conv to accept 1-channel input (L channel)
        self.enc0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
        )  # 64 × 128 × 128

        self.enc1 = resnet.maxpool   # 64 × 64 × 64 (just pooling)
        self.enc2 = resnet.layer1    # 64 × 64 × 64
        self.enc3 = resnet.layer2    # 128 × 32 × 32
        self.enc4 = resnet.layer3    # 256 × 16 × 16
        self.enc5 = resnet.layer4    # 512 × 8 × 8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )  # 512 × 8 × 8

        # ── Decoder: U-Net with skip connections ──────────────────────────────
        self.dec5 = ConvBlock(512,       512, dropout=True)   # 512 × 16 × 16
        self.dec4 = ConvBlock(512 + 256, 256, dropout=True)   # 256 × 32 × 32
        self.dec3 = ConvBlock(256 + 128, 128)                 # 128 × 64 × 64
        self.dec2 = ConvBlock(128 + 64,  64)                  # 64 × 128 × 128
        self.dec1 = ConvBlock(64  + 64,  64)                  # 64 × 256 × 256

        # Final output: predict AB channels
        self.out = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )  # 2 × 256 × 256

        # Freeze early encoder layers (they already know basic features)
        for param in self.enc0.parameters():
            param.requires_grad = False
        for param in self.enc2.parameters():
            param.requires_grad = False

    def forward(self, x):
        # ── Encode ────────────────────────────────────────────────────────────
        e0 = self.enc0(x)           # 64  × 128 × 128
        e1 = self.enc1(e0)          # 64  × 64  × 64
        e2 = self.enc2(e1)          # 64  × 64  × 64
        e3 = self.enc3(e2)          # 128 × 32  × 32
        e4 = self.enc4(e3)          # 256 × 16  × 16
        e5 = self.enc5(e4)          # 512 × 8   × 8

        # ── Bottleneck ────────────────────────────────────────────────────────
        bn = self.bottleneck(e5)    # 512 × 8   × 8

        # ── Decode with skip connections ──────────────────────────────────────
        d5 = self.dec5(bn)                          # 512 × 16  × 16
        d4 = self.dec4(torch.cat([d5, e4], dim=1))  # 256 × 32  × 32
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # 128 × 64  × 64
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # 64  × 128 × 128
        d1 = self.dec1(torch.cat([d2, e0], dim=1))  # 64  × 256 × 256

        return self.out(d1)                         # 2   × 256 × 256


if __name__ == '__main__':
    # Quick test
    G = Generator(pretrained=False)
    x = torch.randn(2, 1, 256, 256)
    out = G(x)
    print(f'Input  : {x.shape}')
    print(f'Output : {out.shape}')
    params = sum(p.numel() for p in G.parameters()) / 1e6
    print(f'Params : {params:.1f}M')
