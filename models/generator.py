"""
Generator — U-Net Architecture
"""

import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, dropout=False):
        super().__init__()
        if down:
            layers = [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.e1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.e2 = UNetBlock(64,  128)
        self.e3 = UNetBlock(128, 256)
        self.e4 = UNetBlock(256, 512)
        self.e5 = UNetBlock(512, 512)

        # Bottleneck (named 'bn' to match Colab saved model)
        self.bn = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.d1 = UNetBlock(512,  512, down=False, dropout=True)
        self.d2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.d3 = UNetBlock(1024, 256, down=False)
        self.d4 = UNetBlock(512,  128, down=False)
        self.d5 = UNetBlock(256,  64,  down=False)

        # Output (named 'out' to match Colab saved model)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, 2, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        bn = self.bn(e5)

        d1 = self.d1(bn)
        d2 = self.d2(torch.cat([d1, e5], dim=1))
        d3 = self.d3(torch.cat([d2, e4], dim=1))
        d4 = self.d4(torch.cat([d3, e3], dim=1))
        d5 = self.d5(torch.cat([d4, e2], dim=1))

        return self.out(torch.cat([d5, e1], dim=1))