"""
Helper Utilities
=================
Common functions used across training, inference, and evaluation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.color import lab2rgb
from PIL import Image


def lab_to_rgb(L_tensor: torch.Tensor, AB_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert LAB tensors back to an RGB numpy array (uint8).

    Args:
        L_tensor  : (1, H, W) or (H, W) — normalized L channel
        AB_tensor : (2, H, W)            — normalized AB channels
    Returns:
        RGB image as np.ndarray (H, W, 3) uint8
    """
    if L_tensor.dim() == 3:
        L_tensor = L_tensor.squeeze(0)

    L  = (L_tensor.cpu().numpy() + 1.0) * 50.0                    # [-1,1] → [0,100]
    AB = AB_tensor.permute(1, 2, 0).cpu().numpy() * 110.0         # [-1,1] → [-110,110]

    lab = np.concatenate([L[:, :, np.newaxis], AB], axis=2)
    rgb = (lab2rgb(lab) * 255).clip(0, 255).astype(np.uint8)
    return rgb


def save_sample_grid(
    L_batch, real_AB_batch, fake_AB_batch,
    save_path: str, epoch: int, n: int = 4
):
    """
    Save a 3-row comparison grid:
        Row 1 — Grayscale inputs
        Row 2 — GAN colorized outputs
        Row 3 — Ground truth colors

    Args:
        L_batch       : (B, 1, H, W) tensor
        real_AB_batch : (B, 2, H, W) tensor
        fake_AB_batch : (B, 2, H, W) tensor
        save_path     : directory to save image
        epoch         : current epoch number
        n             : number of samples to display
    """
    n = min(n, L_batch.size(0))
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))

    titles = ["Grayscale Input", "GAN Colorized", "Ground Truth"]

    for i in range(n):
        gray      = ((L_batch[i].squeeze().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        colorized = lab_to_rgb(L_batch[i], fake_AB_batch[i])
        gt        = lab_to_rgb(L_batch[i], real_AB_batch[i])

        for row, (ax, img) in enumerate(zip(axes[:, i], [gray, colorized, gt])):
            if row == 0:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[row], fontsize=11, fontweight="bold")

    plt.suptitle(f"Epoch {epoch} — Colorization Results", fontsize=14)
    plt.tight_layout()

    out_path = Path(save_path) / f"epoch_{epoch:03d}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return str(out_path)


def count_parameters(model) -> str:
    """Return a formatted string of total trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1_000_000:
        return f"{total / 1_000_000:.2f}M"
    return f"{total / 1_000:.1f}K"


def weights_init(m):
    """
    Custom weight initialization for Conv and BatchNorm layers.
    As recommended in the original Pix2Pix paper.
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
