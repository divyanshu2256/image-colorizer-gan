"""
ColorizationDataset
====================
Loads RGB images, converts to LAB color space, and returns:
    L  channel : grayscale input  (1, H, W) normalized to [-1, 1]
    AB channels: color target     (2, H, W) normalized to [-1, 1]
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.color import rgb2lab
import torch


class ColorizationDataset(Dataset):
    def __init__(self, root_dir: str, size: int = 256, augment: bool = True):
        """
        Args:
            root_dir : path to folder containing .jpg / .png images
            size     : resize all images to (size x size)
            augment  : apply random horizontal flip for training
        """
        self.root  = Path(root_dir)
        self.size  = size
        self.files = sorted(
            list(self.root.glob("*.jpg")) +
            list(self.root.glob("*.jpeg")) +
            list(self.root.glob("*.png"))
        )

        if len(self.files) == 0:
            raise ValueError(f"No images found in {root_dir}")

        tf_list = [transforms.Resize((size, size), transforms.InterpolationMode.LANCZOS)]
        if augment:
            tf_list.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(tf_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        img_np = np.array(img, dtype=np.uint8)

        # Convert to LAB color space
        lab = rgb2lab(img_np).astype("float32")

        # Normalize:
        #   L  : [0, 100]  → [-1, 1]
        #   AB : [-128, 127] → [-1, 1] (approx, clip to [-110, 110])
        L  = lab[:, :, 0] / 50.0 - 1.0
        AB = lab[:, :, 1:] / 110.0

        L_tensor  = torch.tensor(L).unsqueeze(0)         # (1, H, W)
        AB_tensor = torch.tensor(AB).permute(2, 0, 1)    # (2, H, W)

        return L_tensor, AB_tensor
