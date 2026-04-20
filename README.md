# Image Colorization Using Deep Learning

B.Tech Final Year Project — Department of Information Technology  
Institute of Engineering & Technology, DDUGU Gorakhpur (Batch 2025)

## Overview
Automatic colorization of grayscale images using three deep learning models:
- **Pix2Pix GAN** — U-Net generator + PatchGAN discriminator (200 epochs)
- **ResNet GAN** — ResNet34 pretrained encoder + custom decoder (100 epochs)
- **OpenCV Zhang 2016** — Classical baseline (pretrained)

## Results (50 validation images, 512×512)
| Model | PSNR (dB) | SSIM |
|-------|-----------|------|
| ResNet GAN ★ | 22.49 | 0.8929 |
| Pix2Pix GAN | 22.41 | 0.8909 |
| OpenCV Zhang 2016 | 21.18 | 0.8861 |

## Dataset
COCO 2017 — 118,287 training images, 5,000 validation images

## Tech Stack
Python · PyTorch · Flask · OpenCV · scikit-image · Pillow · Vanilla JS

## Run Locally
```bash
pip install -r requirements.txt
python backend.py
# Open http://localhost:5000
```

## Project Structure
backend.py          # Flask web server
colorize_utils.py   # Shared utilities
evaluate.py         # Evaluation pipeline
checkpoint_picker.py# Smart checkpoint selection
compare_models.py   # Model comparison
inference.py        # Command-line inference
index.html          # Web application frontend

## Guide
Under supervision of Dr. Vidya Srivastava, IT Department, IET DDUGU
