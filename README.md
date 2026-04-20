# Image Colorization Using Deep Learning

> B.Tech Final Year Project — Department of Information Technology  
> Institute of Engineering & Technology (IET), DDUGU Gorakhpur — Batch 2026

---

## Overview

This project develops and evaluates three deep learning models for **automatic grayscale image colorization** — the task of predicting realistic colors for black-and-white photographs without any human input.

All models operate in the **CIE LAB color space**: the L channel (lightness/grayscale) is given as input, and the model predicts the A and B channels (color). This ensures output brightness always matches the input.

---

## Models

| Model | Architecture | Training |
|-------|-------------|----------|
| **Pix2Pix GAN** | U-Net generator + PatchGAN discriminator | 200 epochs, COCO 2017 (118k images), two-phase |
| **ResNet GAN** ⭐ | ResNet34 pretrained encoder + custom decoder | 100 epochs, transfer learning |
| **OpenCV Zhang 2016** | Classification CNN (313 AB bins) | Pretrained on ImageNet (baseline) |

---

## Results

Evaluated on **50 COCO 2017 validation images** at 512×512 resolution (boost=1.0, no post-processing):

| Model | PSNR (dB) | SSIM | Epochs |
|-------|-----------|------|--------|
| **ResNet GAN ⭐ (best)** | **22.49** | **0.8929** | 100 |
| Pix2Pix GAN | 22.41 | 0.8909 | 200 |
| OpenCV Zhang 2016 | 21.18 | 0.8861 | — |

**Best scene-specific results** (smart checkpoint selection from 303 evaluated checkpoints):

| Model | Scene | Epoch | PSNR | SSIM |
|-------|-------|-------|------|------|
| ResNet | Portrait | 31 | 23.40 dB | 0.9021 |
| ResNet | Landscape | 8 | 23.20 dB | 0.9196 |
| GAN | Portrait | 178 | 22.29 dB | 0.8983 |
| GAN | Landscape | 131 | 23.41 dB | 0.9219 |

---

## Web Application

A complete Flask web application with:

- **Single Image** tab — drag-and-drop colorization with before/after compare slider
- **Batch Processing** — up to 20 images at once
- **Recent Results** — last 10 colorizations with metrics
- **Model Switching** — switch between GAN / ResNet / OpenCV at runtime
- **Smart Checkpoint Selection** — auto-detects portrait vs landscape and loads the best checkpoint
- **Evaluation Report** — embedded HTML report with PSNR/SSIM per image
- **Training Metrics** tab — live training progress display

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Deep Learning | PyTorch 2.0, torchvision |
| Image Processing | Pillow, scikit-image, OpenCV |
| Web Backend | Flask, flask-cors |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Training | Kaggle (NVIDIA T4 GPU) |
| Dataset | COCO 2017 (118,287 training images) |

---

## Project Structure
image-colorizer-gan/
├── backend.py              # Flask web server — main entry point
├── colorize_utils.py       # Shared utilities (colorization, metrics, enhancement)
├── evaluate.py             # Evaluation pipeline — PSNR/SSIM on validation set
├── checkpoint_picker.py    # Smart checkpoint selection (303 checkpoints evaluated)
├── compare_models.py       # Single-image model comparison with matplotlib figure
├── inference.py            # Command-line colorization tool
├── plot_training.py        # Training progress graph generator
├── update_metrics.py       # Update Training Metrics tab in web app
├── index.html              # Complete web application frontend
├── models/
│   ├── generator.py        # U-Net GAN generator architecture
│   └── generator_resnet.py # ResNet34-based generator architecture
└── best_checkpoints.json   # Best checkpoint per model per scene type

---

## Setup and Run

**Requirements:**
```bash
pip install -r requirements.txt
```

**Start the web application:**
```bash
python backend.py
```
Then open `http://localhost:5000` in your browser.

> **Note:** Place your trained `.pth` checkpoint files in the `checkpoints/` folder before running. Checkpoints are not included in this repository due to file size limits.

**Run evaluation:**
```bash
python evaluate.py --limit 50
```

**Generate training graph:**
```bash
python plot_training.py
```

**Compare models on a single image:**
```bash
python compare_models.py --input dataset/val/000000000139.jpg --boost-display 1.4
```

---

## Training Strategy

**GAN — Two-phase training:**
- Phase 1: 100 epochs × 5,000 images, LR = 5×10⁻⁴
- Phase 2: 100 epochs × 118,287 images, LR = 5×10⁻⁵
- Optimizer: Adam (β₁=0.5), StepLR scheduler

**ResNet GAN — Transfer learning:**
- 100 epochs × 118,287 images, LR = 1×10⁻⁴ (fixed)
- ResNet34 encoder pretrained on ImageNet
- Low LR prevents catastrophic forgetting of pretrained features

---

## References

1. Zhang et al. (2016) — *Colorful Image Colorization* — ECCV
2. Isola et al. (2017) — *Image-to-Image Translation with Conditional GANs* (Pix2Pix) — CVPR
3. He et al. (2016) — *Deep Residual Learning for Image Recognition* — CVPR
4. Ronneberger et al. (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation*

---

## Project Guide

Under the supervision of **Dr. Vidya Srivastava**, Assistant Professor, Department of Information Technology, IET DDUGU Gorakhpur.

---

*Submitted in partial fulfillment for the award of Bachelor of Technology in Information Technology, DDUGU Gorakhpur (Batch 2026)*
