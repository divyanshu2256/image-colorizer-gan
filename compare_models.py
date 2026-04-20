"""
compare_models.py — Single-Image Model Comparison (GAN vs ResNet vs OpenCV)
===========================================================================
PURPOSE : Run all three models on ONE image and produce a matplotlib comparison
          figure plus individual JPEG files. Great for demo/presentation images.

CHECKPOINT SELECTION:
          Reads best_checkpoints.json (portrait keys) first.
          Falls back to latest checkpoint if JSON doesn't exist.

SHARED CODE:
          Imports from utils.py — find_latest, load_best_config, download_file,
          load_opencv_net, colorize_torch, colorize_opencv, detect_image_type,
          auto_enhance

HOW TO USE:
    python compare_models.py                                        # auto-detect
    python compare_models.py --input dataset/val/000000000139.jpg
    python compare_models.py --input photo.jpg                       # metrics at boost=1.0 (fair)
    python compare_models.py --input photo.jpg --boost 1.4           # vivid, same boost for both
    python compare_models.py --input photo.jpg --boost-display 1.4   # accurate metrics + vivid figure
    python compare_models.py --input photo.jpg --no-opencv          # skip OpenCV

OUTPUT:
    comparison/comparison_FILENAME.png    ← matplotlib figure (for presentations)
    comparison/gan_FILENAME.jpg           ← GAN result alone
    comparison/resnet_FILENAME.jpg        ← ResNet result alone
    comparison/opencv_FILENAME.jpg        ← OpenCV result alone
"""

# ── Standard library ───────────────────────────────────────────────────────────
import argparse
import sys

# ── Scientific computing ───────────────────────────────────────────────────────
import numpy as np

# ── Deep learning ──────────────────────────────────────────────────────────────
import torch

# ── Image processing ───────────────────────────────────────────────────────────
from pathlib import Path
from PIL import Image

# ── Quality metrics ────────────────────────────────────────────────────────────
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity   as ssim_fn

# ── Plotting ───────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# ── Shared utilities (utils.py) ────────────────────────────────────────────────
from colorize_utils import (
    DEVICE,               # "cuda" or "cpu"
    find_latest,          # find highest-epoch checkpoint matching a glob pattern
    load_best_config,     # read best_checkpoints.json
    load_opencv_net,      # load Zhang 2016 OpenCV model (auto-downloads)
    colorize_torch,       # LAB colorization with PyTorch model
    colorize_opencv,      # colorization with OpenCV Zhang 2016 model
    detect_image_type,    # Haar Cascade face detection → 'portrait' or 'landscape'
    auto_enhance,         # smart post-processing matching backend.py
)

# ── Command-line arguments ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Compare GAN, ResNet, and OpenCV on a single image"
)
parser.add_argument("--input",     default="dataset/val/000000000139.jpg",
                    help="Input image path")
parser.add_argument("--gan",       default=None,
                    help="GAN checkpoint path. Auto-selected from best_checkpoints.json if omitted.")
parser.add_argument("--resnet",    default=None,
                    help="ResNet checkpoint path. Auto-selected from best_checkpoints.json if omitted.")
parser.add_argument("--output",    default="comparison",
                    help="Output folder (default: comparison/)")
parser.add_argument("--boost",        type=float, default=1.0,
                    help="Boost for METRIC computation. Default 1.0 = raw output = "
                         "matches evaluate.py (fair PSNR/SSIM). Use 1.4 for vivid output.")
parser.add_argument("--boost-display", dest="boost_display", type=float, default=None,
                    help="Boost for the saved FIGURE only (default: same as --boost). "
                         "Use --boost 1.0 --boost-display 1.4 for accurate metrics + vivid images.")
parser.add_argument("--no-opencv", action="store_true",
                    help="Skip OpenCV model")
args = parser.parse_args()

# ── Evaluation resolution ─────────────────────────────────────────────────────
# All three models are evaluated at the same resolution for a fair comparison.
# 512 matches backend.py and evaluate.py — so numbers are consistent everywhere.
# Previously compare_models.py used 256 for OpenCV and 512 for GAN/ResNet,
# which made OpenCV look artificially better (lower resolution = higher PSNR).
EVAL_SIZE = 512

# Resolve display boost — if not set separately, same as metric boost
if args.boost_display is None:
    args.boost_display = args.boost

print("=" * 65)
print("  compare_models.py — GAN vs ResNet vs OpenCV")
print("=" * 65)
print(f"  Device : {DEVICE}")

# ── Checkpoint auto-detection ─────────────────────────────────────────────────
# Use best_checkpoints.json portrait keys (best for demo images with faces)
cfg = load_best_config()   # from utils.py — reads best_checkpoints.json

if not args.gan:
    args.gan = cfg.get("gan_portrait") or find_latest("generator_epoch*.pth")

if not args.resnet:
    args.resnet = cfg.get("resnet_portrait") or find_latest("generator_resnet_epoch*.pth")

print(f"  GAN    : {args.gan    or '❌ not found'}")
print(f"  ResNet : {args.resnet or '❌ not found'}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════

results = {}   # key → dict with model, name, colour, output arrays, metrics

# Load GAN
if args.gan:
    from models.generator import Generator as GAN_Gen
    G_gan = GAN_Gen().to(DEVICE)
    G_gan.load_state_dict(torch.load(args.gan, map_location=DEVICE))
    G_gan.eval()
    gan_name = Path(args.gan).stem.replace("generator_", "")   # human-readable label
    print(f"✅ GAN loaded    : {gan_name}")
    results["gan"] = {"model": G_gan, "name": gan_name, "color": "#4f8ef7", "fn": "torch"}
else:
    print("⚠️  GAN skipped  : checkpoint not found")

# Load ResNet
if args.resnet:
    from models.generator_resnet import Generator as ResNet_Gen
    G_res = ResNet_Gen(pretrained=False).to(DEVICE)
    G_res.load_state_dict(torch.load(args.resnet, map_location=DEVICE))
    G_res.eval()
    resnet_name = Path(args.resnet).stem.replace("generator_", "")
    print(f"✅ ResNet loaded : {resnet_name}")
    results["resnet"] = {"model": G_res, "name": resnet_name, "color": "#3dd68c", "fn": "torch"}
else:
    print("⚠️  ResNet skipped: checkpoint not found")

# Load OpenCV (optional)
if not args.no_opencv:
    try:
        opencv_net = load_opencv_net()   # from utils.py — auto-downloads if missing
        print("✅ OpenCV loaded : Zhang et al. 2016 (pretrained)")
        results["opencv"] = {
            "model": opencv_net, "name": "Zhang2016 (OpenCV)",
            "color": "#f0b429", "fn": "opencv"
        }
    except ImportError:
        print("⚠️  OpenCV skipped: pip install opencv-python")
    except Exception as e:
        print(f"⚠️  OpenCV skipped: {e}")

if not results:
    print("❌ No models available. Exiting.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD INPUT IMAGE
# ══════════════════════════════════════════════════════════════════════════════

if not Path(args.input).exists():
    print(f"❌ Input not found: {args.input}")
    sys.exit(1)

img_orig   = Image.open(args.input).convert("RGB")
img_gray   = img_orig.convert("L").convert("RGB")          # grayscale for display
orig_np = np.array(img_orig.resize((EVAL_SIZE, EVAL_SIZE), Image.LANCZOS))   # ground truth at eval resolution for metrics

print(f"\n📷 Input: {args.input}  ({img_orig.size[0]}×{img_orig.size[1]})")

# Detect scene type once — applied to all models for consistent output
image_type = detect_image_type(img_orig)   # from utils.py
print(f"🔍 Scene type  : {image_type.upper()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RUN ALL MODELS
# colorize_torch / colorize_opencv from utils.py
# auto_enhance from utils.py — same settings as backend.py
# ══════════════════════════════════════════════════════════════════════════════

print()
for key, info in results.items():
    print(f"🎨 Colorizing with {info['name']}...")

    # ── Step 1: colorize at METRIC boost (default=1.0 = raw, matches evaluate.py) ─
    # No enhancement applied — raw output so PSNR/SSIM is directly comparable
    if info["fn"] == "torch":
        metric_pil, _ = colorize_torch(info["model"], img_orig, args.boost)
    else:
        metric_pil, _ = colorize_opencv(info["model"], img_orig, args.boost)

    # Compute PSNR/SSIM — all 3 models at same EVAL_SIZE, no boost amplification
    rgb_eval = np.array(metric_pil.resize((EVAL_SIZE, EVAL_SIZE), Image.LANCZOS))
    p = round(float(psnr_fn(orig_np, rgb_eval, data_range=255)), 2)
    s = round(float(ssim_fn(orig_np, rgb_eval, channel_axis=2, data_range=255)), 4)

    # ── Step 2: colorize at DISPLAY boost for the saved figure ───────────
    # If boost-display == boost, reuse metric_pil — avoids a second inference
    if abs(args.boost_display - args.boost) > 0.01:
        if info["fn"] == "torch":
            disp_pil, _ = colorize_torch(info["model"], img_orig, args.boost_display)
        else:
            disp_pil, _ = colorize_opencv(info["model"], img_orig, args.boost_display)
    else:
        disp_pil = metric_pil   # same boost — no second inference needed

    # Apply smart enhancement to the display image (mirrors backend.py)
    disp_pil, _ = auto_enhance(disp_pil, image_type)
    rgb_out = np.array(disp_pil)   # used for the figure and saved JPEG files

    info["rgb_out"] = rgb_out   # full-res enhanced array for figure
    info["psnr"]    = p
    info["ssim"]    = s
    print(f"   PSNR={p} dB  SSIM={s}")


# ── Print summary table ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  COMPARISON SUMMARY")
print("=" * 65)
print(f"  {'Model':<30} {'PSNR':>8}  {'SSIM':>8}")
print(f"  {'-'*50}")
for key, info in results.items():
    print(f"  {info['name']:<30} {str(info['psnr'])+' dB':>8}  {str(info['ssim']):>8}")
print("=" * 65)

if len(results) > 1:
    best_psnr = max(results.items(), key=lambda x: x[1]["psnr"])
    best_ssim = max(results.items(), key=lambda x: x[1]["ssim"])
    print(f"  🏆 Best PSNR : {best_psnr[1]['name']}  ({best_psnr[1]['psnr']} dB)")
    print(f"  🏆 Best SSIM : {best_ssim[1]['name']}  ({best_ssim[1]['ssim']})")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MATPLOTLIB COMPARISON FIGURE
# Dark-themed layout matching the web UI:
# Grayscale Input | GAN | ResNet | [OpenCV] | Ground Truth
# ══════════════════════════════════════════════════════════════════════════════


n_cols    = len(results) + 2   # +2 for grayscale + ground truth
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
if n_cols == 1:
    axes = [axes]
fig.patch.set_facecolor("#0a0b0e")   # dark background matching web UI

# Build display order: grayscale | models | ground truth
# Display images at 256×256 for the matplotlib figure (smaller = faster rendering)
# Note: this is display-only — metrics above were computed at EVAL_SIZE=512
DISP_SIZE  = 256
orig_disp  = np.array(img_orig.resize((DISP_SIZE, DISP_SIZE), Image.LANCZOS))
gray_disp  = np.array(img_gray.resize((DISP_SIZE, DISP_SIZE), Image.LANCZOS))

plot_items = [("Grayscale Input", gray_disp, "#888888", "")]
for key, info in results.items():
    disp        = np.array(Image.fromarray(info["rgb_out"]).resize((DISP_SIZE, DISP_SIZE), Image.LANCZOS))
    metrics_txt = f"PSNR: {info['psnr']} dB\nSSIM: {info['ssim']}"
    plot_items.append((info["name"], disp, info["color"], metrics_txt))
plot_items.append(("Ground Truth\n(Original)", orig_disp, "#f0b429", ""))

for ax, (title, img, color, metrics) in zip(axes, plot_items):
    ax.imshow(img)
    ax.set_facecolor("#111318")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)   # colour-coded borders per model
        spine.set_linewidth(2.5)
    ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=10)
    if metrics:
        ax.text(0.5, -0.06, metrics, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5,
                color="white", fontfamily="monospace")

fig.suptitle(
    f"Image Colorization — Model Comparison\nInput: {Path(args.input).name}",
    color="white", fontsize=13, y=1.02
)
plt.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

out_dir = Path(args.output)
out_dir.mkdir(exist_ok=True)
stem = Path(args.input).stem

# Save matplotlib comparison figure
cmp_path = out_dir / f"comparison_{stem}.png"
plt.savefig(cmp_path, dpi=150, bbox_inches="tight",
            facecolor="#0a0b0e", edgecolor="none")
plt.show()
print(f"\n✅ Comparison figure → {cmp_path}")

# Save individual model outputs as JPEG
for key, info in results.items():
    out_path = out_dir / f"{key}_{stem}.jpg"
    Image.fromarray(info["rgb_out"]).save(out_path, quality=95)
    print(f"✅ {key.upper()} output     → {out_path}")

print(f"\n✅ All outputs saved to: {out_dir}/")
