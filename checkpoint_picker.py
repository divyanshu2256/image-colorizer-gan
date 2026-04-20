"""
checkpoint_picker.py — Auto-Find Best Checkpoints and Write best_checkpoints.json
==================================================================================
PURPOSE : Scans ALL .pth checkpoints in checkpoints/, evaluates each one on
          validation images, and writes best_checkpoints.json with the best
          checkpoint for each model type × scene type combination.

          backend.py and inference.py read best_checkpoints.json automatically
          to serve the highest-quality result for each image.

JSON WRITTEN (4 model-specific keys):
    {
      "gan_portrait":    "checkpoints/generator_epoch178.pth",
      "gan_landscape":   "checkpoints/generator_epoch131.pth",
      "resnet_portrait": "checkpoints/generator_resnet_epoch31.pth",
      "resnet_landscape":"checkpoints/generator_resnet_epoch8.pth",
      "_meta": { ... }
    }

SHARED CODE:
          Imports from utils.py — load_torch_generator, DEVICE, colorize_torch,
          detect_image_type

HOW TO USE:
    python checkpoint_picker.py                    # test all, 20 images each
    python checkpoint_picker.py --limit 50         # better accuracy, ~45 min on GPU
    python checkpoint_picker.py --type resnet      # only ResNet checkpoints
    python checkpoint_picker.py --type gan         # only GAN checkpoints

OUTPUT:
    best_checkpoints.json    ← read by backend.py and inference.py
    checkpoint_results.json  ← full scores for every checkpoint tested
"""

# ── Standard library ───────────────────────────────────────────────────────────
import argparse
import json
import sys

# ── Scientific computing ───────────────────────────────────────────────────────
import numpy as np

# ── Image processing ───────────────────────────────────────────────────────────
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Quality metrics ────────────────────────────────────────────────────────────
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity   as ssim_fn

# ── Shared utilities (utils.py) ────────────────────────────────────────────────
from colorize_utils import (
    DEVICE,                # "cuda" or "cpu"
    load_torch_generator,  # load GAN or ResNet .pth checkpoint
    detect_image_type,     # Haar Cascade face detector → 'portrait' or 'landscape'
    colorize_torch,        # LAB colorization for scoring each checkpoint
)

# ── Command-line arguments ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Find best checkpoint per model type and scene type"
)
parser.add_argument("--images", default="dataset/val",
                    help="Validation images folder (default: dataset/val)")
parser.add_argument("--limit",  type=int, default=20,
                    help="Images to test per checkpoint per scene type (default: 20). "
                         "50 gives better accuracy.")
parser.add_argument("--type",   default="all", choices=["all", "gan", "resnet"],
                    help="Which checkpoints to test: all | gan | resnet (default: all)")
parser.add_argument("--output", default="best_checkpoints.json",
                    help="Output JSON path (default: best_checkpoints.json)")
parser.add_argument("--size",   type=int, default=512,
                    help="Inference resolution — must match backend.py (default: 512)")
args = parser.parse_args()

print("=" * 65)
print("  checkpoint_picker.py — Find Best Checkpoint Per Scene Type")
print("=" * 65)
print(f"  Device  : {DEVICE}")
print(f"  Images  : {args.images}  (limit {args.limit} per type)")
print(f"  Type    : {args.type} checkpoints")
print(f"  Size    : {args.size}×{args.size}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FIND ALL CHECKPOINT FILES
# ══════════════════════════════════════════════════════════════════════════════

def find_all_checkpoints(ckpt_type="all"):
    """
    Scan checkpoints/ and return a list of (Path, kind) tuples.

    Naming conventions:
      GAN    : generator_epochXX.pth          (no 'resnet' in filename)
      ResNet : generator_resnet_epochXX.pth

    Sorted by epoch number for chronological display in results table.
    """
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        print("❌ checkpoints/ folder not found!")
        sys.exit(1)

    found = []

    # GAN: 'generator_epoch*.pth' excluding any file with 'resnet' in name
    if ckpt_type in ("all", "gan"):
        for p in sorted(
            [f for f in ckpt_dir.glob("generator_epoch*.pth") if "resnet" not in f.name],
            key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)
        ):
            found.append((p, "gan"))

    # ResNet: 'generator_resnet_epoch*.pth'
    if ckpt_type in ("all", "resnet"):
        for p in sorted(
            ckpt_dir.glob("generator_resnet_epoch*.pth"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)
        ):
            found.append((p, "resnet"))

    return found


checkpoints = find_all_checkpoints(args.type)

if not checkpoints:
    print(f"❌ No checkpoints found in checkpoints/ for type='{args.type}'")
    sys.exit(1)

print(f"\n📦 Found {len(checkpoints)} checkpoint(s):")
for p, kind in checkpoints:
    mb = p.stat().st_size / (1024 * 1024)
    print(f"   [{kind.upper():6}] {p.name:<45} {mb:.0f} MB")
print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CATEGORISE VALIDATION IMAGES (portrait vs landscape)
# Uses detect_image_type() from utils.py (Haar Cascade + aspect ratio fallback)
# ══════════════════════════════════════════════════════════════════════════════

print("🔍 Categorising validation images by scene type...")

img_dir = Path(args.images)
if not img_dir.exists():
    print(f"❌ Images folder not found: {args.images}")
    sys.exit(1)

all_imgs   = sorted(img_dir.glob("*.jpg"))
portraits  = []
landscapes = []

# Scan up to 200 images to find enough of each type
for path in tqdm(all_imgs[:200], desc="  Detecting scene types"):
    try:
        img = Image.open(path).convert("RGB")
        # detect_image_type() imported from utils.py — Haar Cascade face detection
        if detect_image_type(img) == "portrait":
            portraits.append(path)
        else:
            landscapes.append(path)
    except Exception:
        pass

portraits  = portraits[:args.limit]    # cap at requested limit
landscapes = landscapes[:args.limit]

print(f"  Portrait  images : {len(portraits)}")
print(f"  Landscape images : {len(landscapes)}")

# Safety fallback: if not enough of one type, use first N images for that type
if not portraits:
    print("  ⚠️  No portrait images found — using first images as portrait test set")
    portraits = list(all_imgs[:args.limit])
if not landscapes:
    print("  ⚠️  No landscape images found — using first images as landscape test set")
    landscapes = list(all_imgs[:args.limit])
print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SCORE ONE IMAGE WITH ONE CHECKPOINT
# Uses colorize_torch() from utils.py with boost=1.0 (raw output for fair eval)
# ══════════════════════════════════════════════════════════════════════════════

def score_image(model, img_path, size):
    """
    Run model on one image and return (psnr, ssim).

    Uses colorize_torch() from utils.py with boost=1.0 — raw model output
    with no colour amplification, for a fair apples-to-apples comparison.

    Returns: (psnr_float, ssim_float) or (None, None) on any error.
    """
    try:
        img_pil = Image.open(img_path).convert("RGB")

        # colorize_torch from utils.py — returns (colorized PIL, original numpy at size×size)
        _, img_np = colorize_torch(model, img_pil, boost=1.0, size=size)

        # We need the colorized numpy — re-run to get both arrays
        # (colorize_torch returns (PIL, numpy_at_size), so use the numpy directly)
        result_pil, img_np = colorize_torch(model, img_pil, boost=1.0, size=size)

        import numpy as np
        from PIL import Image as PILImage
        result_np = np.array(result_pil.resize((size, size), PILImage.LANCZOS))

        p = float(psnr_fn(img_np, result_np, data_range=255))
        s = float(ssim_fn(img_np, result_np, channel_axis=2, data_range=255))
        return round(p, 4), round(s, 4)

    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RUN ALL CHECKPOINTS ON ALL IMAGES
# ══════════════════════════════════════════════════════════════════════════════

# all_results: {checkpoint_name: {path, kind, portrait:{psnr,ssim,n}, landscape:{...}}}
all_results = {}

for ci, (ckpt_path, kind) in enumerate(checkpoints):
    name = ckpt_path.name
    print(f"[{ci+1}/{len(checkpoints)}] {name}  [{kind.upper()}]")

    # load_torch_generator from utils.py
    try:
        model = load_torch_generator(str(ckpt_path), resnet=(kind == "resnet"))
    except Exception as e:
        print(f"  ❌ Load failed: {e}\n")
        continue

    entry = {"path": str(ckpt_path), "kind": kind}

    # Score portrait images
    p_psnrs, p_ssims = [], []
    for img_path in tqdm(portraits, desc="  Portrait ", leave=False):
        p, s = score_image(model, img_path, args.size)
        if p is not None:
            p_psnrs.append(p); p_ssims.append(s)

    if p_psnrs:
        entry["portrait"] = {
            "psnr": round(float(np.mean(p_psnrs)), 4),
            "ssim": round(float(np.mean(p_ssims)), 4),
            "n":    len(p_psnrs),
        }
        print(f"  Portrait  → PSNR {entry['portrait']['psnr']:.4f}  "
              f"SSIM {entry['portrait']['ssim']:.4f}  ({len(p_psnrs)} imgs)")

    # Score landscape images
    l_psnrs, l_ssims = [], []
    for img_path in tqdm(landscapes, desc="  Landscape", leave=False):
        p, s = score_image(model, img_path, args.size)
        if p is not None:
            l_psnrs.append(p); l_ssims.append(s)

    if l_psnrs:
        entry["landscape"] = {
            "psnr": round(float(np.mean(l_psnrs)), 4),
            "ssim": round(float(np.mean(l_ssims)), 4),
            "n":    len(l_psnrs),
        }
        print(f"  Landscape → PSNR {entry['landscape']['psnr']:.4f}  "
              f"SSIM {entry['landscape']['ssim']:.4f}  ({len(l_psnrs)} imgs)")

    # Overall combined score
    all_p, all_s = p_psnrs + l_psnrs, p_ssims + l_ssims
    if all_p:
        entry["overall"] = {
            "psnr": round(float(np.mean(all_p)), 4),
            "ssim": round(float(np.mean(all_s)), 4),
        }

    all_results[name] = entry

    # Free GPU memory before loading the next checkpoint
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()


if not all_results:
    print("❌ No checkpoints could be evaluated.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — FIND BEST CHECKPOINT PER MODEL KIND × SCENE TYPE
# Searches independently for each of the 4 combinations:
#   gan_portrait, gan_landscape, resnet_portrait, resnet_landscape
# Primary metric: SSIM (better perceptual quality indicator for colorization)
# ══════════════════════════════════════════════════════════════════════════════

def best_for(results, kind_filter, scene, metric="ssim"):
    """
    Find the best checkpoint for a specific model kind AND scene type.

    Filters by kind_filter first (avoids ResNet beating GAN for 'gan_portrait' key).
    Returns: (checkpoint_name, score) or (None, 0.0) if no candidates found.
    """
    best_name  = None
    best_score = -1.0
    for name, data in results.items():
        if data.get("kind") != kind_filter:
            continue   # skip wrong model type
        if scene not in data:
            continue   # skip if this scene wasn't evaluated
        score = data[scene].get(metric, -1.0)
        if score > best_score:
            best_score = score
            best_name  = name
    return best_name, best_score


best_gan_port_name,  best_gan_port_ssim  = best_for(all_results, "gan",    "portrait")
best_gan_land_name,  best_gan_land_ssim  = best_for(all_results, "gan",    "landscape")
best_res_port_name,  best_res_port_ssim  = best_for(all_results, "resnet", "portrait")
best_res_land_name,  best_res_land_ssim  = best_for(all_results, "resnet", "landscape")


# ── Print results table ───────────────────────────────────────────────────────
print("\n" + "=" * 82)
print("  FULL RESULTS TABLE  (ranked by SSIM — higher is better)")
print("=" * 82)
print(f"  {'Checkpoint':<45} {'Kind':>6}  "
      f"{'P.PSNR':>7}  {'P.SSIM':>7}  {'L.PSNR':>7}  {'L.SSIM':>7}")
print(f"  {'-'*80}")

for name, data in all_results.items():
    kind = data.get("kind", "?").upper()
    pp   = f"{data['portrait']['psnr']:.4f}"  if "portrait"  in data else "  —  "
    ps   = f"{data['portrait']['ssim']:.4f}"  if "portrait"  in data else "  —  "
    lp   = f"{data['landscape']['psnr']:.4f}" if "landscape" in data else "  —  "
    ls   = f"{data['landscape']['ssim']:.4f}" if "landscape" in data else "  —  "

    marks = []
    if name == best_gan_port_name:  marks.append("◀ GAN portrait")
    if name == best_gan_land_name:  marks.append("◀ GAN landscape")
    if name == best_res_port_name:  marks.append("◀ ResNet portrait")
    if name == best_res_land_name:  marks.append("◀ ResNet landscape")
    mark_str = "  " + "  ".join(marks) if marks else ""

    print(f"  {name:<45} {kind:>6}  {pp:>7}  {ps:>7}  {lp:>7}  {ls:>7}{mark_str}")

print("=" * 82)
if best_gan_port_name:
    print(f"\n  🏆 Best GAN    portrait  : {best_gan_port_name}  (SSIM {best_gan_port_ssim:.4f})")
if best_gan_land_name:
    print(f"  🏆 Best GAN    landscape : {best_gan_land_name}  (SSIM {best_gan_land_ssim:.4f})")
if best_res_port_name:
    print(f"  🏆 Best ResNet portrait  : {best_res_port_name}  (SSIM {best_res_port_ssim:.4f})")
if best_res_land_name:
    print(f"  🏆 Best ResNet landscape : {best_res_land_name}  (SSIM {best_res_land_ssim:.4f})")
print("=" * 82)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — WRITE best_checkpoints.json
# Always writes all 4 model-specific keys.
# backend.py uses: lookup_key = f"{active_model}_{image_type}"
# ══════════════════════════════════════════════════════════════════════════════

def normalize_path(raw_path):
    """Forward slashes — safe on both Windows and Linux/Mac."""
    return str(raw_path).replace("\\", "/")


config = {}   # populated below
meta   = {
    "images_per_type":    args.limit,
    "eval_size":          args.size,
    "checkpoints_tested": len(all_results),
}

if best_gan_port_name:
    config["gan_portrait"] = normalize_path(all_results[best_gan_port_name]["path"])
    meta["gan_portrait_checkpoint"] = best_gan_port_name
    meta["gan_portrait_ssim"]       = best_gan_port_ssim
    meta["gan_portrait_psnr"]       = all_results[best_gan_port_name]["portrait"]["psnr"]

if best_gan_land_name:
    config["gan_landscape"] = normalize_path(all_results[best_gan_land_name]["path"])
    meta["gan_landscape_checkpoint"] = best_gan_land_name
    meta["gan_landscape_ssim"]       = best_gan_land_ssim
    meta["gan_landscape_psnr"]       = all_results[best_gan_land_name]["landscape"]["psnr"]

if best_res_port_name:
    config["resnet_portrait"] = normalize_path(all_results[best_res_port_name]["path"])
    meta["resnet_portrait_checkpoint"] = best_res_port_name
    meta["resnet_portrait_ssim"]       = best_res_port_ssim
    meta["resnet_portrait_psnr"]       = all_results[best_res_port_name]["portrait"]["psnr"]

if best_res_land_name:
    config["resnet_landscape"] = normalize_path(all_results[best_res_land_name]["path"])
    meta["resnet_landscape_checkpoint"] = best_res_land_name
    meta["resnet_landscape_ssim"]       = best_res_land_ssim
    meta["resnet_landscape_psnr"]       = all_results[best_res_land_name]["landscape"]["psnr"]

config["_meta"] = meta

# Save both output files
out_path     = Path(args.output)
results_path = Path("checkpoint_results.json")

with open(out_path, "w")     as f: json.dump(config,      f, indent=2)
with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

print(f"\n✅ best_checkpoints.json → {out_path}")
print(f"   Keys written : {[k for k in config if not k.startswith('_')]}")
print(f"\n✅ Full results  → {results_path}")
print(f"""
{'='*65}
  DONE — backend.py now reads best_checkpoints.json automatically.
  Run: python backend.py
{'='*65}
""")
