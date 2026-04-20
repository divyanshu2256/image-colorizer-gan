"""
inference.py — Command-Line Image Colorization Tool
====================================================
PURPOSE : Colorize a single image (or compare all three models) from the
          command line. Used for quick testing and producing demo images.

SMART CHECKPOINT SELECTION:
          Reads best_checkpoints.json automatically.
          --model gan    → uses best gan_portrait checkpoint
          --model resnet → uses best resnet_portrait checkpoint
          --model auto   → detects image type, picks best checkpoint
          Falls back to latest .pth file if JSON doesn't exist.

SHARED CODE:
          Imports from utils.py — find_latest, load_best_config, load_torch_generator,
          load_opencv_net, colorize_torch, colorize_opencv, detect_image_type,
          auto_enhance, enhance_manual, compute_metrics

HOW TO USE:
    python inference.py --input image.jpg                   # auto-select best checkpoint
    python inference.py --input image.jpg --model resnet    # best ResNet checkpoint
    python inference.py --input image.jpg --model gan       # best GAN checkpoint
    python inference.py --input image.jpg --model opencv    # OpenCV Zhang 2016
    python inference.py --input image.jpg --model checkpoints/generator_epoch130.pth
    python inference.py --input image.jpg --both            # GAN + ResNet side-by-side
    python inference.py --input image.jpg --all             # all 3 models side-by-side
    python inference.py --input image.jpg --no-enhance      # raw output, no post-processing

OUTPUT:
    colorized/colorized_IMAGE.jpg         (single model)
    colorized/IMAGE_comparison.jpg        (--both or --all)
"""

# ── Standard library ───────────────────────────────────────────────────────────
import argparse
import sys
import warnings

# ── Deep learning ──────────────────────────────────────────────────────────────
import torch

# ── Image processing ───────────────────────────────────────────────────────────
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Suppress minor library warnings (PyTorch FutureWarning etc.)
warnings.filterwarnings("ignore")

# ── Shared utilities (utils.py) ────────────────────────────────────────────────
from colorize_utils import (
    DEVICE,                # "cuda" or "cpu"
    find_latest,           # find highest-epoch checkpoint
    load_best_config,      # read best_checkpoints.json
    load_torch_generator,  # load GAN or ResNet .pth checkpoint
    load_opencv_net,       # load Zhang 2016 OpenCV model
    colorize_torch,        # LAB colorization with PyTorch model
    colorize_opencv,       # colorization with OpenCV Zhang 2016 model
    detect_image_type,     # Haar Cascade → 'portrait' or 'landscape'
    auto_enhance,          # smart post-processing matching backend.py
    enhance_manual,        # fixed enhancement (used with --no-auto)
    compute_metrics,       # PSNR and SSIM between two images
)

# ── Comparison image panel colours ────────────────────────────────────────────
PANEL_COLORS = {
    "gan":    "#4f8ef7",   # blue
    "resnet": "#3dd68c",   # green
    "opencv": "#f0b429",   # gold
    "gray":   "#888888",   # grey for Grayscale Input panel
    "gt":     "#ffffff",   # white for Ground Truth panel
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SMART CHECKPOINT SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def resolve_model_path(model_arg, image_type=None):
    """
    Determine which checkpoint file to load based on --model argument and
    optionally the detected image type.

    Priority order:
      1. Direct file path provided → use it as-is
      2. 'opencv' → no .pth needed
      3. 'gan' / 'resnet' → read best_checkpoints.json for {model}_{scene} key
      4. None (auto) → prefer best ResNet from JSON
      5. Fallback → latest .pth file

    Uses load_best_config() and find_latest() from utils.py.

    Returns: (path_str_or_None, kind_str)
      path: checkpoint file path, or None for OpenCV
      kind: 'gan', 'resnet', or 'opencv'
    """
    cfg   = load_best_config()    # from utils.py
    scene = image_type or "portrait"   # default to portrait (safe for mixed images)

    # ── Direct file path ───────────────────────────────────────────────────
    if model_arg and model_arg.lower() not in ("gan", "resnet", "opencv", "auto"):
        kind = "resnet" if "resnet" in model_arg.lower() else "gan"
        return model_arg, kind

    # ── OpenCV — no .pth file needed ──────────────────────────────────────
    if model_arg and model_arg.lower() == "opencv":
        return None, "opencv"

    # ── Explicit 'gan' ────────────────────────────────────────────────────
    if model_arg and model_arg.lower() == "gan":
        key       = f"gan_{scene}"   # e.g. "gan_portrait"
        preferred = cfg.get(key)
        if preferred and Path(preferred).exists():
            print(f"   Smart select [gan]: {key} → {Path(preferred).name}")
            return str(preferred), "gan"
        return find_latest("generator_epoch*.pth"), "gan"

    # ── Explicit 'resnet' ─────────────────────────────────────────────────
    if model_arg and model_arg.lower() == "resnet":
        key       = f"resnet_{scene}"   # e.g. "resnet_portrait"
        preferred = cfg.get(key)
        if preferred and Path(preferred).exists():
            print(f"   Smart select [resnet]: {key} → {Path(preferred).name}")
            return str(preferred), "resnet"
        return find_latest("generator_resnet_epoch*.pth"), "resnet"

    # ── Auto mode — prefer ResNet (best overall metrics) ──────────────────
    if cfg:
        key       = f"resnet_{scene}"
        preferred = cfg.get(key)
        if preferred and Path(preferred).exists():
            print(f"   Smart select [auto → resnet]: {key} → {Path(preferred).name}")
            return str(preferred), "resnet"
        # Try any ResNet key in config
        for fallback_key in ("resnet_portrait", "resnet_landscape"):
            fallback = cfg.get(fallback_key)
            if fallback and Path(fallback).exists():
                print(f"   Smart select [auto → resnet]: {fallback_key} → {Path(fallback).name}")
                return str(fallback), "resnet"

    # Final fallback: latest checkpoint file
    resnet_latest = find_latest("generator_resnet_epoch*.pth")
    if resnet_latest:
        return resnet_latest, "resnet"
    return find_latest("generator_epoch*.pth"), "gan"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL LOADER (with sys.exit on missing checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

def load_torch_model(path, kind):
    """
    Load a checkpoint with an informative error if the file is missing.
    Wraps utils.load_torch_generator() with a user-friendly error message.
    """
    if not path:
        print(f"❌ No checkpoint found for '{kind}'")
        print("   Expected: checkpoints/generator_epoch*.pth")
        print("          or checkpoints/generator_resnet_epoch*.pth")
        sys.exit(1)

    model = load_torch_generator(path, resnet=(kind == "resnet"))   # from utils.py
    print(f"✅ Loaded [{kind.upper()}] → {path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COMPARISON IMAGE BUILDER (for --both and --all modes)
# Layout: Grayscale Input | GAN | ResNet | [OpenCV] | Ground Truth
# ══════════════════════════════════════════════════════════════════════════════

def make_comparison(panels):
    """
    Build a side-by-side comparison image from a list of panels.

    Each panel dict must have:
      label   — text shown above the image
      img_pil — PIL Image
      color   — hex colour for the label and border
      psnr    — optional float (shown below image)
      ssim    — optional float (shown below image)

    Returns a single composite PIL Image on dark background.
    """
    try:
        font_sm = ImageFont.truetype("arial.ttf", 13)
        font_xs = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_sm = ImageFont.load_default()   # fallback on Linux
        font_xs = font_sm

    W, H    = panels[0]["img_pil"].size
    n       = len(panels)
    pad     = 10      # pixels between panels
    label_h = 28      # height for label text above image
    total_w = W * n + pad * (n + 1)
    total_h = H + pad * 2 + label_h * 2
    canvas  = Image.new("RGB", (total_w, total_h), (15, 17, 20))   # near-black background
    draw    = ImageDraw.Draw(canvas)

    for i, p in enumerate(panels):
        x = pad + i * (W + pad)
        canvas.paste(p["img_pil"], (x, pad + label_h))
        draw.text((x + 4, pad + 4), p["label"], fill=p["color"], font=font_sm)
        if p.get("psnr") is not None:
            m = f"PSNR {p['psnr']:.2f} dB   SSIM {p['ssim']:.4f}"
            draw.text((x + 4, H + pad + label_h + 6), m, fill="#aaaaaa", font=font_xs)

    # Add "✓ Best SSIM" badge to winning panel
    scored = [p for p in panels if p.get("ssim") is not None]
    if len(scored) > 1:
        winner = max(scored, key=lambda p: p["ssim"])
        wi     = panels.index(winner)
        wx     = pad + wi * (W + pad) + W - 92
        wc     = winner["color"]
        draw.rectangle([wx, pad + 2, wx + 88, pad + label_h - 2], outline=wc)
        draw.text((wx + 5, pad + 5), "✓  Best SSIM", fill=wc, font=font_xs)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Colorize images using GAN, ResNet, or OpenCV"
    )
    parser.add_argument("--input",      required=True,
                        help="Path to input image file")
    parser.add_argument("--model",      default=None,
                        help='"gan" | "resnet" | "opencv" | checkpoint path. '
                             'Default: auto (smart config, prefers ResNet)')
    parser.add_argument("--both",       action="store_true",
                        help="GAN + ResNet side-by-side comparison")
    parser.add_argument("--all",        action="store_true",
                        help="GAN + ResNet + OpenCV side-by-side comparison")
    parser.add_argument("--output",     default="colorized",
                        help="Output folder (default: colorized/)")
    parser.add_argument("--boost",      type=float, default=1.4,
                        help="AB colour boost multiplier (default: 1.4)")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Skip all post-processing (raw model output)")
    parser.add_argument("--no-auto",    action="store_true",
                        help="Use fixed enhancement instead of smart auto-enhance")
    parser.add_argument("--size",       type=int, default=512,
                        help="Inference resolution in pixels (default: 512)")
    args = parser.parse_args()

    print(f"🖥️  Device : {DEVICE}")

    img_path = Path(args.input)
    if not img_path.exists():
        print(f"❌ Input not found: {args.input}"); sys.exit(1)

    img_orig = Image.open(img_path).convert("RGB")
    gray_pil = img_orig.convert("L").convert("RGB")
    print(f"📷 Input  : {img_path.name}  ({img_orig.size[0]}×{img_orig.size[1]})")

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    stem = img_path.stem

    # ── Multi-model comparison (--both or --all) ──────────────────────────────
    if args.all or args.both:
        model_specs = []

        gan_path, _ = resolve_model_path("gan")
        if gan_path: model_specs.append(("gan", gan_path, "torch"))
        else:        print("⚠️  GAN checkpoint not found — skipping")

        res_path, _ = resolve_model_path("resnet")
        if res_path: model_specs.append(("resnet", res_path, "torch"))
        else:        print("⚠️  ResNet checkpoint not found — skipping")

        if args.all:
            model_specs.append(("opencv", None, "opencv"))

        if not model_specs:
            print("❌ No models available"); sys.exit(1)

        # Load all models
        loaded = {}
        for kind, path, mtype in model_specs:
            if mtype == "torch":
                loaded[kind] = ("torch", load_torch_model(path, kind), path)
            else:
                try:
                    loaded[kind] = ("opencv", load_opencv_net(), None)   # utils.py
                except Exception as e:
                    print(f"⚠️  OpenCV failed: {e}")

        print()
        img_type = detect_image_type(img_orig)   # utils.py
        print(f"🔍 Image type detected: {img_type.upper()}")

        panels     = [{"label": "Grayscale Input",
                       "img_pil": gray_pil.resize(img_orig.size),
                       "color": PANEL_COLORS["gray"]}]
        print_rows = []

        for kind, (mtype, model, path) in loaded.items():
            print(f"🎨 Colorizing with {kind.upper()}...")
            if mtype == "torch":
                result, gt_np = colorize_torch(model, img_orig, args.boost, args.size)
            else:
                result, gt_np = colorize_opencv(model, img_orig, args.boost)

            p, s = compute_metrics(gt_np, result)   # utils.py

            if not args.no_enhance:
                if args.no_auto:
                    result = enhance_manual(result, saturation=1.25, contrast=1.1, sharpness=1.6)
                else:
                    result, img_type = auto_enhance(result, img_type)   # utils.py

            label_str = Path(path).stem if path else "OpenCV_Zhang2016"
            out_path  = out_dir / f"{stem}_{kind}.jpg"
            result.save(out_path, quality=95)

            panels.append({"label": f"{kind.upper()}  {label_str}",
                           "img_pil": result, "color": PANEL_COLORS[kind],
                           "psnr": p, "ssim": s})
            print_rows.append((label_str, p, s))
            print(f"   ✅ PSNR={p} dB  SSIM={s}  → {out_path}")

        # Ground Truth panel (rightmost)
        panels.append({"label": "Ground Truth", "img_pil": img_orig,
                       "color": PANEL_COLORS["gt"]})

        cmp_out = out_dir / f"{stem}_comparison.jpg"
        make_comparison(panels).save(cmp_out, quality=95)

        print(f"\n{'='*55}")
        print(f"  {'Model':<32} {'PSNR':>8}  {'SSIM':>8}")
        print(f"  {'-'*52}")
        for label_str, p, s in print_rows:
            print(f"  {label_str:<32} {p:>7.2f}  {s:>8.4f}")
        print(f"{'='*55}")
        winner = max(print_rows, key=lambda x: x[2])
        print(f"  🏆 Best SSIM : {winner[0]}")
        print(f"{'='*55}")
        print(f"\n✅ Comparison image → {cmp_out}")

    # ── Single model mode ─────────────────────────────────────────────────────
    else:
        detected_type = detect_image_type(img_orig) if args.model is None else None
        if detected_type:
            print(f"🔍 Image type: {detected_type.upper()}  (checking best_checkpoints.json...)")

        model_path, kind = resolve_model_path(args.model, detected_type)

        if kind == "opencv":
            try:
                model = load_opencv_net()   # utils.py
            except ImportError:
                print("❌ opencv-python not installed: pip install opencv-python"); sys.exit(1)
            print(f"\n🎨 Colorizing with OpenCV...")
            result, gt_np = colorize_opencv(model, img_orig, args.boost)
        else:
            model = load_torch_model(model_path, kind)
            print(f"\n🎨 Colorizing with {kind.upper()}...")
            result, gt_np = colorize_torch(model, img_orig, args.boost, args.size)

        p, s = compute_metrics(gt_np, result)   # utils.py

        if not args.no_enhance:
            if args.no_auto:
                result = enhance_manual(result, saturation=1.25, contrast=1.1, sharpness=1.6)
                print("✅ Fixed post-processing applied")
            else:
                img_type = detected_type or detect_image_type(img_orig)
                result, img_type = auto_enhance(result, img_type)   # utils.py

        out_path = out_dir / f"colorized_{stem}.jpg"
        result.save(out_path, quality=95)

        print(f"\n{'='*40}")
        print(f"  PSNR : {p:.2f} dB")
        print(f"  SSIM : {s:.4f}")
        print(f"{'='*40}")
        print(f"✅ Saved → {out_path}")


if __name__ == "__main__":
    main()
