"""
backend.py — Flask Web Server for Image Colorization
=====================================================
PURPOSE:
  This is the MAIN SERVER file. Run this to start the web app.
  It loads all 3 AI models and serves the frontend (index.html).

THREE MODELS:
  1. GAN     → checkpoints/generator_epoch*.pth        (Pix2Pix U-Net, trained from scratch)
  2. ResNet  → checkpoints/generator_resnet_epoch*.pth (ResNet34 encoder, transfer learning)
  3. OpenCV  → models/ (Zhang 2016, auto-downloaded ~125 MB)

SMART FEATURES:
  - Auto-detects portrait vs landscape using Haar Cascade face detection
  - Picks best checkpoint per scene type from best_checkpoints.json
  - Auto-enhances colours differently for portraits vs landscapes
  - Switches checkpoints on-demand when user changes scene type

SHARED CODE:
  Imports from colorize_utils.py:
    DEVICE, load_best_config, load_torch_generator, load_opencv_net,
    colorize_torch, colorize_opencv, detect_image_type, auto_enhance,
    enhance_manual, compute_metrics

RUN:
    python backend.py
    Open: http://localhost:5000
"""

# ── Standard library ───────────────────────────────────────────────────────────
import os          # os.path.basename for checkpoint filenames in /health response
import io          # in-memory byte streams for base64 image encode/decode
import base64      # encode PIL images as base64 strings for JSON transfer
import time        # measure colorization processing time per request
import threading   # load models in background (server starts immediately)
import traceback   # print full stack trace to console when colorization fails
from datetime import datetime  # timestamp each colorization result

# ── Image processing ───────────────────────────────────────────────────────────
from pathlib import Path
from PIL import Image

# ── Deep learning ──────────────────────────────────────────────────────────────
import torch

# ── Web framework ──────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS   # allow browser to call API from any origin

# ── Add project root to Python path (needed to import from models/) ────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

# ── Shared utilities (colorize_utils.py) ──────────────────────────────────────
# All colorization, detection, enhancement, and model loading logic lives here.
# See colorize_utils.py for full documentation of every function.
from colorize_utils import (
    DEVICE,                  # "cuda" or "cpu" — selected at import time
    load_best_config,        # read best_checkpoints.json → {key: checkpoint_path}
    load_torch_generator,    # load GAN or ResNet .pth into memory
    load_opencv_net,         # load Zhang 2016 OpenCV model (auto-downloads if missing)
    colorize_torch,          # LAB colorization with PyTorch model → (PIL, numpy)
    colorize_opencv,         # LAB colorization with OpenCV model → (PIL, numpy)
    detect_image_type,       # Haar Cascade face detection → 'portrait' or 'landscape'
    auto_enhance,            # smart post-processing tuned per scene type
    enhance_manual,          # manual post-processing from UI slider values
    compute_metrics,         # PSNR and SSIM between ground truth and colorized
)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Configure CORS properly so the browser can make POST requests with JSON.
# Without resources/supports_credentials settings, some flask-cors versions
# only allow simple GET requests, blocking POST with Content-Type: application/json.
CORS(app,
     resources={r"/*": {"origins": "*"}},   # allow all origins (local dev)
     supports_credentials=False,             # no cookies needed
     allow_headers=["Content-Type"],         # allow JSON content type
     methods=["GET", "POST", "OPTIONS"])     # explicitly allow OPTIONS preflight


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE
# Written once by load_model() background thread; read-only after startup.
# ══════════════════════════════════════════════════════════════════════════════

G_gan        = None       # loaded GAN PyTorch model (U-Net)
G_resnet     = None       # loaded ResNet PyTorch model (ResNet34 encoder)
opencv_net   = None       # loaded OpenCV DNN (Zhang 2016)
G            = None       # pointer to the currently active PyTorch model
active_model = "resnet"   # active model name: "gan" | "resnet" | "opencv"
model_ready  = False      # True once all models have finished loading
model_error  = None       # error string if loading failed (shown in UI)
gan_path     = None       # full file path to the loaded GAN checkpoint
resnet_path  = None       # full file path to the loaded ResNet checkpoint

# ── Recent results store ───────────────────────────────────────────────────────
# Keeps the last MAX_RECENT colorizations in memory (cleared on server restart).
# Served by GET /recent — displayed in the "Recent Results" tab in index.html.
# Each entry is a dict: { colorized, original, filename, model, checkpoint,
#                         image_type, psnr, ssim, time, timestamp }
MAX_RECENT    = 10        # max number of results to keep
recent_results = []       # most-recent first (index 0 = newest)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL LOADING (background thread at startup)
# Server starts immediately; /health and /colorize wait for model_ready=True.
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """
    Load all 3 models into GPU/CPU memory at server startup.
    Called once in a daemon thread (stops automatically when server stops).

    Load order and preference:
      1. GAN     — find latest generator_epoch*.pth
      2. ResNet  — find latest generator_resnet_epoch*.pth
      3. OpenCV  — load via colorize_utils.load_opencv_net() (auto-downloads)

    Active model default: ResNet > GAN > OpenCV (best PSNR first).
    """
    global G, G_gan, G_resnet, opencv_net
    global model_ready, model_error, active_model, gan_path, resnet_path

    try:
        # ── Helper: find checkpoint with highest epoch number ──────────────────
        def latest(pattern, exclude=None):
            """
            Return str path to latest checkpoint matching pattern, or None.
            exclude: optional substring — skip any file whose name contains it.

            CRITICAL: always pass exclude="resnet" when scanning for GAN checkpoints.
            The glob "generator_epoch*.pth" also matches "generator_resnet_epoch*.pth"
            because * matches anything. Without the exclude, the ResNet .pth file
            gets loaded into the U-Net GAN architecture, causing wrong results.
            """
            ckpts = sorted(
                [p for p in Path("checkpoints").glob(pattern)
                 if not (exclude and exclude in p.name)],
                key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)
            )
            return str(ckpts[-1]) if ckpts else None

        # ── 1. GAN model ──────────────────────────────────────────────────────
        # exclude="resnet" prevents generator_resnet_epoch*.pth from matching
        gan_path = latest("generator_epoch*.pth", exclude="resnet")
        if gan_path:
            try:
                G_gan = load_torch_generator(gan_path, resnet=False)
                print(f"✅ GAN loaded    : {gan_path}")
            except Exception as e:
                print(f"⚠️  GAN load failed: {e}")
                gan_path = None   # clear so /health shows gan_available=False
        else:
            print("⚠️  GAN          : not found in checkpoints/")

        # ── 2. ResNet model ───────────────────────────────────────────────────
        # resnet=True → loads ResNet34 encoder + decoder from models/generator_resnet.py
        resnet_path = latest("generator_resnet_epoch*.pth")
        if resnet_path:
            try:
                G_resnet = load_torch_generator(resnet_path, resnet=True)
                print(f"✅ ResNet loaded : {resnet_path}")
            except Exception as e:
                print(f"⚠️  ResNet load failed: {e}")
                resnet_path = None
        else:
            print("⚠️  ResNet       : not found in checkpoints/")

        # ── 3. OpenCV Zhang 2016 model ─────────────────────────────────────────
        # load_opencv_net from colorize_utils — auto-downloads 3 files (~125 MB) if missing
        try:
            opencv_net = load_opencv_net()
            print(f"✅ OpenCV CNN    : Zhang et al. 2016 (pretrained)")
        except ImportError:
            print("⚠️  OpenCV       : run 'pip install opencv-python'")
        except Exception as e:
            print(f"⚠️  OpenCV load failed: {e}")

        # ── Require at least one model ─────────────────────────────────────────
        if not G_gan and not G_resnet and not opencv_net:
            raise FileNotFoundError(
                "No model found! Put .pth checkpoints in the checkpoints/ folder."
            )

        # ── Set default active model (best overall metrics wins) ───────────────
        # ResNet PSNR 22.49 > GAN PSNR 22.41 > OpenCV PSNR 21.18
        if G_resnet:
            G            = G_resnet
            active_model = "resnet"
        elif G_gan:
            G            = G_gan
            active_model = "gan"
        else:
            active_model = "opencv"   # G stays None — OpenCV uses opencv_net directly

        model_ready = True
        print(f"\n✅ Active model  : {active_model.upper()}")
        print(f"   → Visit: http://localhost:5000\n")

    except Exception as e:
        model_error = str(e)
        print(f"❌ Model load error: {e}")


# Start loading in background — server is available immediately, models load in ~5–10 s
threading.Thread(target=load_model, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BASE64 IMAGE HELPERS
# Backend-specific helpers for browser ↔ server image transfer.
# (Not in colorize_utils because those work with PIL files, not base64 strings.)
# ══════════════════════════════════════════════════════════════════════════════

def b64_to_pil(b64: str) -> Image.Image:
    """
    Convert a base64 image string sent by the browser → PIL Image (RGB).

    Browser format: "data:image/jpeg;base64,/9j/4AAQ..."
    We strip the "data:...;base64," prefix then decode the raw bytes.
    """
    if "," in b64:
        b64 = b64.split(",")[1]   # remove MIME type prefix
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    """
    Convert PIL Image → base64 JPEG string for sending to the browser.
    quality=92: good visual quality with reasonable file size (~80–150 KB).
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SMART CHECKPOINT SELECTION
# Reads best_checkpoints.json to pick the optimal checkpoint per image.
# Falls back gracefully if JSON is missing or a key is not found.
# ══════════════════════════════════════════════════════════════════════════════

def smart_model_for(img):
    """
    Select the best checkpoint for this image, always respecting the user's
    currently active model selection (GAN or ResNet).

    BEHAVIOUR:
      active_model="gan"    → uses GAN,    picks best GAN checkpoint for scene type
      active_model="resnet" → uses ResNet, picks best ResNet checkpoint for scene type
      active_model="opencv" → handled separately in /colorize, never reaches here

    STEPS:
      1. detect_image_type(img) → 'portrait' or 'landscape'
      2. key = f"{active_model}_{image_type}"  e.g. "resnet_portrait"
      3. Look up key in best_checkpoints.json for the preferred checkpoint path
      4. Fast path: if that checkpoint is already loaded in memory → return it
      5. Slow path: load on-demand (~2 s) — happens when scene type changes
      6. Fallback: use the currently loaded model if anything fails

    Returns: (model, kind, image_type, checkpoint_name)
      model:           PyTorch model in eval() mode, ready for inference
      kind:            'gan' or 'resnet'
      image_type:      'portrait' or 'landscape'
      checkpoint_name: filename stem, e.g. 'generator_resnet_epoch31'
    """
    image_type = detect_image_type(img)   # colorize_utils — Haar Cascade + aspect ratio
    cfg        = load_best_config()       # colorize_utils — reads best_checkpoints.json

    # OpenCV is routed separately in /colorize — should never reach this function
    if active_model == "opencv":
        return None, "opencv", image_type, "opencv"

    # No config file → use whichever checkpoint is already in memory
    if not cfg:
        model = G_resnet if active_model == "resnet" else G_gan
        return model, active_model, image_type, "current"

    # Build lookup key: "gan_portrait" | "gan_landscape" | "resnet_portrait" | "resnet_landscape"
    lookup_key     = f"{active_model}_{image_type}"
    preferred_path = cfg.get(lookup_key)   # e.g. "checkpoints/generator_resnet_epoch31.pth"

    if not preferred_path:
        # Key missing (old JSON format or partial checkpoint_picker run)
        print(f"  No key '{lookup_key}' in best_checkpoints.json — using current checkpoint")
        model = G_resnet if active_model == "resnet" else G_gan
        return model, active_model, image_type, "current"

    ckpt_name = Path(preferred_path).stem   # e.g. "generator_resnet_epoch31"
    print(f"  Smart: {active_model.upper()} + {image_type} → {ckpt_name}")

    # Fast path: preferred checkpoint is already loaded in memory
    # Compare resolved paths so Windows vs Linux path differences don't cause mismatches
    if active_model == "gan" and gan_path and \
            Path(gan_path).resolve() == Path(preferred_path).resolve():
        return G_gan, "gan", image_type, ckpt_name

    if active_model == "resnet" and resnet_path and \
            Path(resnet_path).resolve() == Path(preferred_path).resolve():
        return G_resnet, "resnet", image_type, ckpt_name

    # Slow path: load the specific checkpoint on-demand
    # This happens when the best checkpoint for the detected scene type
    # is different from the checkpoint that was loaded at startup
    try:
        m = load_torch_generator(preferred_path, resnet=(active_model == "resnet"))
        print(f"  On-demand loaded: {ckpt_name}  ({image_type})")
        return m, active_model, image_type, ckpt_name

    except Exception as e:
        print(f"  On-demand load failed ({e}) — using fallback")
        model = G_resnet if active_model == "resnet" else G_gan
        return model, active_model, image_type, "current"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COLORIZATION WRAPPERS
# Thin adapters between colorize_utils (returns PIL + numpy) and the
# API endpoints (which need PIL + psnr + ssim).
# ══════════════════════════════════════════════════════════════════════════════

def run_colorize_torch(model, img, color_boost):
    """
    Run PyTorch colorization and return (colorized_PIL, psnr, ssim).

    colorize_torch returns (PIL, original_numpy).
    compute_metrics converts that to PSNR and SSIM floats.
    Both functions come from colorize_utils.
    """
    colorized, gt_np = colorize_torch(model, img, boost=color_boost, size=512)
    p, s = compute_metrics(gt_np, colorized)
    return colorized, p, s


def run_colorize_opencv(img, color_boost):
    """
    Run OpenCV colorization and return (colorized_PIL, psnr, ssim).

    Uses the module-level opencv_net loaded at startup.
    """
    colorized, gt_np = colorize_opencv(opencv_net, img, boost=color_boost, size=512)
    p, s = compute_metrics(gt_np, colorized)
    return colorized, p, s


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def save_to_recent(original_pil, colorized_pil, filename, model, checkpoint,
                   image_type, psnr, ssim, elapsed):
    """
    Save a colorization result to the in-memory recent_results list.

    Stores both the original (grayscale input) and colorized output as
    base64 thumbnail strings (320px wide) to keep memory usage low.

    Inserts at the front of the list (newest first) and trims to MAX_RECENT.

    Args:
      original_pil:  PIL Image — the grayscale input image
      colorized_pil: PIL Image — the colorized result
      filename:      str — original filename (from request, e.g. "photo.jpg")
      model:         str — active model name ("gan" | "resnet" | "opencv")
      checkpoint:    str — checkpoint stem (e.g. "generator_resnet_epoch31")
      image_type:    str — "portrait" or "landscape"
      psnr:          float — PSNR metric
      ssim:          float — SSIM metric
      elapsed:       float — seconds taken
    """
    global recent_results

    def make_thumb(img, max_w=320):
        """Resize PIL image to max_w wide (preserving aspect ratio) and encode as base64 JPEG."""
        w, h    = img.size
        ratio   = min(max_w / w, 1.0)   # never upscale
        new_w   = int(w * ratio)
        new_h   = int(h * ratio)
        thumb   = img.resize((new_w, new_h), Image.LANCZOS)
        buf     = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # Model display colour (matches the web UI colour scheme)
    MODEL_COLORS = {"resnet": "#16a34a", "gan": "#2563eb", "opencv": "#d97706"}

    entry = {
        "colorized":   make_thumb(colorized_pil),       # thumbnail base64 JPEG
        "original":    make_thumb(original_pil),         # thumbnail base64 JPEG
        "filename":    filename or "image.jpg",
        "model":       model,
        "color":       MODEL_COLORS.get(model, "#6b7280"),
        "checkpoint":  checkpoint,
        "image_type":  image_type,
        "psnr":        psnr,
        "ssim":        ssim,
        "time":        elapsed,
        "timestamp":   datetime.now().strftime("%H:%M:%S"),  # e.g. "14:32:07"
        "date":        datetime.now().strftime("%d %b"),     # e.g. "16 Mar"
    }

    # Insert newest first, keep list at MAX_RECENT
    recent_results.insert(0, entry)
    recent_results = recent_results[:MAX_RECENT]


@app.route("/")
def index():
    """Serve the main frontend page — index.html."""
    return send_file("index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static assets (images, CSS, JS) from the /static/ folder."""
    return send_from_directory("static", filename)


@app.route("/health")
def health():
    """
    Health check — called by the browser on page load.

    GET /health
    Response:
      {
        "status":           "ok",
        "model_ready":      true,
        "model_error":      null,
        "device":           "cuda",
        "active_model":     "resnet",
        "gan_available":    true,
        "resnet_available": true,
        "opencv_available": true,
        "gan_path":         "generator_epoch200.pth",
        "resnet_path":      "generator_resnet_epoch100.pth",
        "smart_config":     { ... }
      }
    """
    return jsonify({
        "status":           "ok",
        "model_ready":      model_ready,               # False while loading
        "model_error":      model_error,               # null or error string
        "device":           DEVICE,                    # "cuda" or "cpu"
        "active_model":     active_model,              # current active model name
        "gan_available":    G_gan      is not None,
        "resnet_available": G_resnet   is not None,
        "opencv_available": opencv_net is not None,
        "gan_path":         os.path.basename(gan_path)    if gan_path    else None,
        "resnet_path":      os.path.basename(resnet_path) if resnet_path else None,
        "smart_config":     load_best_config(),        # contents of best_checkpoints.json
    })


@app.route("/recent")
def recent_endpoint():
    """
    Return the last MAX_RECENT colorization results.

    GET /recent
    Response:
      {
        "results": [
          {
            "colorized":  "data:image/jpeg;base64,...",  # thumbnail (320px)
            "original":   "data:image/jpeg;base64,...",  # thumbnail (320px)
            "filename":   "photo.jpg",
            "model":      "resnet",
            "color":      "#16a34a",
            "checkpoint": "generator_resnet_epoch31",
            "image_type": "portrait",
            "psnr":       23.4,
            "ssim":       0.9021,
            "time":       1.2,
            "timestamp":  "14:32:07",
            "date":       "16 Mar"
          },
          ...
        ],
        "count": 5
      }

    Results are newest-first. Empty list [] if nothing has been colorized yet.
    """
    return jsonify({"results": recent_results, "count": len(recent_results)})


@app.route("/switch-model", methods=["POST", "OPTIONS"])
def switch_model():
    """
    Switch the active model when user clicks GAN / ResNet / OpenCV in the UI.

    POST /switch-model
    Request:  { "model": "resnet" }
    Response: { "success": true, "active": "resnet", "model": "generator_resnet_epoch100.pth" }

    Errors:
      404 — requested model's checkpoint was not found at startup
      400 — model name is not "gan", "resnet", or "opencv"
    """
    # Handle CORS preflight — browser sends OPTIONS before POST with JSON
    if request.method == "OPTIONS":
        return jsonify({}), 200

    global G, active_model
    data       = request.get_json()
    model_name = data.get("model", "gan").lower()

    if model_name == "gan":
        if G_gan is None:
            return jsonify({"success": False, "error": "GAN not found in checkpoints/"}), 404
        G            = G_gan
        active_model = "gan"
        name         = os.path.basename(gan_path) if gan_path else "GAN"
        print(f"🔄 Switched → GAN: {name}")
        return jsonify({"success": True, "active": "gan", "model": name})

    elif model_name == "resnet":
        if G_resnet is None:
            return jsonify({"success": False, "error": "ResNet not found in checkpoints/"}), 404
        G            = G_resnet
        active_model = "resnet"
        name         = os.path.basename(resnet_path) if resnet_path else "ResNet"
        print(f"🔄 Switched → ResNet: {name}")
        return jsonify({"success": True, "active": "resnet", "model": name})

    elif model_name == "opencv":
        if opencv_net is None:
            return jsonify({"success": False,
                            "error": "OpenCV model not loaded. Check models/ folder."}), 404
        G            = None   # OpenCV uses opencv_net directly, not the G pointer
        active_model = "opencv"
        print(f"🔄 Switched → OpenCV CNN (Zhang 2016)")
        return jsonify({"success": True, "active": "opencv", "model": "Zhang et al. 2016"})

    return jsonify({"success": False, "error": "Use 'gan', 'resnet', or 'opencv'"}), 400


@app.route("/colorize", methods=["POST", "OPTIONS"])
def colorize_endpoint():
    """
    Main colorization endpoint.

    POST /colorize
    Request JSON:
      {
        "image":       "data:image/jpeg;base64,...",  # input image
        "color_boost": 1.4,   # AB colour multiplier (default 1.4)
        "saturation":  1.0,   # manual slider (1.0 = auto mode)
        "contrast":    1.0,
        "sharpness":   1.0
      }

    Response JSON:
      {
        "success":      true,
        "colorized":    "data:image/jpeg;base64,...",
        "time":         1.2,
        "psnr":         23.28,
        "ssim":         0.8987,
        "active_model": "resnet",
        "checkpoint":   "generator_resnet_epoch31",
        "image_type":   "portrait"
      }

    SMART BOOST LOGIC:
      If user has NOT changed the boost slider from its default (1.4):
        portrait  → boost=1.0 (natural skin tones, avoids artefacts on faces)
        landscape → boost=1.4 (vivid sky/grass/water colours)
      If user HAS changed the slider → use their exact value.

    ENHANCEMENT LOGIC:
      If all sliders at default (1.0) → smart auto_enhance per scene type
      If any slider moved → enhance_manual with the user's chosen values
    """
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if not model_ready:
        return jsonify({"success": False,
                        "error": "Models still loading — please wait a few seconds"}), 503

    try:
        data        = request.get_json()
        color_boost = float(data.get("color_boost", 1.4))
        saturation  = float(data.get("saturation",  1.0))
        contrast    = float(data.get("contrast",    1.0))
        sharpness   = float(data.get("sharpness",   1.0))

        img = b64_to_pil(data["image"])   # base64 string → PIL Image
        t0  = time.time()                 # start timing

        # ── Route to the active model ──────────────────────────────────────────
        if active_model == "opencv":
            if opencv_net is None:
                return jsonify({"success": False, "error": "OpenCV model not available"}), 500

            image_type = detect_image_type(img)   # colorize_utils

            # Smart boost: only override if user hasn't moved the slider (default=1.4)
            if color_boost == 1.4:
                color_boost = 1.0 if image_type == "portrait" else 1.4

            colorized, p, s = run_colorize_opencv(img, color_boost)
            used_model = "opencv"
            ckpt_name  = "Zhang2016"

        else:
            # PyTorch model (GAN or ResNet)
            if G is None:
                return jsonify({"success": False, "error": "No model loaded"}), 500

            # smart_model_for picks the best checkpoint per scene type
            model, used_model, image_type, ckpt_name = smart_model_for(img)
            if model is None:
                return jsonify({"success": False, "error": "Smart model selection failed"}), 500

            # Smart boost (same logic as OpenCV branch above)
            if color_boost == 1.4:
                color_boost = 1.0 if image_type == "portrait" else 1.4

            colorized, p, s = run_colorize_torch(model, img, color_boost)

        # ── Post-processing ────────────────────────────────────────────────────
        user_adjusted = (saturation != 1.0 or contrast != 1.0 or sharpness != 1.0)

        if user_adjusted:
            # Manual mode — apply the user's exact slider values
            # enhance_manual from colorize_utils (only applies values != 1.0)
            colorized = enhance_manual(colorized, saturation, contrast, sharpness)
        else:
            # Auto mode — scene-specific smart enhancement
            # auto_enhance from colorize_utils — returns (PIL, image_type)
            colorized, _ = auto_enhance(colorized, image_type)

        elapsed = round(time.time() - t0, 2)
        print(f"[{used_model.upper()} | {ckpt_name} | {image_type}] "
              f"{elapsed}s | PSNR {p} dB | SSIM {s}")

        # Save result to recent_results (displayed in Recent Results tab)
        # We pass the original input image and the post-processed colorized image
        filename = data.get("filename", "image.jpg")   # frontend sends the filename
        save_to_recent(img, colorized, filename, used_model, ckpt_name,
                       image_type, p, s, elapsed)

        return jsonify({
            "success":      True,
            "colorized":    pil_to_b64(colorized),
            "time":         elapsed,
            "psnr":         p,
            "ssim":         s,
            "active_model": used_model,
            "checkpoint":   ckpt_name,
            "image_type":   image_type,
        })

    except Exception as e:
        traceback.print_exc()   # full stack trace in console for debugging
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/batch", methods=["POST", "OPTIONS"])
def batch_endpoint():
    """
    Batch colorization — process multiple images in a single request.
    Each image is processed independently; a single failure won't stop the rest.

    POST /batch
    Request:
      {
        "images":      ["data:image/jpeg;base64,...", ...],
        "color_boost": 1.4,
        "saturation":  1.0,
        "contrast":    1.0,
        "sharpness":   1.0
      }

    Response:
      {
        "results": [
          { "index": 0, "success": true, "colorized": "...", "psnr": 23.1, "ssim": 0.89 },
          { "index": 1, "success": false, "error": "..." }
        ]
      }
    """
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if not model_ready:
        return jsonify({"success": False, "error": "Models still loading — please wait"}), 503

    try:
        data        = request.get_json()
        color_boost = float(data.get("color_boost", 1.4))
        saturation  = float(data.get("saturation",  1.0))
        contrast    = float(data.get("contrast",    1.0))
        sharpness   = float(data.get("sharpness",   1.0))
        results     = []

        for i, img_b64 in enumerate(data.get("images", [])):
            try:
                img = b64_to_pil(img_b64)

                if active_model == "opencv":
                    img_type = detect_image_type(img)
                    boost    = color_boost if color_boost != 1.4 else (
                        1.0 if img_type == "portrait" else 1.4
                    )
                    colorized, p, s = run_colorize_opencv(img, boost)
                else:
                    # Each image gets its own smart checkpoint selection
                    model, _, img_type, _ = smart_model_for(img)
                    boost = color_boost if color_boost != 1.4 else (
                        1.0 if img_type == "portrait" else 1.4
                    )
                    colorized, p, s = run_colorize_torch(model or G, img, boost)

                # Post-processing
                user_adjusted = (saturation != 1.0 or contrast != 1.0 or sharpness != 1.0)
                if user_adjusted:
                    colorized = enhance_manual(colorized, saturation, contrast, sharpness)
                else:
                    colorized, _ = auto_enhance(colorized, img_type)

                results.append({
                    "index":     i,
                    "success":   True,
                    "colorized": pil_to_b64(colorized),
                    "psnr":      p,
                    "ssim":      s,
                })

            except Exception as e:
                # One image failed — record the error and continue with the rest
                results.append({"index": i, "success": False, "error": str(e)})

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/eval_report/<path:filename>")
def serve_eval_report(filename):
    """Serve evaluation HTML report files from the eval_report/ folder."""
    return send_from_directory("eval_report", filename)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Path("static/gallery").mkdir(parents=True, exist_ok=True)  # for training graph

    print("=" * 55)
    print("  Image Colorization — Three Model Support")
    print("=" * 55)
    print(f"  Device  : {DEVICE}")
    print()
    print("  Loading models in background thread:")
    print("  1. GAN    → checkpoints/generator_epoch*.pth")
    print("  2. ResNet → checkpoints/generator_resnet_epoch*.pth")
    print("  3. OpenCV → models/ (auto-downloaded if missing ~125 MB)")
    print()
    print("  Open browser → http://localhost:5000")
    print("=" * 55)

    # host="0.0.0.0" = accessible from any device on the same local network
    # debug=False    = production mode (no auto-reload on file changes)
    app.run(host="0.0.0.0", port=5000, debug=False)
