"""
colorize_utils.py — Shared Utility Functions for Image Colorization Project
==================================================================
PURPOSE : Central module containing functions used across multiple scripts.
          Import from here instead of copy-pasting code into every file.

USED BY:
    backend.py          — model loading, colorize_torch, detect_image_type, auto_enhance
    evaluate.py         — find_latest_checkpoint, best_checkpoint_for, download_file,
                          load_torch_generator, load_opencv_net, colorize_torch, colorize_opencv
    inference.py        — find_latest, load_best_config, load_torch_model, load_opencv_model,
                          colorize_torch, colorize_opencv, detect_image_type, auto_enhance
    compare_models.py   — find_latest, load_best_config, download_file, load_opencv_net,
                          colorize_torch, colorize_opencv, detect_type, smart_enhance
    checkpoint_picker.py— colorize_torch (via score_image)
    plot_training.py    — find_latest (checkpoint epoch detection)
    update_metrics.py   — find_latest (checkpoint epoch detection)

CONTENTS:
    Section 1 — Checkpoint helpers    (find_latest, load_best_config, best_checkpoint_for)
    Section 2 — File downloader       (download_file)
    Section 3 — Model loaders         (load_torch_generator, load_opencv_net)
    Section 4 — Colorization          (colorize_torch, colorize_opencv)
    Section 5 — Image type detection  (detect_image_type)
    Section 6 — Enhancement           (auto_enhance, enhance_manual)
    Section 7 — Metrics               (compute_metrics)
    Section 8 — Shared constants      (DEVICE, OpenCV URLs/paths)
"""

# ── Standard library ───────────────────────────────────────────────────────────
import json
import urllib.request

# ── Scientific computing ───────────────────────────────────────────────────────
import numpy as np

# ── Image processing ───────────────────────────────────────────────────────────
from pathlib import Path
from PIL import Image, ImageEnhance

# ── Deep learning ──────────────────────────────────────────────────────────────
import torch

# ── Colour space conversion ────────────────────────────────────────────────────
# LAB colour space: L = lightness (model input), AB = colour (model output)
from skimage.color import rgb2lab, lab2rgb

# ── Quality metrics ────────────────────────────────────────────────────────────
from skimage.metrics import peak_signal_noise_ratio as psnr_fn   # higher dB = better
from skimage.metrics import structural_similarity   as ssim_fn   # 0–1, higher = better


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SHARED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# GPU if available, else CPU — shared across all scripts
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── OpenCV Zhang 2016 model file paths ────────────────────────────────────────
MODEL_DIR   = Path("models")
PROTOTXT    = MODEL_DIR / "colorization_deploy_v2.prototxt"     # network architecture
CAFFEMODEL  = MODEL_DIR / "colorization_release_v2.caffemodel"  # trained weights (~125 MB)
POINTS_FILE = MODEL_DIR / "pts_in_hull.npy"                     # 313 AB colour cluster centres

# Mirror download URLs — tried in order if primary fails
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/richzhang/colorization/"
    "caffe/colorization/models/colorization_deploy_v2.prototxt"
)
CAFFEMODEL_URLS = [
    "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1",
    "https://huggingface.co/datasets/holwech/colorization-models/"
    "resolve/main/colorization_release_v2.caffemodel",
]
POINTS_URL = (
    "https://raw.githubusercontent.com/richzhang/colorization/"
    "caffe/colorization/resources/pts_in_hull.npy"
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def find_latest(pattern):
    """
    Find the checkpoint file with the HIGHEST epoch number matching a glob pattern.

    Scans checkpoints/ folder and returns the path with the largest epoch number
    extracted from the filename digits.

    Examples:
      find_latest("generator_epoch*.pth")         → "checkpoints/generator_epoch200.pth"
      find_latest("generator_resnet_epoch*.pth")  → "checkpoints/generator_resnet_epoch100.pth"

    Args:
      pattern: glob pattern string (e.g. "generator_epoch*.pth")

    Returns:
      str path to the latest checkpoint, or None if no files match.
    """
    ckpts = sorted(
        Path("checkpoints").glob(pattern),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)
    )
    return str(ckpts[-1]) if ckpts else None


def load_best_config():
    """
    Read best_checkpoints.json (created by checkpoint_picker.py).

    This file maps model-type + scene-type keys to the best checkpoint path:
      {
        "gan_portrait":    "checkpoints/generator_epoch178.pth",
        "gan_landscape":   "checkpoints/generator_epoch131.pth",
        "resnet_portrait": "checkpoints/generator_resnet_epoch31.pth",
        "resnet_landscape":"checkpoints/generator_resnet_epoch8.pth",
        "_meta": { ... }
      }

    Returns:
      dict from the JSON file, or {} if the file doesn't exist yet.
    """
    cfg_path = Path("best_checkpoints.json")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def best_checkpoint_for(kind, scene="portrait"):
    """
    Return the recommended checkpoint path for a given model kind and scene type.

    Priority:
      1. Read best_checkpoints.json → key = "{kind}_{scene}"
         e.g. "resnet_portrait" → checkpoints/generator_resnet_epoch31.pth
      2. Fall back to the latest .pth file if JSON missing or key not found

    Args:
      kind:  'gan' or 'resnet'
      scene: 'portrait' or 'landscape' (default: 'portrait')

    Returns:
      str path to checkpoint, or None if nothing found.
    """
    cfg = load_best_config()

    if cfg:
        key       = f"{kind}_{scene}"   # e.g. "resnet_portrait"
        preferred = cfg.get(key)
        if preferred and Path(preferred).exists():
            print(f"  📋 best_checkpoints.json [{key}] → {Path(preferred).name}")
            return str(preferred)

    # Fallback: scan for latest checkpoint of this kind
    pattern = "generator_resnet_epoch*.pth" if kind == "resnet" else "generator_epoch*.pth"
    return find_latest(pattern)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FILE DOWNLOADER
# ══════════════════════════════════════════════════════════════════════════════

def download_file(urls, dest):
    """
    Download a file from a list of mirror URLs, trying each in order.

    Shows download progress percentage. Removes partial files on failure.
    Used for auto-downloading the OpenCV Zhang 2016 model files.

    Args:
      urls: str (single URL) or list of str (mirror URLs tried in order)
      dest: pathlib.Path destination file path

    Raises:
      RuntimeError if all mirrors fail.
    """
    name = dest.name
    if isinstance(urls, str):
        urls = [urls]   # normalise single URL to list

    for i, url in enumerate(urls):
        try:
            print(f"  Downloading {name} (mirror {i+1}/{len(urls)})...")

            def hook(c, b, t):
                """Progress callback: c=blocks downloaded, b=block size, t=total size."""
                if t > 0:
                    print(f"\r    {min(100, c * b * 100 // t)}%", end="", flush=True)

            urllib.request.urlretrieve(url, dest, hook)
            print(f"\r  ✅ {name} saved")
            return   # success — stop trying mirrors

        except Exception as e:
            print(f"\n  ❌ Mirror {i+1} failed: {e}")
            if dest.exists():
                dest.unlink()   # remove corrupted partial download

    raise RuntimeError(f"All mirrors failed for {name}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_torch_generator(model_path, resnet=False):
    """
    Load a PyTorch GAN or ResNet generator checkpoint from disk.

    resnet=False → U-Net GAN generator    (models/generator.py)
    resnet=True  → ResNet34 generator     (models/generator_resnet.py)
                   Falls back to base Generator if generator_resnet.py is missing.

    Both models are put into eval() mode (disables dropout/batchnorm training).

    Args:
      model_path: str path to the .pth checkpoint file
      resnet:     bool — which architecture to load

    Returns:
      Loaded model in eval() mode.
    """
    from models.generator import Generator   # U-Net GAN architecture

    if resnet:
        try:
            from models.generator_resnet import Generator as ResNetGenerator
            G = ResNetGenerator(pretrained=False).to(DEVICE)
        except ImportError:
            print("  ⚠️  generator_resnet.py not found — using base Generator")
            G = Generator().to(DEVICE)
    else:
        G = Generator().to(DEVICE)

    # Load saved weight dictionary from the .pth file
    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()   # inference mode
    return G


def load_opencv_net():
    """
    Load the Zhang et al. 2016 colorization model via OpenCV DNN.

    Auto-downloads the 3 required files (~125 MB total) if they are missing:
      - prototxt    : network architecture definition (small text file)
      - caffemodel  : pretrained weights (~125 MB)
      - pts_in_hull : 313 AB colour cluster centre coordinates

    Returns:
      OpenCV DNN network object ready for inference.

    Raises:
      ImportError if opencv-python is not installed.
      RuntimeError if downloads fail.
    """
    import cv2
    MODEL_DIR.mkdir(exist_ok=True)   # create models/ folder if needed

    # Auto-download any missing files
    if not PROTOTXT.exists():    download_file(PROTOTXT_URL,    PROTOTXT)
    if not CAFFEMODEL.exists():  download_file(CAFFEMODEL_URLS, CAFFEMODEL)
    if not POINTS_FILE.exists(): download_file(POINTS_URL,      POINTS_FILE)

    # Load Caffe model via OpenCV DNN
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))

    # Set the 313 AB colour cluster centres (annealing from Zhang 2016 paper §3.3)
    pts    = np.load(str(POINTS_FILE)).transpose().reshape(2, 313, 1, 1)
    class8 = net.getLayerId("class8_ab")      # AB colour output layer
    conv8  = net.getLayerId("conv8_313_rh")   # final convolutional layer
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype=np.float32)]
    # 2.606 = temperature scaling factor from Zhang 2016 paper

    return net


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COLORIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def colorize_torch(model, img_pil, boost=1.4, size=512):
    """
    Colorize an image using a PyTorch model (GAN or ResNet).

    How LAB colorization works:
      1. Resize image to size×size (model's training resolution)
      2. Convert RGB → LAB colour space
      3. Extract L channel (lightness) — this is the model's INPUT
      4. Normalise L from [0,100] to [-1,1]
      5. Model predicts AB channels (colour) — output shape (H,W,2)
      6. Scale AB back to [-110,110] (valid LAB AB range), apply boost
      7. Combine original L + predicted AB → reconstruct full LAB image
      8. Convert LAB → RGB

    boost: multiplies AB values. 1.0 = raw output (use for evaluation/metrics).
           1.4 = slightly more vivid colours (use for demos/web app).
    size:  512 matches backend.py — keeps evaluation metrics consistent.

    Args:
      model:   PyTorch model in eval() mode
      img_pil: PIL Image (RGB)
      boost:   float AB colour multiplier (default 1.4)
      size:    int inference resolution (default 512)

    Returns:
      (colorized_PIL, original_numpy_uint8)
      colorized: PIL Image resized back to original dimensions
      original:  numpy uint8 array at size×size for computing metrics
    """
    orig_size   = img_pil.size
    img_resized = img_pil.resize((size, size), Image.LANCZOS)
    img_np      = np.array(img_resized.convert("RGB"), dtype=np.uint8)

    # Convert to LAB and extract L (lightness = model input)
    lab = rgb2lab(img_np).astype("float32")
    L   = lab[:, :, 0] / 50.0 - 1.0   # normalise: [0,100] → [-1,1]

    # Add batch dim (1) and channel dim (1): (H,W) → (1,1,H,W)
    L_t = torch.tensor(L).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Run model — no_grad skips gradient tracking (saves memory + faster)
    with torch.no_grad():
        AB_pred = model(L_t).squeeze(0).permute(1, 2, 0).cpu().numpy()
        # squeeze(0):    (1,2,H,W) → (2,H,W)
        # permute(1,2,0): channels-first → channels-last → (H,W,2)

    # Scale AB to valid range and apply boost
    AB_pred = np.clip(AB_pred * 110.0 * boost, -110, 110)

    # Reverse L normalisation: [-1,1] → [0,100]
    L_orig = (L + 1.0) * 50.0

    # Reconstruct full LAB image: stack L + AB → (H,W,3)
    lab_out = np.concatenate([L_orig[:, :, np.newaxis], AB_pred], axis=2)

    # Convert LAB → RGB, clip to [0,255], cast to uint8
    rgb_out = (lab2rgb(lab_out) * 255).clip(0, 255).astype(np.uint8)

    # Resize back to original input dimensions
    result = Image.fromarray(rgb_out).resize(orig_size, Image.LANCZOS)
    return result, img_np   # (colorized PIL, ground truth numpy at size×size)


def colorize_opencv(net, img_pil, boost=1.4, size=512):
    """
    Colorize an image using the Zhang et al. 2016 OpenCV model.

    Key differences from our GAN:
      - Trained on ImageNet (1.3M images) vs COCO 118k
      - Colour bin CLASSIFICATION (313 bins), not direct AB regression
      - Internally operates at 224×224; evaluated at `size` resolution
      - Produces conservative colours — less vivid but more stable
      - Our ResNet beats it: 22.49 dB vs 21.18 dB PSNR on 50 images

    Includes global colour cast correction (Zhang model sometimes produces
    a green or yellow tint across the whole image).

    Args:
      net:     OpenCV DNN network from load_opencv_net()
      img_pil: PIL Image (RGB)
      boost:   float AB colour multiplier (default 1.4)
      size:    int evaluation resolution (default 512)

    Returns:
      (colorized_PIL, original_numpy_uint8)
    """
    import cv2

    orig_size = img_pil.size
    img_np    = np.array(img_pil.resize((size, size)).convert("RGB"), dtype=np.uint8)

    # OpenCV uses BGR channel order (reverse of RGB)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Derive clean grayscale (BGR→gray→BGR avoids chroma leakage artifacts)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray3   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)   # 3-channel needed for LAB conversion
    scaled  = gray3.astype(np.float32) / 255.0          # normalise [0,255] → [0,1]

    # Extract L channel from LAB
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L   = cv2.split(lab)[0]   # (H, W) float

    # Resize L to Zhang model's input size and subtract mean (paper protocol)
    Lr = cv2.resize(L, (224, 224)) - 50

    # Run Zhang model — outputs low-resolution AB prediction
    net.setInput(cv2.dnn.blobFromImage(Lr))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))   # (2,H,W) → (H,W,2)

    # Resize AB output to evaluation resolution
    ab = cv2.resize(ab, (size, size))

    # Correct global colour cast (model can produce green/yellow global tint)
    for c in range(2):   # A channel then B channel
        mean = ab[:, :, c].mean()
        if abs(mean) > 5:              # 5 = threshold for significant cast
            ab[:, :, c] -= mean * 0.5  # pull half the global shift towards zero

    # Apply boost and clamp to valid AB range
    ab = np.clip(ab * boost, -100, 100)

    # Reconstruct: L + predicted AB → LAB → BGR → RGB
    out     = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    out     = np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2BGR), 0, 1)
    rgb_out = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    result = Image.fromarray(rgb_out).resize(orig_size, Image.LANCZOS)
    return result, img_np   # (colorized PIL, ground truth numpy at size×size)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — IMAGE TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_image_type(img_pil):
    """
    Classify an image as 'portrait' (has a face) or 'landscape' (no face).

    Method:
      1. Run OpenCV Haar Cascade frontal face detector
      2. If face found → 'portrait'
      3. Fallback if OpenCV unavailable: tall image (height > width × 1.1) → 'portrait'

    Used by backend.py, inference.py, and compare_models.py to:
      - Pick the best checkpoint for the scene type
      - Apply the correct enhancement profile

    Args:
      img_pil: PIL Image

    Returns:
      'portrait' or 'landscape'
    """
    try:
        import cv2
        arr  = np.array(img_pil.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Haar Cascade: pre-trained face detector
        fc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Parameters:
        #   scaleFactor=1.1  → scan at 10% scale steps to catch different face sizes
        #   minNeighbors=5   → require 5 overlapping detections (reduces false positives)
        #   minSize=(30,30)  → ignore faces smaller than 30×30 px
        if len(fc.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))) > 0:
            return "portrait"

    except Exception:
        pass   # OpenCV not installed — fall back to aspect ratio

    # Aspect ratio fallback: images significantly taller than wide are likely portraits
    w, h = img_pil.size
    return "portrait" if h > w * 1.1 else "landscape"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENHANCEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def auto_enhance(img_pil, image_type=None):
    """
    Apply smart post-processing based on the detected scene type.

    Automatically detects portrait vs landscape if image_type is not provided.
    These settings mirror backend.py's auto_enhance() exactly, so command-line
    output matches what users see in the web app.

    PORTRAIT settings (face in image):
      0.85 saturation → reduces wrong skin/beard colour tones from GAN
      1.15 contrast   → adds depth to face features
      1.80 sharpness  → sharpens hair, beard, and eye detail
      1.05 brightness → slight natural brightness lift

    LANDSCAPE settings (nature/city/architecture):
      1.25 saturation → vivid sky blue, grass green, water colours
      1.10 contrast   → adds scene depth and separation
      1.60 sharpness  → sharpens trees, buildings, and edges

    Args:
      img_pil:    PIL Image to enhance
      image_type: 'portrait' | 'landscape' | None (auto-detect if None)

    Returns:
      (enhanced_PIL_Image, image_type_str)
    """
    if image_type is None:
        image_type = detect_image_type(img_pil)

    if image_type == "portrait":
        img_pil = ImageEnhance.Color(img_pil).enhance(0.85)       # tame oversaturated skin
        img_pil = ImageEnhance.Contrast(img_pil).enhance(1.15)    # face contrast depth
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.80)   # hair/eye sharpness
        img_pil = ImageEnhance.Brightness(img_pil).enhance(1.05)  # brightness lift
    else:
        img_pil = ImageEnhance.Color(img_pil).enhance(1.25)       # vivid landscape colours
        img_pil = ImageEnhance.Contrast(img_pil).enhance(1.10)    # scene depth
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.60)   # edge sharpness

    return img_pil, image_type


def enhance_manual(img_pil, saturation=1.0, contrast=1.0, sharpness=1.0):
    """
    Apply manual enhancement values (from UI sliders or fixed defaults).

    Only applies each enhancement if the value differs from 1.0 (neutral).
    Used when the user has manually adjusted the sliders in the web app,
    or when --no-auto flag is passed to inference.py.

    Args:
      img_pil:    PIL Image
      saturation: float (1.0 = no change, >1.0 = more vivid, <1.0 = less vivid)
      contrast:   float (1.0 = no change, >1.0 = more contrast)
      sharpness:  float (1.0 = no change, >1.0 = sharper)

    Returns:
      Enhanced PIL Image.
    """
    if saturation != 1.0:
        img_pil = ImageEnhance.Color(img_pil).enhance(saturation)
    if contrast != 1.0:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    if sharpness != 1.0:
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(sharpness)
    return img_pil


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(gt_np, colorized_pil_or_np):
    """
    Compute PSNR and SSIM between a ground truth array and a colorized image.

    Automatically handles size mismatch by resizing colorized to match gt_np.

    PSNR (Peak Signal-to-Noise Ratio):
      - Measured in dB — higher is better
      - >22 dB = good colourization
      - Formula: 10 × log10(MAX² / MSE)

    SSIM (Structural Similarity Index):
      - Range 0–1 — higher is better
      - >0.85 = high structural quality
      - Measures luminance, contrast, and structure simultaneously

    Args:
      gt_np:               numpy uint8 array (H,W,3) — ground truth colour image
      colorized_pil_or_np: PIL Image or numpy uint8 array — model output

    Returns:
      (psnr_float, ssim_float) — both rounded to reasonable precision.
    """
    # Convert PIL to numpy if needed
    if hasattr(colorized_pil_or_np, "resize"):
        # It's a PIL Image — resize to match ground truth dimensions
        pred_np = np.array(
            colorized_pil_or_np.resize(
                (gt_np.shape[1], gt_np.shape[0]), Image.LANCZOS
            )
        )
    else:
        pred_np = colorized_pil_or_np   # already numpy

    p = round(float(psnr_fn(gt_np, pred_np, data_range=255)), 2)
    s = round(float(ssim_fn(gt_np, pred_np, channel_axis=2, data_range=255)), 4)
    return p, s
