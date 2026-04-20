"""
evaluate.py — Evaluate GAN, ResNet, and OpenCV Models on Validation Images
===========================================================================
PURPOSE : Measures how good each model is using two standard metrics:
            PSNR  (Peak Signal-to-Noise Ratio, in dB) — higher = more accurate colours
            SSIM  (Structural Similarity Index, 0–1)   — higher = better structure/texture

          Generates an HTML report with side-by-side image comparisons
          and saves per-model JSON files for update_metrics.py.

CHECKPOINT SELECTION:
          1st: Reads best_checkpoints.json (created by checkpoint_picker.py)
          2nd: Falls back to the latest .pth file if the JSON doesn't exist

SHARED CODE:
          All shared utilities live in utils.py — imports from there.
          find_latest, best_checkpoint_for, download_file,
          load_torch_generator, load_opencv_net, colorize_torch, colorize_opencv

HOW TO USE:
    python evaluate.py                     # all 3 models, up to 500 images
    python evaluate.py --limit 50          # recommended — 50 images is enough
    python evaluate.py --no-opencv         # skip OpenCV (faster run)
    python evaluate.py --model checkpoints/generator_epoch178.pth  # specific file

OUTPUT:
    eval_report/eval_report.html       ← open in a browser to view results
    eval_report/metrics_*.json         ← raw numbers consumed by update_metrics.py
"""

# ── Standard library ───────────────────────────────────────────────────────────
import json
import argparse

# ── Scientific computing ───────────────────────────────────────────────────────
import numpy as np

# ── Image processing ───────────────────────────────────────────────────────────
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Quality metrics (used locally in evaluate_model) ──────────────────────────
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity   as ssim_fn

# ── Shared utilities (utils.py) ────────────────────────────────────────────────
# See utils.py for full documentation of each function
from colorize_utils import (
    best_checkpoint_for,    # read best_checkpoints.json → get portrait/landscape key
    load_torch_generator,   # load GAN or ResNet .pth checkpoint
    load_opencv_net,        # load Zhang 2016 OpenCV model (auto-downloads files)
    colorize_torch,         # LAB colorization with a PyTorch model
    colorize_opencv,        # colorization with OpenCV Zhang 2016 model
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EVALUATION LOOP
# Runs one model across all images and collects PSNR/SSIM per image
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(colorize_fn, files, label):
    """
    Evaluate one model across all images and return results + summary dict.

    colorize_fn — callable: image_path_str → (ground_truth_np, colorized_np)
                  IMPORTANT: colorize_fn should use boost=1.0 (raw output)
                  so that PSNR/SSIM measures the true model accuracy, not boosted output.
    files       — list of image Path objects
    label       — model name string (used in output filenames and HTML report)

    Returns:
      results_list — list of per-image dicts: {gt, colorized, grayscale, psnr, ssim}
      summary_dict — averaged metrics across all images
    """
    psnr_scores = []   # PSNR value for each image
    ssim_scores = []   # SSIM value for each image
    results     = []   # per-image result dicts for the HTML report

    for f in tqdm(files, desc=f"  [{label}]"):
        try:
            gt, colorized = colorize_fn(str(f))

            # Compute quality metrics — both higher = better
            p = float(psnr_fn(gt, colorized, data_range=255))
            s = float(ssim_fn(gt, colorized, channel_axis=2, data_range=255))
            psnr_scores.append(p)
            ssim_scores.append(s)

            # Build grayscale version for display in HTML report
            gray_ch = np.array(Image.fromarray(gt).convert("L"))   # (H,W) single channel
            gray    = np.stack([gray_ch] * 3, axis=2)              # (H,W,3) by repeating

            results.append({
                "gt":        gt,          # original colour image (numpy uint8)
                "colorized": colorized,   # model output (numpy uint8)
                "grayscale": gray,        # grayscale input for report display
                "psnr":      round(p, 2),
                "ssim":      round(s, 4),
            })

        except Exception as e:
            print(f"  ⚠️  Skipped {f.name}: {e}")

    if not psnr_scores:
        raise RuntimeError(f"No images successfully evaluated for {label}")

    # Compute statistics across all evaluated images
    summary = {
        "label":      label,
        "avg_psnr":   round(float(np.mean(psnr_scores)), 4),   # main headline metric
        "avg_ssim":   round(float(np.mean(ssim_scores)), 4),
        "min_psnr":   round(float(np.min(psnr_scores)),  4),   # worst single image
        "max_psnr":   round(float(np.max(psnr_scores)),  4),   # best single image
        "num_images": len(results),
    }
    return results, summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HTML REPORT GENERATOR
# Produces eval_report/eval_report.html with embedded images (self-contained)
# ══════════════════════════════════════════════════════════════════════════════

def save_html_report(all_results, out_dir, summaries):
    """
    Generate eval_report.html with summary cards, winner banner, and
    a side-by-side image comparison table (first 50 images).

    All images are base64-embedded (JPEG, quality=82) so the file is
    fully self-contained and works offline.

    all_results — dict: label → list of per-image result dicts
    out_dir     — Path where the HTML file will be saved
    summaries   — list of summary dicts from evaluate_model()
    """
    import base64, io

    def b64(arr):
        """Encode a numpy uint8 image array as an inline base64 JPEG string."""
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()

    best_psnr_val = max(s["avg_psnr"] for s in summaries)
    best_ssim_val = max(s["avg_ssim"] for s in summaries)

    # Colour scheme matching index.html
    MODEL_COLORS = {"gan": "#2563eb", "resnet": "#16a34a", "opencv": "#d97706"}
    MODEL_BG     = {"gan": "#dbeafe", "resnet": "#dcfce7", "opencv": "#fef3c7"}
    MODEL_BORDER = {"gan": "#93c5fd", "resnet": "#86efac", "opencv": "#fcd34d"}

    def model_color(label):
        """Return (text, background, border) colours for a model label string."""
        l = label.lower()
        for k in MODEL_COLORS:
            if k in l:
                return MODEL_COLORS[k], MODEL_BG[k], MODEL_BORDER[k]
        return "#374151", "#f0f2f5", "#dde1e8"

    # ── Summary cards ──────────────────────────────────────────────────────────
    cards_html = ""
    for s in summaries:
        mc, mbg, mbd = model_color(s["label"])
        bp = " &nbsp;✓ Best" if s["avg_psnr"] == best_psnr_val and len(summaries) > 1 else ""
        bs = " &nbsp;✓ Best" if s["avg_ssim"] == best_ssim_val and len(summaries) > 1 else ""
        cp = "#16a34a" if s["avg_psnr"] == best_psnr_val else "#d97706"
        cs = "#16a34a" if s["avg_ssim"] == best_ssim_val else "#d97706"
        cards_html += f"""
        <div class="model-card" style="border-color:{mbd}; border-left:5px solid {mc};">
          <div class="model-title" style="color:{mc};">{s["label"]}</div>
          <div class="metrics-row">
            <div class="metric"><div class="metric-val" style="color:{cp};">{s["avg_psnr"]}</div>
              <div class="metric-label">Avg PSNR (dB){bp}</div></div>
            <div class="metric"><div class="metric-val" style="color:{cs};">{s["avg_ssim"]}</div>
              <div class="metric-label">Avg SSIM{bs}</div></div>
            <div class="metric"><div class="metric-val">{s["num_images"]}</div>
              <div class="metric-label">Images Evaluated</div></div>
            <div class="metric"><div class="metric-val">{s["min_psnr"]}</div>
              <div class="metric-label">Min PSNR</div></div>
            <div class="metric"><div class="metric-val">{s["max_psnr"]}</div>
              <div class="metric-label">Max PSNR</div></div>
          </div>
        </div>"""

    # ── Winner banner ──────────────────────────────────────────────────────────
    winner_html = ""
    if len(summaries) > 1:
        wp = max(summaries, key=lambda x: x["avg_psnr"])["label"]
        ws = max(summaries, key=lambda x: x["avg_ssim"])["label"]
        winner_html = f"""
        <div class="winners-row">
          <div class="winner-box" style="border-color:#86efac; background:#dcfce7;">
            <div style="font-size:0.78rem;font-weight:600;color:#6b7280;text-transform:uppercase;">🏆 Best PSNR</div>
            <div style="font-size:1.1rem;font-weight:700;color:#16a34a;margin-top:4px;">{wp}</div></div>
          <div class="winner-box" style="border-color:#93c5fd; background:#dbeafe;">
            <div style="font-size:0.78rem;font-weight:600;color:#6b7280;text-transform:uppercase;">🏆 Best SSIM</div>
            <div style="font-size:1.1rem;font-weight:700;color:#2563eb;margin-top:4px;">{ws}</div></div>
          <div class="winner-box" style="border-color:#dde1e8; background:#f0f2f5;">
            <div style="font-size:0.78rem;font-weight:600;color:#6b7280;text-transform:uppercase;">📊 Metric Guide</div>
            <div style="font-size:0.82rem;color:#374151;margin-top:4px;">PSNR &gt; 22 dB = Good &nbsp;|&nbsp; SSIM &gt; 0.85 = High</div></div>
        </div>"""

    # ── Comparison table ───────────────────────────────────────────────────────
    labels   = list(all_results.keys())
    n_models = len(labels)

    def get_color(label):
        mc, _, _ = model_color(label)
        return mc

    th_imgs = "<th>Grayscale</th>" + "".join(
        f'<th style="color:{get_color(l)}">{l}</th>' for l in labels
    ) + "<th>Ground Truth</th>"
    th_metrics = "".join(
        f'<th style="color:{get_color(l)}">{l}<br>'
        f'<span style="font-weight:400;font-size:0.78em;">PSNR / SSIM</span></th>'
        for l in labels
    )

    table_html = (
        '<h2 class="section-title">Side-by-Side Comparison — First 50 Images</h2>'
        f'<table><thead><tr>{th_imgs}{th_metrics}</tr></thead><tbody>'
    )

    row_count = min(50, min(len(r) for r in all_results.values()))
    rows_data = [all_results[l] for l in labels]

    for i in range(row_count):
        psnr_vals  = [rows_data[j][i]["psnr"] for j in range(n_models)]
        ssim_vals  = [rows_data[j][i]["ssim"] for j in range(n_models)]
        best_p, best_s = max(psnr_vals), max(ssim_vals)

        gray_td = f'<td><img src="data:image/jpeg;base64,{b64(rows_data[0][i]["grayscale"])}"/></td>'
        col_tds = "".join(
            f'<td><img src="data:image/jpeg;base64,{b64(rows_data[j][i]["colorized"])}"/></td>'
            for j in range(n_models)
        )
        gt_td   = f'<td><img src="data:image/jpeg;base64,{b64(rows_data[0][i]["gt"])}"/></td>'

        metric_tds = ""
        for j in range(n_models):
            p  = rows_data[j][i]["psnr"]
            s  = rows_data[j][i]["ssim"]
            cp = "#16a34a" if p == best_p else "#d97706"
            cs = "#16a34a" if s == best_s else "#d97706"
            metric_tds += (
                f'<td><span style="color:{cp};font-weight:600;">{p:.2f} dB</span>'
                f'<br><span style="color:{cs};">{s:.4f}</span></td>'
            )

        row_bg = "background:#f9fafb;" if i % 2 == 0 else ""
        table_html += f'<tr style="{row_bg}">{gray_td}{col_tds}{gt_td}{metric_tds}</tr>'

    table_html += "</tbody></table>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Colorization Evaluation Report</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter','Segoe UI',Arial,sans-serif;background:#f5f6f8;color:#111827;font-size:15px}}
  .page-header{{background:#fff;border-bottom:2px solid #dde1e8;padding:1.2rem 2rem;display:flex;align-items:center;justify-content:space-between}}
  .page-title{{font-size:1.4rem;font-weight:700}}.page-sub{{font-size:.82rem;color:#6b7280;margin-top:3px}}
  .page-badge{{font-size:.78rem;font-weight:600;padding:.3rem .85rem;border-radius:6px;background:#dbeafe;color:#2563eb;border:1.5px solid #93c5fd}}
  .content{{max-width:1400px;margin:0 auto;padding:2rem}}
  .model-card{{background:#fff;border:1.5px solid #dde1e8;border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
  .model-title{{font-size:1rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1rem}}
  .metrics-row{{display:flex;gap:1rem;flex-wrap:wrap}}
  .metric{{background:#f5f6f8;border:1px solid #dde1e8;border-radius:10px;padding:1rem 1.4rem;text-align:center;min-width:120px}}
  .metric-val{{font-size:1.6rem;font-weight:700;line-height:1.1}}.metric-label{{font-size:.75rem;font-weight:600;color:#6b7280;margin-top:.35rem;text-transform:uppercase;letter-spacing:.04em}}
  .winners-row{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem}}
  .winner-box{{flex:1;min-width:180px;padding:1rem 1.2rem;border:1.5px solid;border-radius:10px;text-align:center}}
  .section-title{{font-size:1rem;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.08em;margin:2rem 0 1rem 0;padding-bottom:.6rem;border-bottom:2px solid #dde1e8}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;border:1.5px solid #dde1e8}}
  thead{{position:sticky;top:0;z-index:2}}
  th{{background:#f0f2f5;padding:.85rem .7rem;font-size:.78rem;font-weight:700;text-transform:uppercase;color:#374151;border-bottom:2px solid #dde1e8}}
  td{{padding:.55rem .65rem;text-align:center;border-bottom:1px solid #e5e7eb;font-size:.85rem;vertical-align:middle}}
  tr:hover td{{background:#eff6ff!important}}
  img{{width:108px;height:108px;object-fit:cover;border-radius:6px;border:1px solid #dde1e8;display:block;margin:0 auto}}
  .page-footer{{text-align:center;padding:1.5rem;font-size:.78rem;color:#9ca3af;border-top:1px solid #dde1e8;margin-top:2rem;background:#fff}}
</style>
</head>
<body>
<div class="page-header">
  <div>
    <div class="page-title">🎨 Image Colorization — Evaluation Report</div>
    <div class="page-sub">Automated evaluation · {summaries[0]["num_images"]} validation images · GAN · ResNet · OpenCV</div>
  </div>
  <div class="page-badge">{len(summaries)} Model{"s" if len(summaries)>1 else ""} Evaluated</div>
</div>
<div class="content">{cards_html}{winner_html}{table_html}</div>
<div class="page-footer">Image Colorization — B.Tech Final Year IT Project &nbsp;·&nbsp; Generated by evaluate.py</div>
</body></html>"""

    report_path = out_dir / "eval_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\n📊 HTML report → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _eval_fn_torch(G, path):
    """
    Wrapper for colorize_torch that returns (gt_numpy, colorized_numpy).

    colorize_torch returns (colorized_PIL, original_numpy).
    evaluate_model expects (ground_truth_numpy, colorized_numpy).
    This wrapper: swaps the order + converts colorized PIL to numpy uint8.
    """
    img          = Image.open(path).convert("RGB")
    colorized_pil, gt_np = colorize_torch(G, img, boost=1.0)
    # Resize colorized to match gt dimensions before converting
    colorized_np = np.array(
        colorized_pil.resize((gt_np.shape[1], gt_np.shape[0]), Image.LANCZOS)
    )
    return gt_np, colorized_np


def _eval_fn_opencv(net, path):
    """
    Wrapper for colorize_opencv that returns (gt_numpy, colorized_numpy).
    Same swap as _eval_fn_torch — colorize_opencv also returns (PIL, numpy).
    """
    img          = Image.open(path).convert("RGB")
    colorized_pil, gt_np = colorize_opencv(net, img, boost=1.0)
    colorized_np = np.array(
        colorized_pil.resize((gt_np.shape[1], gt_np.shape[0]), Image.LANCZOS)
    )
    return gt_np, colorized_np


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GAN, ResNet, and OpenCV colorization models"
    )
    parser.add_argument("--model",     default=None,
                        help="Specific checkpoint path. Auto-detects all models if omitted.")
    parser.add_argument("--images",    default="dataset/val",
                        help="Validation images folder (default: dataset/val)")
    parser.add_argument("--output",    default="eval_report",
                        help="Output folder for HTML report and JSON files")
    parser.add_argument("--limit",     type=int, default=500,
                        help="Max images per model. 50 is recommended. (default: 500)")
    parser.add_argument("--no-opencv", action="store_true",
                        help="Skip OpenCV model (faster)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    # ── Build list of (colorize_fn, label) pairs ──────────────────────────────
    # NOTE: boost=1.0 is used in all lambdas below.
    # Raw output (no amplification) is essential for fair PSNR/SSIM measurement.
    models_to_run = []

    if args.model:
        is_resnet = "resnet" in args.model.lower()
        G         = load_torch_generator(args.model, resnet=is_resnet)
        label     = Path(args.model).stem
        models_to_run.append(
            (lambda p, G=G: _eval_fn_torch(G, p), label)
        )

    else:
        # Auto-detect using best_checkpoints.json portrait keys
        gan_path = best_checkpoint_for("gan", scene="portrait")
        if gan_path:
            G_gan = load_torch_generator(gan_path, resnet=False)
            label = Path(gan_path).stem
            models_to_run.append(
                (lambda p, G=G_gan: _eval_fn_torch(G, p), label)
            )
            print(f"✅ GAN loaded    : {gan_path}")
        else:
            print("⚠️  GAN: no checkpoint found in checkpoints/")

        resnet_path = best_checkpoint_for("resnet", scene="portrait")
        if resnet_path:
            G_res = load_torch_generator(resnet_path, resnet=True)
            label = Path(resnet_path).stem
            models_to_run.append(
                (lambda p, G=G_res: _eval_fn_torch(G, p), label)
            )
            print(f"✅ ResNet loaded : {resnet_path}")
        else:
            print("⚠️  ResNet: no checkpoint found in checkpoints/")

        if not args.no_opencv:
            try:
                opencv_net = load_opencv_net()
                models_to_run.append(
                    (lambda p, net=opencv_net: _eval_fn_opencv(net, p), "OpenCV_Zhang2016")
                )
                print(f"✅ OpenCV loaded : Zhang et al. 2016")
            except ImportError:
                print("⚠️  OpenCV skipped — pip install opencv-python")
            except Exception as e:
                print(f"⚠️  OpenCV skipped: {e}")

    if not models_to_run:
        print("❌ No models found. Exiting.")
        return

    img_dir = Path(args.images)
    files   = sorted(img_dir.glob("*.jpg"))[:args.limit]
    print(f"\n📁 {len(files)} validation images from: {img_dir}")
    if not files:
        print(f"❌ No .jpg files found! Check: {img_dir}")
        return

    all_results, all_summaries = {}, []

    for colorize_fn, label in models_to_run:
        print(f"\n{'='*50}\n🔧 Evaluating: {label}")
        results, summary = evaluate_model(colorize_fn, files, label)
        all_results[label]   = results
        all_summaries.append(summary)
        json_path = out_dir / f"metrics_{label}.json"
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"💾 Metrics saved → {json_path}")

    print(f"\n{'='*55}\n  📈  EVALUATION SUMMARY\n{'='*55}")
    print(f"  {'Model':<38} {'PSNR':>8}  {'SSIM':>8}")
    print(f"  {'-'*55}")
    for s in all_summaries:
        print(f"  {s['label']:<38} {s['avg_psnr']:>7.2f} dB  {s['avg_ssim']:>8.4f}")
    print(f"{'='*55}")

    if len(all_summaries) > 1:
        best_p = max(all_summaries, key=lambda x: x["avg_psnr"])
        best_s = max(all_summaries, key=lambda x: x["avg_ssim"])
        print(f"  🏆 Best PSNR : {best_p['label']}  ({best_p['avg_psnr']} dB)")
        print(f"  🏆 Best SSIM : {best_s['label']}  ({best_s['avg_ssim']})")
        print(f"{'='*55}")

    save_html_report(all_results, out_dir, all_summaries)


if __name__ == "__main__":
    main()
