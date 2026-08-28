import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Simple tqdm fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nets.cbam_res_unet_exp11 import CBAMResUnetExp11
from utils.utils import cvtColor, preprocess_input, resize_image


DEFAULT_MASK_DIR = SCRIPT_DIR / "miou_out" / "detection-results-optimized-final"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pupil_params_results"
DEFAULT_WEIGHT_PATH = SCRIPT_DIR / "logs" / "final_exp11_miou98_16_epoch070.pth"
DEFAULT_LPW_IMAGE_DIR = ROOT / "VOCdevkit" / "VOC2007" / "JPEGImages"
DEFAULT_LPW_VAL_TXT = ROOT / "VOCdevkit" / "VOC2007" / "ImageSets" / "Segmentation" / "val.txt"


def to_binary_mask(mask):
    """Convert a mask-like array to a 0/255 uint8 binary mask."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.float32)
    if mask.max() <= 1.0:
        mask = mask * 255.0
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def clean_binary_mask(mask, median_kernel=3, morph_kernel=5):
    """Remove small noise and smooth the binary pupil mask."""
    mask = to_binary_mask(mask)
    if median_kernel and median_kernel >= 3:
        if median_kernel % 2 == 0:
            median_kernel += 1
        mask = cv2.medianBlur(mask, median_kernel)

    if morph_kernel and morph_kernel >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return to_binary_mask(mask)


def find_largest_contour(mask):
    """Return the largest external contour and all external contours."""
    mask = to_binary_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, contours
    largest = max(contours, key=cv2.contourArea)
    return largest, contours


def ellipse_residual_score(contour, ellipse, image_shape):
    """
    Estimate ellipse fitting quality from normalized algebraic residual.

    For points on a perfect ellipse:
        (x'/a)^2 + (y'/b)^2 = 1
    The average absolute deviation from 1 is mapped to [0, 1].
    """
    (cx, cy), (axis_a, axis_b), angle = ellipse
    semi_a = max(axis_a, axis_b) / 2.0
    semi_b = min(axis_a, axis_b) / 2.0
    if semi_a <= 1e-6 or semi_b <= 1e-6:
        return 0.0, float("inf")

    points = contour.reshape(-1, 2).astype(np.float32)
    theta = np.deg2rad(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = points[:, 0] - cx
    y = points[:, 1] - cy
    xr = x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t

    residual = np.abs((xr / semi_a) ** 2 + (yr / semi_b) ** 2 - 1.0)
    mean_residual = float(np.mean(residual))
    score = float(np.exp(-3.0 * mean_residual))
    return float(np.clip(score, 0.0, 1.0)), mean_residual


def compute_confidence(largest_area, total_mask_area, contour, ellipse, image_shape):
    """
    Combine connected-component dominance, ellipse fit quality, and circularity.

    The final confidence is normalized to [0, 1]:
    - area_ratio_score: largest contour area / total foreground area
    - fit_quality_score: how closely contour points follow the fitted ellipse
    - circularity_score: 4*pi*area/perimeter^2, with mild tolerance for elliptical pupils
    """
    if total_mask_area <= 0:
        return 0.0, {
            "area_ratio": 0.0,
            "fit_quality": 0.0,
            "circularity": 0.0,
            "ellipse_residual": float("inf"),
        }

    area_ratio = float(np.clip(largest_area / total_mask_area, 0.0, 1.0))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        circularity = 0.0
    else:
        circularity = float(4.0 * np.pi * largest_area / (perimeter * perimeter))
    circularity_score = float(np.clip(circularity / 0.85, 0.0, 1.0))

    fit_quality, ellipse_residual = ellipse_residual_score(contour, ellipse, image_shape)
    confidence = 0.35 * area_ratio + 0.40 * fit_quality + 0.25 * circularity_score
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return confidence, {
        "area_ratio": area_ratio,
        "fit_quality": fit_quality,
        "circularity": circularity,
        "ellipse_residual": ellipse_residual,
    }


def extract_pupil_from_mask(mask, do_clean=True, min_area=20):
    """
    Extract pupil center and ellipse parameters from a binary segmentation mask.

    Args:
        mask: numpy array containing a binary mask, either 0/1 or 0/255.
        do_clean: whether to apply median filtering and morphology.
        min_area: minimum largest-contour area in pixels.

    Returns:
        None if extraction fails, otherwise a dictionary with center, axes,
        angle, confidence, and quality sub-scores.
    """
    original_mask = to_binary_mask(mask)
    processed_mask = clean_binary_mask(original_mask) if do_clean else original_mask
    largest, contours = find_largest_contour(processed_mask)
    if largest is None:
        return None

    largest_area = float(cv2.contourArea(largest))
    total_mask_area = float(np.count_nonzero(processed_mask))
    if largest_area < min_area:
        return None
    if len(largest) < 5:
        return None

    ellipse = cv2.fitEllipse(largest)
    (cx, cy), (axis_a, axis_b), angle = ellipse
    major_axis = float(max(axis_a, axis_b))
    minor_axis = float(min(axis_a, axis_b))
    confidence, quality = compute_confidence(largest_area, total_mask_area, largest, ellipse, processed_mask.shape)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "angle": float(angle),
        "confidence": confidence,
        "area": largest_area,
        "total_mask_area": total_mask_area,
        "area_ratio": quality["area_ratio"],
        "fit_quality": quality["fit_quality"],
        "ellipse_residual": quality["ellipse_residual"],
        "circularity": quality["circularity"],
        "contour": largest,
        "ellipse": ellipse,
        "mask": processed_mask,
    }


def draw_pupil_overlay(image, result, output_path=None):
    """Draw fitted ellipse, center point, and confidence on an image."""
    if isinstance(image, Image.Image):
        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image.copy()
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    canvas = image_bgr.copy()
    if result is not None:
        cv2.ellipse(canvas, result["ellipse"], (0, 255, 0), 2)
        center = (int(round(result["cx"])), int(round(result["cy"])))
        cv2.circle(canvas, center, 3, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"({result['cx']:.1f}, {result['cy']:.1f}) conf={result['confidence']:.2f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if output_path:
        cv2.imwrite(str(output_path), canvas)
    return canvas


def load_exp11_model(weight_path, device):
    model = CBAMResUnetExp11(num_classes=2, pretrained=False)
    checkpoint = torch.load(str(weight_path), map_location=device)
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        checkpoint = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.to(device).eval()
    return model


def predict_mask_with_model(model, image, device, input_shape=(256, 256), threshold=0.5):
    image = cvtColor(image)
    original_w, original_h = image.size
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    with torch.no_grad():
        tensor = torch.from_numpy(image_data).to(device)
        pred = model(tensor)[0]
        pred = F.softmax(pred.permute(1, 2, 0), dim=-1).cpu().numpy()
        pred = pred[
            int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
            int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw),
        ]
        prob = np.array(Image.fromarray(pred[:, :, 1]).resize((original_w, original_h), Image.BILINEAR))
        return np.where(prob > threshold, 255, 0).astype(np.uint8)


def read_lpw_ids(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f.readlines() if line.strip()]


def process_mask_directory(
    mask_dir,
    output_csv,
    image_dir=None,
    overlay_dir=None,
    image_ids=None,
    model_weight=None,
    threshold=0.5,
    use_gpu=True,
):
    """
    Process all masks in a directory and save pupil center results as CSV.

    If a mask is missing and model_weight/image_dir are provided, the mask is
    predicted by the experiment-11 model before ellipse fitting.
    """
    mask_dir = Path(mask_dir)
    output_csv = Path(output_csv)
    image_dir = Path(image_dir) if image_dir else None
    overlay_dir = Path(overlay_dir) if overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if image_ids is None:
        image_ids = sorted(p.stem for p in mask_dir.glob("*.png"))

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = None
    rows = []
    for image_id in tqdm(image_ids, desc="Extract pupil centers"):
        mask_path = mask_dir / f"{image_id}.png"
        image_path = image_dir / f"{image_id}.jpg" if image_dir else None
        source = "saved_mask"

        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
        elif image_path and image_path.exists() and model_weight:
            if model is None:
                model = load_exp11_model(model_weight, device)
            image = Image.open(image_path).convert("RGB")
            mask = predict_mask_with_model(model, image, device, threshold=threshold)
            source = "model_prediction"
        else:
            rows.append([image_id, "missing_mask", "", "", "", "", "", "", "", "", "", "", source])
            continue

        result = extract_pupil_from_mask(mask)
        if result is None:
            rows.append([image_id, "failed", "", "", "", "", "", "", "", "", "", "", source])
            continue

        rows.append([
            image_id,
            "ok",
            f"{result['cx']:.3f}",
            f"{result['cy']:.3f}",
            f"{result['major_axis']:.3f}",
            f"{result['minor_axis']:.3f}",
            f"{result['angle']:.3f}",
            f"{result['confidence']:.4f}",
            f"{result['area_ratio']:.4f}",
            f"{result['fit_quality']:.4f}",
            f"{result['circularity']:.4f}",
            f"{result['ellipse_residual']:.6f}",
            source,
        ])

        if overlay_dir and image_path and image_path.exists():
            image = Image.open(image_path).convert("RGB")
            draw_pupil_overlay(image, result, overlay_dir / f"{image_id}.png")

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "status",
            "cx",
            "cy",
            "major_axis",
            "minor_axis",
            "angle",
            "confidence",
            "area_ratio",
            "fit_quality",
            "circularity",
            "ellipse_residual",
            "source",
        ])
        writer.writerows(rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract LPW pupil center coordinates with ellipse fitting and confidence scores."
    )
    parser.add_argument("--mask-dir", default=str(DEFAULT_MASK_DIR), help="Directory containing predicted binary masks.")
    parser.add_argument("--image-dir", default=str(DEFAULT_LPW_IMAGE_DIR), help="LPW image directory for visualization/fallback prediction.")
    parser.add_argument("--split-file", default=str(DEFAULT_LPW_VAL_TXT), help="LPW split file. Use empty string to process all masks.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory under experiment 11.")
    parser.add_argument("--csv-name", default="lpw_pupil_center_coordinates.csv", help="Output CSV filename.")
    parser.add_argument("--overlay", action="store_true", help="Save visualization overlays.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional number of images to process.")
    parser.add_argument("--model-weight", default=str(DEFAULT_WEIGHT_PATH), help="Fallback experiment-11 weight path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fallback model prediction threshold.")
    parser.add_argument("--cpu", action="store_true", help="Disable GPU for fallback model prediction.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_csv = output_dir / args.csv_name
    overlay_dir = output_dir / "ellipse_overlays" if args.overlay else None

    image_ids = None
    if args.split_file:
        image_ids = read_lpw_ids(args.split_file)
        if args.num_samples is not None:
            image_ids = image_ids[: args.num_samples]

    rows = process_mask_directory(
        mask_dir=args.mask_dir,
        output_csv=output_csv,
        image_dir=args.image_dir,
        overlay_dir=overlay_dir,
        image_ids=image_ids,
        model_weight=Path(args.model_weight) if args.model_weight else None,
        threshold=args.threshold,
        use_gpu=not args.cpu,
    )

    ok_count = sum(1 for r in rows if r[1] == "ok")
    print(f"Processed: {len(rows)}")
    print(f"Success: {ok_count}")
    print(f"Saved CSV: {output_csv}")
    if overlay_dir:
        print(f"Saved overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
