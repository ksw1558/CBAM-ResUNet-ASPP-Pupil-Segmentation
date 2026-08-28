import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.append(str(ROOT))

from nets.cbam_res_unet_exp11 import CBAMResUnetExp11  # noqa: E402


DEFAULT_WEIGHT = SCRIPT_DIR / "logs" / "final_exp11_miou98_16_epoch070.pth"
DEFAULT_VIDEO_DIR = ROOT / "resources" / "videos"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "video_predictions"
INPUT_SIZE = 256


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.maximum(mask_u8, holes).astype(np.uint8)


def refine_mask(prob_map: np.ndarray, threshold: float = 0.31, min_area: int = 30) -> np.ndarray:
    binary_mask = (prob_map > threshold).astype(np.uint8)
    filled_mask = fill_holes(binary_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1
        if areas[largest_idx - 1] >= min_area:
            filled_mask = (labels == largest_idx).astype(np.uint8)

    smoothed = cv2.medianBlur(filled_mask, 3)
    blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (3, 3), 0)
    return (blurred > 0.5).astype(np.uint8)


def load_model(weight_path: Path, device: torch.device) -> torch.nn.Module:
    model = CBAMResUnetExp11(num_classes=2, pretrained=False)
    checkpoint = torch.load(str(weight_path), map_location=device)
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        checkpoint = checkpoint["net"]
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_mask(model: torch.nn.Module, frame_bgr: np.ndarray, device: torch.device, threshold: float) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0, 1].detach().cpu().numpy()

    small_mask = refine_mask(prob, threshold=threshold)
    mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def measure_pupil(mask: np.ndarray) -> Optional[dict]:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None

    result = {
        "area": area,
        "center_x": "",
        "center_y": "",
        "diameter_px": "",
        "major_axis_px": "",
        "minor_axis_px": "",
        "method": "",
        "contour": contour,
    }

    if len(contour) >= 5:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        major_axis = float(max(axis_a, axis_b))
        minor_axis = float(min(axis_a, axis_b))
        result.update(
            {
                "center_x": float(cx),
                "center_y": float(cy),
                "diameter_px": float((major_axis + minor_axis) / 2.0),
                "major_axis_px": major_axis,
                "minor_axis_px": minor_axis,
                "method": "ellipse",
                "ellipse": ((float(cx), float(cy)), (float(axis_a), float(axis_b)), float(angle)),
            }
        )
        return result

    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2.0
        cy = y + h / 2.0

    diameter = 2.0 * np.sqrt(area / np.pi)
    result.update(
        {
            "center_x": float(cx),
            "center_y": float(cy),
            "diameter_px": float(diameter),
            "major_axis_px": float(diameter),
            "minor_axis_px": float(diameter),
            "method": "equivalent_circle",
        }
    )
    return result


def draw_overlay(frame_bgr: np.ndarray, mask: np.ndarray, measurement: Optional[dict]) -> np.ndarray:
    overlay = frame_bgr.copy()

    if measurement is None:
        cv2.putText(
            overlay,
            "No pupil detected",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    contour = measurement.get("contour")
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 180, 0), 1)

    if "ellipse" in measurement:
        cv2.ellipse(overlay, measurement["ellipse"], (0, 255, 0), 3)

    cx = int(round(measurement["center_x"]))
    cy = int(round(measurement["center_y"]))

    cv2.circle(overlay, (cx, cy), 3, (0, 255, 255), -1)
    cv2.putText(
        overlay,
        f"Center=({measurement['center_x']:.1f}, {measurement['center_y']:.1f})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def process_video(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    max_frames: Optional[int],
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_for_bar = min(total_frames, max_frames) if total_frames > 0 else max_frames
    else:
        total_for_bar = total_frames if total_frames > 0 else None

    video_stem = video_path.stem
    video_output_dir = output_dir / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = video_output_dir / f"{video_stem}_exp11_realtime_ellipse.mp4"
    csv_path = video_output_dir / f"{video_stem}_pupil_params.csv"
    prediction_csv_path = video_output_dir / "prediction.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    rows = []
    prediction_rows = []
    frame_idx = 0
    start_time = time.time()

    progress = tqdm(total=total_for_bar, desc=f"Processing {video_path.name}")
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        mask = predict_mask(model, frame, device, threshold)
        measurement = measure_pupil(mask)
        overlay = draw_overlay(frame, mask, measurement)

        overlay_writer.write(overlay)
        current_time = frame_idx / fps if fps > 0 else float(frame_idx)

        if measurement is None:
            rows.append([frame_idx, "no_pupil", "", "", "", "", "", "", ""])
        else:
            prediction_rows.append(
                [
                    f"{current_time:.6f}",
                    f"{measurement['center_x']:.3f}",
                    f"{measurement['center_y']:.3f}",
                ]
            )
            rows.append(
                [
                    frame_idx,
                    "ok",
                    f"{measurement['center_x']:.3f}",
                    f"{measurement['center_y']:.3f}",
                    f"{measurement['diameter_px']:.3f}",
                    f"{measurement['major_axis_px']:.3f}",
                    f"{measurement['minor_axis_px']:.3f}",
                    f"{measurement['area']:.3f}",
                    measurement["method"],
                ]
            )

        frame_idx += 1
        progress.update(1)

    progress.close()
    cap.release()
    overlay_writer.release()

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "status",
                "center_x",
                "center_y",
                "diameter_px",
                "major_axis_px",
                "minor_axis_px",
                "area_px",
                "method",
            ]
        )
        writer.writerows(rows)

    with prediction_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "pred_center_x", "pred_center_y"])
        writer.writerows(prediction_rows)

    elapsed = time.time() - start_time
    speed = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"Saved realtime ellipse video: {overlay_path}")
    print(f"Saved pupil CSV: {csv_path}")
    print(f"Saved prediction CSV: {prediction_csv_path}")
    print(f"Frames: {frame_idx}, speed: {speed:.2f} FPS")


def collect_videos(video_dir: Path, video_name: Optional[str]) -> List[Path]:
    if video_name:
        path = video_dir / video_name
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]

    suffixes = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp11 CBAM-ResUNet-ASPP on pupil videos with CUDA.")
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Input video folder.")
    parser.add_argument("--video_name", type=str, default=None, help="Process only one video file in video_dir.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT, help="Exp11 model weight path.")
    parser.add_argument("--threshold", type=float, default=0.31, help="Pupil probability threshold.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU fallback when CUDA is unavailable.")
    args = parser.parse_args()

    if not args.video_dir.exists():
        raise FileNotFoundError(args.video_dir)
    if not args.weight.exists():
        raise FileNotFoundError(args.weight)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available in this Python environment. "
            "Please run with: conda activate pupil_cuda116"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Weight: {args.weight}")
    print(f"Input videos: {args.video_dir}")
    print(f"Output dir: {args.output_dir}")

    videos = collect_videos(args.video_dir, args.video_name)
    if not videos:
        print("No video files found.")
        return

    model = load_model(args.weight, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        process_video(
            video_path=video_path,
            model=model,
            device=device,
            output_dir=args.output_dir,
            threshold=args.threshold,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
