import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nets.cbam_res_unet_exp11 import CBAMResUnetExp11  # noqa: E402


DEFAULT_VIDEO_DIR = ROOT / "resources" / "videos"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "video_diameter_results"
DEFAULT_WEIGHT = SCRIPT_DIR / "logs" / "final_exp11_miou98_16_epoch070.pth"
INPUT_SIZE = 256


def load_model(weight_path: Path, device: torch.device) -> torch.nn.Module:
    model = CBAMResUnetExp11(num_classes=2, pretrained=False)
    checkpoint = torch.load(str(weight_path), map_location=device)
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        checkpoint = checkpoint["net"]
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.maximum(mask_u8, holes).astype(np.uint8)


def refine_mask(prob_map: np.ndarray, threshold: float = 0.31, min_area: int = 30) -> np.ndarray:
    binary = (prob_map > threshold).astype(np.uint8)
    binary = fill_holes(binary)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1
        if areas[largest_idx - 1] >= min_area:
            binary = (labels == largest_idx).astype(np.uint8)

    binary = cv2.medianBlur(binary, 3)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return (binary > 0).astype(np.uint8)


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


def fit_pupil_ellipse(mask: np.ndarray, min_area: int = 30) -> Optional[dict]:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area or len(contour) < 5:
        return None

    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (axis_a, axis_b), angle = ellipse
    major_axis = float(max(axis_a, axis_b))
    minor_axis = float(min(axis_a, axis_b))
    pupil_diameter = float((major_axis + minor_axis) / 2.0)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "pupil_diameter": pupil_diameter,
        "ellipse": ellipse,
    }


def draw_overlay(frame_bgr: np.ndarray, result: Optional[dict]) -> np.ndarray:
    overlay = frame_bgr.copy()
    if result is None:
        cv2.putText(
            overlay,
            "No pupil detected",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    cv2.ellipse(overlay, result["ellipse"], (0, 200, 0), 2)
    center = (int(round(result["cx"])), int(round(result["cy"])))
    cv2.circle(overlay, center, 3, (0, 0, 255), -1)
    cv2.putText(
        overlay,
        f"Diameter = {result['pupil_diameter']:.2f} px",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 0),
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
    max_frames: Optional[int] = None,
    save_overlay: bool = True,
    fill_nan: bool = True,
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
    if max_frames is not None and total_frames > 0:
        total_for_bar = min(total_frames, max_frames)
    else:
        total_for_bar = total_frames if total_frames > 0 else None

    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = video_output_dir / f"{video_path.stem}_diameter.csv"
    overlay_path = video_output_dir / f"{video_path.stem}_overlay.mp4"

    writer = None
    if save_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    records = []
    frame_id = 0
    progress = tqdm(total=total_for_bar, desc=f"Processing {video_path.name}")

    while True:
        if max_frames is not None and frame_id >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        time_sec = frame_id / fps
        mask = predict_mask(model, frame, device, threshold=threshold)
        result = fit_pupil_ellipse(mask)

        if result is None:
            if fill_nan:
                records.append({
                    "frame_id": frame_id,
                    "time": time_sec,
                    "cx": np.nan,
                    "cy": np.nan,
                    "major_axis": np.nan,
                    "minor_axis": np.nan,
                    "pupil_diameter": np.nan,
                })
        else:
            records.append({
                "frame_id": frame_id,
                "time": time_sec,
                "cx": result["cx"],
                "cy": result["cy"],
                "major_axis": result["major_axis"],
                "minor_axis": result["minor_axis"],
                "pupil_diameter": result["pupil_diameter"],
            })

        if writer is not None:
            writer.write(draw_overlay(frame, result))

        frame_id += 1
        progress.update(1)

    progress.close()
    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(
        records,
        columns=["frame_id", "time", "cx", "cy", "major_axis", "minor_axis", "pupil_diameter"],
    )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"Saved diameter CSV: {csv_path}")
    if save_overlay:
        print(f"Saved overlay video: {overlay_path}")


def collect_videos(video_dir: Path) -> list[Path]:
    suffixes = {".mp4", ".avi"}
    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate per-frame pupil diameter from videos using Exp11 segmentation and ellipse fitting.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Input video directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT, help="Exp11 model weight path.")
    parser.add_argument("--threshold", type=float, default=0.31, help="Pupil probability threshold.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick testing.")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay video generation.")
    parser.add_argument("--skip-failed", action="store_true", help="Skip failed frames instead of writing NaN rows.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback if CUDA is unavailable.")
    args = parser.parse_args()

    if not args.video_dir.exists():
        raise FileNotFoundError(args.video_dir)
    if not args.weight.exists():
        raise FileNotFoundError(args.weight)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Please run in the pupil_cuda116 environment or pass --allow-cpu.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")

    videos = collect_videos(args.video_dir)
    if not videos:
        print("No .mp4 or .avi videos found.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weight, device)
    for video_path in videos:
        process_video(
            video_path=video_path,
            model=model,
            device=device,
            output_dir=args.output_dir,
            threshold=args.threshold,
            max_frames=args.max_frames,
            save_overlay=not args.no_overlay,
            fill_nan=not args.skip_failed,
        )


if __name__ == "__main__":
    main()
