"""Create a QC-oriented pupil prediction video from an input video and CSV results."""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def robust_limits(values: np.ndarray, multiplier: float = 4.0) -> tuple[float, float]:
    """Return median/MAD limits without allowing a nearly constant series to collapse."""
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    robust_sigma = max(1.4826 * mad, 1.0)
    return median - multiplier * robust_sigma, median + multiplier * robust_sigma


def build_quality_labels(df: pd.DataFrame) -> list[str]:
    valid = df["status"].eq("ok")
    diameter = pd.to_numeric(df["diameter_px"], errors="coerce").to_numpy(float)
    major = pd.to_numeric(df["major_axis_px"], errors="coerce").to_numpy(float)
    minor = pd.to_numeric(df["minor_axis_px"], errors="coerce").to_numpy(float)
    area = pd.to_numeric(df["area_px"], errors="coerce").to_numpy(float)

    valid_diameter = diameter[valid.to_numpy() & np.isfinite(diameter)]
    valid_area = area[valid.to_numpy() & np.isfinite(area)]
    diameter_low, diameter_high = robust_limits(valid_diameter)
    area_low, area_high = robust_limits(valid_area)

    labels = []
    previous_diameter = np.nan
    for idx in range(len(df)):
        if not valid.iat[idx] or not np.isfinite(diameter[idx]):
            labels.append("NO DETECTION")
            continue

        axis_ratio = major[idx] / minor[idx] if minor[idx] > 0 else np.inf
        jump = abs(diameter[idx] - previous_diameter) if np.isfinite(previous_diameter) else 0.0
        if (
            diameter[idx] < diameter_low
            or diameter[idx] > diameter_high
            or area[idx] < area_low
            or area[idx] > area_high
            or axis_ratio > 1.8
            or jump > 8.0
        ):
            labels.append("CHECK")
        else:
            labels.append("VALID")
        previous_diameter = diameter[idx]
    return labels


def draw_text_box(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, padding = 0.55, 2, 8
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x - padding, y - height - padding), (x + width + padding, y + baseline + padding), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_curve_panel(frame: np.ndarray, history: deque[float], current_diameter: float) -> None:
    height, width = frame.shape[:2]
    panel_w = min(370, width - 32)
    panel_h = 104
    x0, y0 = width - panel_w - 16, height - panel_h - 14
    panel = frame[y0 : y0 + panel_h, x0 : x0 + panel_w]
    shade = panel.copy()
    shade[:] = (18, 18, 18)
    cv2.addWeighted(shade, 0.78, panel, 0.22, 0, panel)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 1)
    cv2.putText(frame, "Diameter history (px)", (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)

    values = np.asarray(history, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return
    low, high = np.percentile(values, [2, 98])
    margin = max((high - low) * 0.25, 5.0)
    low, high = low - margin, high + margin
    plot_x0, plot_x1 = x0 + 12, x0 + panel_w - 12
    plot_y0, plot_y1 = y0 + 30, y0 + panel_h - 22
    cv2.line(frame, (plot_x0, plot_y1), (plot_x1, plot_y1), (125, 125, 125), 1)
    cv2.line(frame, (plot_x0, plot_y0), (plot_x0, plot_y1), (125, 125, 125), 1)

    smoothed = pd.Series(history, dtype=float).interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).median().to_numpy()
    points = []
    count = len(history)
    for index, value in enumerate(smoothed):
        if not np.isfinite(value):
            continue
        x = int(plot_x0 + (plot_x1 - plot_x0) * index / max(count - 1, 1))
        y = int(plot_y1 - (value - low) / (high - low) * (plot_y1 - plot_y0))
        points.append((x, int(np.clip(y, plot_y0, plot_y1))))
    if len(points) >= 2:
        cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, (55, 215, 255), 2, cv2.LINE_AA)
    if np.isfinite(current_diameter):
        cv2.putText(frame, f"{current_diameter:.1f}", (x0 + panel_w - 57, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)


def enhance_video(video_path: Path, csv_path: Path, output_path: Path, alpha: float, history_frames: int) -> None:
    df = pd.read_csv(csv_path)
    labels = build_quality_labels(df)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write video: {output_path}")

    history: deque[float] = deque(maxlen=history_frames)
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index >= len(df):
            writer.write(frame)
            frame_index += 1
            continue

        row = df.iloc[frame_index]
        diameter = pd.to_numeric(pd.Series([row["diameter_px"]]), errors="coerce").iat[0]
        label = labels[frame_index]
        if label != "NO DETECTION":
            center = (int(round(float(row["center_x"]))), int(round(float(row["center_y"]))))
            axes = (float(row["major_axis_px"]), float(row["minor_axis_px"]))
            ellipse = (center, (int(round(axes[0])), int(round(axes[1]))), 0)
            mask_layer = frame.copy()
            cv2.ellipse(mask_layer, ellipse, (25, 125, 25), -1)
            cv2.addWeighted(mask_layer, alpha, frame, 1.0 - alpha, 0, frame)
            outline = (0, 215, 0) if label == "VALID" else (0, 165, 255)
            cv2.ellipse(frame, ellipse, outline, 2, cv2.LINE_AA)
            cv2.circle(frame, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
            draw_text_box(frame, f"Diameter  {diameter:.1f} px", (14, 32), (245, 245, 245))
            history.append(float(diameter))
        else:
            history.append(np.nan)

        status_color = (0, 215, 0) if label == "VALID" else ((0, 165, 255) if label == "CHECK" else (0, 0, 255))
        status_text = {"VALID": "Tracking normal", "CHECK": "Review frame", "NO DETECTION": "No pupil"}[label]
        draw_text_box(frame, status_text, (14, 65), status_color)
        draw_curve_panel(frame, history, float(diameter) if np.isfinite(diameter) else np.nan)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    print(f"Saved enhanced QC video: {output_path}")
    print(f"Frames written: {frame_index}; QC labels: {dict(pd.Series(labels).value_counts())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add mask, quality-control label, and diameter curve to a pupil result video.")
    parser.add_argument("--video", type=Path, required=True, help="Source eye video.")
    parser.add_argument("--csv", type=Path, required=True, help="Per-frame pupil parameter CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--alpha", type=float, default=0.14, help="Mask opacity from 0 to 1.")
    parser.add_argument("--history-frames", type=int, default=150, help="Number of past frames in the curve panel.")
    args = parser.parse_args()
    enhance_video(args.video, args.csv, args.output, args.alpha, args.history_frames)


if __name__ == "__main__":
    main()
