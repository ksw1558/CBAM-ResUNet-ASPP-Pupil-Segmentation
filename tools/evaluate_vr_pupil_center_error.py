import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VR_REQUIRED_COLUMNS = ["time", "center_x", "center_y"]
VR_OPTIONAL_COLUMNS = ["pupil_size", "confidence"]
PRED_COLUMNS = ["time", "pred_center_x", "pred_center_y"]


def parse_time_series(series):
    """Parse normal timestamps and VR timestamps like 2024-10-25 10:16:40:945."""
    text = series.astype(str).str.strip()
    fixed = text.str.replace(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):(\d+)$",
        r"\1.\2",
        regex=True,
    )
    parsed = pd.to_datetime(fixed, errors="coerce")
    if parsed.isna().all():
        numeric = pd.to_numeric(text, errors="coerce")
        parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
    return parsed


def parse_numeric_time(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def read_vr_data(path):
    df = pd.read_csv(path)
    missing = [c for c in VR_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"VR CSV missing columns: {missing}")

    keep_cols = VR_REQUIRED_COLUMNS + [c for c in VR_OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep_cols].copy()
    df["time_dt"] = parse_time_series(df["time"])
    numeric_cols = [c for c in ["center_x", "center_y", "pupil_size", "confidence"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_dt", "center_x", "center_y"])
    df = df.sort_values("time_dt").drop_duplicates("time_dt", keep="first")
    df["time_sec"] = (df["time_dt"] - df["time_dt"].min()).dt.total_seconds()
    return df


def read_prediction_data(path):
    df = pd.read_csv(path)
    missing = [c for c in PRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {missing}")

    df = df[PRED_COLUMNS].copy()
    df["time_numeric"] = parse_numeric_time(df["time"])
    df["time_dt"] = parse_time_series(df["time"])
    for col in ["pred_center_x", "pred_center_y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_dt", "pred_center_x", "pred_center_y"])
    df = df.sort_values("time_dt").drop_duplicates("time_dt", keep="first")
    if df["time_numeric"].notna().mean() > 0.95:
        df["time_sec"] = df["time_numeric"]
    else:
        df["time_sec"] = (df["time_dt"] - df["time_dt"].min()).dt.total_seconds()
    return df


def optionally_convert_vr_to_pixels(vr_df, image_width=None, image_height=None):
    """Use this only when VR centers are normalized [0, 1] but predictions are pixels."""
    if image_width is None or image_height is None:
        return vr_df
    vr_df = vr_df.copy()
    vr_df["center_x"] = vr_df["center_x"] * float(image_width)
    vr_df["center_y"] = vr_df["center_y"] * float(image_height)
    return vr_df


def align_by_nearest_time(vr_df, pred_df, tolerance_ms=None, time_mode="auto"):
    pred_is_relative = pred_df["time_numeric"].notna().mean() > 0.95
    use_relative = time_mode == "relative" or (time_mode == "auto" and pred_is_relative)

    if use_relative:
        tolerance = None if tolerance_ms is None else float(tolerance_ms) / 1000.0
        pred_for_merge = pred_df.sort_values("time_sec")
        vr_for_merge = vr_df.sort_values("time_sec")
        aligned = pd.merge_asof(
            pred_for_merge,
            vr_for_merge,
            on="time_sec",
            direction="nearest",
            tolerance=tolerance,
            suffixes=("_pred", "_vr"),
        )
    else:
        tolerance = None if tolerance_ms is None else pd.Timedelta(milliseconds=float(tolerance_ms))
        pred_for_merge = pred_df.sort_values("time_dt")
        vr_for_merge = vr_df.sort_values("time_dt")
        aligned = pd.merge_asof(
            pred_for_merge,
            vr_for_merge,
            on="time_dt",
            direction="nearest",
            tolerance=tolerance,
            suffixes=("_pred", "_vr"),
        )
    aligned = aligned.dropna(subset=["center_x", "center_y", "pred_center_x", "pred_center_y"])
    return aligned


def compute_error(aligned):
    aligned = aligned.copy()
    aligned["error"] = np.sqrt(
        (aligned["pred_center_x"] - aligned["center_x"]) ** 2
        + (aligned["pred_center_y"] - aligned["center_y"]) ** 2
    )
    return aligned


def summarize_errors(df):
    return {
        "Mean Error": df["error"].mean(),
        "Median Error": df["error"].median(),
        "Std Error": df["error"].std(ddof=1),
        "Max Error": df["error"].max(),
        "Min Error": df["error"].min(),
    }


def save_error_summary(stats, out_path):
    rows = [{"Metric": metric, "Value": f"{value:.6f}"} for metric, value in stats.items()]
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def set_sci_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def save_error_histogram(df, out_path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    ax.hist(df["error"], bins=50, color="#1f4e79", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Error")
    ax.set_ylabel("count")
    ax.set_title("Error Histogram")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_error_time_curve(df, out_path):
    x_col = "time_sec" if "time_sec" in df.columns else "time_dt"
    plot_df = df.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    ax.plot(plot_df[x_col], plot_df["error"], color="#1f4e79", linewidth=1.0)
    ax.set_xlabel("Time (s)" if x_col == "time_sec" else "Time")
    ax.set_ylabel("Error")
    ax.set_title("Error vs Time")
    ax.grid(color="0.9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if x_col != "time_sec":
        fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_error_confidence(df, out_path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    ax.scatter(df["confidence"], df["error"], s=10, color="#1f4e79", alpha=0.65, edgecolors="none")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Error")
    ax.set_title("Error vs Confidence")
    ax.grid(color="0.9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_outputs(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = output_dir / "evaluation_results.csv"
    summary_csv = output_dir / "error_summary.csv"
    hist_png = output_dir / "error_histogram.png"
    time_png = output_dir / "error_time_curve.png"
    conf_png = output_dir / "error_confidence.png"

    export_cols = [
        "time_pred",
        "time_vr",
        "time_sec",
        "center_x",
        "center_y",
        "pred_center_x",
        "pred_center_y",
        "pupil_size",
        "confidence",
        "error",
    ]
    df = df.copy()
    if "time_pred" not in df.columns and "time_x" in df.columns:
        df = df.rename(columns={"time_x": "time_pred", "time_y": "time_vr"})
    existing = [c for c in export_cols if c in df.columns]
    df[existing].to_csv(result_csv, index=False, encoding="utf-8-sig")
    stats = summarize_errors(df)
    save_error_summary(stats, summary_csv)

    set_sci_style()
    save_error_histogram(df, hist_png)
    save_error_time_curve(df, time_png)
    if "confidence" in df.columns:
        save_error_confidence(df, conf_png)

    return result_csv, summary_csv, hist_png, time_png, conf_png


def main():
    parser = argparse.ArgumentParser(description="Evaluate pupil center localization error using VR eye-tracking reference data.")
    parser.add_argument(
        "--vr-csv",
        default="cleaned_eye_tracking_data.csv",
        help="Cleaned VR reference CSV with time, center_x, center_y, pupil_size, confidence.",
    )
    parser.add_argument(
        "--pred-csv",
        default="prediction.csv",
        help="Model prediction CSV with time, pred_center_x, pred_center_y.",
    )
    parser.add_argument(
        "--output-dir",
        default="vr_pupil_center_evaluation",
        help="Directory for evaluation_results.csv and figures.",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=None,
        help="Optional maximum nearest-time distance in milliseconds. If omitted, always uses nearest match.",
    )
    parser.add_argument("--image-width", type=float, default=None, help="Optional image width for converting normalized VR x to pixels.")
    parser.add_argument("--image-height", type=float, default=None, help="Optional image height for converting normalized VR y to pixels.")
    parser.add_argument(
        "--time-mode",
        choices=["auto", "relative", "absolute"],
        default="auto",
        help="Use relative seconds, absolute timestamps, or auto-detect from prediction time.",
    )
    args = parser.parse_args()

    vr_df = read_vr_data(Path(args.vr_csv))
    vr_df = optionally_convert_vr_to_pixels(vr_df, args.image_width, args.image_height)
    pred_df = read_prediction_data(Path(args.pred_csv))

    aligned = align_by_nearest_time(vr_df, pred_df, args.tolerance_ms, args.time_mode)
    if aligned.empty:
        raise RuntimeError("No aligned samples. Check timestamps or increase --tolerance-ms.")

    evaluated = compute_error(aligned)
    stats = summarize_errors(evaluated)
    result_csv, summary_csv, hist_png, time_png, conf_png = save_outputs(evaluated, Path(args.output_dir))

    print("Pupil Center Localization Error Evaluation")
    print("=" * 48)
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")
    print(f"Samples: {len(evaluated)}")
    print("=" * 48)
    print(f"Saved CSV: {result_csv}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved histogram: {hist_png}")
    print(f"Saved time curve: {time_png}")
    print(f"Saved confidence plot: {conf_png}")


if __name__ == "__main__":
    main()
