import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "video_diameter_analysis"


def collect_raw_csvs(analysis_dir: Path) -> List[Path]:
    return sorted(analysis_dir.glob("*/*_pupil_params_raw.csv"))


def to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def append_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    reasons = reasons.astype(str)
    reasons.loc[mask] = reasons.loc[mask].apply(lambda old: reason if old == "" else f"{old};{reason}")
    return reasons


def robust_filter_one_video(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Objective QC for the final paper curve: geometry, confidence, IQR, and local Hampel filtering."""
    work = to_numeric(
        df,
        [
            "frame_id",
            "time_s",
            "cx",
            "cy",
            "major_axis",
            "minor_axis",
            "angle",
            "pupil_diameter",
            "confidence",
            "fit_quality",
            "ellipse_residual",
        ],
    ).sort_values("frame_id")

    keep = pd.Series(True, index=work.index)
    reasons = pd.Series("", index=work.index, dtype=object)

    invalid = (
        work[["major_axis", "minor_axis", "pupil_diameter", "time_s"]].isna().any(axis=1)
        | (work["major_axis"] <= 0)
        | (work["minor_axis"] <= 0)
        | (work["pupil_diameter"] <= 0)
    )
    keep &= ~invalid
    reasons = append_reason(reasons, invalid, "invalid_geometry")

    if "confidence" in work.columns:
        low_conf = work["confidence"].notna() & (work["confidence"] < 0.60)
        keep &= ~low_conf
        reasons = append_reason(reasons, low_conf, "confidence_lt_0.60")

    residual = work.loc[keep, "ellipse_residual"].dropna()
    if len(residual) >= 10:
        q1, q3 = residual.quantile([0.25, 0.75])
        high = q3 + 1.5 * (q3 - q1)
        bad_residual = work["ellipse_residual"].notna() & (work["ellipse_residual"] > high)
        keep &= ~bad_residual
        reasons = append_reason(reasons, bad_residual, "ellipse_residual_iqr_high")

    fit_quality = work.loc[keep, "fit_quality"].dropna()
    if len(fit_quality) >= 10:
        q1, q3 = fit_quality.quantile([0.25, 0.75])
        low = max(0.0, q1 - 1.5 * (q3 - q1))
        bad_fit = work["fit_quality"].notna() & (work["fit_quality"] < low)
        keep &= ~bad_fit
        reasons = append_reason(reasons, bad_fit, "fit_quality_iqr_low")

    diameter = work.loc[keep, "pupil_diameter"].dropna()
    if len(diameter) >= 10:
        q1, q3 = diameter.quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        bad_diameter = work["pupil_diameter"].notna() & (
            (work["pupil_diameter"] < low) | (work["pupil_diameter"] > high)
        )
        keep &= ~bad_diameter
        reasons = append_reason(reasons, bad_diameter, "diameter_iqr_outlier")

    # Hampel filter on the remaining sequence removes isolated spikes while preserving step-like real trends.
    stable = work.loc[keep].sort_values("frame_id")
    if len(stable) >= 15:
        y = stable["pupil_diameter"]
        rolling_median = y.rolling(window=15, center=True, min_periods=5).median()
        abs_dev = (y - rolling_median).abs()
        rolling_mad = abs_dev.rolling(window=15, center=True, min_periods=5).median()
        local_threshold = np.maximum(8.0, 4.0 * 1.4826 * rolling_mad.fillna(0.0))
        hampel_bad = abs_dev > local_threshold
        if hampel_bad.any():
            bad_indices = stable.index[hampel_bad.to_numpy()]
            keep.loc[bad_indices] = False
            reasons = append_reason(reasons, keep.index.isin(bad_indices), "local_hampel_spike")

    work["qc_keep_paper"] = keep
    work["qc_reason_paper"] = reasons.replace("", np.nan)
    return work.loc[work["qc_keep_paper"]].copy(), work.loc[~work["qc_keep_paper"]].copy()


def build_continuous_time(filtered_frames: List[pd.DataFrame], gap_s: float) -> pd.DataFrame:
    pieces = []
    offset = 0.0
    for df in filtered_frames:
        if df.empty:
            continue
        piece = df.sort_values("time_s").copy()
        start = float(piece["time_s"].min())
        piece["continuous_time_s"] = piece["time_s"] - start + offset
        duration = float(piece["continuous_time_s"].max() - offset)
        offset += duration + gap_s
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def save_reference_style_curve(plot_df: pd.DataFrame, output_path: Path, smooth_window: int = 3) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    for _, group in plot_df.groupby("source_video", sort=True):
        group = group.sort_values("continuous_time_s").copy()
        y = group["pupil_diameter"].rolling(window=smooth_window, center=True, min_periods=1).mean()
        ax.plot(group["continuous_time_s"], y, color="#1f4e79", linewidth=2.0)

    ax.set_title("Temporal Variation of Pupil Diameter", fontsize=18, pad=12)
    ax.set_xlabel("Time (s)", fontsize=15)
    ax.set_ylabel("Pupil Diameter (px)", fontsize=15)
    ax.grid(True, color="0.90", linewidth=0.9)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    ymin = float(plot_df["pupil_diameter"].quantile(0.01))
    ymax = float(plot_df["pupil_diameter"].quantile(0.99))
    pad = max(5.0, (ymax - ymin) * 0.12)
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_final_chinese_curve(plot_df: pd.DataFrame, output_path: Path, smooth_window: int = 21) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    groups = [(name, group.sort_values("continuous_time_s").copy()) for name, group in plot_df.groupby("source_video", sort=True)]
    y_min = float(plot_df["pupil_diameter"].quantile(0.01))
    y_max = float(plot_df["pupil_diameter"].quantile(0.99))
    y_pad = max(5.0, (y_max - y_min) * 0.14)
    upper_label_y = y_max + y_pad * 0.42

    raw_label_used = False
    smooth_label_used = False
    previous_end = None
    for idx, (name, group) in enumerate(groups, start=1):
        x = group["continuous_time_s"]
        y = group["pupil_diameter"]
        smooth_y = y.rolling(window=smooth_window, center=True, min_periods=1).mean()

        if previous_end is not None:
            boundary = (previous_end + float(x.min())) / 2.0
            ax.axvline(boundary, color="0.78", linestyle="--", linewidth=1.0, zorder=0)
        previous_end = float(x.max())

        ax.plot(
            x,
            y,
            color="0.72",
            linewidth=0.75,
            alpha=0.80,
            label="清洗后直径" if not raw_label_used else None,
            zorder=1,
        )
        ax.plot(
            x,
            smooth_y,
            color="black",
            linewidth=2.0,
            label="平滑趋势" if not smooth_label_used else None,
            zorder=2,
        )
        raw_label_used = True
        smooth_label_used = True

        label_x = (float(x.min()) + float(x.max())) / 2.0
        ax.text(label_x, upper_label_y, f"Video {idx}", ha="center", va="bottom", fontsize=11, color="0.25")

    ax.set_xlabel("时间 / s", fontsize=15)
    ax.set_ylabel("瞳孔直径 / px", fontsize=15)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, color="0.90", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.legend(frameon=False, fontsize=12, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paper-style combined temporal pupil diameter curve.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--gap-s", type=float, default=2.0, help="Gap inserted between videos on the x-axis.")
    parser.add_argument("--smooth-window", type=int, default=3, help="Small rolling mean window for visual stability.")
    args = parser.parse_args()

    csv_paths = collect_raw_csvs(args.analysis_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No *_pupil_params_raw.csv files found in {args.analysis_dir}")

    filtered_frames = []
    rejected_frames = []
    summary_rows = []
    for csv_path in csv_paths:
        raw = pd.read_csv(csv_path)
        filtered, rejected = robust_filter_one_video(raw)
        filtered_frames.append(filtered)
        rejected_frames.append(rejected)
        source = str(raw["source_video"].dropna().iloc[0]) if "source_video" in raw.columns and raw["source_video"].notna().any() else csv_path.parent.name
        values = filtered["pupil_diameter"].dropna()
        summary_rows.append(
            {
                "video_id": csv_path.parent.name,
                "source_video": source,
                "raw_frames": len(raw),
                "kept_frames": len(filtered),
                "removed_frames": len(raw) - len(filtered),
                "removed_ratio": (len(raw) - len(filtered)) / len(raw) if len(raw) else 0.0,
                "mean_diameter": float(values.mean()) if len(values) else np.nan,
                "std_diameter": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "min_diameter": float(values.min()) if len(values) else np.nan,
                "max_diameter": float(values.max()) if len(values) else np.nan,
            }
        )

    combined = build_continuous_time(filtered_frames, gap_s=args.gap_s)
    rejected_all = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    combined_csv = args.analysis_dir / "all_videos_paper_curve_filtered.csv"
    rejected_csv = args.analysis_dir / "all_videos_paper_curve_rejected.csv"
    summary_csv = args.analysis_dir / "all_videos_paper_curve_summary.csv"
    figure_path = args.analysis_dir / "all_videos_temporal_variation_paper.png"
    final_figure_path = args.analysis_dir / "pupil_diameter_time_curve_final.png"

    combined.to_csv(combined_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    rejected_all.to_csv(rejected_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    save_reference_style_curve(combined, figure_path, smooth_window=args.smooth_window)
    save_final_chinese_curve(combined, final_figure_path)

    print(f"Saved figure: {figure_path}")
    print(f"Saved final Chinese figure: {final_figure_path}")
    print(f"Saved filtered data: {combined_csv}")
    print(f"Saved rejected data: {rejected_csv}")
    print(f"Saved summary: {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
