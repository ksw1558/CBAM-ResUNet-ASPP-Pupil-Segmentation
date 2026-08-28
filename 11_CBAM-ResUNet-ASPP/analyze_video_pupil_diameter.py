import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "video_diameter_results_full"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "video_diameter_analysis"
DIAMETER_COLUMNS = ["frame_id", "time", "cx", "cy", "major_axis", "minor_axis", "pupil_diameter"]


def set_sci_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def read_diameter_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in DIAMETER_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df = df[DIAMETER_COLUMNS].copy()
    for col in DIAMETER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["video_id"] = csv_path.stem.replace("_diameter", "")
    return df


def collect_data(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.rglob("*_diameter.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_diameter.csv files found in {input_dir}")
    frames = [read_diameter_csv(path) for path in csv_files]
    return pd.concat(frames, ignore_index=True)


def filter_valid_diameter(df: pd.DataFrame, min_diameter=10, max_diameter=300) -> pd.DataFrame:
    df = df.dropna(subset=["pupil_diameter", "time"]).copy()
    return df[(df["pupil_diameter"] >= min_diameter) & (df["pupil_diameter"] <= max_diameter)].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    values = df["pupil_diameter"]
    return pd.DataFrame(
        [
            {"Metric": "Mean Diameter", "Value": f"{values.mean():.6f}"},
            {"Metric": "Std Diameter", "Value": f"{values.std(ddof=1):.6f}"},
            {"Metric": "Max Diameter", "Value": f"{values.max():.6f}"},
            {"Metric": "Min Diameter", "Value": f"{values.min():.6f}"},
        ]
    )


def summarize_by_video(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for video_id, group in df.groupby("video_id", sort=True):
        values = group["pupil_diameter"]
        rows.extend(
            [
                {"Video": video_id, "Metric": "Mean Diameter", "Value": f"{values.mean():.6f}"},
                {"Video": video_id, "Metric": "Std Diameter", "Value": f"{values.std(ddof=1):.6f}"},
                {"Video": video_id, "Metric": "Max Diameter", "Value": f"{values.max():.6f}"},
                {"Video": video_id, "Metric": "Min Diameter", "Value": f"{values.min():.6f}"},
            ]
        )
    return pd.DataFrame(rows)


def save_curve(df: pd.DataFrame, out_path: Path):
    set_sci_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    offset = 0.0
    for video_id, group in df.groupby("video_id", sort=True):
        group = group.sort_values("time")
        x = group["time"].to_numpy(dtype=float) + offset
        y = group["pupil_diameter"].to_numpy(dtype=float)
        ax.plot(x, y, color="#1f4e79", linewidth=1.5)
        if len(group) > 0:
            offset = x[-1] + 1.0

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil Diameter (px)")
    ax.set_title("Temporal Variation of Pupil Diameter")
    ax.grid(color="0.9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_histogram(df: pd.DataFrame, out_path: Path):
    set_sci_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    ax.hist(
        df["pupil_diameter"],
        bins=30,
        color="#c7c7c7",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xlabel("Pupil Diameter (px)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Pupil Diameter")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def add_continuous_time(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    offset = 0.0
    for video_id, group in df.groupby("video_id", sort=True):
        group = group.sort_values("time").copy()
        group["continuous_time"] = group["time"] + offset
        if len(group) > 0:
            offset = float(group["continuous_time"].iloc[-1]) + 1.0
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def save_smoothed_curve(df: pd.DataFrame, out_path: Path, window=5):
    set_sci_style()
    plot_df = add_continuous_time(df)
    plot_df["smooth_diameter"] = plot_df.groupby("video_id", group_keys=False)["pupil_diameter"].apply(
        lambda s: s.rolling(window=window, center=True, min_periods=1).mean()
    )
    mean_val = float(plot_df["smooth_diameter"].mean())
    std_val = float(plot_df["smooth_diameter"].std(ddof=1))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    ax.plot(
        plot_df["continuous_time"],
        plot_df["pupil_diameter"],
        color="gray",
        linewidth=1.0,
        alpha=0.15,
        label="Original",
    )
    ax.plot(
        plot_df["continuous_time"],
        plot_df["smooth_diameter"],
        color="#1f77b4",
        linewidth=2.5,
        label="Smoothed",
    )
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Pupil Diameter (px)", fontsize=14)
    ax.set_title("Temporal Variation of Pupil Diameter (Smoothed)", fontsize=18)
    ax.text(
        0.97,
        0.94,
        f"Mean = {mean_val:.2f} px\nStd = {std_val:.2f} px",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, boxstyle="round,pad=0.3"),
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper right", bbox_to_anchor=(0.98, 0.82))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_final_histogram(df: pd.DataFrame, out_path: Path):
    set_sci_style()
    values = df["pupil_diameter"].dropna()
    mean_val = float(values.mean())
    std_val = float(values.std(ddof=1))

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    ax.hist(values, bins=30, color="#4c72b0", edgecolor="black", linewidth=0.5, alpha=0.7, density=True)
    sns.kdeplot(values, color="red", linewidth=2, ax=ax, label="KDE")
    ax.axvline(mean_val, color="darkred", linewidth=2, linestyle="--", label="Mean")
    ax.text(
        0.03,
        0.94,
        f"Mean = {mean_val:.2f} px\nStd = {std_val:.2f} px",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.85, boxstyle="round,pad=0.3"),
    )
    ax.set_xlabel("Pupil Diameter (px)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Distribution of Pupil Diameter", fontsize=18)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate SCI-style pupil diameter statistics and figures.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing *_diameter.csv files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for tables and figures.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = collect_data(args.input_dir)
    filtered_df = filter_valid_diameter(raw_df)
    final_df = filter_valid_diameter(raw_df, min_diameter=100, max_diameter=200)

    filtered_df.to_csv(args.output_dir / "diameter_filtered_data.csv", index=False, encoding="utf-8-sig")
    summary = summarize(filtered_df)
    summary.to_csv(args.output_dir / "diameter_summary.csv", index=False, encoding="utf-8-sig")
    summarize_by_video(filtered_df).to_csv(args.output_dir / "diameter_summary_by_video.csv", index=False, encoding="utf-8-sig")

    save_curve(filtered_df, args.output_dir / "pupil_diameter_curve.png")
    save_histogram(filtered_df, args.output_dir / "pupil_diameter_histogram.png")
    final_df.to_csv(args.output_dir / "diameter_filtered_data_final.csv", index=False, encoding="utf-8-sig")
    save_smoothed_curve(final_df, args.output_dir / "pupil_diameter_curve_smooth.png", window=5)
    save_final_histogram(final_df, args.output_dir / "pupil_diameter_histogram_final.png")

    print(f"Input rows: {len(raw_df)}")
    print(f"Valid rows after filtering: {len(filtered_df)}")
    print(f"Saved: {args.output_dir / 'diameter_summary.csv'}")
    print(f"Saved: {args.output_dir / 'pupil_diameter_curve.png'}")
    print(f"Saved: {args.output_dir / 'pupil_diameter_histogram.png'}")
    print(f"Final valid rows after 100-200 px filtering: {len(final_df)}")
    print(f"Saved: {args.output_dir / 'pupil_diameter_curve_smooth.png'}")
    print(f"Saved: {args.output_dir / 'pupil_diameter_histogram_final.png'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
