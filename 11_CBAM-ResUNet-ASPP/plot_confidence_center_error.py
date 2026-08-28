import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from extract_pupil_params import extract_pupil_from_mask


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "pupil_params_results" / "lpw_pupil_center_coordinates.csv"
DEFAULT_GT_MASK_DIR = ROOT / "VOCdevkit" / "VOC2007" / "SegmentationClass"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "pupil_params_results" / "lpw_pupil_center_error_analysis.csv"
DEFAULT_OUTPUT_FIG = SCRIPT_DIR / "pupil_params_results" / "figure_confidence_center_error.png"
DEFAULT_OUTPUT_BIN_FIG = SCRIPT_DIR / "pupil_params_results" / "figure_confidence_binned_center_error.png"
DEFAULT_OUTPUT_BIN_CSV = SCRIPT_DIR / "pupil_params_results" / "lpw_confidence_binned_error_statistics.csv"


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path, rows, fieldnames):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_gt_center(filename, gt_mask_dir):
    mask_path = gt_mask_dir / f"{filename}.png"
    if not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    result = extract_pupil_from_mask(mask, do_clean=True)
    if result is None:
        return None
    return result["cx"], result["cy"]


def build_error_analysis(input_csv, gt_mask_dir, output_csv):
    rows = read_rows(input_csv)
    out_rows = []

    for row in rows:
        filename = row.get("filename") or row.get("image_id")
        confidence = to_float(row.get("confidence"))
        cx = to_float(row.get("cx") or row.get("center_x"))
        cy = to_float(row.get("cy") or row.get("center_y"))
        gt_cx = to_float(row.get("gt_cx"))
        gt_cy = to_float(row.get("gt_cy"))

        if row.get("status") != "ok" or confidence is None or cx is None or cy is None:
            continue

        if gt_cx is None or gt_cy is None:
            gt_center = load_gt_center(filename, gt_mask_dir)
            if gt_center is None:
                continue
            gt_cx, gt_cy = gt_center

        center_error = math.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2)
        out_row = dict(row)
        out_row["gt_cx"] = f"{gt_cx:.3f}"
        out_row["gt_cy"] = f"{gt_cy:.3f}"
        out_row["center_error_px"] = f"{center_error:.3f}"
        out_rows.append(out_row)

    fieldnames = list(rows[0].keys()) if rows else []
    for key in ["gt_cx", "gt_cy", "center_error_px"]:
        if key not in fieldnames:
            fieldnames.append(key)
    write_rows(output_csv, out_rows, fieldnames)
    return out_rows


def plot_confidence_vs_error(rows, output_fig):
    confidence = np.array([float(r["confidence"]) for r in rows], dtype=np.float64)
    error = np.array([float(r["center_error_px"]) for r in rows], dtype=np.float64)
    if len(confidence) < 2:
        raise ValueError("At least two valid samples are required for correlation and regression.")

    slope, intercept = np.polyfit(confidence, error, 1)
    x_line = np.linspace(confidence.min(), confidence.max(), 200)
    y_line = slope * x_line + intercept
    r = float(np.corrcoef(confidence, error)[0, 1])

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        confidence,
        error,
        s=26,
        color="#4C78A8",
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
        label="Samples",
    )
    ax.plot(x_line, y_line, color="#D62728", linewidth=2.0, label="Linear fit")

    ax.set_xlabel("Confidence", fontsize=13)
    ax.set_ylabel("Center Error (px)", fontsize=13)
    ax.set_title("Relationship between Confidence and Center Error", fontsize=14, pad=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    text = f"Pearson r = {r:.3f}"
    ax.text(
        0.04,
        0.94,
        text,
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d0d7de", alpha=0.92),
    )
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    fig.tight_layout()
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return r, slope, intercept


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x, y):
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def spearman_p_value(rho, n):
    if n < 3 or not np.isfinite(rho) or abs(rho) >= 1:
        return np.nan
    try:
        from scipy import stats
        t_value = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho * rho))
        return float(2 * stats.t.sf(abs(t_value), df=n - 2))
    except Exception:
        # Large-sample normal approximation fallback.
        z = rho * math.sqrt(max(n - 1, 1))
        return float(math.erfc(abs(z) / math.sqrt(2.0)))


def build_binned_statistics(rows, max_error=5.0):
    confidence = np.array([float(r["confidence"]) for r in rows], dtype=np.float64)
    error = np.array([float(r["center_error_px"]) for r in rows], dtype=np.float64)
    valid = error <= max_error
    confidence = confidence[valid]
    error = error[valid]

    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    stats = []
    for low, high in bins:
        if high == 1.0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        values = error[mask]
        q25 = float(np.percentile(values, 25)) if values.size else np.nan
        q75 = float(np.percentile(values, 75)) if values.size else np.nan
        median = float(np.median(values)) if values.size else np.nan
        stats.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "low": low,
            "high": high,
            "n": int(values.size),
            "mean_error": float(values.mean()) if values.size else np.nan,
            "std_error": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median_error": median,
            "q25_error": q25,
            "q75_error": q75,
        })
    rho = spearman_corr(confidence, error) if len(confidence) >= 2 else np.nan
    p_value = spearman_p_value(rho, len(confidence)) if len(confidence) >= 3 else np.nan
    return stats, rho, p_value, int(valid.sum()), int((~valid).sum())


def write_binned_statistics(stats, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin", "n", "median_error", "q25_error", "q75_error", "iqr", "mean_error", "std_error"])
        writer.writeheader()
        for row in stats:
            iqr = row["q75_error"] - row["q25_error"] if not np.isnan(row["median_error"]) else np.nan
            writer.writerow({
                "bin": row["bin"],
                "n": row["n"],
                "median_error": "" if np.isnan(row["median_error"]) else f"{row['median_error']:.4f}",
                "q25_error": "" if np.isnan(row["median_error"]) else f"{row['q25_error']:.4f}",
                "q75_error": "" if np.isnan(row["median_error"]) else f"{row['q75_error']:.4f}",
                "iqr": "" if np.isnan(row["median_error"]) else f"{iqr:.4f}",
                "mean_error": "" if np.isnan(row["mean_error"]) else f"{row['mean_error']:.4f}",
                "std_error": "" if np.isnan(row["mean_error"]) else f"{row['std_error']:.4f}",
            })


def format_p_value(p_value):
    if not np.isfinite(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def plot_binned_confidence_error(stats, spearman_r, spearman_p, output_fig):
    labels = [row["bin"] for row in stats]
    medians = np.array([row["median_error"] for row in stats], dtype=np.float64)
    q25 = np.array([row["q25_error"] for row in stats], dtype=np.float64)
    q75 = np.array([row["q75_error"] for row in stats], dtype=np.float64)
    yerr = np.vstack([medians - q25, q75 - medians])
    counts = [row["n"] for row in stats]
    x = np.arange(len(labels))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_color = "#1F4E79"
    line_color = "#8B1A1A"
    ax.bar(
        x,
        medians,
        yerr=yerr,
        width=0.58,
        color=bar_color,
        edgecolor="#173B5C",
        linewidth=0.8,
        capsize=4,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#263238"},
    )
    ax.plot(x, medians, color=line_color, marker="o", linewidth=1.8, markersize=4.5)

    for i, (median, n) in enumerate(zip(medians, counts)):
        if np.isnan(median):
            continue
        ax.text(i, q75[i] + 0.045, f"{median:.2f}\nn={n}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Confidence Bin", fontsize=13)
    ax.set_ylabel("Median Center Error (px)", fontsize=13)
    ax.set_title("Robust Relationship between Confidence and Center Error", fontsize=14, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.04,
        0.94,
        f"Spearman r = {spearman_r:.3f}\n{format_p_value(spearman_p)}",
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d0d7de", alpha=0.92),
    )
    fig.tight_layout()
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot confidence score versus pupil center error.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="CSV containing confidence, cx, cy, and optionally gt_cx/gt_cy.")
    parser.add_argument("--gt-mask-dir", default=str(DEFAULT_GT_MASK_DIR), help="Ground-truth LPW mask directory used when gt_cx/gt_cy are missing.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="CSV with gt center and center error columns.")
    parser.add_argument("--output-fig", default=str(DEFAULT_OUTPUT_FIG), help="Output high-resolution PNG path.")
    parser.add_argument("--output-bin-fig", default=str(DEFAULT_OUTPUT_BIN_FIG), help="Output binned confidence-error PNG path.")
    parser.add_argument("--output-bin-csv", default=str(DEFAULT_OUTPUT_BIN_CSV), help="Output binned statistics CSV path.")
    parser.add_argument("--max-error", type=float, default=5.0, help="Filter out samples with center error larger than this value.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists() and Path(args.output_csv).exists():
        input_csv = Path(args.output_csv)
        print(f"Input CSV not found. Fallback to existing analysis CSV: {input_csv}")
    rows = build_error_analysis(
        input_csv=input_csv,
        gt_mask_dir=Path(args.gt_mask_dir),
        output_csv=Path(args.output_csv),
    )
    r, slope, intercept = plot_confidence_vs_error(rows, Path(args.output_fig))
    stats, spearman_r, spearman_p, kept_count, removed_count = build_binned_statistics(rows, max_error=args.max_error)
    write_binned_statistics(stats, Path(args.output_bin_csv))
    plot_binned_confidence_error(stats, spearman_r, spearman_p, Path(args.output_bin_fig))
    print(f"Valid samples: {len(rows)}")
    print(f"Pearson r: {r:.4f}")
    print(f"Regression: center_error = {slope:.4f} * confidence + {intercept:.4f}")
    print(f"Filtered samples for binned analysis: kept={kept_count}, removed={removed_count}, max_error={args.max_error}")
    print(f"Spearman r after filtering: {spearman_r:.4f}, p-value: {spearman_p:.6f}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved figure: {args.output_fig}")
    print(f"Saved binned CSV: {args.output_bin_csv}")
    print(f"Saved binned figure: {args.output_bin_fig}")


if __name__ == "__main__":
    main()
