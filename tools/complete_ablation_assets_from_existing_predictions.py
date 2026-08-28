import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "teacher_report_assets" / "ablation_plan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_DIR = ROOT / "VOCdevkit" / "VOC2007" / "SegmentationClass"


COMPLETED_VARIANTS = [
    {
        "id": "A1",
        "model": "Base U-Net",
        "role": "Baseline",
        "backbone": "VGG16",
        "residual": "No",
        "attention": "No",
        "scale": "No",
        "loss": "CE+Dice",
        "pred_dir": ROOT / "01_Base_Unet_Exp" / "miou_out" / "detection-results",
    },
    {
        "id": "A2",
        "model": "CA-UNet",
        "role": "Coord attention",
        "backbone": "VGG16",
        "residual": "No",
        "attention": "CA",
        "scale": "No",
        "loss": "CE+Dice",
        "pred_dir": ROOT / "02_CBAM_UNet_Exp" / "miou_out" / "detection-results",
        "override": {"mIoU": 95.13, "Dice": 95.08, "Recall": 99.71},
        "derived_reliable": False,
        "note": "mIoU/Dice/Recall use existing project record; preserved full masks are inconsistent for derived precision/center metrics.",
    },
    {
        "id": "A3",
        "model": "CBAM-UNet",
        "role": "CBAM attention",
        "backbone": "VGG16",
        "residual": "No",
        "attention": "CBAM",
        "scale": "No",
        "loss": "CE+Dice",
        "pred_dir": ROOT / "03_CBAM_UNet_Focal_Exp" / "miou_out" / "detection-results",
    },
    {
        "id": "A4",
        "model": "ResUNet",
        "role": "Residual baseline",
        "backbone": "ResNet50",
        "residual": "Yes",
        "attention": "No",
        "scale": "No",
        "loss": "CE+Dice",
        "pred_dir": ROOT / "08_ResUNet_Exp" / "miou_out" / "detection-results",
    },
    {
        "id": "A5",
        "model": "CBAM-ResUNet",
        "role": "Residual + CBAM",
        "backbone": "ResNet50",
        "residual": "Yes",
        "attention": "CBAM",
        "scale": "No",
        "loss": "CE+Dice",
        "pred_dir": ROOT / "09_CBAM_ResUNet_Exp" / "miou_out" / "detection-results",
        "override": {"mIoU": 97.32, "Dice": 97.32, "Recall": 97.42},
    },
    {
        "id": "A6",
        "model": "CA-ResUNet",
        "role": "Residual + CA",
        "backbone": "ResNet50",
        "residual": "Yes",
        "attention": "CA",
        "scale": "No",
        "loss": "CE+Dice",
        "pred_dir": ROOT / "10_CA-ResUNet V4" / "miou_out" / "detection-results",
        "override": {"mIoU": 97.56, "Dice": 97.56, "Recall": 98.70},
        "derived_reliable": False,
        "note": "Only 2 LPW masks are preserved locally; reported LPW metrics use existing record.",
    },
    {
        "id": "A7",
        "model": "CBAM-ResUNet-ASPP",
        "role": "Full model",
        "backbone": "ResNet50",
        "residual": "Yes",
        "attention": "CBAM",
        "scale": "ASPP",
        "loss": "Tversky+OHEM",
        "pred_dir": ROOT / "11_CBAM-ResUNet-ASPP" / "miou_out" / "detection-results-optimized-final",
        "override": {"mIoU": 98.16, "Dice": 98.11, "Recall": 98.61},
    },
]


PENDING_EXPERIMENTS = [
    ["P1", "ResNet50-UNet", "Backbone only without residual decoder", "Need training/evaluation"],
    ["P2", "ResUNet+SE", "Replace CBAM by SE attention", "Need implementation/training"],
    ["P3", "CBAM-ResUNet+PPM", "Replace ASPP by PPM/PSP module", "Need implementation/training"],
    ["P4", "Full model w/o Edge Loss", "Remove edge-weighted loss", "Need training/evaluation"],
    ["P5", "Full model w/o Centroid Loss", "Remove centroid constraint", "Need training/evaluation"],
    ["P6", "Full model w/o OHEM/Tversky", "Use basic CE+Dice under same architecture", "Need confirmed checkpoint"],
]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = font(30, True)
FONT_SUBTITLE = font(18)
FONT_HEADER = font(16, True)
FONT_CELL = font(14)

COLORS = {
    "dark": (28, 32, 40),
    "muted": (88, 96, 110),
    "grid": (204, 212, 224),
    "header": (235, 240, 247),
    "known": (232, 246, 241),
    "final": (255, 242, 225),
    "pending": (249, 249, 249),
}


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0


def centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def compute_metrics(pred_dir):
    pred_paths = sorted(pred_dir.glob("*.png"))
    tp = fp = fn = tn = 0
    center_errors = []
    valid = 0
    missing = 0

    for pred_path in pred_paths:
        gt_path = GT_DIR / pred_path.name
        if not gt_path.exists():
            missing += 1
            continue
        gt = load_mask(gt_path)
        pred = load_mask(pred_path)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST)) > 0

        tp += int(np.logical_and(pred, gt).sum())
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(~pred, gt).sum())
        tn += int(np.logical_and(~pred, ~gt).sum())

        cg = centroid(gt)
        cp = centroid(pred)
        if cg is not None and cp is not None:
            center_errors.append(float(np.linalg.norm(cp - cg)))
        valid += 1

    pupil_iou = tp / (tp + fp + fn) if tp + fp + fn else 0
    bg_iou = tn / (tn + fp + fn) if tn + fp + fn else 0
    miou = (pupil_iou + bg_iou) / 2
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    center_error = float(np.mean(center_errors)) if center_errors else None

    return {
        "n": valid,
        "missing": missing,
        "mIoU": miou * 100,
        "Dice": dice * 100,
        "Recall": recall * 100,
        "Precision": precision * 100,
        "CenterError": center_error,
    }


def completed_rows():
    rows = []
    for item in COMPLETED_VARIANTS:
        metrics = compute_metrics(item["pred_dir"])
        override = item.get("override", {})
        row = {
            **item,
            **metrics,
            "mIoU": override.get("mIoU", metrics["mIoU"]),
            "Dice": override.get("Dice", metrics["Dice"]),
            "Recall": override.get("Recall", metrics["Recall"]),
            "note": item.get("note", "Computed from preserved LPW masks" if not override else "Metrics use existing project record; center/precision computed when masks are available"),
        }
        if item.get("derived_reliable") is False:
            row["Precision"] = None
            row["CenterError"] = None
        rows.append(row)
    return rows


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def save_csvs(rows):
    completed_csv = OUT_DIR / "ablation_completed_results.csv"
    with open(completed_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Model", "Role", "Backbone", "Residual", "Attention", "Scale", "Loss", "N masks", "mIoU", "Dice", "Recall", "Precision", "Center Error(px)", "Note"])
        for r in rows:
            writer.writerow([r["id"], r["model"], r["role"], r["backbone"], r["residual"], r["attention"], r["scale"], r["loss"], r["n"], fmt(r["mIoU"]), fmt(r["Dice"]), fmt(r["Recall"]), fmt(r["Precision"]), fmt(r["CenterError"]), r["note"]])

    pending_csv = OUT_DIR / "ablation_pending_experiments.csv"
    with open(pending_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Experiment", "Purpose", "Status"])
        writer.writerows(PENDING_EXPERIMENTS)
    return completed_csv, pending_csv


def draw_wrapped(draw, box, text, fnt, fill=COLORS["dark"], line_h=17, align="center"):
    x0, y0, x1, y1 = box
    width = x1 - x0 - 8
    words = str(text).replace("/", "/ ").split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=fnt)
        if bbox[2] - bbox[0] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        lines = [""]
    y = y0 + max(2, (y1 - y0 - len(lines) * line_h) / 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        x = x0 + 7 if align == "left" else x0 + (x1 - x0 - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h


def table_image(filename, title, subtitle, header, table_rows, widths, row_h=52):
    margin_x, margin_top, margin_bottom = 34, 28, 32
    title_h = 76
    w = margin_x * 2 + sum(widths)
    h = margin_top + title_h + row_h * (len(table_rows) + 1) + margin_bottom
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin_x, margin_top), title, font=FONT_TITLE, fill=COLORS["dark"])
    draw.text((margin_x, margin_top + 40), subtitle, font=FONT_SUBTITLE, fill=COLORS["muted"])
    x0, y0 = margin_x, margin_top + title_h
    x = x0
    for i, head in enumerate(header):
        draw.rectangle((x, y0, x + widths[i], y0 + row_h), fill=COLORS["header"], outline=COLORS["grid"])
        draw_wrapped(draw, (x, y0, x + widths[i], y0 + row_h), head, FONT_HEADER)
        x += widths[i]

    for r_i, row in enumerate(table_rows):
        y = y0 + row_h * (r_i + 1)
        fill = COLORS["final"] if "Full" in str(row[1]) else COLORS["known"]
        x = x0
        for c_i, val in enumerate(row):
            draw.rectangle((x, y, x + widths[c_i], y + row_h), fill=fill, outline=COLORS["grid"])
            draw_wrapped(draw, (x, y, x + widths[c_i], y + row_h), val, FONT_CELL, align="left" if c_i in (1, 2, len(row) - 1) else "center")
            x += widths[c_i]
    out = OUT_DIR / filename
    img.save(out, dpi=(300, 300))
    return out


def make_images(rows):
    completed_table = [
        [r["id"], r["model"], r["role"], r["attention"], r["scale"], fmt(r["mIoU"]), fmt(r["Dice"]), fmt(r["Recall"]), fmt(r["Precision"]), fmt(r["CenterError"]), str(r["n"])]
        for r in rows
    ]
    completed_img = table_image(
        "table_ablation_completed_results.png",
        "Completed LPW Ablation Results",
        "Only rows with preserved prediction masks or existing project records are included.",
        ["ID", "Model", "Role", "Att", "Scale", "mIoU", "Dice", "Recall", "Prec.", "Center Err.", "N"],
        completed_table,
        [58, 190, 170, 76, 76, 74, 74, 74, 74, 100, 58],
        row_h=54,
    )

    pending_img = table_image(
        "table_ablation_pending_experiments.png",
        "Pending Ablation Experiments",
        "These rows cannot be completed from current files; they require new training/evaluation.",
        ["ID", "Experiment", "Purpose", "Status"],
        PENDING_EXPERIMENTS,
        [62, 220, 420, 210],
        row_h=58,
    )
    return completed_img, pending_img


def update_teacher_md(rows):
    path = OUT_DIR / "ablation_completed_summary_for_teacher.md"
    best = max(rows, key=lambda r: r["mIoU"])
    content = f"""# LPW 消融实验已完成数据说明

本次更新只使用当前项目中已经保存的 LPW 预测 mask、已有项目记录和最终模型记录，不编造未训练模型的数据。

## 已完成结果

- 已生成 `ablation_completed_results.csv`
- 已生成 `table_ablation_completed_results.png`
- 最优模型：{best['model']}
- LPW mIoU：{best['mIoU']:.2f}%
- Dice：{best['Dice']:.2f}%
- Recall：{best['Recall']:.2f}%

## 仍需训练/评估的实验

`ablation_pending_experiments.csv` 和 `table_ablation_pending_experiments.png` 中列出的实验当前没有对应权重或预测 mask，因此不能直接补数。

建议后续如果老师要求完整“去模块/替换模块”实验，优先补：

1. Full model w/o CBAM
2. Full model w/o ASPP
3. Full model w/o Centroid Loss
4. ResUNet + SE
5. CBAM-ResUNet + PPM

## 论文表述建议

当前可先使用“已完成消融结果表”证明从 U-Net、CBAM-UNet、ResUNet 到最终 CBAM-ResUNet-ASPP 的性能提升；待补实验作为计划项或补充实验。
"""
    path.write_text(content, encoding="utf-8")
    return path


def main():
    rows = completed_rows()
    completed_csv, pending_csv = save_csvs(rows)
    completed_img, pending_img = make_images(rows)
    summary = update_teacher_md(rows)
    print(f"Saved: {completed_csv}")
    print(f"Saved: {pending_csv}")
    print(f"Saved: {completed_img}")
    print(f"Saved: {pending_img}")
    print(f"Saved: {summary}")


if __name__ == "__main__":
    main()



