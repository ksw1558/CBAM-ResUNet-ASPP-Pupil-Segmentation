import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

VOC_ROOT = os.path.join(ROOT, "VOCdevkit", "VOC2007")
IMAGE_DIR = os.path.join(VOC_ROOT, "JPEGImages")
LABEL_DIR = os.path.join(VOC_ROOT, "SegmentationClass")
VAL_LIST = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "val.txt")

PRED_DIR = os.path.join(CURRENT_DIR, "miou_out", "detection-results")
OUT_DIR = os.path.join(CURRENT_DIR, "visualize_results")


def binary_mask(image):
    return (np.array(image.convert("L")) > 0).astype(np.uint8)


def mask_to_rgb(image):
    return Image.fromarray(binary_mask(image) * 255, "L").convert("RGB")


def overlay_image(image, label, pred):
    label_arr = binary_mask(label)
    pred_arr = binary_mask(pred)
    overlay = np.zeros((label_arr.shape[0], label_arr.shape[1], 4), dtype=np.uint8)
    overlay[(label_arr == 1) & (pred_arr == 1)] = [0, 255, 0, 120]
    overlay[(label_arr == 1) & (pred_arr == 0)] = [255, 0, 0, 140]
    overlay[(label_arr == 0) & (pred_arr == 1)] = [0, 0, 255, 120]
    return Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay, "RGBA")).convert("RGB")


def save_comparison(image_id, image, label, pred):
    thumb = (420, 300)
    margin = 18
    gap = 16
    title_h = 48
    font = ImageFont.load_default()

    panels = [
        ("Original Image", image.convert("RGB")),
        ("Ground Truth", mask_to_rgb(label)),
        ("Prediction (CBAM-ResUNet)", mask_to_rgb(pred)),
        ("Overlay (Green:TP, Red:FN, Blue:FP)", overlay_image(image, label, pred)),
    ]

    canvas_w = margin * 2 + thumb[0] * len(panels) + gap * (len(panels) - 1)
    canvas_h = margin * 2 + title_h + thumb[1]
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (title, panel) in enumerate(panels):
        x = margin + idx * (thumb[0] + gap)
        y = margin
        draw.text((x, y), title, font=font, fill=(20, 24, 32))
        resample = Image.BILINEAR if idx in (0, 3) else Image.NEAREST
        canvas.paste(panel.resize(thumb, resample), (x, y + title_h))

    os.makedirs(OUT_DIR, exist_ok=True)
    canvas.save(os.path.join(OUT_DIR, f"{image_id}.png"))


def main(num_samples=10):
    if not os.path.exists(PRED_DIR):
        raise FileNotFoundError(f"Prediction directory not found: {PRED_DIR}")

    with open(VAL_LIST, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    generated = 0
    for image_id in image_ids[:num_samples]:
        image_path = os.path.join(IMAGE_DIR, image_id + ".jpg")
        label_path = os.path.join(LABEL_DIR, image_id + ".png")
        pred_path = os.path.join(PRED_DIR, image_id + ".png")
        if not (os.path.exists(image_path) and os.path.exists(label_path) and os.path.exists(pred_path)):
            continue

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")
        pred = Image.open(pred_path).convert("L")
        if pred.size != label.size:
            pred = pred.resize(label.size, Image.NEAREST)
        if image.size != label.size:
            image = image.resize(label.size, Image.BILINEAR)

        save_comparison(image_id, image, label, pred)
        generated += 1

    print(f"Generated {generated} images in: {OUT_DIR}")


if __name__ == "__main__":
    main()
