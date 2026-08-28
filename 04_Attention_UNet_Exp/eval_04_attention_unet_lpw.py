import os
import sys
import warnings
import csv

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from nets.attention_unet import AttentionGateUnet  # noqa: E402
from utils.utils_metrics import compute_mIoU  # noqa: E402


def get_centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def compute_binary_metrics(label_dir, pred_dir, image_ids):
    tp = fp = fn = tn = 0
    center_errors = []
    valid_count = 0

    for image_id in image_ids:
        label_path = os.path.join(label_dir, image_id + ".png")
        pred_path = os.path.join(pred_dir, image_id + ".png")
        if not os.path.exists(label_path) or not os.path.exists(pred_path):
            continue

        label = np.asarray(Image.open(label_path).convert("L")) > 0
        pred = np.asarray(Image.open(pred_path).convert("L")) > 0
        if label.shape != pred.shape:
            continue

        tp += int(np.logical_and(label, pred).sum())
        fp += int(np.logical_and(~label, pred).sum())
        fn += int(np.logical_and(label, ~pred).sum())
        tn += int(np.logical_and(~label, ~pred).sum())

        label_center = get_centroid(label)
        pred_center = get_centroid(pred)
        if label_center is not None and pred_center is not None:
            center_errors.append(float(np.linalg.norm(label_center - pred_center)))
        valid_count += 1

    pupil_iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    background_iou = tn / (tn + fp + fn) if tn + fp + fn else 0.0
    return {
        "N masks": valid_count,
        "mIoU": (pupil_iou + background_iou) * 50.0,
        "Pupil IoU": pupil_iou * 100.0,
        "Dice": (2 * tp / (2 * tp + fp + fn) * 100.0) if 2 * tp + fp + fn else 0.0,
        "Recall": (tp / (tp + fn) * 100.0) if tp + fn else 0.0,
        "Precision": (tp / (tp + fp) * 100.0) if tp + fp else 0.0,
        "Center Error(px)": float(np.mean(center_errors)) if center_errors else np.nan,
    }


def save_metrics_csv(metrics, output_path):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow([
            f"{value:.4f}" if isinstance(value, float) else value
            for value in metrics.values()
        ])


def preprocess_image(image_path, input_size=256):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0), original_size


def evaluate():
    print("=" * 60)
    print("Evaluate Experiment 04: Attention U-Net on LPW (VOC format)")
    print("=" * 60)

    exp_dir = os.path.join(ROOT_DIR, "04_Attention_UNet_Exp")
    model_path = os.path.join(exp_dir, "logs", "best_epoch_weights.pth")
    output_dir = os.path.join(exp_dir, "miou_out", "detection-results")
    os.makedirs(output_dir, exist_ok=True)

    voc_root = os.path.join(ROOT_DIR, "VOCdevkit", "VOC2007")
    val_list = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    image_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weight not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionGateUnet(num_classes=2, pretrained=False, backbone="vgg")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    with open(val_list, "r", encoding="utf-8") as f:
        image_ids = [line.strip().split()[0] for line in f if line.strip()]

    for image_id in tqdm(image_ids, desc="Predict"):
        image_path = os.path.join(image_dir, image_id + ".jpg")
        label_path = os.path.join(label_dir, image_id + ".png")
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue

        x, _ = preprocess_image(image_path, input_size=256)
        x = x.to(device)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0].astype(np.uint8)

        target_size = Image.open(label_path).size
        pred_img = Image.fromarray(pred).resize(target_size, Image.NEAREST)
        pred_img.save(os.path.join(output_dir, image_id + ".png"))

    _, ious, recalls, precisions = compute_mIoU(
        label_dir, output_dir, image_ids, num_classes=2, name_classes=["background", "pupil"]
    )
    metrics = compute_binary_metrics(label_dir, output_dir, image_ids)
    save_metrics_csv(metrics, os.path.join(exp_dir, "metrics.csv"))

    print("=" * 60)
    print(f"mIoU: {np.nanmean(ious) * 100:.2f}%")
    print(f"Pupil IoU: {ious[1] * 100:.2f}%")
    print(f"Dice: {metrics['Dice']:.2f}%")
    print(f"Pupil Recall: {recalls[1] * 100:.2f}%")
    print(f"Pupil Precision: {precisions[1] * 100:.2f}%")
    print(f"Center Error: {metrics['Center Error(px)']:.2f}px")
    print(f"Saved metrics: {os.path.join(exp_dir, 'metrics.csv')}")


if __name__ == "__main__":
    evaluate()
