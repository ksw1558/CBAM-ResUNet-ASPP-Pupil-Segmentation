import argparse
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")


def find_project_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(current, "VOCdevkit")) and os.path.exists(os.path.join(current, "nets")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(start_path)
        current = parent


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = find_project_root(CURRENT_DIR)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from nets.cbam_res_unet_exp11 import CBAMResUnetExp11  # noqa: E402
from utils.utils_metrics import compute_mIoU  # noqa: E402


def load_model(weight_path, device):
    model = CBAMResUnetExp11(num_classes=2, pretrained=False)
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        checkpoint = checkpoint["net"]
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate(input_size=320):
    folder_name = "11_CBAM-ResUNet-ASPP"
    weight_path = os.path.join(ROOT_PATH, folder_name, "logs", "final_exp11_miou98_16_epoch070.pth")
    output_dir = os.path.join(ROOT_PATH, folder_name, "miou_out", "detection-results")

    voc_root = os.path.join(ROOT_PATH, "VOCdevkit", "VOC2007")
    val_list_path = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    image_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Final checkpoint not found: {weight_path}")
    if not os.path.exists(val_list_path):
        raise FileNotFoundError(f"Validation split not found: {val_list_path}")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weight_path, device)

    with open(val_list_path, "r", encoding="utf-8") as f:
        val_ids = [line.strip() for line in f if line.strip()]

    success_count = 0
    for image_id in tqdm(val_ids, desc="Evaluating ACR-UNet"):
        image_path = os.path.join(image_dir, image_id + ".jpg")
        label_path = os.path.join(label_dir, image_id + ".png")
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")
        target_size = label.size

        image_resized = image.resize((input_size, input_size), Image.BILINEAR)
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_np = np.transpose(image_np, (2, 0, 1))
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0].astype(np.uint8)

        pred_img = Image.fromarray(pred, mode="L").resize(target_size, Image.NEAREST)
        pred_img.save(os.path.join(output_dir, image_id + ".png"))
        success_count += 1

    hist, ious, recall, precision = compute_mIoU(
        label_dir,
        output_dir,
        val_ids,
        num_classes=2,
        name_classes=["background", "pupil"],
    )

    mean_iou = float(np.nanmean(ious) * 100)
    pupil_iou = float(ious[1] * 100)
    pupil_recall = float(recall[1] * 100)
    pupil_precision = float(precision[1] * 100)
    dice = 2 * pupil_precision * pupil_recall / (pupil_precision + pupil_recall + 1e-8)
    accuracy = float(np.diag(hist).sum() / (hist.sum() + 1e-8) * 100)

    print("\nACR-UNet final evaluation")
    print(f"Checkpoint : {weight_path}")
    print(f"Val images : {success_count}/{len(val_ids)}")
    print(f"mIoU       : {mean_iou:.2f}%")
    print(f"Pupil IoU  : {pupil_iou:.2f}%")
    print(f"Dice       : {dice:.2f}%")
    print(f"Recall     : {pupil_recall:.2f}%")
    print(f"Precision  : {pupil_precision:.2f}%")
    print(f"Accuracy   : {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate final ACR-UNet checkpoint on VOC validation split.")
    parser.add_argument("--input-size", type=int, default=320, help="square input size used for evaluation")
    args = parser.parse_args()
    evaluate(input_size=args.input_size)
