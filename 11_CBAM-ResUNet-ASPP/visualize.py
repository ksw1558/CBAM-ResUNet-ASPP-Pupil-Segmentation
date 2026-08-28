import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Locate the project root so imports work from either the project root or this folder.
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)

while True:
    if os.path.exists(os.path.join(current_dir, "VOCdevkit")):
        root_path = current_dir
        break
    if os.path.exists(os.path.join(current_dir, "nets")):
        root_path = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        root_path = current_dir
        break
    current_dir = parent

sys.path.append(root_path)

try:
    from nets.cbam_res_unet_exp11 import CBAMResUnetExp11
except ImportError:
    print("Cannot import nets.cbam_res_unet_exp11.CBAMResUnetExp11")
    sys.exit(1)


def visualize_predictions(num_samples=10, input_size=320):
    print("=" * 60)
    print("CBAM-ResUNet-ASPP (Experiment 11) visualization")
    print("=" * 60)

    folder_name = "11_CBAM-ResUNet-ASPP"

    # Prefer the final experiment-11 checkpoint; fall back to any available .pth.
    logs_dir = os.path.join(root_path, folder_name, "logs")
    best_model_path = os.path.join(logs_dir, "final_exp11_miou98_16_epoch070.pth")
    last_model_path = os.path.join(logs_dir, "last_epoch_weights.pth")

    if os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(last_model_path):
        model_path = last_model_path
    else:
        pth_files = [f for f in os.listdir(logs_dir) if f.endswith('.pth')] if os.path.exists(logs_dir) else []
        if pth_files:
            pth_files.sort(key=lambda x: os.path.getsize(os.path.join(logs_dir, x)), reverse=True)
            model_path = os.path.join(logs_dir, pth_files[0])
        else:
            print("No checkpoint file found.")
            return

    output_dir = os.path.join(root_path, folder_name, "visualize_results")
    os.makedirs(output_dir, exist_ok=True)

    voc_root = os.path.join(root_path, "VOCdevkit", "VOC2007")
    val_list_path = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    img_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")

    with open(val_list_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    img_names = val_ids[:num_samples]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint: {model_path}")
    print(f"Device: {device}")

    model = CBAMResUnetExp11(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    print(f"Generating visualizations for {len(img_names)} samples...")

    success_count = 0
    for img_id in tqdm(img_names, desc="Visualizing"):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        label_path = os.path.join(label_dir, img_id + ".png")
        save_id = img_id

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')

            original_size = image.size

            image_resized = image.resize((input_size, input_size), Image.BILINEAR)
            img_np = np.array(image_resized, dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]

            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_img.resize(original_size, Image.NEAREST)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(label, cmap='gray')
            axes[1].set_title('Ground Truth Mask', fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(pred_resized, cmap='gray')
            axes[2].set_title('Prediction (CBAM-ResUNet-ASPP)', fontsize=12)
            axes[2].axis('off')

            label_np = np.array(label.resize(original_size, Image.NEAREST))
            pred_np = np.array(pred_resized)

            overlay = np.zeros((*original_size[::-1], 4), dtype=np.uint8)
            overlay[(label_np > 0) & (pred_np > 0)] = [0, 255, 0, 128]
            overlay[(label_np > 0) & (pred_np == 0)] = [255, 0, 0, 128]
            overlay[(label_np == 0) & (pred_np > 0)] = [0, 0, 255, 128]

            axes[3].imshow(image)
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay (Green:TP, Red:FN, Blue:FP)', fontsize=12)
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{save_id}.png"), dpi=150, bbox_inches='tight')
            plt.close()

            success_count += 1

        except Exception as e:
            print(f"Failed to process {img_id}: {e}")
            continue

    print(f"\nVisualization complete. Generated {success_count} images.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CBAM-ResUNet-ASPP visualization')
    parser.add_argument('--num_samples', type=int, default=10, help='number of samples to visualize')
    parser.add_argument('--input_size', type=int, default=320, help='square input size for the model')

    args = parser.parse_args()

    visualize_predictions(num_samples=args.num_samples, input_size=args.input_size)

