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

# === 智能定位项目根目录 ===
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
    from nets.cbam_res_unet import CBAMResUnet
except ImportError:
    print("❌ 找不到 nets.cbam_res_unet.CBAMResUnet")
    sys.exit(1)


def visualize_predictions(num_samples=10):
    print("=" * 60)
    print("🎨 CBAM-ResUNet 可视化结果生成")
    print("=" * 60)

    folder_name = "09_CBAM_ResUNet_Exp"
    model_path = os.path.join(root_path, folder_name, "logs", "best_epoch_weights.pth")
    output_dir = os.path.join(root_path, folder_name, "visualize_results")
    os.makedirs(output_dir, exist_ok=True)

    voc_root = os.path.join(root_path, "VOCdevkit", "VOC2007")
    val_list_path = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    img_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        print("❌ 权重不存在！")
        return

    model = CBAMResUnet(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    with open(val_list_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]

    sample_ids = val_ids[:num_samples]
    print(f"📸 正在生成 {len(sample_ids)} 个样本的可视化结果...")

    for img_id in tqdm(sample_ids, desc="可视化中"):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        label_path = os.path.join(label_dir, img_id + ".png")

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')

            image_resized = image.resize((256, 256), Image.BILINEAR)
            img_np = np.array(image_resized, dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]

            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_img.resize(image.size, Image.NEAREST)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image)
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(label, cmap='gray')
            axes[1].set_title('Ground Truth', fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(pred_resized, cmap='gray')
            axes[2].set_title('Prediction (CBAM-ResUNet)', fontsize=12)
            axes[2].axis('off')

            label_np = np.array(label)
            pred_np = np.array(pred_resized)
            overlay = np.zeros((*label_np.shape, 4), dtype=np.uint8)
            overlay[(label_np > 0) & (pred_np > 0)] = [0, 255, 0, 128]
            overlay[(label_np > 0) & (pred_np == 0)] = [255, 0, 0, 128]
            overlay[(label_np == 0) & (pred_np > 0)] = [0, 0, 255, 128]

            axes[3].imshow(image)
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay (Green:TP, Red:FN, Blue:FP)', fontsize=12)
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{img_id}.png"), dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️ 处理 {img_id} 时出错: {e}")
            continue

    print(f"✅ 可视化完成！结果保存到: {output_dir}")


if __name__ == "__main__":
    visualize_predictions()
