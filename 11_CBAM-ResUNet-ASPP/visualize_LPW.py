import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
import cv2

warnings.filterwarnings('ignore')

# === 1. 智能定位项目根目录 ===
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
print(f"📂 已定位项目根目录: {root_path}")

try:
    from nets.cbam_res_unet_exp11 import CBAMResUnetExp11
except ImportError:
    print("❌ 找不到 nets.cbam_res_unet_exp11.CBAMResUnetExp11")
    sys.exit(1)


def fill_holes(mask):
    """空洞填充"""
    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.maximum(mask_u8, holes).astype(np.uint8)


def adaptive_threshold_segmentation(prob_map, global_threshold=0.31):
    """
    自适应阈值分割：
    1. 全局阈值二值化
    2. 空洞填充
    3. 最大连通域过滤
    4. 边缘平滑
    """
    binary_mask = (prob_map > global_threshold).astype(np.uint8)
    filled_mask = fill_holes(binary_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)

    if num_labels <= 1:
        final_mask = filled_mask
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            final_mask = filled_mask
        else:
            largest_idx = np.argmax(areas) + 1
            largest_area = areas[np.argmax(areas)]

            if largest_area < 30:
                final_mask = binary_mask
            else:
                final_mask = (labels == largest_idx).astype(np.uint8)

    final_mask = cv2.medianBlur(final_mask, 3)
    blurred = cv2.GaussianBlur(final_mask.astype(np.float32), (3, 3), 0)
    final_mask = (blurred > 0.5).astype(np.uint8)

    return final_mask


def predict_and_visualize():
    """使用CBAM-ResUNet-ASPP模型预测VOC数据集并生成可视化mask"""
    print("=" * 80)
    print("🚀 实验11 CBAM-ResUNet-ASPP - VOC数据集预测与可视化")
    print("=" * 80)

    # 2. 路径配置
    folder_name = "11_CBAM-ResUNet-ASPP"
    model_path = os.path.join(root_path, folder_name, "logs", "final_exp11_miou98_16_epoch070.pth")

    # 输出目录
    pred_raw_dir = os.path.join(root_path, folder_name, "miou_out", "detection-results")
    pred_vis_dir = os.path.join(root_path, folder_name, "miou_out", "visualization")

    # VOC 路径
    voc_root = os.path.join(root_path, "VOCdevkit", "VOC2007")
    val_list_path = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    img_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")

    os.makedirs(pred_raw_dir, exist_ok=True)
    os.makedirs(pred_vis_dir, exist_ok=True)

    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📂 正在加载权重: {model_path}")

    if not os.path.exists(model_path):
        print("❌ 权重文件不存在！")
        return

    model = CBAMResUnetExp11(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    print("✅ 模型加载成功！")

    # 4. 读取验证集列表
    with open(val_list_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    print(f"📸 开始推理 {len(val_ids)} 张 VOC 验证集图片...")
    print("⚙️  策略: Softmax + 自适应阈值(0.31) + 空洞填充 + 边缘平滑")

    # 5. 推理循环
    success_count = 0
    best_global_thresh = 0.31

    for img_id in tqdm(val_ids, desc="推理中"):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        label_path = os.path.join(label_dir, img_id + ".png")

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue

        try:
            # 获取标签原图尺寸
            label_img = Image.open(label_path)
            target_size = label_img.size

            # 预处理
            image = Image.open(img_path).convert('RGB')
            image_resized = image.resize((256, 256), Image.BILINEAR)

            img_np = np.array(image_resized, dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.softmax(output, dim=1)
                pupil_prob = pred[0, 1].cpu().numpy()

            # 自适应阈值分割
            refined_mask = adaptive_threshold_segmentation(pupil_prob, global_threshold=best_global_thresh)

            # 保存原始预测结果 (值为0和1)
            pred_img = Image.fromarray(refined_mask.astype(np.uint8))
            pred_resized = pred_img.resize(target_size, Image.NEAREST)
            pred_resized.save(os.path.join(pred_raw_dir, img_id + ".png"))

            # 生成可视化mask (0->黑色背景, 1->白色瞳孔区域)
            mask_visual = (np.array(pred_resized) * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_visual, "L")
            mask_img.save(os.path.join(pred_vis_dir, img_id + ".png"))

            success_count += 1
        except Exception as e:
            continue

    print(f"\n{'='*80}")
    print(f"✅ 推理完成！成功处理 {success_count} 张图片")
    print(f"📁 原始预测结果: {pred_raw_dir}")
    print(f"📁 可视化mask: {pred_vis_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    predict_and_visualize()
