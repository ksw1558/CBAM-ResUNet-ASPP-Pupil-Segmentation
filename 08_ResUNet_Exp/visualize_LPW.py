import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings

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
    from nets.res_unet import ResUnet
except ImportError:
    print("❌ 找不到 nets.res_unet.ResUnet")
    sys.exit(1)


def predict_and_visualize():
    """使用ResUNet模型预测VOC数据集并生成可视化mask"""
    print("=" * 60)
    print("🚀 实验08 ResUNet - VOC数据集预测与可视化")
    print("=" * 60)

    # 2. 路径配置
    folder_name = "08_ResUNet_Exp"
    model_path = os.path.join(root_path, folder_name, "logs", "best_epoch_weights.pth")

    # 输出目录：miou_out/detection-results (原始预测) 和 miou_out/visualization (可视化mask)
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

    input_size = 256
    model = ResUnet(num_classes=2)

    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print(f"✅ 模型加载成功！(输入尺寸：{input_size}x{input_size})")

    # 4. 读取验证集列表
    with open(val_list_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    print(f"📸 开始推理 {len(val_ids)} 张 VOC 验证集图片...")

    # 5. 推理循环
    success_count = 0
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
            image_resized = image.resize((input_size, input_size), Image.BILINEAR)

            img_np = np.array(image_resized, dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]

            # 保存原始预测结果 (值为0和1)
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_img.resize(target_size, Image.NEAREST)
            pred_resized.save(os.path.join(pred_raw_dir, img_id + ".png"))

            # 生成可视化mask (0->黑色背景, 1->白色瞳孔区域)
            mask_visual = (np.array(pred_resized) * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_visual, "L")
            mask_img.save(os.path.join(pred_vis_dir, img_id + ".png"))

            success_count += 1
        except Exception as e:
            continue

    print(f"\n{'=' * 60}")
    print(f"✅ 推理完成！成功处理 {success_count} 张图片")
    print(f"📁 原始预测结果: {pred_raw_dir}")
    print(f"📁 可视化mask: {pred_vis_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    predict_and_visualize()
