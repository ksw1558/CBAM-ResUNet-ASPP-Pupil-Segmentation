import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# === 1. 智能定位项目根目录 ===
# 逻辑：向上查找，直到找到包含 "VOCdevkit" 或 "nets" 的目录
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

# 导入工具
from utils.utils_metrics import compute_mIoU

try:
    from nets.unet import Unet
except ImportError:
    print("❌ 找不到 nets.unet.Unet，请检查模型文件。")
    sys.exit(1)


def eval_base_unet_voc():
    print("=" * 60)
    print("🚀 评估 01_Base_Unet 在 VOC 上的性能")
    print("=" * 60)

    # 2. 路径配置
    folder_name = "01_Base_Unet_Exp"
    model_path = os.path.join(root_path, folder_name, "logs", "best_epoch_weights.pth")
    output_dir = os.path.join(root_path, folder_name, "miou_out", "detection-results")

    # VOC 路径
    voc_root = os.path.join(root_path, "VOCdevkit", "VOC2007")
    val_list_path = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    img_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")

    os.makedirs(output_dir, exist_ok=True)

    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📂 正在加载权重: {model_path}")

    if not os.path.exists(model_path):
        print("❌ 权重文件不存在！")
        return

    model = Unet(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint: checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print("✅ 模型加载成功！")

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
            # 获取原图尺寸 (用于 Resize 回原大小)
            label_img = Image.open(label_path)
            target_size = label_img.size  # (width, height)

            # 预处理 (固定输入 256x256)
            image = Image.open(img_path).convert('RGB')
            image_resized = image.resize((256, 256), Image.BILINEAR)

            img_np = np.array(image_resized, dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]

            # 关键修复：将 256x256 的预测结果 Resize 回原图尺寸
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_img.resize(target_size, Image.NEAREST)

            # 保存结果
            pred_resized.save(os.path.join(output_dir, img_id + ".png"))
            success_count += 1
        except Exception as e:
            continue

    print(f"\n✅ 推理完成！成功保存 {success_count} 张预测图。")

    # 6. 计算 mIoU
    print("📊 正在计算 mIoU...")
    try:
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            label_dir, output_dir, val_ids, num_classes=2, name_classes=["background", "pupil"]
        )

        mIoU = np.nanmean(IoUs) * 100
        pupil_IoU = IoUs[1] * 100

        print("\n" + "=" * 60)
        print("🏆 最终评估结果")
        print("=" * 60)
        print(f"   mIoU      : {mIoU:.2f}%")
        print(f"   Pupil IoU : {pupil_IoU:.2f}%")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 计算失败: {e}")


if __name__ == "__main__":
    eval_base_unet_voc()
