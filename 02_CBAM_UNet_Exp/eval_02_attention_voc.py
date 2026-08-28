import os
import sys
import torch
import numpy as np
from PIL import Image
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
print(f"📂 已定位项目根目录: {root_path}")

from utils.utils_metrics import compute_mIoU

# 注意：Attention 模型通常也是基于 Unet 类实现的，如果报错找不到类，请尝试改为 from nets.unet import Unet
try:
    from nets.unet import Unet
except ImportError:
    # 如果你的 Attention 模型定义在单独的类里，请取消下面这行的注释并修改类名
    # from nets.attention_unet import AttentionUnet as Unet
    pass


def eval_model_voc():
    print("=" * 60)
    print("🚀 评估 02_Attention_Unet 在 VOC 上的性能")
    print("=" * 60)

    # 1. 修改这里：指定实验文件夹
    folder_name = "02_CBAM_UNet_Exp"

    model_path = os.path.join(root_path, folder_name, "logs", "best_epoch_weights.pth")
    output_dir = os.path.join(root_path, folder_name, "miou_out", "detection-results")

    voc_root = os.path.join(root_path, "VOCdevkit", "VOC2007")
    val_list_path = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    img_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")

    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📂 正在加载权重: {model_path}")

    if not os.path.exists(model_path):
        print("❌ 权重不存在！")
        return

    # 注意：这里使用 Unet 类加载 Attention 模型的权重，因为结构通常是兼容的
    model = Unet(num_classes=2)

    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint: checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print("✅ 模型加载成功！")

    with open(val_list_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    print(f"📸 开始推理 {len(val_ids)} 张 VOC 验证集图片...")

    # 3. 推理循环
    success_count = 0
    for img_id in tqdm(val_ids, desc="推理中"):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        label_path = os.path.join(label_dir, img_id + ".png")

        if not os.path.exists(img_path) or not os.path.exists(label_path): continue

        try:
            # 获取标签原图尺寸
            label_img = Image.open(label_path)
            target_size = label_img.size

            image = Image.open(img_path).convert('RGB')
            image_resized = image.resize((256, 256), Image.BILINEAR)

            img_np = np.array(image_resized, dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]

            # 尺寸对齐
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred_resized = pred_img.resize(target_size, Image.NEAREST)
            pred_resized.save(os.path.join(output_dir, img_id + ".png"))
            success_count += 1
        except Exception as e:
            continue

    print(f"\n✅ 推理完成！成功保存 {success_count} 张。")

    # 4. 计算指标
    print("📊 正在计算 mIoU...")
    try:
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            label_dir, output_dir, val_ids, num_classes=2, name_classes=["background", "pupil"]
        )
        print(f"\n🏆 mIoU: {np.nanmean(IoUs) * 100:.2f}% | Pupil IoU: {IoUs[1] * 100:.2f}%")
    except Exception as e:
        print(f"❌ 计算失败: {e}")


if __name__ == "__main__":
    eval_model_voc()


