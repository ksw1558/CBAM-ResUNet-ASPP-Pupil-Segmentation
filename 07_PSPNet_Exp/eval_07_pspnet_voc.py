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

from utils.utils_metrics import compute_mIoU

try:
    from nets.pspnet import PSPNet
except ImportError:
    print("❌ 找不到 nets.pspnet.PSPNet")
    sys.exit(1)


def eval_pspnet_voc():
    print("=" * 60)
    print("🚀 评估 07_PSPNet 在 VOC 上的性能")
    print("=" * 60)

    # 2. 路径配置
    folder_name = "07_PSPNet_Exp"
    model_path = os.path.join(root_path, folder_name, "logs", "best_epoch_weights.pth")
    output_dir = os.path.join(root_path, folder_name, "miou_out", "detection-results")

    voc_root = os.path.join(root_path, "VOCdevkit", "VOC2007")
    val_list_path = os.path.join(voc_root, "ImageSets", "Segmentation", "val.txt")
    img_dir = os.path.join(voc_root, "JPEGImages")
    label_dir = os.path.join(voc_root, "SegmentationClass")

    os.makedirs(output_dir, exist_ok=True)

    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📂 正在加载权重: {model_path}")

    if not os.path.exists(model_path):
        print("❌ 权重不存在！")
        return

    # PSPNet 推荐输入 512，若显存不足请改为 256
    input_size = 512
    model = PSPNet(num_classes=2)

    checkpoint = torch.load(model_path, map_location=device)
    if 'net' in checkpoint: checkpoint = checkpoint['net']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    print(f"✅ 模型加载成功！(输入尺寸：{input_size}x{input_size})")

    with open(val_list_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    print(f"📸 开始推理 {len(val_ids)} 张 VOC 验证集图片...")

    # 4. 推理循环
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
            image_resized = image.resize((input_size, input_size), Image.BILINEAR)

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

    # 5. 计算指标
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
    eval_pspnet_voc()
