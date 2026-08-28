import os
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

EXP_DIR = os.path.join(CURRENT_DIR, "01_Base_Unet_Exp")
PRED_DIR = os.path.join(EXP_DIR, "miou_out", "detection-results")
OUT_DIR = os.path.join(EXP_DIR, "miou_out", "visualization")

VOC_ROOT = os.path.join(CURRENT_DIR, "VOCdevkit", "VOC2007")
VAL_LIST = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "val.txt")


def main():
    """将二值预测结果转换为可视化的白色mask"""
    if not os.path.exists(PRED_DIR):
        raise FileNotFoundError(f"预测目录不存在: {PRED_DIR}")

    with open(VAL_LIST, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"📊 开始生成 {len(image_ids)} 张可视化mask...")

    generated = 0
    for image_id in image_ids:
        pred_path = os.path.join(PRED_DIR, image_id + ".png")

        if not os.path.exists(pred_path):
            continue

        try:
            # 读取二值预测结果 (值为0和1)
            pred = Image.open(pred_path).convert("L")
            pred_array = np.array(pred)

            # 转换为可视化mask: 0->黑色背景, 1->白色瞳孔区域
            mask_visual = (pred_array * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_visual, "L")

            save_path = os.path.join(OUT_DIR, f"{image_id}.png")
            mask_img.save(save_path)
            generated += 1

        except Exception as e:
            print(f"❌ 处理 {image_id} 时出错: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(f"✅ 完成！成功生成 {generated} 张可视化mask")
    print(f"📁 保存位置: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
