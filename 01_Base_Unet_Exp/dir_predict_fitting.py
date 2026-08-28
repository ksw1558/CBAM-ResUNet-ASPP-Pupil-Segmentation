# ---------------------------------------------------------#
#   ## 批量预测脚本 (子文件夹规范版)
#   位置：放在 01_Base_Unet_Exp/ 文件夹下
#   功能：与当前实验权重对齐，批量生成预测结果
# ---------------------------------------------------------#
import os
import cv2
import numpy as np
import csv
import sys
from PIL import Image
from tqdm import tqdm

# --- 修改点 1：将上一级目录加入系统路径，以便导入根目录的 unet.py ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unet import Unet

if __name__ == "__main__":
    # --- 修改点 2：路径全部改为基于当前实验文件夹的相对路径 ---
    # 获取当前脚本所在文件夹的绝对路径 (即 .../01_Base_Unet_Exp/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 输入路径：通常 VOCdevkit 在根目录，所以要往上走一层
    dir_origin_path = os.path.join(current_dir, "..", "VOCdevkit/VOC2007/JPEGImages/")

    # 输出路径：直接放在当前实验文件夹下
    dir_save_path = os.path.join(current_dir, "img_out_fitting/")
    csv_save_path = os.path.join(current_dir, "pupil_data_voc.csv")

    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    # 初始化模型
    unet = Unet()
    unet.mix_type = 1

    results_list = []

    if not os.path.exists(dir_origin_path):
        print(f"错误：找不到原始图片路径 {dir_origin_path}")
    else:
        img_names = os.listdir(dir_origin_path)
        print(f"\n>>> 实验 01 批量检测启动，总计 {len(img_names)} 张图片")

        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dir_origin_path, img_name)
                try:
                    image = Image.open(image_path)
                except:
                    continue

                r_image = unet.detect_image(image)
                mask_bgr = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                final_output = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                cx, cy, axes_l, axes_s, angle = 0, 0, 0, 0, 0

                if contours:
                    best_cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(best_cnt) > 100 and len(best_cnt) >= 5:
                        ellipse = cv2.fitEllipse(best_cnt)
                        cv2.ellipse(final_output, ellipse, (0, 255, 0), 2)
                        cv2.circle(final_output, (int(ellipse[0][0]), int(ellipse[0][1])), 3, (0, 0, 255), -1)
                        (cx, cy), (axes_a, axes_b), angle = ellipse
                        axes_l, axes_s = max(axes_a, axes_b), min(axes_a, axes_b)

                results_list.append({
                    "图片名": img_name, "中心X": round(cx, 2), "中心Y": round(cy, 2),
                    "长轴": round(axes_l, 2), "短轴": round(axes_s, 2), "角度": round(angle, 2)
                })
                cv2.imwrite(os.path.join(dir_save_path, img_name), final_output)

        with open(csv_save_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["图片名", "中心X", "中心Y", "长轴", "短轴", "角度"])
            writer.writeheader()
            writer.writerows(results_list)

        print(f"\n✅ 实验 01 数据已对齐保存至: {current_dir}")