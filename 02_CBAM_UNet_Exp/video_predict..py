import os
import cv2
import sys
import time
import numpy as np
from PIL import Image

# 1. 环境与路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_path)

from unet import Unet


# 导入你刚才写的 V2 版后处理逻辑，确保视频里也没有“大色块”干扰
def post_process_mask_v2(mask_array):
    mask_8u = mask_array.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)
    if num_labels <= 1: return mask_array
    h, w = mask_array.shape
    img_center = np.array([w / 2, h / 2])
    best_label, min_dist = -1, float('inf')
    for i in range(1, num_labels):
        if stats[i, 4] < 20: continue
        dist = np.linalg.norm(centroids[i] - img_center)
        if dist < min_dist:
            min_dist = dist
            best_label = i
    refined_mask = np.zeros_like(mask_array)
    if best_label != -1: refined_mask[labels == best_label] = 1
    return refined_mask


if __name__ == "__main__":
    # 2. 加载实验 02 最佳权重
    model_path = os.path.join(current_dir, "logs", "best_epoch_weights.pth")
    print(f"正在加载注意力增强模型: {model_path}")
    unet = Unet(model_path=model_path)

    # 3. 视频路径
    video_src_dir = os.path.join(root_path, "resources", "videos")
    video_save_dir = os.path.join(current_dir, "video_out")
    os.makedirs(video_save_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_src_dir) if f.endswith(('.mp4', '.avi'))]

    for video_name in video_files:
        video_path = os.path.join(video_src_dir, video_name)
        capture = cv2.VideoCapture(video_path)

        # 获取视频属性
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        save_path = os.path.join(video_save_dir, f"attention_res_{video_name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        print(f"正在处理视频: {video_name} ...")

        while True:
            res, frame = capture.read()
            if not res: break

            # A. 格式转换 RGB -> PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            # B. 模型预测 (获取 0-1 掩膜)
            mask_png = unet.get_miou_png(img_pil)
            mask_arr = np.array(mask_png)

            # C. ⭐ 调用 V2 空间过滤逻辑，抹除画面边缘干扰
            refined_mask = post_process_mask_v2(mask_arr)

            # D. 拟合绿圈并绘制
            mask_255 = (refined_mask * 255).astype(np.uint8)
            # 缩放掩膜回到原视频大小进行绘图
            mask_rescaled = cv2.resize(mask_255, (width, height), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask_rescaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 5:  # 至少5个点才能拟合椭圆
                    ellipse = cv2.fitEllipse(cnt)
                    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

            out.write(frame)

        capture.release()
        out.release()
        print(f"视频已生成: {save_path}")

    print("\n✅ 所有视频处理完成！你可以去 video_out 文件夹验收结果了。")