# ---------------------------------------------------------#
#   ## 视频批量预测脚本 (资源池架构版)
#   位置：01_Base_Unet_Exp/video_predict_cbam.py
# ---------------------------------------------------------#
import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 确保能找到上一级目录的 unet.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unet import Unet

if __name__ == "__main__":
    # 1. 初始化模型
    unet = Unet()
    unet.mix_type = 1

    # 2. 路径溯源 (重点修改)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 指向新的全局资源池路径
    input_dir = os.path.join(current_dir, "..", "resources", "videos")
    # 输出依然保持在当前实验文件夹下
    output_dir = os.path.join(current_dir, "video_out")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 自动扫描 resources/videos 下的所有 mp4 文件
    video_extensions = ('.mp4', '.avi', '.mov')
    all_videos = [f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]

    print(f"\n>>> 资源池扫描完成，发现 {len(all_videos)} 段待处理视频...")

    for video_name in all_videos:
        video_path = os.path.join(input_dir, video_name)
        save_path = os.path.join(output_dir, f"result_{video_name}")

        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="f")

        while True:
            ref, frame = capture.read()
            if not ref:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            r_image = unet.detect_image(image)

            mask_bgr = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                best_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(best_cnt) > 100 and len(best_cnt) >= 5:
                    ellipse = cv2.fitEllipse(best_cnt)
                    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                    cv2.circle(frame, (int(ellipse[0][0]), int(ellipse[0][1])), 3, (0, 0, 255), -1)

            out.write(frame)
            pbar.update(1)

        pbar.close()
        capture.release()
        out.release()

    cv2.destroyAllWindows()
    print("\n🎉 资源池视频批量任务已全部完成！")