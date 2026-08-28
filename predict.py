"""瞳孔分割交互式预测工具 - 带鲁棒椭圆拟合

本脚本提供单张图片的交互式瞳孔分割与参数提取功能。
主要特性：
    - 自动过滤文字干扰和噪点
    - 基于最大连通域的瞳孔区域筛选
    - 最小二乘法椭圆拟合
    - 可视化中心坐标和椭圆边界

使用方法：
    >>> python predict.py
    >>> # 按提示输入图片路径即可

Author: Project contributor
日期: 2026.06
版本: Final Release v1.0
"""
# ----------------------------------------------------#
#   瞳孔分割 + 鲁棒椭圆拟合工具 (Robust Ellipse Fitting)
#   功能：自动过滤文字干扰、噪点，仅拟合最大面积瞳孔
# ----------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
import copy

# 核心导入：确保你的根目录下有修改过的那个 unet.py
from unet import Unet

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   参数设置：确保 input_shape 与你训练时一致 (建议 256 或 512)
    # -------------------------------------------------------------------------#
    mode = "predict"
    count = False
    name_classes = ["background", "pupil"]

    # 初始化模型
    unet = Unet()

    if mode == "predict":
        print("\n" + "=" * 50)
        print("已启动瞳孔高精度拟合模式 (Filtering Noise Mode)")
        print("=" * 50)

        while True:
            img_path = input('请输入图片路径 (例如 VOCdevkit/VOC2007/JPEGImages/01_01_0001.jpg):')
            try:
                # 1. 加载原始图片
                image = Image.open(img_path)
            except:
                print('图片打开失败，请检查路径是否正确！')
                continue
            else:
                # ---------------------------------------------------------#
                # 步骤 A：获取纯净的分割 Mask (不带原图混合)
                # ---------------------------------------------------------#
                temp_mix_type = unet.mix_type
                unet.mix_type = 1  # 强制设为 1，只输出黑红分割图，防止原图反光干扰

                # 得到分割后的 PIL 图片
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)

                # ---------------------------------------------------------#
                # 步骤 B：OpenCV 预处理
                # ---------------------------------------------------------#
                # 将分割结果转为 OpenCV BGR 格式
                mask_bgr = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
                # 转灰度 -> 二值化 (由于瞳孔是红色 128,0,0，灰度值较低，阈值设为 30 即可提取)
                gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

                # ---------------------------------------------------------#
                # 步骤 C：轮廓筛选与椭圆拟合 (核心改进)
                # ---------------------------------------------------------#
                # 提取所有闭合轮廓
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 准备一张干净的原图用于绘图展示
                final_output = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

                if contours:
                    # ⭐ 关键逻辑：按照面积对轮廓进行降序排序
                    # 这样 contours[0] 永远是图片中面积最大的那个物体 (瞳孔)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    pupil_cnt = contours[0]

                    # 设定最小面积阈值 (例如 100 像素)，防止把微小噪点当成瞳孔
                    if cv2.contourArea(pupil_cnt) > 100 and len(pupil_cnt) >= 5:
                        # 仅对这一个最大面积轮廓进行最小二乘法椭圆拟合
                        ellipse = cv2.fitEllipse(pupil_cnt)

                        # 在原图上绘制拟合结果
                        # 绿色椭圆线条 (thickness=2)
                        cv2.ellipse(final_output, ellipse, (0, 255, 0), 2)
                        # 红色几何中心点 (radius=3, filled)
                        center_x, center_y = ellipse[0]
                        cv2.circle(final_output, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

                        print(f"成功定位！中心: ({center_x:.1f}, {center_y:.1f}), 面积: {cv2.contourArea(pupil_cnt):.0f}")
                    else:
                        print("警告：识别到的最大区域面积过小，疑似噪点，已忽略。")
                else:
                    print("错误：未在图像中发现任何瞳孔分割区域。")

                # ---------------------------------------------------------#
                # 步骤 D：结果展示
                # ---------------------------------------------------------#
                unet.mix_type = temp_mix_type  # 还原设置

                cv2.imshow("Final Result (Largest Contour Only)", final_output)
                print("\n>>> 窗口已弹出。按键盘【任意键】继续下一次预测...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    else:
        print("当前仅支持 predict 模式进行椭圆拟合测试。")
