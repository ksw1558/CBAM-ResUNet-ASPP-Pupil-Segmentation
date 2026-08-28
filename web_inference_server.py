"""
Web Inference Server for Pupil Segmentation (Experiment 11)
===========================================================
基于 Flask 的 Web 推理服务器，提供 REST API 接口用于瞳孔分割预测。

主要功能：
1. 单张图片预测 (/api/predict/image)
   - 输入：RGB 图片（JPG/PNG/BMP）
   - 输出：分割 Mask、叠加图、瞳孔中心坐标、直径等参数
   
2. 批量图片预测 (/api/predict/batch)
   - 输入：多张 RGB 图片
   - 输出：CSV 报告、ZIP 压缩包（包含所有结果）
   
3. 视频预测 (/api/predict/video)
   - 输入：眼动追踪视频（MP4/AVI）
   - 输出：逐帧分析视频、GIF 预览、CSV 时间序列数据

技术栈：
- 后端框架：Flask
- 深度学习：PyTorch + Experiment 11 CBAM-ResUNet-ASPP
- 图像处理：OpenCV, PIL
- 部署方式：本地开发服务器（127.0.0.1:5000）

作者：Pupil Segmentation Project Team
版本：v1.0
日期：2026
"""

import csv
import importlib.util
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename


# ==================== 全局配置 ====================
ROOT = Path(__file__).resolve().parent  # 项目根目录
EXP11_DIR = ROOT / "11_CBAM-ResUNet-ASPP"  # 实验11目录（最终模型 ACR-UNet）
WEIGHT_PATH = EXP11_DIR / "logs" / "final_exp11_miou98_16_epoch070.pth"  # 预训练权重路径（mIoU=98.16%）
OUTPUT_ROOT = EXP11_DIR / "web_predictions"  # 输出文件根目录
UPLOAD_ROOT = OUTPUT_ROOT / "_uploads"  # 临时上传文件目录
INPUT_SIZE = 256  # 模型输入尺寸（256×256 像素）

# 将项目根目录添加到 Python 模块搜索路径
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_exp11_helpers():
    """
    动态加载实验11的工具函数模块（extract_pupil_params.py）
    
    Returns:
        module: 包含 load_exp11_model() 和 extract_pupil_from_mask() 等函数的模块对象
    """
    helper_path = EXP11_DIR / "extract_pupil_params.py"
    spec = importlib.util.spec_from_file_location("exp11_extract_pupil_params", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 加载实验11辅助函数
helpers = load_exp11_helpers()

# 创建 Flask 应用实例
app = Flask(__name__)

# 全局变量：模型实例（延迟加载，首次请求时初始化）
model = None

# 设备选择：优先使用 GPU（CUDA），否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 线程锁：保证模型加载和推理的线程安全
model_lock = Lock()


# ==================== CORS 中间件 ====================
@app.after_request
def add_cors_headers(response):
    """
    为所有响应添加 CORS 头，允许跨域请求（前端网页访问）
    
    Args:
        response: Flask 响应对象
        
    Returns:
        response: 添加了 CORS 头的响应对象
    """
    response.headers["Access-Control-Allow-Origin"] = "*"  # 允许任意域名访问
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"  # 允许的请求头
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"  # 允许的 HTTP 方法
    return response


# ==================== API 端点 ====================

@app.route("/api/health", methods=["GET", "OPTIONS"])
def health():
    """
    健康检查端点
    
    Returns:
        JSON: 服务器状态信息（模型是否加载、GPU 是否可用等）
    """
    return jsonify(
        {
            "ok": True,
            "model": "Experiment 11 CBAM-ResUNet-ASPP",  # 模型名称
            "device": str(device),  # 当前设备（cuda/cpu）
            "cuda_available": torch.cuda.is_available(),  # CUDA 是否可用
            "weight_exists": WEIGHT_PATH.exists(),  # 权重文件是否存在
            "weight_path": str(WEIGHT_PATH),  # 权重文件路径
            "loaded": model is not None,  # 模型是否已加载
        }
    )


@app.route("/outputs/<path:filename>", methods=["GET"])
def outputs(filename):
    """
    静态文件服务：提供预测结果的下载链接
    
    Args:
        filename: 相对路径（相对于 OUTPUT_ROOT）
        
    Returns:
        File: 文件内容（图片或 CSV）
    """
    return send_from_directory(OUTPUT_ROOT, filename, as_attachment=False)


@app.route("/api/predict/image", methods=["POST", "OPTIONS"])
def predict_image():
    """
    单张图片预测端点
    
    Request:
        POST /api/predict/image
        Content-Type: multipart/form-data
        Body:
            - image: 图片文件（JPG/PNG/BMP）
            - threshold: 可选，二值化阈值（默认 0.31）
            
    Response:
        JSON:
            {
                "filename": "test.png",
                "status": "ok",
                "cx": 128.5,          # 瞳孔中心 X 坐标
                "cy": 130.2,          # 瞳孔中心 Y 坐标
                "diameter": 45.3,     # 瞳孔直径（像素）
                "major_axis": 46.1,   # 椭圆长轴
                "minor_axis": 44.5,   # 椭圆短轴
                "area": 1623.7,       # 瞳孔面积（像素²）
                "confidence": 0.9876, # 置信度
                "angle": 12.3,        # 椭圆角度（度）
                "overlay_url": "/outputs/image_.../overlay.png",  # 叠加图 URL
                "mask_url": "/outputs/image_.../mask.png"         # Mask URL
            }
    """
    if request.method == "OPTIONS":
        return "", 204  # CORS 预检请求
    
    # 获取上传的图片文件
    file = request.files.get("image")
    if file is None:
        return error("缺少 image 文件字段", 400)

    # 获取二值化阈值（默认 0.31）
    threshold = get_threshold()
    
    # 创建任务目录（用于存储本次预测的所有输出）
    job_dir = make_job_dir("image")
    original_name = safe_name(file.filename or "image.png")
    upload_path = job_dir / original_name
    
    # 保存上传的文件
    file.save(upload_path)

    # 执行预测
    image = Image.open(upload_path).convert("RGB")
    result, mask = infer_image(image, threshold)
    
    # 保存 Mask 和叠加图
    mask_path = job_dir / "mask.png"
    overlay_path = job_dir / "overlay.png"
    Image.fromarray(mask).save(mask_path)
    save_overlay(image, mask, result, overlay_path)

    # 构建返回结果
    row = result_to_row(original_name, result)
    return jsonify(
        {
            **row,
            "status": row["status"],
            "overlay_url": output_url(overlay_path),  # 叠加图下载链接
            "mask_url": output_url(mask_path),        # Mask 下载链接
            "rows": [row],
        }
    )


@app.route("/api/predict/batch", methods=["POST", "OPTIONS"])
def predict_batch():
    """
    批量图片预测端点
    
    Request:
        POST /api/predict/batch
        Content-Type: multipart/form-data
        Body:
            - images[]: 多个图片文件
            - threshold: 可选，二值化阈值（默认 0.31）
            
    Response:
        JSON:
            {
                "count": 10,                    # 总图片数
                "ok_count": 9,                  # 成功预测数
                "rows": [...],                  # 每张图片的结果列表
                "first_overlay_url": "...",     # 第一张图的叠加图 URL
                "csv_url": "/outputs/.../batch_pupil_params.csv",  # CSV 报告 URL
                "zip_url": "/outputs/.../batch_outputs.zip",       # ZIP 压缩包 URL
                "summary": {                    # 统计摘要
                    "status": "ok",
                    "center_x": 128.5,
                    "center_y": 130.2,
                    "diameter_px": 45.3,
                    ...
                }
            }
    """
    if request.method == "OPTIONS":
        return "", 204
    
    # 获取所有上传的图片文件
    files = request.files.getlist("images")
    files = [file for file in files if file and is_image_name(file.filename)]
    if not files:
        return error("没有收到可预测的图片文件", 400)

    threshold = get_threshold()
    job_dir = make_job_dir("batch")
    
    # 创建子目录
    upload_dir = job_dir / "uploads"      # 原始上传图片
    overlay_dir = job_dir / "overlays"    # 叠加图
    mask_dir = job_dir / "masks"          # Mask 图
    upload_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, file in enumerate(files, start=1):
        original_name = file.filename or f"image_{index}.png"
        stored_name = f"{index:04d}_{safe_name(Path(original_name).name)}"  # 规范化文件名
        upload_path = upload_dir / stored_name
        file.save(upload_path)

        try:
            # 执行单张图片预测
            image = Image.open(upload_path).convert("RGB")
            result, mask = infer_image(image, threshold)
            
            # 保存结果
            mask_path = mask_dir / f"{Path(stored_name).stem}_mask.png"
            overlay_path = overlay_dir / f"{Path(stored_name).stem}_overlay.png"
            Image.fromarray(mask).save(mask_path)
            save_overlay(image, mask, result, overlay_path)
            
            row = result_to_row(original_name, result)
            row["overlay_url"] = output_url(overlay_path)
            rows.append(row)
        except Exception as exc:
            # 记录失败信息
            rows.append({"filename": original_name, "status": "failed", "error": str(exc)})

    # 生成 CSV 报告
    csv_path = job_dir / "batch_pupil_params.csv"
    write_rows_csv(csv_path, rows)
    
    # 打包 ZIP 文件
    zip_path = job_dir / "batch_outputs.zip"
    make_zip(zip_path, [csv_path, overlay_dir, mask_dir])

    # 计算统计信息
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_count = len(ok_rows)
    first_overlay_url = next((row.get("overlay_url") for row in ok_rows if row.get("overlay_url")), "")
    summary = summarize_rows(ok_rows)
    summary["count"] = len(rows)
    summary["ok_count"] = ok_count
    
    return jsonify(
        {
            "count": len(rows),
            "ok_count": ok_count,
            "rows": rows,
            "first_overlay_url": first_overlay_url,
            "csv_url": output_url(csv_path),
            "zip_url": output_url(zip_path),
            "summary": summary,
        }
    )


@app.route("/api/predict/video", methods=["POST", "OPTIONS"])
def predict_video():
    """
    视频预测端点
    
    Request:
        POST /api/predict/video
        Content-Type: multipart/form-data
        Body:
            - video: 视频文件（MP4/AVI/MOV）
            - threshold: 可选，二值化阈值（默认 0.31）
            
    Response:
        JSON:
            {
                "status": "ok",
                "rows": [...],              # 前 300 帧的结果（避免响应过大）
                "count": 1500,              # 总帧数
                "video_url": "/outputs/.../video_exp11_overlay.mp4",  # 完整预测视频 URL
                "preview_url": "/outputs/.../video_preview.gif",      # GIF 预览 URL
                "gif_url": "/outputs/.../video_preview.gif",          # 同 preview_url
                "csv_url": "/outputs/.../video_pupil_params.csv",     # CSV 时间序列 URL
                "summary": {                # 统计摘要
                    "status": "ok",
                    "frames": 1500,
                    "fps": 25.0,
                    "center_x": 128.5,
                    "center_y": 130.2,
                    "diameter_px": 45.3,
                    ...
                }
            }
    """
    if request.method == "OPTIONS":
        return "", 204
    
    # 获取上传的视频文件
    file = request.files.get("video")
    if file is None:
        return error("缺少 video 文件字段", 400)

    threshold = get_threshold()
    job_dir = make_job_dir("video")
    video_name = safe_name(file.filename or "video.mp4")
    video_path = job_dir / video_name
    file.save(video_path)

    # 执行视频预测（逐帧处理）
    rows, overlay_path, preview_path, csv_path, summary = infer_video(video_path, job_dir, threshold)
    
    return jsonify(
        {
            "status": "ok",
            "rows": rows[:300],  # 只返回前 300 帧（避免 JSON 过大）
            "count": len(rows),
            "video_url": output_url(overlay_path),
            "preview_url": output_url(preview_path) if preview_path else output_url(overlay_path),
            "gif_url": output_url(preview_path) if preview_path else "",
            "csv_url": output_url(csv_path),
            "summary": summary,
        }
    )


# ==================== 核心推理函数 ====================

def get_model():
    """
    获取或加载实验11模型（单例模式，线程安全）
    
    Returns:
        nn.Module: 加载好的 PyTorch 模型（CBAM-ResUNet-ASPP）
    """
    global model
    with model_lock:
        if model is None:
            # 检查权重文件是否存在
            if not WEIGHT_PATH.exists():
                raise FileNotFoundError(f"模型权重不存在: {WEIGHT_PATH}")
            
            # 加载模型到指定设备
            loaded = helpers.load_exp11_model(WEIGHT_PATH, device)
            
            # 启用 CUDA 加速（如果使用 GPU）
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True
            
            model = loaded
        return model


def infer_image(image, threshold):
    """
    对单张图片执行瞳孔分割和参数提取
    
    Args:
        image: PIL.Image 对象（RGB 格式）
        threshold: 二值化阈值（0.0-1.0，默认 0.31）
        
    Returns:
        tuple: (result_dict, mask_array)
            - result_dict: 瞳孔参数字典（中心坐标、直径、面积等）
            - mask_array: 二值化 Mask（uint8，0 或 255）
    """
    net = get_model()
    
    # 记录原始尺寸（用于后续恢复）
    original_w, original_h = image.size
    
    # 调整到模型输入尺寸（256×256）
    image_resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    
    # 预处理：归一化 + 转置（HWC → CHW）
    img_np = np.array(image_resized, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    
    # 转换为 PyTorch Tensor 并移动到设备
    tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

    # 执行推理（禁用梯度计算以节省内存）
    with model_lock:
        with torch.no_grad():
            output = net(tensor)
            # 取瞳孔类别的概率图（第 1 通道）
            prob = F.softmax(output, dim=1)[0, 1].detach().cpu().numpy()

    # 后处理：二值化 + 形态学优化
    small_mask = refine_mask(prob, threshold=threshold)
    
    # 恢复到原始尺寸
    mask = cv2.resize(small_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    
    # 提取瞳孔参数（中心、直径、椭圆拟合等）
    result = helpers.extract_pupil_from_mask(mask)
    
    return result, mask


def fill_holes(mask):
    """
    填充 Mask 中的孔洞（洪水填充算法）
    
    Args:
        mask: 二值化数组（uint8，0 或 1）
        
    Returns:
        ndarray: 填充后的 Mask（uint8）
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape
    
    # 创建洪水填充掩码（需要比原图大 2 像素）
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # 从左上角 (0,0) 开始填充背景
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    
    # 孔洞 = 未被填充的区域
    holes = (flood == 0).astype(np.uint8)
    
    # 合并原始 Mask 和孔洞
    return np.maximum(mask_u8, holes).astype(np.uint8)


def refine_mask(prob_map, threshold=0.31, min_area=30):
    """
    优化概率图为高质量 Mask
    
    处理步骤：
    1. 二值化（根据阈值）
    2. 填充孔洞
    3. 连通域分析（选择最可能是瞳孔的区域）
    4. 中值滤波 + 高斯模糊（平滑边缘）
    
    Args:
        prob_map: 概率图（float32，0.0-1.0）
        threshold: 二值化阈值（默认 0.31）
        min_area: 最小连通域面积（过滤噪声，默认 30 像素）
        
    Returns:
        ndarray: 优化后的二值 Mask（uint8，0 或 1）
    """
    # Step 1: 二值化
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    # Step 2: 填充孔洞
    filled_mask = fill_holes(binary_mask)

    # Step 3: 连通域分析（选择最佳候选区域）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
    
    if num_labels > 1:
        candidates = []
        h, w = prob_map.shape
        img_center = (w // 2, h // 2)  # 图像中心点
        
        # 遍历所有连通域（排除背景 label=0）
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue  # 跳过太小的区域（噪声）
            
            # 获取区域几何信息
            x1, y1, bw, bh, area_val = stats[i]
            center_x, center_y = centroids[i]
            
            # 特征 1: 距离图像中心的远近（瞳孔通常在中间）
            dist_to_center = np.sqrt((center_x - img_center[0])**2 + (center_y - img_center[1])**2)
            
            # 特征 2: 形状紧凑度（圆形区域的 bbox_area / actual_area 接近 1）
            bbox_area = bw * bh
            compactness = area_val / (bbox_area + 1e-6)
            
            # 特征 3: 平均概率（模型对该区域的置信度）
            region_prob = np.mean(prob_map[labels == i])
            
            # 综合评分公式：
            # score = 0.4*(距离得分) + 0.3*(形状得分) + 0.3*(概率得分)
            max_dist = np.sqrt(w**2 + h**2) / 2
            norm_dist = dist_to_center / (max_dist + 1e-6)
            score = (1 - norm_dist) * 0.4 + compactness * 0.3 + region_prob * 0.3
            
            candidates.append((score, i, area, region_prob))
        
        if candidates:
            # 选择得分最高的区域作为瞳孔
            best_score, best_idx, best_area, best_prob = max(candidates, key=lambda x: x[0])
            filled_mask = (labels == best_idx).astype(np.uint8)

    # Step 4: 平滑处理（去除锯齿边缘）
    smoothed = cv2.medianBlur(filled_mask, 3)  # 中值滤波（去噪）
    blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (3, 3), 0)  # 高斯模糊（抗锯齿）
    
    return (blurred > 0.5).astype(np.uint8)


def infer_video(video_path, job_dir, threshold):
    """
    对视频逐帧执行瞳孔分割
    
    Args:
        video_path: 视频文件路径
        job_dir: 输出目录
        threshold: 二值化阈值
        
    Returns:
        tuple: (rows, overlay_path, preview_path, csv_path, summary)
            - rows: 每帧的结果列表
            - overlay_path: 完整预测视频路径
            - preview_path: GIF 预览路径
            - csv_path: CSV 时间序列路径
            - summary: 统计摘要
    """
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("视频尺寸读取失败")

    # 准备输出文件
    overlay_path = job_dir / f"{video_path.stem}_exp11_overlay.mp4"  # 完整预测视频
    preview_path = job_dir / f"{video_path.stem}_preview.gif"         # GIF 预览
    csv_path = job_dir / f"{video_path.stem}_pupil_params.csv"        # CSV 报告
    
    # 创建视频写入器
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("预测视频写入器打开失败")
    
    rows = []
    gif_frames = []
    
    # GIF 采样策略：每秒 8 帧，最多 180 帧（避免文件过大）
    gif_stride = max(1, int(round(fps / 8.0)))
    gif_max_frames = 180
    
    frame_idx = 0
    start = time.time()

    # 逐帧处理
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # 转换颜色空间（BGR → RGB）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        
        # 执行单帧预测
        result, mask = infer_image(image, threshold)
        
        # 生成叠加图（在原图上绘制 Mask 和瞳孔参数）
        overlay = make_overlay_bgr(frame, mask, result)
        writer.write(overlay)
        
        # 采样 GIF 帧
        if frame_idx % gif_stride == 0 and len(gif_frames) < gif_max_frames:
            gif_frames.append(to_gif_frame(overlay))
        
        # 记录结果
        rows.append(video_row(frame_idx, result))
        frame_idx += 1

    # 释放资源
    cap.release()
    writer.release()
    
    # 生成 GIF 预览
    if gif_frames:
        duration = int(1000 / 8)  # 每帧持续时间（毫秒）
        gif_frames[0].save(
            preview_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
    else:
        preview_path = None
    
    # 生成 CSV 报告
    write_rows_csv(csv_path, rows)

    # 计算统计信息
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary = summarize_rows(ok_rows)
    summary["frames"] = len(rows)
    summary["fps"] = round(len(rows) / max(time.time() - start, 1e-6), 2)  # 实际处理速度
    summary["preview_type"] = "gif" if preview_path else "mp4"
    
    return rows, overlay_path, preview_path, csv_path, summary


def to_gif_frame(frame_bgr):
    """
    将 BGR 帧转换为适合 GIF 的 RGB 图像（缩小尺寸）
    
    Args:
        frame_bgr: OpenCV BGR 帧（ndarray）
        
    Returns:
        PIL.Image: RGB 图像（最大宽度 720 像素）
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    
    # 限制最大宽度（减小 GIF 文件大小）
    max_width = 720
    if image.width > max_width:
        new_height = int(image.height * max_width / image.width)
        image = image.resize((max_width, new_height), Image.BILINEAR)
    
    return image


def save_overlay(image, mask, result, output_path):
    """
    保存叠加图（原图 + Mask + 瞳孔参数标注）
    
    Args:
        image: PIL.Image 对象（RGB）
        mask: 二值化 Mask（uint8）
        result: 瞳孔参数字典
        output_path: 输出路径
    """
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = make_overlay_bgr(bgr, mask, result)
    cv2.imwrite(str(output_path), overlay)


def make_overlay_bgr(frame_bgr, mask, result):
    """
    生成叠加图（在原图上绘制绿色 Mask 和瞳孔参数）
    
    Args:
        frame_bgr: OpenCV BGR 帧（ndarray）
        mask: 二值化 Mask（uint8）
        result: 瞳孔参数字典（或 None）
        
    Returns:
        ndarray: 叠加后的 BGR 图像
    """
    overlay = frame_bgr.copy()
    mask_u8 = (mask > 0).astype(np.uint8)
    
    # 创建绿色图层
    color_layer = np.zeros_like(overlay)
    color_layer[:, :, 1] = 220  # G 通道设为 220（绿色）
    
    # 混合：Mask 区域用 55% 原图 + 45% 绿色
    overlay = np.where(mask_u8[:, :, None] > 0, cv2.addWeighted(overlay, 0.55, color_layer, 0.45, 0), overlay)

    if result is None:
        # 未检测到瞳孔
        cv2.putText(overlay, "No pupil detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA)
        return overlay

    # 绘制椭圆轮廓（绿色）
    cv2.ellipse(overlay, result["ellipse"], (0, 255, 0), 2)
    
    # 绘制中心点（红色）
    center = (int(round(result["cx"])), int(round(result["cy"])))
    cv2.circle(overlay, center, 4, (0, 0, 255), -1)
    
    # 绘制文本信息（中心坐标 + 直径）
    cv2.putText(
        overlay,
        f"Center=({result['cx']:.1f}, {result['cy']:.1f}) D={diameter(result):.1f}px",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    
    return overlay


def result_to_row(filename, result):
    """
    将瞳孔参数转换为 CSV 行字典
    
    Args:
        filename: 文件名
        result: 瞳孔参数字典（或 None）
        
    Returns:
        dict: CSV 行数据
    """
    if result is None:
        return {"filename": filename, "status": "no_pupil"}
    
    return {
        "filename": filename,
        "status": "ok",
        "cx": round(result["cx"], 3),           # 中心 X
        "cy": round(result["cy"], 3),           # 中心 Y
        "center_x": round(result["cx"], 3),     # 兼容字段
        "center_y": round(result["cy"], 3),     # 兼容字段
        "diameter": round(diameter(result), 3), # 直径
        "diameter_px": round(diameter(result), 3),  # 兼容字段
        "major_axis": round(result["major_axis"], 3),   # 椭圆长轴
        "minor_axis": round(result["minor_axis"], 3),   # 椭圆短轴
        "area": round(result["area"], 3),       # 面积
        "confidence": round(result["confidence"], 4),   # 置信度
        "angle": round(result["angle"], 3),     # 椭圆角度
    }


def video_row(frame_idx, result):
    """
    将视频帧结果转换为 CSV 行字典
    
    Args:
        frame_idx: 帧索引
        result: 瞳孔参数字典
        
    Returns:
        dict: CSV 行数据（包含 frame 字段）
    """
    row = result_to_row(str(frame_idx), result)
    row["frame"] = frame_idx
    return row


def summarize_rows(rows):
    """
    计算多行结果的统计摘要（平均值）
    
    Args:
        rows: 结果列表
        
    Returns:
        dict: 统计摘要
    """
    if not rows:
        return {"status": "no_pupil"}
    
    keys = ["center_x", "center_y", "diameter_px", "major_axis", "minor_axis", "area", "confidence"]
    summary = {"status": "ok"}
    
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        if values:
            summary[key] = round(float(np.mean(values)), 3)
    
    # 添加兼容字段
    summary["cx"] = summary.get("center_x")
    summary["cy"] = summary.get("center_y")
    summary["diameter"] = summary.get("diameter_px")
    
    return summary


def diameter(result):
    """
    计算瞳孔直径（椭圆长短轴的平均值）
    
    Args:
        result: 瞳孔参数字典
        
    Returns:
        float: 直径（像素）
    """
    return float((result["major_axis"] + result["minor_axis"]) / 2.0)


def get_threshold():
    """
    从请求中获取二值化阈值
    
    Returns:
        float: 阈值（0.01-0.99，默认 0.31）
    """
    raw = request.form.get("threshold", request.args.get("threshold", "0.31"))
    try:
        value = float(raw)
    except ValueError:
        value = 0.31
    return min(max(value, 0.01), 0.99)


def make_job_dir(prefix):
    """
    创建任务输出目录（带时间戳，避免冲突）
    
    Args:
        prefix: 目录前缀（image/batch/video）
        
    Returns:
        Path: 任务目录路径
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 生成唯一目录名：prefix_YYYYMMDD_HHMMSS_mmm
    job_dir = OUTPUT_ROOT / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    
    return job_dir


def output_url(path):
    """
    生成文件的相对 URL（用于前端访问）
    
    Args:
        path: 文件绝对路径
        
    Returns:
        str: 相对 URL（如 /outputs/image_.../overlay.png）
    """
    rel = Path(path).resolve().relative_to(OUTPUT_ROOT.resolve())
    return "/outputs/" + "/".join(rel.parts)


def is_image_name(name):
    """
    检查文件名是否为支持的图片格式
    
    Args:
        name: 文件名
        
    Returns:
        bool: 是否为图片
    """
    return Path(name or "").suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def safe_name(name):
    """
    清理文件名（移除危险字符）
    
    Args:
        name: 原始文件名
        
    Returns:
        str: 安全的文件名
    """
    cleaned = secure_filename(name)
    return cleaned or "file"


def write_rows_csv(path, rows):
    """
    将结果列表写入 CSV 文件
    
    Args:
        path: CSV 文件路径
        rows: 结果列表
    """
    fieldnames = [
        "filename",
        "frame",
        "status",
        "center_x",
        "center_y",
        "diameter_px",
        "major_axis",
        "minor_axis",
        "area",
        "confidence",
        "angle",
        "error",
        "overlay_url",
    ]
    
    # 使用 utf-8-sig 编码（Excel 兼容）
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_zip(zip_path, items):
    """
    创建 ZIP 压缩包
    
    Args:
        zip_path: ZIP 文件路径
        items: 要压缩的文件/目录列表
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            item = Path(item)
            if item.is_file():
                zf.write(item, item.name)
            elif item.is_dir():
                for path in item.rglob("*"):
                    if path.is_file():
                        zf.write(path, str(path.relative_to(item.parent)))


def error(message, status):
    """
    生成错误响应
    
    Args:
        message: 错误消息
        status: HTTP 状态码
        
    Returns:
        tuple: (JSON 响应, 状态码)
    """
    return jsonify({"ok": False, "error": message}), status


@app.errorhandler(Exception)
def handle_exception(exc):
    """
    全局异常处理器
    
    Args:
        exc: 异常对象
        
    Returns:
        tuple: (JSON 响应, 500)
    """
    return error(str(exc), 500)


# ==================== 启动服务器 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 11 CBAM-ResUNet-ASPP web inference server")
    print(f"Project root: {ROOT}")
    print(f"Weight: {WEIGHT_PATH}")
    print(f"Device: {device}")
    print("Open desktop HTML, keep API address as: http://127.0.0.1:5000")
    print("=" * 70)
    
    # 启动 Flask 开发服务器
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
