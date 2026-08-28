# 实验 11：ACR-UNet 最终模型说明

本目录是毕业设计最终模型 `ACR-UNet` 的主要交付目录。模型结构为 **ResNet50 编码器 + ASPP 多尺度模块 + CBAM 注意力 + U-Net 解码器**，用于瞳孔二分类分割和后续瞳孔参数提取。

## 核心文件

| 文件或目录 | 作用 |
|---|---|
| `train.py` | 最终模型训练脚本 |
| `eval_final.py` | 加载最终权重并在 VOC 验证集上计算 mIoU、Dice、Recall、Precision |
| `visualize.py` | 生成验证集分割可视化图 |
| `visualize_LPW.py` | LPW 数据集可视化脚本 |
| `predict_videos_gpu.py` | 视频批量分割推理 |
| `extract_pupil_params.py` | 基于 mask 提取瞳孔中心、长轴、短轴、角度 |
| `calculate_video_pupil_diameter.py` | 对视频逐帧计算瞳孔直径 |
| `analyze_video_pupil_diameter.py` | 生成瞳孔直径时序分析图和统计结果 |
| `plot_confidence_center_error.py` | 置信度与中心误差分析 |
| `logs/` | 最终权重和训练日志 |
| `miou_out/` | 评估时生成的预测 mask |
| `pupil_params_results/` | 瞳孔中心和几何参数结果 |
| `video_predictions/` | 视频推理输出 |
| `video_diameter_analysis/` | 视频直径分析结果 |

## 最终权重

推荐使用：

```text
logs/final_exp11_miou98_16_epoch070.pth
```

该权重对应项目最终汇报指标：

| 指标 | 数值 |
|---|---:|
| mIoU | 98.16% |
| Dice | 98.11% |
| Recall | 98.61% |
| Pixel Accuracy | 99.91% |

## 训练

从本目录运行：

```bash
python train.py
```

训练脚本默认读取项目根目录下的：

```text
../VOCdevkit/VOC2007/
```

输入尺寸为 `320 × 320`。如果要换数据集，需要修改 `train.py` 中的 `VOCdevkit_path`。

## 评估

```bash
python eval_final.py
```

该脚本会：

1. 加载 `logs/final_exp11_miou98_16_epoch070.pth`。
2. 读取 `VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt`。
3. 将预测结果保存到 `miou_out/detection-results/`。
4. 输出 mIoU、Pupil IoU、Dice、Recall、Precision 和 Accuracy。

## 可视化

```bash
python visualize.py --num_samples 10
```

输出目录：

```text
visualize_results/
```

## 视频推理与瞳孔直径分析

视频推理：

```bash
python predict_videos_gpu.py --video_dir ../resources/videos
```

瞳孔参数提取：

```bash
python extract_pupil_params.py
```

瞳孔直径时序分析：

```bash
python analyze_video_pupil_diameter.py
```

## 模型代码位置

训练结构：

```text
../nets/cbam_res_unet_v5.py
```

最终推理结构：

```text
../nets/cbam_res_unet_exp11.py
```

保留两个结构文件是为了兼容训练阶段和最终权重加载阶段的命名差异。

## 交付说明

交给老师时，本目录中最重要的是：

```text
train.py
eval_final.py
visualize.py
predict_videos_gpu.py
logs/final_exp11_miou98_16_epoch070.pth
pupil_params_results/
video_diameter_analysis/
```

`miou_out/` 和 `visualize_results/` 属于可重新生成的中间结果，不是必须保留项。
