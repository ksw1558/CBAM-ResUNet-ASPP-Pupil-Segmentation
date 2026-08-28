# 基于 ACR-UNet 的瞳孔分割与参数提取

本科毕业设计项目，方向为医学图像分割与计算机视觉。项目目标是从眼部图像或视频中分割瞳孔区域，并进一步提取瞳孔中心坐标、长短轴和直径等参数。

## 项目概述

本项目以 U-Net 语义分割框架为基础，围绕瞳孔区域边界弱、反光干扰、睫毛遮挡和前景占比小等问题，设计并验证了最终模型 **ACR-UNet**。该模型融合了：

- CBAM 注意力机制：增强瞳孔区域的通道和空间响应。
- Residual 残差结构：提升深层网络训练稳定性。
- ASPP 多尺度上下文模块：增强不同瞳孔尺度和边界形态的适应能力。
- 边界与几何约束损失：改善瞳孔边界和中心定位质量。

最终模型位于 `11_CBAM-ResUNet-ASPP/`。本仓库只开源代码与项目说明，不直接上传模型权重和数据集；推荐权重文件名为：

```text
11_CBAM-ResUNet-ASPP/logs/final_exp11_miou98_16_epoch070.pth
```

## 最终结果

主实验数据集为 LPW 瞳孔数据集，工程中整理为 VOC2007 语义分割格式。目录名 `VOCdevkit/VOC2007/` 表示数据组织格式，不是 PASCAL VOC 自然图像数据集。

| 指标 | 数值 |
|---|---:|
| mIoU | 98.16% |
| Dice | 98.11% |
| Recall | 98.61% |
| Pixel Accuracy | 99.91% |

对比实验和消融实验结果在本地完整项目中保存于：

```text
teacher_report/ablation_plan/ablation_completed_results.csv
teacher_report/ablation_plan/ablation_teacher_final_results.csv
```

## 目录结构

```text
Pupil_Segmentation_Project/
├── nets/                         # 模型结构、注意力模块、损失函数
├── utils/                        # 数据加载、训练、评估和回调工具
├── 01_Base_Unet_Exp/             # 基础 U-Net 对比实验
├── 02_CBAM_UNet_Exp/             # CBAM-UNet 对比实验
├── 03_CBAM_UNet_Focal_Exp/       # CBAM-UNet + Focal Loss
├── 04_Attention_UNet_Exp/        # Attention U-Net 对比实验
├── 05_UNet_PlusPlus_Exp/         # UNet++ 对比实验
├── 06_DeepLabV3_Exp/             # DeepLabV3+ 对比实验
├── 07_PSPNet_Exp/                # PSPNet 对比实验
├── 08_ResUNet_Exp/               # ResUNet 对比实验
├── 09_CBAM_ResUNet_Exp/          # CBAM-ResUNet 对比实验
├── 10_CA-ResUNet V4/             # CA-ResUNet 对比实验
├── 11_CBAM-ResUNet-ASPP/         # ACR-UNet 最终模型
├── tools/                        # 论文、图表和实验整理辅助脚本
├── README.md                     # 项目主说明
├── DEPENDENCIES.md               # 外部库和环境说明
├── PROJECT_HANDOVER.md           # 项目交接说明
├── requirements.txt              # Python 依赖
└── requirements_research.txt     # 推荐研究环境依赖
```

未上传到 GitHub 的大文件目录包括：

```text
VOCdevkit/        # LPW 数据集的 VOC 格式版本
resources/        # 视频、补充数据和外部数据
teacher_report/   # 汇报材料、论文图表和部分实验结果
logs/             # 训练日志与权重
*.pth             # 模型权重
```

## 环境准备

推荐使用已有 GPU 环境：

| 项目 | 建议 |
|---|---|
| Python | 3.8 到 3.10 |
| CUDA | 11.6 |
| PyTorch | 1.13.1+cu116 |
| 环境名 | `pupil_cuda116` |

安装依赖：

```bash
conda activate pupil_cuda116
cd /d <project_root>
pip install -r requirements.txt
```

外部库和项目内部模块的区别见 `DEPENDENCIES.md`。

如果是从 GitHub 克隆本仓库，需另外准备数据集和权重文件。数据集应按下文 VOC2007 格式放置，最终模型权重应放到：

```text
11_CBAM-ResUNet-ASPP/logs/final_exp11_miou98_16_epoch070.pth
```

## 数据集格式

训练代码默认读取 VOC2007 格式数据：

```text
VOCdevkit/VOC2007/
├── JPEGImages/                 # 原始图像，.jpg
├── SegmentationClass/          # 二值 mask，.png，0=背景，1=瞳孔
└── ImageSets/Segmentation/
    ├── train.txt
    ├── val.txt
    ├── trainval.txt
    └── test.txt
```

如需重新生成划分：

```bash
python voc_annotation.py
```

注意：重新生成划分会影响复现实验指标，交付时建议保留当前 `train.txt` 和 `val.txt`。

## 常用操作

训练最终模型：

```bash
cd 11_CBAM-ResUNet-ASPP
python train.py
```

评估最终模型：

```bash
cd 11_CBAM-ResUNet-ASPP
python eval_final.py
```

可视化验证集预测：

```bash
cd 11_CBAM-ResUNet-ASPP
python visualize.py --num_samples 10
```

视频推理：

```bash
cd 11_CBAM-ResUNet-ASPP
python predict_videos_gpu.py --video_dir ../resources/videos
```

瞳孔直径时序分析：

```bash
cd 11_CBAM-ResUNet-ASPP
python analyze_video_pupil_diameter.py
```

## 实验结果汇总

| 实验 | 模型 | 角色 | mIoU |
|---|---|---|---:|
| 01 | Base U-Net | 基线模型 | 94.03% |
| 02 | CA-UNet | 注意力对比 | 95.13% |
| 03 | CBAM-UNet | 注意力对比 | 95.57% |
| 04 | Attention U-Net | 注意力门控对比 | 97.50% |
| 05 | UNet++ | 嵌套跳跃连接对比 | 97.27% |
| 08 | ResUNet | 残差结构对比 | 97.83% |
| 09 | CBAM-ResUNet | 残差 + 注意力 | 97.32% |
| 10 | CA-ResUNet | 坐标注意力对比 | 97.56% |
| 11 | ACR-UNet | 最终模型 | 98.16% |

完整表格以本地完整项目中的 `teacher_report/ablation_plan/ablation_completed_results.csv` 为准。GitHub 仓库未上传 `teacher_report/` 大文件目录。

## 交付建议

交给老师时建议重点说明以下文件：

| 路径 | 说明 |
|---|---|
| `README.md` | 项目总说明 |
| `PROJECT_HANDOVER.md` | 交付说明 |
| `DEPENDENCIES.md` | 环境和外部库说明 |
| `11_CBAM-ResUNet-ASPP/README_项目结构说明.md` | 最终模型目录说明 |
| `11_CBAM-ResUNet-ASPP/logs/final_exp11_miou98_16_epoch070.pth` | 最终推理权重，本地保存，未上传 GitHub |
| `teacher_report/ablation_plan/ablation_completed_results.csv` | 实验对比结果，本地保存，未上传 GitHub |

大体积外部数据集建议单独存放，不和项目代码一起上传 GitHub。OpenEDS 本地整理路径为：

```text
<external_dataset_dir>/OpenEDS_Kaggle
```

## 常见问题

**运行时报 `No module named nets` 或 `No module named utils` 怎么办？**  
请从项目根目录或实验目录运行脚本，不要把脚本单独复制到其他位置运行。

**为什么目录叫 VOCdevkit？**  
这是语义分割任务常用的数据组织格式。本项目中的数据是 LPW 瞳孔数据，不是 PASCAL VOC 自然图像数据。

**瞳孔直径单位是什么？**  
当前输出为像素单位。如需毫米单位，需要结合相机标定或实际成像比例换算。

**复现指标和 README 不一致怎么办？**  
优先检查最终权重、验证集划分和输入尺寸是否一致。当前最终模型权重为 `final_exp11_miou98_16_epoch070.pth`，输入尺寸为 320×320。

---

最后更新：2026-06-20
如果需要联系我，请发送邮箱：liu_kun226@163.com

