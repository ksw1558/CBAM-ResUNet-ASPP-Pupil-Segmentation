# 毕业设计项目交付说明

项目名称：基于 ACR-UNet 的瞳孔分割与参数提取

交付日期：2026-06-20

## 老师优先查看

| 路径 | 说明 |
|---|---|
| `README.md` | 项目总体说明、结果、运行方式 |
| `11_CBAM-ResUNet-ASPP/` | 最终模型目录 |
| `11_CBAM-ResUNet-ASPP/README_项目结构说明.md` | 最终模型使用说明 |
| `11_CBAM-ResUNet-ASPP/logs/final_exp11_miou98_16_epoch070.pth` | 最终权重 |
| `teacher_report/ablation_plan/ablation_completed_results.csv` | 对比实验和消融实验结果 |
| `DEPENDENCIES.md` | 环境与外部库说明 |

## 最终模型

最终模型为 `ACR-UNet`，对应实验目录：

```text
11_CBAM-ResUNet-ASPP/
```

模型结构：

```text
ResNet50 Encoder + ASPP + CBAM + U-Net Decoder
```

推荐权重：

```text
11_CBAM-ResUNet-ASPP/logs/final_exp11_miou98_16_epoch070.pth
```

## 最终指标

| 指标 | 数值 |
|---|---:|
| mIoU | 98.16% |
| Dice | 98.11% |
| Recall | 98.61% |
| Pixel Accuracy | 99.91% |

## 运行方式

安装环境：

```bash
conda activate pupil_cuda116
pip install -r requirements.txt
```

评估最终模型：

```bash
cd 11_CBAM-ResUNet-ASPP
python eval_final.py
```

可视化：

```bash
python visualize.py --num_samples 10
```

## 数据说明

项目内 `VOCdevkit/VOC2007/` 是 LPW 瞳孔数据的 VOC 格式版本：

```text
JPEGImages/          原始图像
SegmentationClass/   二值瞳孔 mask
ImageSets/           训练/验证/测试划分
```

OpenEDS 大数据集已单独整理到：

```text
<external_dataset_dir>/OpenEDS_Kaggle
```

不建议把 OpenEDS 和项目主体一起压缩交付。

## 注意事项

1. 当前项目不是有效 Git 仓库，建议以文件夹或压缩包方式交付。
2. 若重新运行 `voc_annotation.py`，训练/验证划分会变化，可能导致指标和 README 中记录不完全一致。
3. 大权重文件体积较大，如通过网络提交，可只保留最终权重和必要结果文件。
