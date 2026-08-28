# 项目外部库说明

## 什么是外部库
外部库指的是项目运行时需要从 Python 环境中安装的第三方包，不是本项目自己写的代码。
本项目中：

- `torch`、`torchvision`、`numpy`、`opencv-python`、`Pillow`、`matplotlib` 等属于外部库。
- `nets/`、`utils/`、`unet.py`、各实验目录里的 `train.py` 属于项目内部代码，不需要用 `pip install` 安装。
- `VOCdevkit/`、`resources/`、`model_data/`、各实验 `logs/` 属于数据、权重或实验资源，也不是外部库。

## 推荐环境
本项目训练和推理建议使用已有 GPU 环境：
| 项目 | 建议 |
|---|---|
| Python | 3.8 到 3.10 更稳妥 |
| CUDA | 11.6 |
| PyTorch | 1.13.1+cu116 |
| 环境名 | `pupil_cuda116` |

当前项目的代码和权重主要按 `torch==1.13.1+cu116` 整理。不要随意升级到新的 PyTorch 大版本，否则可能出现权重加载、CUDA、算子兼容问题。

## 安装方式
在 Anaconda Prompt 或 PowerShell 中进入项目目录：

```bash
conda activate pupil_cuda116
cd /d <project_root>
pip install -r requirements.txt
```

`requirements.txt` 已包含 PyTorch CUDA 11.6 的下载源：

```text
--extra-index-url https://download.pytorch.org/whl/cu116
```

如果你已经在 `pupil_cuda116` 环境中装好了 PyTorch，通常不需要重复安装。
## 依赖分类
| 类别 | 外部库 | 项目用途 |
|---|---|---|
| 深度学习 | `torch`, `torchvision`, `torchaudio` | 模型训练、权重加载、GPU 推理 |
| 数值计算 | `numpy`, `scipy` | 数据处理、指标计算、曲线平滑 |
| 图像处理 | `opencv-python`, `Pillow` | 图像读取、mask 后处理、椭圆拟合、视频处理 |
| 进度显示 | `tqdm` | 训练、验证、批量推理进度条 |
| 绘图分析 | `matplotlib`, `pandas`, `seaborn` | 损失曲线、指标表、瞳孔直径时序图 |
| Web 演示 | `flask`, `werkzeug` | `web_inference_server.py` 的网页推理服务 |
| 论文工具 | `python-docx`, `PyMuPDF`, `pypdf` | 论文文档和参考资料处理 |

## 项目内部模块

这些目录和文件是本项目源码，不属于外部库：

| 路径 | 作用 |
|---|---|
| `nets/` | 模型结构、注意力模块、损失函数 |
| `utils/` | 数据加载、训练循环、评价指标、回调函数 |
| `unet.py` | 推理封装类 |
| `01_Base_Unet_Exp/` 到 `11_CBAM-ResUNet-ASPP/` | 各实验训练、评估、可视化脚本 |
| `tools/` | 论文、消融实验、图表生成等辅助脚本 |

运行脚本时如果报 `ModuleNotFoundError: No module named 'nets'` 或 `No module named 'utils'`，通常不是外部库没装，而是运行位置不对。应在项目根目录运行脚本：

```bash
cd /d <project_root>
python 11_CBAM-ResUNet-ASPP\train.py
```

## 不建议放进项目的内容
以下内容不应复制进项目目录，应放在 Conda/Python 环境中：
- `site-packages/`
- `venv/`、`.venv/`、`env/`
- `Lib/`
- `Scripts/`
- 手动下载的第三方库源码，除非明确要修改其源码
如果这些目录被放进项目，会让项目体积变大，也容易造成导入混乱。
