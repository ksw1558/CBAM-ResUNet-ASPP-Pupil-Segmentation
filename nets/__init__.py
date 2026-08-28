#nets 模块总览
#这个文件夹集中存放项目中各个语义分割网络、注意力模块、骨干网络和损失函数。
#答辩时可以把它理解成“模型结构库”：训练脚本只负责调用这里的模型，真正的网络结构都在 nets 中定义。

#一、基础骨干与基础模型
#vgg.py:VGG16 特征提取骨干，主要服务于早期的 U-Net、PSPNet、ResUNet 等实验。

#unet.py:基础 U-Net 分割模型，对应实验 01，也作为部分旧实验的 fallback 模型。

#res_unet.py:ResUNet 模块，核心是残差卷积块和带跳跃连接的上采样模块。实验11的最终模型也复用了其中的unetUp上采样结构。

#resnet50_dilated.py
    # 空洞卷积版ResNet50，用于扩大感受野，同时保留较高的特征图分辨率。
    # 实验11训练模型 cbam_res_unet_v5.py 会使用它作为编码器。


#二、注意力与多尺度模块
#cbam.py:CBAM注意力模块，包含通道注意力和空间注意力，用于突出瞳孔相关特征。

#attention.py:坐标注意力CoordAtt的实现，主要服务于Attention U-Net/CA类实验。

#ca.py:CoordAtt 的兼容导出文件，让旧代码可以通过nets.ca导入坐标注意力。

#aspp.py:ASPP空洞空间金字塔池化模块，通过不同dilation的卷积分支提取多尺度上下文。实验11中用它增强瓶颈层特征。


#三、各实验模型文件
#cbam_res_unet.py:CBAM + ResUNet 的基础实现，提供 ResCBAMBlock 和 unetUp 上采样模块。
    #实验 11 的最终推理模型会复用这里的CBAM上采样模块。

#cbam_res_unet_v2.py/cbam_res_unet_v3.py:旧版 CBAM-ResUNet 实验入口，主要用于实验09的兼容。

#cbam_res_unet_v5.py:实验 11 的训练模型版本，结构为 ResNet50Dilated + ASPP + CBAM + U-Net 解码器。

#cbam_res_unet_exp11.py:实验 11 的最终推理模型结构，用于加载 final_exp11_miou98_16_epoch070.pth。
    #答辩和演示时最应该重点介绍这个文件。

#ca_res_unet_v4.py
    #CA-ResUNet V4，对应实验10，使用坐标注意力增强特征表达。

#deeplabv3_plus.py：DeepLabV3+ 风格分割模型，对应实验 06。

#pspnet.py：PSPNet风格分割模型，对应实验07，使用金字塔池化聚合上下文。

#transunet.py：TransUNet实验入口，对应实验04。

#vm_unet.py：VM-UNet实验入口，对应实验05。

#ablation_unet_variants.py：消融实验模型，用于对比ASPP、CBAM、CA等模块的贡献。


#四、损失函数文件
#ce_loss.py：交叉熵损失，作为基础像素分类损失或备用损失。

#dice_loss.py：Dice Loss，关注预测区域和真实区域的重叠，适合瞳孔这种小目标分割。

#focal_loss.py：Focal Loss，降低简单背景像素权重，强化边缘、反光、遮挡等困难像素。

#boundary_loss.py：Boundary Loss，使用 Sobel 算子约束预测边缘和真实边缘一致。

#geometry_loss.py：实验 11 使用的瞳孔几何感知组合损失，主要包含 Tversky 和 OHEM Focal。

#adaptive_loss.py：旧消融实验使用的自适应组合损失。

#五、训练辅助文件
#unet_training.py：早期实验的训练辅助函数，包括权重初始化、学习率调度、旧版损失函数等。



#1. cbam_res_unet_exp11.py：最终模型结构。
#2. cbam.py：CBAM 注意力机制。
#3. aspp.py：多尺度上下文提取。
#4. geometry_loss.py、boundary_loss.py、dice_loss.py、focal_loss.py：实验 11 的损失函数。
#5. 其他模型文件只作为对比实验说明，例如 UNet、DeepLabV3+、PSPNet、TransUNet、VM-UNet。

