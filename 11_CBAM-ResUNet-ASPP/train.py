#训练脚本
import os
import sys
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
# 训练脚本位于实验11目录下，下面几行用于把项目根目录加入sys.path。
# 这样train.py可以直接导入根目录中的nets/和utils/公共模块。
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# 模型、损失函数和训练工具都拆分在不同模块中，训练入口只负责把它们组装起来。
from nets.cbam_res_unet_v5 import CBAMResUnetV5
from nets.geometry_loss import PupilGeometryLoss
from nets.boundary_loss import BoundaryLoss  # 【新增】导入Boundary Loss
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # 1. 基础训练配置：是否使用GPU、随机种子、类别数、输入尺寸等。
    # 本项目是二分类语义分割：背景=0，瞳孔=1。
    Cuda = True
    seed = 11
    fp16 = True
    num_classes = 2  #二分类
    backbone = "resnet50"  #主干特征提取网络
    input_shape = [320, 320]  #输入分辨率

    # 2. 权重配置：从最终模型权重继续微调，避免从零开始训练。
    pretrained = True
    model_path = "logs/final_exp11_miou98_16_epoch070.pth"

    # 3. 训练轮数配置：这里不做冻结阶段，直接全量微调到120epoch。
    Init_Epoch = 0
    Freeze_Epoch = 0
    UnFreeze_Epoch = 120

    # 320x320 输入和 ResNet50 编码器占显存较多，因此 batch size 设置较小。
    Freeze_batch_size = 4  # 【修改】降低batch size
    Unfreeze_batch_size = 2  # 【修改】降低batch size
    Freeze_Train = False

    # 4. 优化器、学习率和保存策略。
    optimizer_type = "adam"
    Init_lr = 1e-5
    Min_lr = Init_lr * 0.01
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    VOCdevkit_path = os.path.join(root_path, 'VOCdevkit')

    # 5. 分割损失配置：Dice/Focal 关注区域重叠和难样本，类别权重提高瞳孔类别的重要性。
    dice_loss = True
    focal_loss = True
    cls_weights = np.array([1.0, 2.0], np.float32)
    num_workers = 0

    seed_everything(seed)

    # 实验11模型。模型主体是ResNet50编码器+CBAM注意力+ASPP多尺度模块+U-Net解码器。
    model = CBAMResUnetV5(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()

    # 7. 几何约束损失：把 Tversky、Focal OHEM 等组合起来，提升小目标瞳孔区域的分割稳定性。
    # 【修改】新的Loss组合：Dice(0.5) + Focal(0.3) + Boundary(0.2)
    geometry_loss_fn = PupilGeometryLoss(
        w_tversky=0.5,      # 【修改】从1.0降到0.5
        w_focal_ohem=0.3,   # 【修改】从8.0降到0.3（相对权重）
        w_tv=0.0,           # 【修改】去掉TV Loss
        w_centroid=0.0,     # 【修改】去掉质心约束
        alpha=0.3,
        beta=0.7,
        ohem_ratio=0.7,
        centroid_start_threshold=0.1,
        centroid_max_dist=50.0
    )
    
    # 8. 边界损失：让模型更关注瞳孔边缘，方便后续椭圆拟合和直径计算。
    # 【新增】Boundary Loss
    boundary_loss_fn = BoundaryLoss(weight=0.2)

    # 9. 加载已有权重。这里只加载名字和张量形状都匹配的层，避免结构微调时加载失败。
    if model_path != "":
        print(f'Loading weights: {model_path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(model_path, map_location=device)
        if 'net' in pretrained_dict:
            pretrained_dict = pretrained_dict['net']
        
        # 只保留可以匹配当前模型结构的参数。
        # 适配不同分辨率的权重加载
        model_dict = model.state_dict()
        matched_dict = {k: v for k, v in pretrained_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
        
        missing_keys = set(model_dict.keys()) - set(matched_dict.keys())
        if missing_keys:
            print(f"⚠️ 以下层将从头初始化: {missing_keys}")
        
        model.load_state_dict(matched_dict, strict=False)
        print("✅ 权重加载成功（部分层重新初始化以适配320x320）")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_exp14_320x320_boundary_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    # 10. 混合精度训练：在支持 CUDA 的机器上减少显存占用并加快训练。
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    # 11. 多GPU封装和cuDNN加速。
    # 数据处理与并行化
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        model_train = model

    # 12. 读取 VOC 格式数据集划分文件，train.txt 用于训练，val.txt 用于验证和 mIoU 评估。
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, \
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    print(f"Train samples: {num_train}, Val samples: {num_val}")

    # 13. 解冻编码器，所有层都参与训练；如果显存不足，可以改为冻结 backbone 只训练解码器。
    # 直接进入全量训练阶段
    if Cuda:
        model_train.module.unfreeze_backbone()
    else:
        model_train.unfreeze_backbone()
    batch_size = Unfreeze_batch_size
    print("✅ 全量微调模式：所有层参数可更新")

    optimizer = optim.Adam(model.parameters(), Init_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=Min_lr)

    # 14. 数据加载器：负责读取图像和标签、resize 到 input_shape，并组织成 batch。
    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=unet_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)

    eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                 eval_flag=eval_flag, period=eval_period)

    print(f"🚀 实验14启动：320x320分辨率 + Boundary Loss")
    print(f"   Loss组合: Dice(0.5) + Focal(0.3) + Boundary(0.2)")
    print(f"   Batch Size: {batch_size} (降低以适配高分辨率)")
    
    # 15. 主训练循环：每个 epoch 更新学习率，调用 fit_one_epoch 完成训练、验证、保存权重和记录日志。
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        lr_scheduler.step()

        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                      num_train // batch_size, num_val // batch_size, gen, gen_val, UnFreeze_Epoch, Cuda,
                      dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir,
                      adaptive_loss_fn=geometry_loss_fn,
                      boundary_loss_fn=boundary_loss_fn)
        print(f"Epoch {epoch + 1}/{UnFreeze_Epoch} Finished.")

    loss_history.writer.close()
    print("Training Complete.")
