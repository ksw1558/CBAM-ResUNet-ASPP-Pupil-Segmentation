import os
import sys
import datetime
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. 自动处理路径：确保能找到父目录下的 nets 和 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # --- 基础配置 ---
    Cuda = True
    seed = 11
    fp16 = True  # 开启混合精度，训练更快
    num_classes = 2
    backbone = "vgg"
    input_shape = [256, 256]

    # --- ⭐ 关键修改：实验 02 权重加载策略 ---
    # 由于结构改变，不再加载实验 01 的 pth 文件
    pretrained = True  # 设为 True 会加载 VGG16 的公共权重（不影响你的 Attention 层）
    model_path = ""  # 留空，不加载之前的 Unet 全模型权重

    # --- 训练阶段设置 ---
    Init_Epoch = 0
    Freeze_Epoch = 30  # 冻结阶段：先练注意力层和解码器
    UnFreeze_Epoch = 100  # 总 Epoch

    Freeze_batch_size = 8
    Unfreeze_batch_size = 4
    Freeze_Train = True

    # --- 优化器与保存路径 ---
    optimizer_type = "adam"
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    save_period = 10
    save_dir = 'logs'  # 对应 02_CBAM_UNet_Exp/logs
    eval_flag = True
    eval_period = 5

    # 数据集路径
    VOCdevkit_path = os.path.join(root_path, 'VOCdevkit')
    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 0

    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 实例化模型：此时会自动调用你修改后的 nets/unet.py (含 CA 模块)
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()

    # ⭐ 如果不加载 model_path，则对新加入的注意力层进行初始化
    if model_path == "":
        weights_init(model)

    # 3. 日志记录
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # 4. 数据加载
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 显示配置
    show_config(
        num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type,
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    # 5. 训练循环
    if True:
        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs = 16
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, 1e-4), 1e-4)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, 1e-6), 1e-6)

        optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(0.9, 0.999))
        lr_scheduler_func = get_lr_scheduler("cos", Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)

        eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                     eval_flag=eval_flag, period=eval_period)

        UnFreeze_flag = False
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                model.unfreeze_backbone()
                # 重新构建 Data Loader
                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=unet_dataset_collate)
                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          num_train // batch_size, num_val // batch_size, gen, gen_val, UnFreeze_Epoch, Cuda,
                          dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, 0)

        loss_history.writer.close()

