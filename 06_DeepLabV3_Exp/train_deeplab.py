import os
import sys
import datetime
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

# 获取当前脚本所在文件夹的父目录（即项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))

# 将根目录加入 Python 的搜索路径
if root_path not in sys.path:
    sys.path.append(root_path)

from nets.deeplabv3_plus import DeepLabV3Plus
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda: 有显卡必开
    # ---------------------------------#
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = True

    # -----------------------------------------------------#
    #   num_classes: 瞳孔分割设为 2 (背景 + 瞳孔)
    # -----------------------------------------------------#
    num_classes = 2

    # -----------------------------------------------------#
    #   从零训练，pretrained_backbone 设为 False
    # -----------------------------------------------------#
    pretrained_backbone = False
    model_path = ""

    # 瞳孔图像 [256, 256]
    input_shape = [256, 256]

    # ------------------------------------------------------------------#
    #   训练阶段设置
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 30
    Freeze_batch_size = 8
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4

    Freeze_Train = True

    # 优化器设置
    optimizer_type = "adam"
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    save_period = 10

    # 确保日志目录是绝对路径
    save_dir = os.path.join(current_dir, 'logs')

    eval_flag = True
    eval_period = 5

    # 数据集路径
    VOCdevkit_path = os.path.join(root_path, 'VOCdevkit')

    # ------------------------------------------------------------------#
    #   损失函数：开启 Dice Loss 处理小目标
    # ------------------------------------------------------------------#
    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 0

    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    # 实例化模型
    model = DeepLabV3Plus(num_classes=num_classes, pretrained_backbone=pretrained_backbone).train()

    # 初始化权重
    if not pretrained_backbone:
        weights_init(model)

    # 记录日志
    if local_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = False
        cudnn.deterministic = True
        model_train = model_train.cuda()




    # 读取数据集
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr,
            optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers,
            num_train=num_train, num_val=num_val
        )

    # 训练循环
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练。")

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path,
                                         log_dir, Cuda, eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小。")

                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                                 worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                                     worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
                          dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                          save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
