import os
import sys
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# 1. 导入 V3 模型 (架构不变)
from nets.cbam_res_unet_v3 import CBAMResUnetV3
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    seed = 11
    fp16 = True
    num_classes = 2
    backbone = "vgg"
    input_shape = [256, 256]

    pretrained = True
    # 2. 加载 V3 的最佳权重进行微调
    model_path = os.path.join(root_path, '09_CBAM_ResUNet_Exp', 'logs', 'best_epoch_weights.pth')

    # 3. 训练参数设置 (针对微调优化)
    Init_Epoch = 0
    Freeze_Epoch = 5  # 仅冻结 5 轮，因为主要目的是调 Loss 权重
    UnFreeze_Epoch = 30  # 跑 30 轮足够了

    Freeze_batch_size = 8
    Unfreeze_batch_size = 4
    Freeze_Train = True

    optimizer_type = "adam"
    Init_lr = 1e-5  # 学习率设为 1e-5，非常小的微调步长，保护已有特征
    Min_lr = Init_lr * 0.01
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    VOCdevkit_path = os.path.join(root_path, 'VOCdevkit')

    dice_loss = True
    focal_loss = True

    # 4. 【方案二核心修改】：降低瞳孔权重，强迫模型关注边缘细节
    cls_weights = np.array([1.0, 2.0], np.float32)

    num_workers = 0

    seed_everything(seed)

    # 5. 实例化模型
    model = CBAMResUnetV3(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()

    if model_path == "":
        print('⚠️ 未设置权重路径，将从头开始训练')
    else:
        print(f'🚀 正在加载 V3 权重进行 Loss 权重微调: {model_path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(model_path, map_location=device)
        if 'net' in pretrained_dict:
            pretrained_dict = pretrained_dict['net']

        model.load_state_dict(pretrained_dict, strict=False)
        print(f"✅ 成功加载参数，开始微调 Loss Weights。")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_v3_weights_tune_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    # 数据处理与并行化
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        model_train = model

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

    if True:
        if Freeze_Train:
            if hasattr(model, 'freeze_backbone'):
                if Cuda:
                    model_train.module.freeze_backbone()
                else:
                    model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        optimizer = optim.Adam(model.parameters(), Init_lr, betas=(0.9, 0.999), weight_decay=1e-4)
        lr_scheduler_func = get_lr_scheduler("cos", Init_lr, Min_lr, UnFreeze_Epoch)

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
                if Cuda:
                    model_train.module.unfreeze_backbone()
                else:
                    model_train.unfreeze_backbone()
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
