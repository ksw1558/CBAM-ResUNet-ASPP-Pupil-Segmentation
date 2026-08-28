import os
import sys
import datetime
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
    fp16 = True  # 开启混合精度，训练速度大幅提升
    num_classes = 2
    backbone = "vgg"
    input_shape = [256, 256]

    # --- ⭐ 汇报冲刺策略：权重加载与训练规划 ---
    pretrained = True  # 加载 VGG16 预训练权重
    # 实验 04 结构变动，建议 model_path 留空进行重新训练，以防 key 不匹配报错
    model_path = ""

    Init_Epoch = 0
    Freeze_Epoch = 30  # 建议增加到 30，让 CBAM 模块充分预热
    UnFreeze_Epoch = 100  # 必须回到 100，确保模型完全收敛

    Freeze_batch_size = 8
    Unfreeze_batch_size = 4
    Freeze_Train = True

    # --- 优化器与保存路径 ---
    optimizer_type = "adam"
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    save_period = 5
    save_dir = 'logs'
    eval_flag = True
    eval_period = 5

    # 数据集路径
    VOCdevkit_path = os.path.join(root_path, 'VOCdevkit')

    # ------------------------------------------------------------------#
    #   损失函数：开启 Focal Loss 处理难分样本，增加瞳孔权重
    # ------------------------------------------------------------------#
    dice_loss = True
    focal_loss = True
    cls_weights = np.array([1.0, 5.0], np.float32)  # 瞳孔权重提升至 5 倍
    num_workers = 0

    seed_everything(seed)

    # 2. 实例化模型：此时已包含 CBAM
    # 此时会自动调用你修改后的 nets/unet.py
    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()

    # 初始化新模块权重（CBAM 层会被 Kaiming 初始化）
    if model_path == "":
        weights_init(model)
    else:
        # 如果有特定权重加载需求，使用非严格加载
        print(f'Loading weights from {model_path}...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        # 只加载形状匹配的权重（跳过 CBAM）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # 3. 日志记录准备
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

    # 4. 读取数据集索引
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 打印配置信息，方便汇报截图
    show_config(
        num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, \
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    # 5. 训练循环
    if True:
        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 优化器设置
        optimizer = optim.Adam(model.parameters(), Init_lr, betas=(0.9, 0.999), weight_decay=1e-4)
        lr_scheduler_func = get_lr_scheduler("cos", Init_lr, Min_lr, UnFreeze_Epoch)

        # 数据加载器
        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)

        # 验证评估回调
        eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                     eval_flag=eval_flag, period=eval_period)

        UnFreeze_flag = False
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 自动解冻逻辑
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                model.unfreeze_backbone()
                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=unet_dataset_collate)
                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 单代训练
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          num_train // batch_size, num_val // batch_size, gen, gen_val, UnFreeze_Epoch, Cuda,
                          dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, 0)

        loss_history.writer.close()
