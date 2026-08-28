import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from nets.ablation_unet_variants import AblationResUnet
from nets.geometry_loss import PupilGeometryLoss
from nets.boundary_loss import BoundaryLoss
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config
from utils.utils_fit import fit_one_epoch


VARIANT_MAP = {
    "full": "full",
    "no_cbam": "no_cbam",
    "no_aspp": "no_aspp",
    "se": "se",
    "ppm": "ppm",
    "no_centroid": "full",
    "no_edge": "full",
    "basic_loss": "full",
}


class SimpleWriter:
    def close(self):
        return None


class SimpleLossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        self.writer = SimpleWriter()
        os.makedirs(self.log_dir, exist_ok=True)

    def append_loss(self, epoch, loss, val_loss=None):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), "a", encoding="utf-8") as f:
            f.write(f"{loss}\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), "a", encoding="utf-8") as f:
            f.write(f"{val_loss}\n")


class NoEvalCallback:
    def on_epoch_end(self, epoch, model):
        return None


def load_partial_weights(model, weight_path):
    if not weight_path or not os.path.exists(weight_path):
        print(f"No initial weight loaded: {weight_path}")
        return
    checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        checkpoint = checkpoint["net"]
    model_dict = model.state_dict()
    matched = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and tuple(v.shape) == tuple(model_dict[k].shape)
    }
    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded {len(matched)}/{len(model_dict)} tensors from {weight_path}")


def build_losses(name):
    if name == "basic_loss":
        return None, None, True, False

    w_centroid = 0.0 if name == "no_centroid" else 0.1
    geometry_loss_fn = PupilGeometryLoss(
        w_tversky=1.0,
        w_focal_ohem=8.0,
        w_tv=0.0,
        w_centroid=w_centroid,
        alpha=0.3,
        beta=0.7,
        ohem_ratio=0.7,
        centroid_start_threshold=0.1,
        centroid_max_dist=50.0,
    )
    boundary_loss_fn = None
    if name not in ("no_edge",):
        boundary_loss_fn = BoundaryLoss(weight=0.2)
    return geometry_loss_fn, boundary_loss_fn, False, False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=sorted(VARIANT_MAP.keys()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-pretrained-load", action="store_true")
    parser.add_argument("--eval-period", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    cuda = torch.cuda.is_available()
    if not cuda:
        print("WARNING: CUDA is not available. Training will be very slow.")

    num_classes = 2
    input_shape = [args.input_size, args.input_size]
    vocdevkit_path = str(ROOT / "VOCdevkit")
    save_dir = ROOT / "teacher_report_assets" / "ablation_plan" / "trained_variants" / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    model_variant = VARIANT_MAP[args.name]
    model = AblationResUnet(num_classes=num_classes, pretrained=False, variant=model_variant).train()
    if not args.no_pretrained_load:
        load_partial_weights(model, str(ROOT / "11_CBAM-ResUNet-ASPP" / "logs" / "final_exp11_miou98_16_epoch070.pth"))

    geometry_loss_fn, boundary_loss_fn, dice_loss, focal_loss = build_losses(args.name)
    cls_weights = np.array([1.0, 2.0], np.float32)

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    log_dir = save_dir / f"loss_{args.name}_{time_str}"
    loss_history = SimpleLossHistory(str(log_dir))

    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    else:
        model_train = model

    with open(ROOT / "VOCdevkit" / "VOC2007" / "ImageSets" / "Segmentation" / "train.txt", "r") as f:
        train_lines = f.readlines()
    with open(ROOT / "VOCdevkit" / "VOC2007" / "ImageSets" / "Segmentation" / "val.txt", "r") as f:
        val_lines = f.readlines()

    show_config(
        num_classes=num_classes,
        backbone="resnet50",
        model_path="final_exp11_miou98_16_epoch070.pth",
        input_shape=input_shape,
        Init_Epoch=0,
        Freeze_Epoch=0,
        UnFreeze_Epoch=args.epochs,
        Freeze_batch_size=args.batch_size,
        Unfreeze_batch_size=args.batch_size,
        Freeze_Train=False,
        Init_lr=args.lr,
        Min_lr=args.lr * 0.01,
        optimizer_type="adam",
        save_period=5,
        save_dir=str(save_dir),
        num_workers=args.num_workers,
        num_train=len(train_lines),
        num_val=len(val_lines),
    )

    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)

    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, vocdevkit_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, vocdevkit_path)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=cuda, drop_last=True, collate_fn=unet_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=cuda, drop_last=True, collate_fn=unet_dataset_collate)
    eval_callback = NoEvalCallback()

    scaler = torch.cuda.amp.GradScaler() if args.fp16 and cuda else None
    print(f"Start ablation training: {args.name}, model_variant={model_variant}, epochs={args.epochs}")
    for epoch in range(args.epochs):
        fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer, epoch,
            len(train_lines) // args.batch_size, len(val_lines) // args.batch_size,
            gen, gen_val, args.epochs, cuda, dice_loss, focal_loss, cls_weights,
            num_classes, bool(scaler), scaler, 5, str(save_dir),
            adaptive_loss_fn=geometry_loss_fn,
            boundary_loss_fn=boundary_loss_fn,
        )
        lr_scheduler.step()
    torch.save(model.state_dict(), save_dir / "final_weights.pth")
    loss_history.writer.close()
    print(f"Training complete: {save_dir / 'final_weights.pth'}")


if __name__ == "__main__":
    main()
