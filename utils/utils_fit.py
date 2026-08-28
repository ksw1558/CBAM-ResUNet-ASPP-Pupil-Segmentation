import os
import math
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
from nets.focal_loss import Focal_Loss
from nets.dice_loss import Dice_Loss
from nets.ce_loss import CE_Loss


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                  epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, 
                  cls_weights, num_classes, fp16, scaler, save_period, save_dir,
                  adaptive_loss_fn=None, boundary_loss_fn=None):
    # 实验11的总损失由四类损失叠加：
    # 1. adaptive_loss_fn: PupilGeometryLoss，内部主要启用 Tversky + OHEM Focal。
    # 2. boundary_loss_fn: BoundaryLoss，约束预测边缘和真实边缘一致。
    # 3. Dice_Loss: 提高预测区域和真实瞳孔区域的重叠度。
    # 4. Focal_Loss: 强化边缘、反光、遮挡等困难像素。
    dice_loss_fn = Dice_Loss(num_classes=num_classes) if dice_loss else None
    focal_loss_fn = Focal_Loss() if focal_loss else None
    
    total_loss = 0
    total_loss_val = 0
    total_f_score = 0
    nans = 0

    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()
                
                # === 源头清洗 ===
                if labels.dim() == 4:
                    if labels.shape[1] == 3:
                        labels = labels[:, 0, :, :]
                    elif labels.shape[1] == 1:
                        labels = labels.squeeze(1)

            optimizer.zero_grad()
            if not fp16:
                outputs = model_train(imgs)
                
                if adaptive_loss_fn is not None:
                    loss_components = adaptive_loss_fn(outputs, pngs)
                    loss = loss_components[0] if isinstance(loss_components, tuple) else loss_components
                else:
                    ce_loss_fn = CE_Loss(weight=weights, num_classes=num_classes)
                    loss = ce_loss_fn(outputs, pngs)
                
                if boundary_loss_fn is not None:
                    pred_prob = torch.softmax(outputs, dim=1)
                    boundary_loss = boundary_loss_fn(pred_prob, pngs)
                    loss = loss + boundary_loss
                
                if dice_loss and dice_loss_fn is not None:
                    main_dice = dice_loss_fn(outputs, pngs)
                    loss = loss + main_dice
                
                if focal_loss and focal_loss_fn is not None:
                    main_focal = focal_loss_fn(outputs, pngs)
                    loss = loss + main_focal

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite training loss at epoch {epoch + 1}, iteration {iteration + 1}. "
                        "Stop training and lower the learning rate or disable mixed precision."
                    )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model_train(imgs)
                    
                    if adaptive_loss_fn is not None:
                        loss_components = adaptive_loss_fn(outputs, pngs)
                        loss = loss_components[0] if isinstance(loss_components, tuple) else loss_components
                    else:
                        ce_loss_fn = CE_Loss(weight=weights, num_classes=num_classes)
                        loss = ce_loss_fn(outputs, pngs)
                    
                    if boundary_loss_fn is not None:
                        pred_prob = torch.softmax(outputs, dim=1)
                        boundary_loss = boundary_loss_fn(pred_prob, pngs)
                        loss = loss + boundary_loss
                    
                    if dice_loss and dice_loss_fn is not None:
                        main_dice = dice_loss_fn(outputs, pngs)
                        loss = loss + main_dice
                    
                    if focal_loss and focal_loss_fn is not None:
                        main_focal = focal_loss_fn(outputs, pngs)
                        loss = loss + main_focal

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite training loss at epoch {epoch + 1}, iteration {iteration + 1}. "
                            "Stop training and lower the learning rate or disable mixed precision."
                        )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            
            with torch.no_grad():
                _f_score = f_score(outputs, pngs)
                if torch.isnan(_f_score):
                    nans += 1
                else:
                    total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score': total_f_score / (iteration + 1 - nans),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

                outputs = model_train(imgs)
                
                if adaptive_loss_fn is not None:
                    loss_components = adaptive_loss_fn(outputs, pngs)
                    val_loss = loss_components[0] if isinstance(loss_components, tuple) else loss_components
                else:
                    ce_loss_fn = CE_Loss(weight=weights, num_classes=num_classes)
                    val_loss = ce_loss_fn(outputs, pngs)
                
                if boundary_loss_fn is not None:
                    pred_prob = torch.softmax(outputs, dim=1)
                    boundary_loss = boundary_loss_fn(pred_prob, pngs)
                    val_loss = val_loss + boundary_loss
                
                if dice_loss and dice_loss_fn is not None:
                    main_dice = dice_loss_fn(outputs, pngs)
                    val_loss = val_loss + main_dice
                
                if focal_loss and focal_loss_fn is not None:
                    main_focal = focal_loss_fn(outputs, pngs)
                    val_loss = val_loss + main_focal

                if not torch.isfinite(val_loss):
                    raise RuntimeError(
                        f"Non-finite validation loss at epoch {epoch + 1}, iteration {iteration + 1}. "
                        "The current model has likely diverged."
                    )

                total_loss_val += val_loss.item()

            pbar.set_postfix(**{'total_loss_val': total_loss_val / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('Finish Validation')
    
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, total_loss_val / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1, model)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, total_loss_val / epoch_step_val))
    
    if save_period and save_period > 0 and (epoch + 1) % save_period == 0 and epoch + 1 != Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, total_loss_val / epoch_step_val)))

    if epoch + 1 == Epoch:
        print('Save final model to final_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "final_weights.pth"))

    if len(loss_history.val_loss) <= 1 or (loss_history.val_loss[-1] < min(loss_history.val_loss[:-1])):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
