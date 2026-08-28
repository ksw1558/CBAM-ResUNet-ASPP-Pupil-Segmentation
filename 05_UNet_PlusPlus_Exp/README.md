# Experiment 05 - UNet++

This experiment uses a UNet++ / Nested U-Net architecture with dense skip pathways.

Reference implementation style:
- 4uiiurz1/pytorch-nested-unet, UNet++ PyTorch implementation.

Main files:

```text
train.py                         # train UNet++ on LPW dataset in VOC format
eval_05_unetplusplus_lpw.py      # evaluate best_epoch_weights.pth on LPW validation set
```

Latest LPW validation result:

```text
N masks: 327
mIoU: 97.27%
Pupil IoU: 94.67%
Dice: 97.26%
Recall: 97.18%
Precision: 97.35%
Center Error: 1.77 px
Best training-curve mIoU: 97.55%
```

Expected weight files after retraining:

```text
logs/best_epoch_weights.pth
logs/last_epoch_weights.pth
logs/final_weights.pth
```

Old VM-UNet-related scripts were removed because they did not match this experiment directory.
