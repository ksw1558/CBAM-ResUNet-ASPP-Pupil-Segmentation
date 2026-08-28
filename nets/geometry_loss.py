import torch
import torch.nn as nn
import torch.nn.functional as F


class PupilGeometryLoss(nn.Module):
    """瞳孔几何感知组合损失。

    这个类本身是一个组合损失，内部包含 4 个子项：
    1. Tversky Loss：类似 Dice，但可以分别控制漏检 FN 和误检 FP 的惩罚。
    2. OHEM Focal Loss：重点学习 loss 最大的一部分困难像素。
    3. TV Loss：平滑预测概率图，减少边缘锯齿。
    4. Centroid Loss：约束预测瞳孔中心和真实瞳孔中心接近。

    在当前实验11的 train.py 中，w_tv=0.0、w_centroid=0.0，
    所以实际主要启用的是 Tversky Loss 和 OHEM Focal Loss。
    """

    def __init__(self, w_tversky=1.0, w_focal_ohem=8.0, w_tv=0.5, w_centroid=0.1,
                 alpha=0.3, beta=0.7, focal_alpha=0.25, focal_gamma=2.0,
                 smooth=1e-5, ohem_ratio=0.7, centroid_start_threshold=0.1,
                 centroid_max_dist=50.0):
        super(PupilGeometryLoss, self).__init__()

        # 各子损失的权重，由 train.py 传入。权重为 0 表示该项暂不参与总损失。
        self.w_tversky = w_tversky
        self.w_focal_ohem = w_focal_ohem
        self.w_tv = w_tv
        self.w_centroid = w_centroid

        # Tversky 参数：alpha 控制 FN 权重，beta 控制 FP 权重。
        self.tversky_alpha = alpha
        self.tversky_beta = beta
        self.smooth = smooth

        # Focal Loss 参数：alpha 平衡类别，gamma 放大困难样本权重。
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # OHEM 参数：只保留 loss 最大的前 ohem_ratio 比例像素。
        self.ohem_ratio = ohem_ratio
        self.centroid_start_threshold = centroid_start_threshold
        self.centroid_max_dist = centroid_max_dist

        self.current_epoch = 0

    def tversky_loss(self, inputs, target, smooth=1):
        """Tversky Loss：用于控制瞳孔区域的漏检和误检。

        Args:
            inputs: 模型输出 logits，形状为 [B, C, H, W]
            target: 真实类别标签，形状为 [B, H, W]，类别 0=背景，1=瞳孔
        """
        # 兼容 [B,1,H,W] 或 [B,3,H,W] 形式的标签，统一整理为 [B,H,W]。
        if target.dim() == 4:
            if target.shape[1] == 3:
                target = target[:, 0, :, :]
            elif target.shape[1] == 1:
                target = target.squeeze(1)

        target = target.long()

        # 将 logits 转为概率，只取第 1 类瞳孔概率参与计算。
        inputs = torch.softmax(inputs, dim=1)
        t = (target == 1).float()

        assert t.dim() == 3, f"t should be 3D [B,H,W], but got {t.shape}"
        assert inputs.dim() == 4, f"inputs should be 4D [B,C,H,W], but got {inputs.shape}"

        # TP: 正确预测为瞳孔；FP: 背景误预测为瞳孔；FN: 瞳孔漏检为背景。
        tp = (inputs[:, 1, :, :] * t).sum()
        fp = (inputs[:, 1, :, :] * (1 - t)).sum()
        fn = ((1 - inputs[:, 1, :, :]) * t).sum()

        tversky = (tp + smooth) / (tp + self.tversky_alpha * fn + self.tversky_beta * fp + smooth)

        # 指标越大越好，损失需要越小越好，所以返回 1 - Tversky。
        return 1 - tversky

    def focal_loss_ohem(self, inputs, target):
        """OHEM Focal Loss：在线困难样本挖掘 + Focal Loss。

        Focal Loss 让模型关注分错的像素；OHEM 再进一步只保留 loss 最大的一批像素。
        对瞳孔边缘、反光、遮挡等困难区域更有帮助。
        """
        # reduction='none' 保留每个像素自己的交叉熵，后面才能筛选困难像素。
        ce_loss = F.cross_entropy(inputs, target.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss_pixel = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss

        B, H, W = focal_loss_pixel.shape
        total_pixels = B * H * W

        # 计算需要保留的困难像素数量。
        n_hard = int(total_pixels * self.ohem_ratio)

        # 将所有像素 loss 展平并降序排序，取第 n_hard 大的值作为阈值。
        loss_flat = focal_loss_pixel.view(-1)
        sorted_loss, _ = torch.sort(loss_flat, descending=True)

        # 防止极端情况下索引越界。
        n_hard = max(1, min(n_hard, len(sorted_loss)))
        threshold = sorted_loss[n_hard - 1]

        # 只保留 loss 大于等于阈值的困难像素。
        hard_mask = (focal_loss_pixel >= threshold).float()

        # 对困难像素求平均，作为 OHEM Focal Loss。
        ohem_loss = (focal_loss_pixel * hard_mask).sum() / (hard_mask.sum() + 1e-8)

        return ohem_loss

    def tv_loss(self, inputs):
        """Total Variation Loss：平滑瞳孔概率图，减少边缘锯齿。"""
        probs = F.softmax(inputs, dim=1)[:, 1, :, :]
        diff_x = torch.pow(probs[:, :, :-1] - probs[:, :, 1:], 2)
        diff_y = torch.pow(probs[:, :-1, :] - probs[:, 1:, :], 2)
        return diff_x.mean() + diff_y.mean()

    def centroid_loss(self, inputs, target):
        """质心距离损失：约束预测瞳孔中心接近真实瞳孔中心。

        该项用于让预测结果的几何中心更稳定，便于后续瞳孔中心定位和直径分析。
        """
        probs = F.softmax(inputs, dim=1)[:, 1, :, :]

        # 兼容不同标签格式，统一为 [B,H,W]。
        if target.dim() == 4:
            if target.shape[1] == 3:
                target = target[:, 0, :, :]
            elif target.shape[1] == 1:
                target = target.squeeze(1)

        t = (target == 1).float()

        if t.dim() == 3:
            t = t.unsqueeze(1)

        H, W = probs.shape[2], probs.shape[3]

        # 构造每个像素的坐标网格，用概率加权平均得到预测质心。
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=probs.device),
            torch.arange(W, dtype=torch.float32, device=probs.device),
            indexing='ij'
        )
        y_coords = y_coords.view(1, 1, H, W)
        x_coords = x_coords.view(1, 1, H, W)

        pred_sum = probs.sum()
        if pred_sum < 1e-6:
            return torch.tensor(0.0, device=inputs.device)

        pred_centroid_x = (probs * x_coords).sum() / pred_sum
        pred_centroid_y = (probs * y_coords).sum() / pred_sum

        gt_sum = t.sum()
        if gt_sum < 1e-6:
            if pred_sum < 1e-6:
                return torch.tensor(0.0, device=inputs.device)
            else:
                return torch.tensor(float('inf'), device=inputs.device)

        gt_centroid_x = (t * x_coords).sum() / gt_sum
        gt_centroid_y = (t * y_coords).sum() / gt_sum

        centroid_dist = torch.sqrt(
            (pred_centroid_x - gt_centroid_x) ** 2 +
            (pred_centroid_y - gt_centroid_y) ** 2
        )

        # 限制最大距离，避免训练早期梯度过大。
        centroid_dist = torch.clamp(centroid_dist, max=self.centroid_max_dist)

        return centroid_dist

    def forward(self, inputs, target, epoch=None):
        """计算几何感知组合损失。

        Returns:
            total_loss, loss_tversky, loss_focal_ohem, loss_tv, loss_centroid
        """
        if epoch is not None:
            self.current_epoch = epoch

        # 1. Tversky Loss：控制前景区域的漏检和误检。
        loss_tversky = self.tversky_loss(inputs, target)

        # 2. OHEM Focal Loss：强化困难像素，特别是边缘和干扰区域。
        loss_focal_ohem = self.focal_loss_ohem(inputs, target)

        # 3. TV Loss：平滑概率图。当前实验11中该项权重为 0，仅保留代码扩展性。
        loss_tv = self.tv_loss(inputs)

        # 4. Centroid Loss：当分割已经比较稳定后再启用。当前实验11中权重为 0。
        loss_centroid = torch.tensor(0.0, device=inputs.device)

        if loss_tversky < self.centroid_start_threshold:
            try:
                loss_centroid = self.centroid_loss(inputs, target)
                if not torch.isfinite(loss_centroid):
                    loss_centroid = torch.tensor(0.0, device=inputs.device)
            except Exception:
                loss_centroid = torch.tensor(0.0, device=inputs.device)

        # 按权重合成总损失。训练循环只把 total_loss 继续叠加 Boundary/Dice/Focal。
        total_loss = (self.w_tversky * loss_tversky +
                      self.w_focal_ohem * loss_focal_ohem +
                      self.w_centroid * loss_centroid +
                      self.w_tv * loss_tv)

        return total_loss, loss_tversky, loss_focal_ohem, loss_tv, loss_centroid
