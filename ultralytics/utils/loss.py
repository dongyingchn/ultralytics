# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing
    on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        self.with_3d = False
        if hasattr(m, "cv4"):
            self.with_3d = True

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        # feats = preds[1] if isinstance(preds, tuple) else preds
        if self.with_3d:
            feats = preds[0] if isinstance(preds, tuple) else preds
        else:
            feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        if "ignore" in batch:  # If ignore column is provided
            targets = torch.cat((targets, batch["ignore"].view(-1, 1)), 1)
        else:
            targets = torch.cat((targets, torch.zeros(targets.shape[0], 1, device=targets.device)), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        gt_labels, gt_bboxes, gt_ignore = targets.split((1, 4, 1), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ---- ignore 标签掩码构建 ----
        # gt_ignore: [batch, n_gts, 1]
        gt_ignore = gt_ignore.squeeze(-1)  # [batch, n_gts]
        ignore_mask = torch.zeros_like(fg_mask, dtype=torch.bool)  # [batch, n_anchors]
        for b in range(batch_size):
            if gt_ignore.shape[1] == 0:
                continue
            idx = target_gt_idx[b][fg_mask[b]].long().clamp(0, gt_ignore.shape[1] - 1)
            ignore_mask[b, fg_mask[b]] = gt_ignore[b, idx].bool()
        # loss_mask: 所有 anchor，正样本且未被ignore
        loss_mask = fg_mask & (~ignore_mask)

        # ---- 类别损失，所有anchor都计算，但被ignore的正样本不反传loss ----
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # [batch, n_anchors, n_cls]
        bce_loss[ignore_mask] = 0.0  # 被ignore的正样本损失屏蔽
        loss[1] = bce_loss.sum() / target_scores_sum

        # ---- bbox 和 dfl 损失，仅对未被ignore的正样本计算 ----
        if loss_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                loss_mask,
            )
        # original loss
        # # Cls loss
        # # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # # Bbox loss
        # if fg_mask.sum():
        #     loss[0], loss[2] = self.bbox_loss(
        #         pred_distri,
        #         pred_bboxes,
        #         anchor_points,
        #         target_bboxes / stride_tensor,
        #         target_scores,
        #         target_scores_sum,
        #         fg_mask,
        #     )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]

# ======== 3D Detection Loss (完全向量化版本，新增) ========
    
def stats(x): return (x.mean().item(), x.std().item(), x.min().item(), x.max().item())

class v8Detection3DLoss:
    """
    Criterion for YOLOv8 detection + 3D regression head (Detect3D), fully vectorized (no per-image Python loop).

    训练期 Detect3D.forward 返回: (feats_levels, extra3d_levels)
      - feats_levels: list[Tensor], 与 v8DetectionLoss 一致，用于 2D 匹配与损失
      - extra3d_levels: list[Tensor]，各层 3D 回归通道 (B, n3d, H, W)

    batch 需要包含（由数据集/Format/collate 提供）：
      - 'batch_idx', 'cls', 'bboxes'（与 v8DetectionLoss 一致）
      - 'labels_3d'(N,9), 'faces_3d'(N,4,7), 'has_3d_mask'(N,), 'vehicle_mask'(N,),
        'face_vis_mask'(N,4), 'face_weight'(N,4)

    超参（可在 model.args 中设置）：
      - lambda_base3d=1.0
      - lambda_faces_xyz=1.0
      - lambda_faces_proj=0.5
      - lambda_faces_vis=0.5
      - lambda_faces_score=0.2
      - angle_as_sincos=True
    """

    def __init__(self, model, tal_topk: int = 10):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]  # Detect()/Detect3D()
        # m = model.model.detect

        # 2D 基础参数（同 v8DetectionLoss）
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # 3D 参数
        self.n3d = getattr(m, "n3d", 37)  # 默认 Detect3D.extra_3d_dims
        self.base_dims = 9
        self.face_dims = 7
        self.num_faces = 4

        self.face_dims_out = 6

        # 3D loss 权重
        self.lambda_base3d = getattr(h, "lambda_base3d", 1.0)
        self.lambda_base3d_xyz = getattr(h, "lambda_base3d_xyz", self.hyp.cls)
        self.lambda_base3d_proj = getattr(h, "lambda_base3d_proj", 1.0) #self.hyp.box)
        self.lambda_base3d_size = getattr(h, "lambda_base3d_size", self.hyp.cls)
        self.lambda_base3d_angle = getattr(h, "lambda_base3d_angle", self.hyp.cls * 1.2)
        self.lambda_base3d_cutcls = getattr(h, "lambda_base3d_cutcls", self.hyp.cls)
        
        self.lambda_faces_xyz = getattr(h, "lambda_faces_xyz", 0.2)
        self.lambda_faces_proj = getattr(h, "lambda_faces_proj", 1.0) #self.hyp.box * 1.2)
        self.lambda_faces_size = getattr(h, "lambda_faces_size", self.hyp.cls)
        self.lambda_faces_vis = getattr(h, "lambda_faces_vis", 0.15)
        self.lambda_faces_score = getattr(h, "lambda_faces_score", 1.0)
        self.angle_as_sincos = getattr(h, "angle_as_sincos", False)
        self.angle_as_cls = getattr(h, "angle_as_cls", True)

        self.klLoss = nn.KLDivLoss(reduction='sum')
        self.L1loss_noredu = nn.L1Loss(reduction='none')
        self.BCEloss = nn.BCEWithLogitsLoss(reduction='sum')
        self.L1loss = nn.L1Loss(reduction='sum')
        self.smoothL1loss = nn.SmoothL1Loss(reduction='sum')

        # train_3d flag: control whether to compute 3D losses.
        # Default: False (2D-only training). To enable 3D supervision set model.args.train_3d = True.
        self.train_3d = bool(getattr(h, "train_3d", False))

    # ---------- 与 v8DetectionLoss 一致的工具 ----------
    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def preprocess_3d(
        self,
        batch_idx_flat: torch.Tensor,
        labels_3d_flat: torch.Tensor,
        faces_3d_flat: torch.Tensor,
        has3d_flat: torch.Tensor,
        vehicle_flat: torch.Tensor,
        face_vis_flat: torch.Tensor,
        face_weight_flat: torch.Tensor,
        batch_size: int,
    ):
        """
        将拼接的 3D GT（与 cls/bboxes 对应顺序）按图像分组并 padding。
        返回:
           gt3d:      (B, M, 9)
           faces:     (B, M, 4, 7)
           has3d:     (B, M)
           vehicle:   (B, M)
           face_vis:  (B, M, 4)
           face_w:    (B, M, 4)
        """
        i = batch_idx_flat.view(-1)
        if i.numel() == 0:
            maxn = 0
        else:
            _, counts = i.unique(return_counts=True)
            maxn = counts.to(dtype=torch.int32).max().item()

        gt3d = torch.full((batch_size, maxn, self.base_dims), -1.0, device=self.device)
        faces = torch.full((batch_size, maxn, self.num_faces, self.face_dims), -1.0, device=self.device)
        has3d = torch.zeros((batch_size, maxn), dtype=torch.bool, device=self.device)
        vehicle = torch.zeros((batch_size, maxn), dtype=torch.bool, device=self.device)
        face_vis = torch.zeros((batch_size, maxn, self.num_faces), dtype=torch.bool, device=self.device)
        face_w = torch.zeros((batch_size, maxn, self.num_faces), dtype=faces.dtype, device=self.device)

        for j in range(batch_size):
            matches = (i == j)
            if not matches.any():
                continue
            n = int(matches.sum().item())
            gt3d[j, :n] = labels_3d_flat[matches].to(self.device)
            faces[j, :n] = faces_3d_flat[matches].to(self.device)
            has3d[j, :n] = has3d_flat[matches].to(self.device)
            vehicle[j, :n] = vehicle_flat[matches].to(self.device)
            face_vis[j, :n] = face_vis_flat[matches].to(self.device)
            face_w[j, :n] = face_weight_flat[matches].to(self.device)

        return gt3d, faces, has3d, vehicle, face_vis, face_w

    def angle_loss(self, pred_angle: torch.Tensor, gt_angle: torch.Tensor) -> torch.Tensor:
        if pred_angle.numel() == 0:
            return pred_angle.new_tensor(0.0)
        if self.angle_as_sincos:
            return F.smooth_l1_loss(torch.sin(pred_angle), torch.sin(gt_angle), reduction="mean") + \
                   F.smooth_l1_loss(torch.cos(pred_angle), torch.cos(gt_angle), reduction="mean")
        
        elif self.angle_as_cls:
            import math
            PI = math.pi
            gt_angle = gt_angle.view(-1,1)
            delta_0 = gt_angle
            delta_1 = gt_angle - PI / 2
            delta_2 = gt_angle + PI / 2
            ang_mask = (torch.abs(gt_angle - PI) < torch.abs(gt_angle + PI)).float()
            delta_3 = (gt_angle - PI)*ang_mask + (gt_angle + PI)*(1-ang_mask)
            angles = torch.cat([delta_0, delta_1, delta_2, delta_3], dim=1)
            # angle classification
            ang_cls = (PI*0.5 - torch.abs(angles)) / (PI*0.5)
            ang_cls[ang_cls < 0] = 0
            assert torch.sum(torch.sum(ang_cls, dim=1) > 1.001) == 0
            assert torch.sum(torch.sum(ang_cls, dim=1) < 0.999) == 0
            gt_angle_ = torch.cat([ang_cls, angles], dim=1)

            pred_angle_prob = torch.log_softmax(pred_angle[:, :4], dim=1)
            l_angle_cls = self.klLoss(pred_angle_prob, gt_angle_[:, :4])

            valid_mask = torch.abs(gt_angle_[:, 4:]) < (PI*0.5)
            yaw_loss = self.L1loss_noredu(pred_angle[:, 4:].tanh()*(PI*0.5), gt_angle_[:, 4:]) * valid_mask
            l_angle_reg = torch.sum(yaw_loss)

            # l_angle = l_angle_cls / pred_angle.shape[0] + l_angle_reg / valid_mask.sum().clamp_min(1.0)

            return l_angle_cls / pred_angle.shape[0], l_angle_reg / valid_mask.sum().clamp_min(1.0)
        else:
            return F.smooth_l1_loss(pred_angle, gt_angle, reduction="mean")

    def cutcls_loss(self, pred_cutcls: torch.Tensor, gt_faces: torch.Tensor) -> torch.Tensor:
        f_c = torch.sum(gt_faces[:, 0, [0,1,2,5,6]],dim=1)==-4
        t_c = torch.sum(gt_faces[:, 1, [0,1,2,5,6]],dim=1)==-4
        l_c = torch.sum(gt_faces[:, 2, [0,1,2,5,6]],dim=1)==-4
        r_c = torch.sum(gt_faces[:, 3, [0,1,2,5,6]],dim=1)==-4

        cut_cls_label = torch.zeros_like(f_c).float()
        cut_cls_label[torch.where(t_c & l_c & r_c)] = 1  ##cut_in 
        cut_cls_label[torch.where(f_c & l_c & r_c)] = 2  ##cut_out

        gt_cutcls = torch.full_like(pred_cutcls, 0)
        gt_cutcls[torch.arange(len(cut_cls_label)), cut_cls_label.long()] = 1.0  ###### tcls[i]#####

        l_cutcls = self.BCEloss(pred_cutcls, gt_cutcls)
        return l_cutcls

    # ---------- 主计算 ----------
    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          total_loss * batch_size, loss_vec.detach()
        其中 loss_vec = [box, cls, dfl, base3d, faces]
        """
        # 解析 Detect3D 训练输出
        # assert isinstance(preds, (list, tuple)) and isinstance(preds[0], list), \
        #     "v8Detection3DLossVectorized expects Detect3D training outputs: (feats_levels, extra3d_levels)"

        # training
        if isinstance(preds, (list, tuple)) and isinstance(preds[0], list):
            feats, extra3d_levels = preds
        # eval
        elif isinstance(preds, (list, tuple)) and isinstance(preds[1], tuple):
            feats, extra3d_levels = preds[1]
        else:
            raise TypeError(
                "v8Detection3DLossVectorized expects Detect3D training outputs: (feats_levels, extra3d_levels)"
            )

        loss_vec = torch.zeros(13, device=self.device)  # box, cls, dfl, base3d, faces, cutcls

        # 2D 展平 (与 v8DetectionLoss 同)
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, N, nc)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (B, N, reg*4)

        # 3D 展平 (B, N, n3d)
        bs = feats[0].shape[0]
        pred_extra3d = torch.cat([e.view(bs, self.n3d, -1) for e in extra3d_levels], 2).permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # --- 新增：从 batch 中获取 ROI ID ---
        # roi_ids_per_image 的形状为 [batch_size]，值为 0 (Wide), 1 (Tele) 等
        roi_ids_per_image = batch.get("roi_id", torch.zeros(batch_size, device=self.device, dtype=torch.int8))

        # Targets 与匹配
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # (B,M,1), (B,M,4)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (B, N, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # 2D Cls
        loss_vec[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # 2D Box + DFL
        if fg_mask.sum():
            l_iou, l_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            loss_vec[0] = l_iou
            loss_vec[2] = l_dfl

        # If 3D training disabled, skip the heavy 3D processing and return only 2D losses.
        if not self.train_3d:
            # Apply 2D gains
            loss_vec[0] *= self.hyp.box
            loss_vec[1] *= self.hyp.cls
            loss_vec[2] *= self.hyp.dfl

            total_loss = loss_vec[:3].sum()
            return total_loss * batch_size, loss_vec.detach()

        # ---------------- 3D GT 预处理（按图像分组 + padding） ----------------
        labels_3d_flat = batch.get("labels_3d", torch.zeros(0, self.base_dims, device=self.device)).to(self.device)
        faces_3d_flat = batch.get("faces_3d", torch.zeros(0, self.num_faces, self.face_dims, device=self.device)).to(self.device)
        has3d_flat = batch.get("has_3d_mask", torch.zeros(0, dtype=torch.bool, device=self.device)).to(self.device)
        vehicle_flat = batch.get("vehicle_mask", torch.zeros(0, dtype=torch.bool, device=self.device)).to(self.device)
        face_vis_flat = batch.get("face_vis_mask", torch.zeros(0, self.num_faces, dtype=torch.bool, device=self.device)).to(self.device)
        face_weight_flat = batch.get("face_weight", torch.zeros(0, self.num_faces, device=self.device)).to(self.device)

        gt3d, faces3d, has3d, vehicle, face_vis, face_w = self.preprocess_3d(
            batch["batch_idx"].to(self.device),
            labels_3d_flat,
            faces_3d_flat,
            has3d_flat,
            vehicle_flat,
            face_vis_flat,
            face_weight_flat,
            batch_size,
        )

        # gt3d: x3d_ori: 0, y3d_ori: 1, z3d_ori: 2, 13d: 3, h3d: 4, w3d: 5, rot_y: 6, xc_ori: 7, yc_ori: 8,
        # faces3d: x3d: 0, y3d: 1, z3d: 2, xc: 3, yc: 4, score: 5, is occ: 6,

        # ---------------- 完全向量化：一次性在所有正样本上 gather ----------------
        pos = fg_mask.nonzero(as_tuple=False)  # (P, 2) with [b_idx, a_idx]
        P = pos.shape[0]

        if P == 0:
            # 保持计算图（避免DDP未用梯度报错的风险）
            loss_vec[3] = loss_vec[3] + (pred_extra3d * 0).sum() * 0.0
            loss_vec[4] = loss_vec[4] + (pred_extra3d * 0).sum() * 0.0
        else:
            b_idx, a_idx = pos[:, 0], pos[:, 1]
            # 正样本对应的 GT 索引（图内编号）
            gi = target_gt_idx[b_idx, a_idx]  # (P,)

            # --- 新增：为每个正样本获取其 ROI ID ---
            # 使用 b_idx (每个正样本所属的图片索引) 来索引 roi_ids_per_image
            # target_roi_ids 的形状为 [P,]，值为 0 或 1
            target_roi_ids = roi_ids_per_image[b_idx]

            # 正样本 3D 预测 (P, n3d)
            pred3d_pos = pred_extra3d[b_idx, a_idx]  # (P, n3d)

            # 分割为 基础3D + 车辆4面
            pred_faces = pred3d_pos[:, :self.num_faces*self.face_dims_out].view(-1, self.num_faces, self.face_dims_out)  # (P, 4, 6)
            pred_base = pred3d_pos[:, self.num_faces*self.face_dims_out:]                          # (P, 6+8+3)

            # GT 3D（按图像分组后，使用 (b_idx, gi) gather）
            gt_base_pos = gt3d[b_idx, gi]             # (P, 9)
            gt_faces_pos = faces3d[b_idx, gi]         # (P, 4, 7)
            has3d_pos = has3d[b_idx, gi]              # (P,)
            vehicle_pos = vehicle[b_idx, gi]          # (P,)
            face_vis_pos = face_vis[b_idx, gi]        # (P, 4)
            face_w_pos = face_w[b_idx, gi]            # (P, 4)

            ap_px = (anchor_points[a_idx] * stride_tensor[a_idx]).to(pred3d_pos.dtype)       # (P, 2) 像素
            st_pos = stride_tensor[a_idx].to(pred3d_pos.dtype) 

            # ---- 基础3D：仅在 has3d 上监督（稳定均值化）
            m = has3d_pos
            if m.any():
                depth_mask = gt_base_pos[:, 2] <= 50.0
                lateral_mask = torch.abs(gt_base_pos[:, 0]) < 35.0
                height_mask = torch.abs(gt_base_pos[:, 1]) < 10.0
                m = m & depth_mask & lateral_mask & height_mask

                m_sum = m.sum().clamp_min(1)
                # xyz, lwh, proj 以元素均值统计；旋转使用角度损失

                gt_base_pos_norm = torch.zeros_like(gt_base_pos)
                gt_base_pos_norm[m, 0] = gt_base_pos[m, 0] / 35.0
                gt_base_pos_norm[m, 1] = gt_base_pos[m, 1] / 5.0

                # --- 条件化 Z 坐标归一化 (Base 3D) ---
                # 约定：ID 1 代表 'Tele_ROI'
                tele_mask = (target_roi_ids == 1) & m
                wide_mask = (target_roi_ids != 1) & m
                # 应用长焦策略
                gt_base_pos_norm[tele_mask, 2] = (gt_base_pos[tele_mask, 2] / 2.0 - 40.0) / 40.0
                # 应用广角/默认策略
                gt_base_pos_norm[wide_mask, 2] = (gt_base_pos[wide_mask, 2] - 40.0) / 40.0
                # ---------------------------------------------

                # gt_base_pos_norm[m, 2] = (gt_base_pos[m, 2] - 40) / 40

                gt_base_pos_norm[m, 3:6] = gt_base_pos[m, 3:6] / 18.0

                l_xyz = self.L1loss(pred_base[m, 0], gt_base_pos_norm[m, 2]) / m_sum
                
                gt_proj = gt_base_pos[:, 7:9]                  # (P,2) 归一化GT
                gt_proj_px = gt_proj * imgsz[[1, 0]]           # (P,2) 像素GT
                gt_proj_off  = (gt_proj_px - ap_px) / st_pos   # (P,2) 偏移归一到 cell
                # gt_proj_off = gt_base_pos[:, 7:9]
                l_proj = self.L1loss(pred_base[m, 1:2].tanh()*6, gt_proj_off[m, :1]) / m_sum

                # def stats(x): return (x.mean().item(), x.std().item(), x.min().item(), x.max().item())
                # print("gt_proj_off stats:", stats(gt_proj_off[m, :1]), " pred_base proj stats:", stats(pred_base[m, 1:2].tanh()*6))

                l_lwh = self.L1loss(pred_base[m, 3:6], gt_base_pos_norm[m, 3:6]) / (m_sum * 3)

                l_rot_cls, l_rot_reg = self.angle_loss(pred_base[m, 6:14], gt_base_pos[m, 6])
                
                # base3d_loss = self.lambda_base3d_xyz*l_xyz + \
                #             self.lambda_base3d_proj*l_proj + \
                #             self.lambda_base3d_size*l_lwh + \
                #             self.lambda_base3d_angle*l_rot
            
            else:
                l_xyz = torch.zeros(1, device=pred3d_pos.device)
                l_proj = torch.zeros(1, device=pred3d_pos.device)
                l_lwh = torch.zeros(1, device=pred3d_pos.device)
                l_rot_cls = torch.zeros(1, device=pred3d_pos.device)
                l_rot_reg = torch.zeros(1, device=pred3d_pos.device)

            # ---- 车辆 4 面：仅在 vehicle 上；几何/投影只监督可见(is_vis)面并以 score 作为权重
            vm = vehicle_pos
            if vm.any():
                depth_mask_vm = gt_base_pos[:, 2] < 50.0
                lateral_mask_vm = torch.abs(gt_base_pos[:, 0]) < 35.0
                height_mask_vm = torch.abs(gt_base_pos[:, 1]) < 10.0
                vm = vm & depth_mask_vm & lateral_mask_vm & height_mask_vm

                pv = vm.sum().item()
                p_faces = pred_faces[vm]        # (Pv, 4, 7)
                g_faces = gt_faces_pos[vm]      # (Pv, 4, 7)
                vis = face_vis_pos[vm].float()  # (Pv, 4)
                w = face_w_pos[vm]              # (Pv, 4)

                w[w>0.3] = 1.0
                w[w<=0.3] = 0.0

                valid_face = vis * w

                g_faces_norm = torch.zeros_like(g_faces)
                g_faces_norm[:, :, 0] = g_faces[:, :, 0] / 35.0
                g_faces_norm[:, :, 1] = g_faces[:, :, 1] / 5.0
                # g_faces_norm[:, :, 2] = (g_faces[:, :, 2] - 40) / 40

                # --- 条件化 Z 坐标归一化 (Faces 3D) ---
                # `vm` 已经是 vehicle_pos 的掩码，形状为 [P,]
                # 我们需要将 target_roi_ids 扩展以匹配 g_faces 的形状
                # target_roi_ids 的形状是 [P,], 我们需要 [Pv, 4]
                # `g_faces` 的形状是 [Pv, 4, 7]，其中 Pv 是 vm.sum()
                
                # 获取 vehicle 正样本对应的 ROI ID
                roi_ids_for_vehicles = target_roi_ids[vm] # 形状 [Pv,]

                # 创建掩码
                tele_mask_v = (roi_ids_for_vehicles == 1) # 形状 [Pv,]
                wide_mask_v = (roi_ids_for_vehicles != 1) # 形状 [Pv,]

                # g_faces 的形状是 [Pv, 4, 7]
                # tele_mask_v[:, None] 的形状是 [Pv, 1]，可以广播到 [Pv, 4]
                # 应用长焦策略
                g_faces_norm[tele_mask_v, :, 2] = (g_faces[tele_mask_v, :, 2] / 2.0 - 40.0) / 40.0 
                # 应用广角/默认策略
                g_faces_norm[wide_mask_v, :, 2] = (g_faces[wide_mask_v, :, 2] - 40.0) / 40.0
                # ---------------------------------------------

                gt_base = gt_base_pos[vm]  # (Pv, 9)
                gt_faces_with_size = torch.zeros((g_faces.shape[0], g_faces.shape[1], 9), device=g_faces.device, dtype=g_faces.dtype)
                gt_faces_with_size[:, :, :7] = g_faces
                gt_faces_with_size[:, :2, 7:9] = gt_base[:, 4:6].unsqueeze(1).repeat(1, 2, 1)
                gt_faces_with_size[:, 2:, 7:9] = gt_base[:, 3:5].unsqueeze(1).repeat(1, 2, 1)

                # xyz (Pv,4,3) -> mean over last -> (Pv,4)
                l_xyz_face = self.L1loss_noredu(p_faces[:, :, 0], g_faces_norm[:, :, 2])
                
                # proj (Pv,4,2) -> (Pv,4)
                # gt_proj_face_off = gt_faces_pos[:, :, 3:5]                                        # (P,4,2) 像素GT
                gt_proj_face = gt_faces_pos[:, :, 3:5]                                              # (P,4,2) 归一化GT
                gt_proj_face_px = gt_proj_face * imgsz[[1, 0]]                                      # (P,4,2) 像素GT
                gt_proj_face_off = (gt_proj_face_px - ap_px[:, None, :]) / st_pos[:, None, :]       # (P,4,2)
                l_proj_face = self.L1loss_noredu(p_faces[:, :, 1:2].tanh()*6, gt_proj_face_off[vm,:, :1]).mean(-1)

                # def stats(x): return (x.mean().item(), x.std().item(), x.min().item(), x.max().item())
                # print("gt_proj_face stats:", stats(gt_proj_face_off[vm,:, :1]), " pred_proj_face stats:", stats(p_faces[:, :, 1:2].tanh()*6))
                
                gt_faces_with_size_norm = gt_faces_with_size[:, :, 7:9] / 18.0
                l_size_face = self.L1loss_noredu(p_faces[:, :, 3:5], gt_faces_with_size_norm).mean(-1)

                # 可见性 BCE（所有面）
                # l_vis_face = F.binary_cross_entropy_with_logits(p_faces[:, :, 5], g_faces[:, :, 6], reduction="none")
                l_vis_face = self.L1loss_noredu(p_faces[:, :, 5], g_faces[:, :, 5])
                # print("g_faces stats:", stats(g_faces[:, :, 5]), " p_faces stats:", stats(p_faces[:, :, 5]))

                # score 回归（仅可见）
                # l_score_face = F.smooth_l1_loss(torch.sigmoid(p_faces[:, :, 5]), g_faces[:, :, 5], reduction="none")

                l_cutcls = self.cutcls_loss(pred_base[vm, 14:], g_faces)
                l_cutcls = l_cutcls / pv

                # 可见面计数（用于归一化）
                denom = valid_face.sum().clamp_min(1.0)

                faces_xyz = (l_xyz_face * valid_face).sum() / denom
                faces_proj = (l_proj_face * valid_face).sum() / denom
                faces_size = (l_size_face * valid_face).sum() / denom
                # faces_vis = (l_vis_face * valid_face).sum() / denom
                faces_vis = l_vis_face.mean()
                # faces_score = (l_score_face * vis).sum() / denom
                
            else:
                faces_xyz = torch.zeros(1, device=pred3d_pos.device)
                faces_proj = torch.zeros(1, device=pred3d_pos.device)
                faces_size = torch.zeros(1, device=pred3d_pos.device)
                faces_vis = torch.zeros(1, device=pred3d_pos.device)
                l_cutcls = torch.zeros(1, device=pred3d_pos.device)
            
            loss_vec[3] = self.lambda_base3d_xyz * l_xyz
            loss_vec[4] = self.lambda_base3d_proj * l_proj
            loss_vec[5] = self.lambda_base3d_size * l_lwh
            loss_vec[6] = self.lambda_base3d_angle * l_rot_cls
            loss_vec[7] = self.lambda_base3d_angle * l_rot_reg

            loss_vec[8] = self.lambda_faces_xyz * faces_xyz
            loss_vec[9] = self.lambda_faces_proj * faces_proj
            loss_vec[10] = self.lambda_faces_size * faces_size
            loss_vec[11] = self.lambda_faces_vis * faces_vis

            loss_vec[12] = self.lambda_base3d_cutcls * l_cutcls

        # 2D gains
        loss_vec[0] *= self.hyp.box
        loss_vec[1] *= self.hyp.cls
        loss_vec[2] *= self.hyp.dfl

        total_loss = loss_vec.sum()
        return total_loss * batch_size, loss_vec.detach()
# ======== 3D Detection Loss (完全向量化版本) 结束 ========