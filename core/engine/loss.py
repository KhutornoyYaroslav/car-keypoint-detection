import torch
import numpy as np
from torch import nn
from typing import List
import torch.nn.functional as F
from core.utils.ops import xywh2xyxy, xyxy2xywh
from core.utils.metrics import bbox_iou
from core.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors


# TODO: this values for human pose...
# OKS_SIGMA = (
#     np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
#     / 10.0
# )


class DFLoss(nn.Module):
    def __init__(self, bins: int = 16) -> None:
        super().__init__()
        self.bins = bins

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.bins - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, dfl_bins: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(dfl_bins) if dfl_bins > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # dfl
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.bins - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.bins), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula # TODO: testing this variant
        # e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


# class KeypointLoss(nn.Module):
#     def __init__(self, sigmas) -> None:
#         super().__init__()
#         self.sigmas = sigmas

#     def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
#         d = (pred_kpts[..., 0] - gt_kpts[..., 0]).abs() + (pred_kpts[..., 1] - gt_kpts[..., 1]).abs()
#         kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
#         # return kpt_loss_factor * ((d / (area + 1e-9))* kpt_mask).mean() # TODO: testing...
#         return kpt_loss_factor * (d * kpt_mask).mean()


class DetectionLoss:
    def __init__(self,
                 num_classes: int,
                 strides: List[float],
                 dfl_bins: int,
                 loss_box_k: float,
                 loss_dfl_k: float,
                 loss_cls_k: float,
                 device: torch.device,
                 tal_topk: int = 10,
                 ):
        self.loss_box_k = loss_box_k
        self.loss_dfl_k = loss_dfl_k
        self.loss_cls_k = loss_cls_k

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stride = strides
        self.nc = num_classes
        self.no = num_classes + dfl_bins * 4
        self.dfl_bins = dfl_bins
        self.device = device
        self.use_dfl = dfl_bins > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.dfl_bins).to(device)
        self.proj = torch.arange(self.dfl_bins, dtype=torch.float, device=device)

    # def preprocess(self, targets, batch_size, scale_tensor):
    #     """Preprocesses the target counts and matches with the input batch size to output a tensor."""
    #     nl, ne = targets.shape # (n_targets, 1 + 4 + num_classes)
    #     if nl == 0:
    #         out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
    #     else:
    #         i = targets[:, 0]  # images indexes
    #         _, counts = i.unique(return_counts=True)
    #         counts = counts.to(dtype=torch.int32)
    #         out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
    #         for j in range(batch_size):
    #             matches = i == j
    #             n = matches.sum()
    #             if n:
    #                 out[j, :n] = targets[matches, 1:] # (bs, num_max_boxes, 4 + num_classes)
    #         out[..., :4] = xywh2xyxy(out[..., :4].mul_(scale_tensor))
    #     return out

    def preprocess(self, targets: torch.Tensor):
        # remove zero padding assuming padding is from the end of tensors
        bs, nb, ne = targets.shape
        bboxes, scores = targets.split((4, self.nc), 2)
        max_bboxes = torch.count_nonzero(bboxes.sum(-1), dim=1).max()
        if max_bboxes == 0:
            return torch.zeros(bs, 0, ne, device=self.device)
        targets = targets[:, :max_bboxes, :]
        assert targets.shape == (bs, max_bboxes, ne)
        return targets.contiguous()

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, targets):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.dfl_bins * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        # (n_targets, 1 + 4 + num_classes) = (n_targets, 1) + (n_targets, 4) + (n_targets, num_classes)
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["bboxes"], batch["cls"]), 1)
        # (bs, n_max_boxes, cls + 4), scale_tensor = (w, h, w, h)
        # targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        targets = self.preprocess(targets)

        gt_bboxes, gt_scores = targets.split((4, self.nc), 2)  # xywh, num_classes
        gt_bboxes = xywh2xyxy(gt_bboxes)
        gt_bboxes = gt_bboxes.mul(imgsz[[1, 0, 1, 0]]) # scaled to image size
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_scores,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.loss_box_k  # box gain
        loss[1] *= self.loss_cls_k  # cls gain
        loss[2] *= self.loss_dfl_k  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class PoseLoss(DetectionLoss):
    def __init__(self,
                 num_classes: int,
                 kpt_shape,
                 strides: List[float],
                 dfl_bins: int,
                 loss_box_k: float,
                 loss_dfl_k: float,
                 loss_cls_k: float,
                 loss_pose_k: float,
                 loss_kobj_k: float,
                 device: torch.device,
                 tal_topk: int = 10,
                 ):
        super().__init__(num_classes, strides, dfl_bins, loss_box_k, loss_dfl_k, loss_cls_k, device, tal_topk)
        self.kpt_shape = kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        # is_pose = self.kpt_shape == (17, 3)
        nkpt = self.kpt_shape[0] # number of keypoints
        # sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        sigmas = torch.ones(nkpt, device=self.device) / nkpt

        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        self.loss_pose_k = loss_pose_k
        self.loss_kobj_k = loss_kobj_k

    def __call__(self, preds, targets, keypoints):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.dfl_bins * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        # batch_idx = batch["batch_idx"].view(-1, 1)
        # targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        # targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # TODO: check remove padding to keypoints
        # targets = self.preprocess(targets)

        gt_bboxes, gt_scores = targets.split((4, self.nc), 2)  # xywh, num_classes
        gt_bboxes = xywh2xyxy(gt_bboxes)
        gt_bboxes = gt_bboxes.mul(imgsz[[1, 0, 1, 0]]) # scaled to image size
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_scores,
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

            # TODO: implement it
            # keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints = keypoints.to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.loss_box_k  # box gain
        loss[1] *= self.loss_pose_k  # pose gain
        loss[2] *= self.loss_kobj_k  # kobj gain
        loss[3] *= self.loss_cls_k  # cls gain
        loss[4] *= self.loss_dfl_k  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    # TODO: remove 'batch_idx' arg later
    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, stride_tensor, target_bboxes, pred_kpts
    ):
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
        batched_keypoints = keypoints

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[2], keypoints.shape[3])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

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
