#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import torch
import torch.nn.functional as F
import torchmetrics as tm
from sklearn.metrics import roc_auc_score

from semgaze.utils.common import generate_binary_gaze_heatmap, is_point_in_box


class Distance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_point_pred: torch.Tensor,
        gaze_point_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        mask = inout_gt == 1
        if mask.any():
            self.sum_dist += (gaze_point_gt[mask] - gaze_point_pred[mask]).pow(2).sum(1).sqrt().sum()
            self.num_obs += mask.sum()

    def compute(self):
        if self.num_obs != 0:
            dist = self.sum_dist / self.num_obs  # type: ignore
        else:
            dist = torch.tensor(-1000.0, device=self.device)
        return dist


class GFTestDistance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist_to_avg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_avg_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_min_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, gaze_point_pred: torch.Tensor, gaze_point_gt: torch.Tensor):
        for k, (gp_pred, gp_gt) in enumerate(zip(gaze_point_pred, gaze_point_gt)):
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points
            
            # Compute average gaze point
            gp_gt_avg = gp_gt.mean(0)
            # Compute distance from pred to avg gt point
            self.sum_dist_to_avg += (gp_gt_avg - gp_pred).pow(2).sum().sqrt()
            # Compute avg distance between pred and gt points
            self.sum_avg_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().mean()
            # Compute min distance between pred and gt points
            self.sum_min_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().min()
        self.num_obs += len(gaze_point_pred)

    def compute(self):
        dist_to_avg = self.sum_dist_to_avg / self.num_obs
        avg_dist = self.sum_avg_dist / self.num_obs
        min_dist = self.sum_min_dist / self.num_obs
        return dist_to_avg, avg_dist, min_dist


class GFTestAUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        """
        Computes AUC for GazeFollow Test set. The AUC is computed for each image in the batch, after resizing the predicted
        heatmap to the original size of the image. The ground-truth binary heatmap is generated from the ground-truth gaze
        point(s) in the original image size. At the end, the mean is returned.
        """

        super().__init__()
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_pt: torch.Tensor,
        img_size: torch.Tensor,
    ):
        for hm_pred, gp_gt, img_wh in zip(gaze_heatmap_pred, gaze_pt, img_size):
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points
            if gp_gt.numel() == 0:
                continue

            img_w = int(img_wh[0].item())
            img_h = int(img_wh[1].item())
            if (img_w <= 0) or (img_h <= 0):
                continue

            hm_pred = F.interpolate(
                hm_pred.unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            hm_gt_binary = generate_binary_gaze_heatmap(gp_gt, size=(img_h, img_w))
            try:
                auc = roc_auc_score(
                    hm_gt_binary.detach().cpu().flatten().numpy(),
                    hm_pred.detach().cpu().flatten().numpy(),
                )
            except ValueError:
                # Skip degenerate edge-cases where ROC AUC is undefined.
                continue

            self.sum_auc += torch.tensor(float(auc), device=self.device)
            self.num_obs += 1

    def compute(self):
        if self.num_obs == 0:
            return torch.tensor(0.0, device=self.device)
        auc = self.sum_auc / self.num_obs
        return auc


class VATAUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, resolution: int = 64, sigma: int = 3):
        """
        Computes AUC for VAT-style single gaze point supervision.
        AUC is computed only for in-frame samples (inout==1).
        Mirrors Gazelle VAT protocol:
        evaluate directly on 64x64 heatmap space with a tolerance region
        around the single ground-truth gaze point.
        """
        super().__init__()
        self.resolution = int(resolution)
        self.sigma = int(sigma)
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_pt: torch.Tensor,
        img_size: torch.Tensor,
        inout: torch.Tensor,
    ):
        # img_size is unused on purpose: VAT AUC is computed in fixed heatmap space.
        del img_size

        res = self.resolution
        sigma = self.sigma
        three_sigma = 3 * sigma

        for hm_pred, gp_gt, io in zip(gaze_heatmap_pred, gaze_pt, inout):
            if float(io.item()) < 0.5:
                continue

            if hm_pred.shape[-2:] != (res, res):
                hm_pred = F.interpolate(
                    hm_pred.unsqueeze(0).unsqueeze(0),
                    size=(res, res),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            # Gazelle VAT uses a rectangular tolerance region around GT point.
            target_map = torch.zeros((res, res), device=hm_pred.device, dtype=torch.int)
            gx = float(gp_gt[0].item()) * res
            gy = float(gp_gt[1].item()) * res
            ul_x = max(0, int(gx - three_sigma))
            ul_y = max(0, int(gy - three_sigma))
            br_x = min(int(gx + three_sigma + 1), res - 1)
            br_y = min(int(gy + three_sigma + 1), res - 1)
            if (br_x > ul_x) and (br_y > ul_y):
                target_map[ul_y:br_y, ul_x:br_x] = 1
            try:
                auc = roc_auc_score(
                    target_map.detach().cpu().flatten().numpy(),
                    hm_pred.detach().cpu().flatten().numpy(),
                )
            except ValueError:
                continue

            self.sum_auc += torch.tensor(float(auc), device=self.device)
            self.num_obs += 1

    def compute(self):
        if self.num_obs == 0:
            return torch.tensor(0.0, device=self.device)
        return self.sum_auc / self.num_obs


class GazeAccuracy(tm.Metric):
    higher_is_better = True
    full_state_update: bool = False
    
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, gaze_point_pred: torch.Tensor, gaze_bbox_gt: torch.Tensor):
        isin = is_point_in_box(gaze_point_pred, gaze_bbox_gt).diag()
        self.correct += isin.sum()
        self.total += gaze_point_pred.size(0)

    def compute(self):
        return self.correct.float() / self.total

    
class MultiAccuracy(tm.Metric):
    higher_is_better = True
    full_state_update: bool = False
    
    def __init__(self, top_k: int = 1, ignore_index = None):
        super().__init__()
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).any(dim=1)
            preds = preds[mask]
            target = target[mask]
        
        # Get the top k predictions
        top_k_preds = preds.topk(self.top_k, dim=1)[1]
        
        # Check if any of the top k predictions match any of the target classes
        target = target.unsqueeze(1) # Expand dims of target for broadcasting
        correct = torch.any(top_k_preds.unsqueeze(2) == target, dim=1).any(dim=1).sum()
        self.correct += correct
        self.total += target.size(0)

    def compute(self):
        return self.correct.float() / self.total
