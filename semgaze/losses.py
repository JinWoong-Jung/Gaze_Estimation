#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_info_nce_loss_batch_local(emb_pred, emb_gt, io_gt, logit_scale):
    # Legacy SemGaze behavior: negatives are only unique labels in the current batch.
    mask = io_gt.bool()
    emb_gt, emb_pred = emb_gt[mask], emb_pred[mask]
    if emb_pred.numel() == 0:
        return torch.tensor(0.0, device=emb_pred.device, dtype=emb_pred.dtype)
    emb_gt, labels = torch.unique(emb_gt, dim=0, return_inverse=True)
    logits = torch.matmul(emb_pred, emb_gt.t()) * logit_scale.exp()
    return F.cross_entropy(logits, labels)


def compute_info_nce_loss_vocab(
    emb_pred,
    label_id_gt,
    io_gt,
    logit_scale,
    vocab_emb,
    margin_type: str = "none",
    margin: float = 0.0,
    easy_margin: bool = False,
):
    # Use in-frame samples with valid label ids.
    mask = io_gt.bool() & (label_id_gt >= 0)
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, device=emb_pred.device, dtype=emb_pred.dtype)

    emb_pred = F.normalize(emb_pred[mask], p=2, dim=-1)
    labels = label_id_gt[mask].long()
    vocab_emb = F.normalize(vocab_emb, p=2, dim=-1).to(emb_pred.device, emb_pred.dtype)

    cosine = torch.matmul(emb_pred, vocab_emb.t())
    margin_type = str(margin_type).lower()

    if (margin > 0.0) and (margin_type in {"arcface", "cosface"}):
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).to(cosine.dtype)
        if margin_type == "cosface":
            cosine = cosine - one_hot * margin
        elif margin_type == "arcface":
            cos_m = math.cos(margin)
            sin_m = math.sin(margin)
            th = math.cos(math.pi - margin)
            mm = math.sin(math.pi - margin) * margin
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0.0, 1.0))
            phi = cosine * cos_m - sine * sin_m
            if easy_margin:
                phi = torch.where(cosine > 0.0, phi, cosine)
            else:
                phi = torch.where(cosine > th, phi, cosine - mm)
            cosine = one_hot * phi + (1.0 - one_hot) * cosine

    logits = cosine * logit_scale.exp()
    loss = F.cross_entropy(logits, labels)
    return loss


def compute_info_nce_loss(*args, **kwargs):
    """
    Backward-compatible wrapper.
    - Legacy call: compute_info_nce_loss(emb_pred, emb_gt, io_gt, logit_scale)
    - Vocab call : compute_info_nce_loss(..., vocab_emb=..., ...)
    """
    if "vocab_emb" in kwargs:
        return compute_info_nce_loss_vocab(*args, **kwargs)
    if len(args) == 4:
        return compute_info_nce_loss_batch_local(*args)
    raise TypeError("Unsupported compute_info_nce_loss call signature.")


def compute_dist_loss(gp_pred, gp_gt, io_gt):
    dist_loss = (gp_pred - gp_gt).pow(2).sum(dim=1)
    dist_loss = torch.mul(dist_loss, io_gt)
    dist_loss = torch.sum(dist_loss) / torch.sum(io_gt)
    return dist_loss


def compute_heatmap_loss(hm_pred, hm_gt, io_gt):
    # BCELoss is not autocast-safe in mixed precision. Compute it in fp32 with autocast disabled.
    if torch.is_autocast_enabled():
        with torch.autocast(device_type=hm_pred.device.type, enabled=False):
            heatmap_loss = F.binary_cross_entropy(
                hm_pred.float(),
                hm_gt.float(),
                reduction="none",
            ).mean([1, 2])
    else:
        heatmap_loss = F.binary_cross_entropy(hm_pred, hm_gt, reduction="none").mean([1, 2])
    heatmap_loss = torch.mul(heatmap_loss, io_gt)
    heatmap_loss = torch.sum(heatmap_loss) / torch.sum(io_gt)
    return heatmap_loss


def compute_angular_loss(gv_pred, gv_gt, io_gt):
    angular_loss = (1 - torch.einsum("ij,ij->i", gv_pred, gv_gt)) / 2
    angular_loss = torch.mul(angular_loss, io_gt)
    angular_loss = torch.sum(angular_loss) / torch.sum(io_gt)
    return angular_loss


def compute_alignment_loss(
    emb_pred,
    emb_gt,
    valid_mask,
    loss_type: str = "cosine",
    logit_scale=None,
):
    mask = valid_mask.bool()
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, device=emb_pred.device, dtype=emb_pred.dtype)

    emb_pred = emb_pred[mask]
    emb_gt = emb_gt[mask]

    if loss_type == "cosine":
        emb_pred = F.normalize(emb_pred, p=2, dim=-1)
        emb_gt = F.normalize(emb_gt, p=2, dim=-1)
        return (1 - torch.sum(emb_pred * emb_gt, dim=-1)).mean()

    if loss_type == "mse":
        return F.mse_loss(emb_pred, emb_gt, reduction="none").mean(dim=-1).mean()

    if loss_type == "infonce":
        emb_pred = F.normalize(emb_pred, p=2, dim=-1)
        emb_gt = F.normalize(emb_gt, p=2, dim=-1)
        n = emb_pred.size(0)
        labels = torch.arange(n, device=emb_pred.device)
        if logit_scale is None:
            scale = 1.0
        else:
            scale = logit_scale.exp()
        logits = emb_pred @ emb_gt.t() * scale
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i + loss_t)

    raise ValueError(f"Unsupported alignment loss type: {loss_type}")


def compute_relational_distillation_loss(
    emb_pred,
    emb_gt,
    valid_mask,
    tau_student: float = 0.07,
    tau_teacher: float = 0.07,
    detach_teacher: bool = True,
    remove_diagonal: bool = True,
    min_valid: int = 2,
):
    mask = valid_mask.bool()
    num_valid = int(torch.sum(mask).item())
    min_valid = max(2, int(min_valid))
    if num_valid < min_valid:
        return torch.tensor(0.0, device=emb_pred.device, dtype=emb_pred.dtype)

    student = F.normalize(emb_pred[mask], p=2, dim=-1)
    teacher = F.normalize(emb_gt[mask], p=2, dim=-1)

    if detach_teacher:
        teacher = teacher.detach()

    tau_student = max(float(tau_student), 1e-6)
    tau_teacher = max(float(tau_teacher), 1e-6)

    sim_student = (student @ student.t()) / tau_student
    sim_teacher = (teacher @ teacher.t()) / tau_teacher

    if remove_diagonal:
        diag_mask = torch.eye(num_valid, device=sim_student.device, dtype=torch.bool)
        sim_student = sim_student.masked_fill(diag_mask, -1e4)
        sim_teacher = sim_teacher.masked_fill(diag_mask, -1e4)

    teacher_prob = F.softmax(sim_teacher, dim=-1)
    student_log_prob = F.log_softmax(sim_student, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
