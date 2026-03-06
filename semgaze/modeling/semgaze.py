#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

# ==================================================================================================================
#                                                      IMPORTS                                                     #
# ==================================================================================================================
import os
import math
import json
from termcolor import colored
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Type, Union

import wandb
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import DINOv3ViTModel
try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

import torchmetrics as tm
import pytorch_lightning as pl

from semgaze.modeling.encoder import GazeEncoder, SpatialInputTokenizer, ViTEncoder
from semgaze.modeling.decoder import GazeDecoder
from semgaze.losses import (
    compute_heatmap_loss,
    compute_angular_loss,
    compute_info_nce_loss,
    compute_info_nce_loss_batch_local,
    compute_alignment_loss,
)
from semgaze.metrics import Distance, GFTestAUC, GFTestDistance, MultiAccuracy, GazeAccuracy
from semgaze.utils.common import spatial_argmax2d, dark_coordinate_decoding

TERM_COLOR = "cyan"


def _cfg_get(cfg, path, default):
    value = cfg
    try:
        for part in path.split("."):
            value = getattr(value, part)
        return value
    except Exception:
        return default


def _get_cosine_schedule_with_warmup_torch(optimizer, num_warmup_steps, num_training_steps):
    num_warmup_steps = int(max(0, num_warmup_steps))
    num_training_steps = int(max(1, num_training_steps))

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

# ==================================================================================================================
#                                                   SEMGAZE MODULE                                                 #
# ==================================================================================================================
class SemGazeModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = SemGaze(
            image_size=cfg.model.semgaze.image_size,
            patch_size=cfg.model.semgaze.patch_size, 
            token_dim=cfg.model.semgaze.token_dim, 
            gaze_vec_dim=cfg.model.semgaze.gaze_vec_dim, 
            image_encoder_name=cfg.model.semgaze.image_encoder_name,
            decoder_depth=cfg.model.semgaze.decoder_depth, 
            decoder_num_heads=cfg.model.semgaze.decoder_num_heads, 
            decoder_label_emb_dim=512,
            alignment_feature_dim=_cfg_get(cfg, "alignment.feature_dim", 768),
        )
        self.image_encoder_lora_enabled = False
        self._apply_image_encoder_lora()
        self.feature_map_size = cfg.model.semgaze.image_size // cfg.model.semgaze.patch_size
        
        self.dataset = cfg.experiment.dataset
        if self.dataset == "gazefollow":
            self.num_train_samples = cfg.data.gf.num_train_samples
            self.vocab_size = cfg.data.gf.vocab_size
        elif self.dataset == "gazehoi":
            self.num_train_samples = cfg.data.gazehoi.num_train_samples
            self.vocab_size = cfg.data.gazehoi.vocab_size
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.") 
            
        self.num_steps_in_epoch = math.ceil(self.num_train_samples / cfg.train.batch_size)
        
        # Define Metrics
        self.metrics = nn.ModuleDict({
            "val_dist": Distance(), 
            "test_dist": GFTestDistance() if self.dataset == "gazefollow" else Distance(),
            "test_acc@1": tm.Accuracy(task="multiclass", num_classes=self.vocab_size, top_k=1, ignore_index=-1),
            "test_acc@3": tm.Accuracy(task="multiclass", num_classes=self.vocab_size, top_k=3, ignore_index=-1),
        })
        
        if self.dataset == "gazefollow":
            self.metrics["test_multi_acc@1"] = MultiAccuracy(top_k=1, ignore_index=-1)
            self.metrics["test_auc"] = GFTestAUC()
        if self.dataset == "gazehoi":
            self.metrics["val_gaze_acc"] = GazeAccuracy()
            self.metrics["test_gaze_acc"] = GazeAccuracy()

        # Define logit scale parameter for loss computation
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / cfg.model.semgaze.temp_init_value)))
        self.align_logit_scale = nn.Parameter(torch.log(torch.tensor(1 / _cfg_get(cfg, "alignment.temp_init_value", 0.07))))
        self.align_enabled = bool(_cfg_get(cfg, "alignment.enabled", False))
        self.align_train_only = bool(_cfg_get(cfg, "alignment.train_only", True))
        self.align_layer_index = int(_cfg_get(cfg, "alignment.layer_index", 1))
        self.align_loss_type = str(_cfg_get(cfg, "alignment.loss_type", "cosine")).lower()
        self.align_weight = float(_cfg_get(cfg, "loss.weight_align", 0.0))
        self.label_objective = str(_cfg_get(cfg, "loss.label_objective", "legacy_infonce")).lower()
        self.label_margin_type = str(_cfg_get(cfg, "loss.label_margin_type", "none")).lower()
        self.label_margin = float(_cfg_get(cfg, "loss.label_margin", 0.0))
        self.label_easy_margin = bool(_cfg_get(cfg, "loss.label_easy_margin", False))
        self.register_buffer("vocab_emb", torch.empty(0), persistent=False)
        self.vocab = None
        
        # Initialize Weights
        self._init_weights()
        
    def _resolve_image_encoder_lora_targets(self, encoder, configured_targets):
        linear_suffixes = sorted(
            {name.split(".")[-1] for name, module in encoder.named_modules() if isinstance(module, nn.Linear)}
        )
        if isinstance(configured_targets, str):
            configured_targets = [configured_targets]
        configured_targets = [str(t) for t in configured_targets]

        if "auto" in configured_targets:
            preferred = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "query", "key", "value", "out_proj",
                "qkv", "proj",
            ]
            targets = [name for name in preferred if name in linear_suffixes]
            if targets:
                return targets
            raise ValueError(
                "Could not auto-resolve image encoder LoRA target modules. "
                f"Available linear module names: {linear_suffixes[:50]}"
            )

        targets = [name for name in configured_targets if name in linear_suffixes]
        if len(targets) == 0:
            raise ValueError(
                "Configured image encoder LoRA target_modules do not match any linear modules. "
                f"Configured={configured_targets}, available={linear_suffixes[:50]}"
            )
        return targets

    def _apply_image_encoder_lora(self):
        enabled = bool(_cfg_get(self.cfg, "model.semgaze.image_encoder_lora.enabled", False))
        if not enabled:
            return
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise ImportError("LoRA is enabled but `peft` is not installed. Install with `pip install peft`.")

        target_modules_cfg = _cfg_get(self.cfg, "model.semgaze.image_encoder_lora.target_modules", ["auto"])
        target_modules = self._resolve_image_encoder_lora_targets(self.model.encoder, target_modules_cfg)
        lora_cfg = LoraConfig(
            r=int(_cfg_get(self.cfg, "model.semgaze.image_encoder_lora.r", 8)),
            lora_alpha=int(_cfg_get(self.cfg, "model.semgaze.image_encoder_lora.alpha", 16)),
            lora_dropout=float(_cfg_get(self.cfg, "model.semgaze.image_encoder_lora.dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.model.encoder = get_peft_model(self.model.encoder, lora_cfg)
        self.image_encoder_lora_enabled = True
        print(colored(f"Enabled image encoder LoRA (targets={target_modules}).", TERM_COLOR))
        try:
            self.model.encoder.print_trainable_parameters()
        except Exception:
            pass


    def _init_weights(self):
        if self.cfg.model.weights is not None:
            model_ckpt = torch.load(self.cfg.model.weights, map_location="cpu", weights_only=False)  
            missing, unexpected = self.load_state_dict(model_ckpt["state_dict"], strict=False)
            print(colored(f"Loaded the model pre-trained weights from {self.cfg.model.weights}.", TERM_COLOR))
            if len(missing) > 0:
                print(colored(f"Missing keys while loading checkpoint: {missing}", TERM_COLOR))
            if len(unexpected) > 0:
                print(colored(f"Unexpected keys while loading checkpoint: {unexpected}", TERM_COLOR))
            del model_ckpt
        else:
            # Load Gaze360 Weights for Gaze Encoder Backbone
            gaze_backbone_ckpt = torch.load(self.cfg.model.pretraining.gaze_backbone, map_location="cpu")
            gaze_backbone_weights = OrderedDict([
                (name.replace("base_head.", ""), value) 
                for name, value in gaze_backbone_ckpt["model_state_dict"].items() 
                if "base_head" in name
            ])
            self.model.gaze_encoder.backbone.load_state_dict(gaze_backbone_weights, strict=True)
            print(colored(f"Loaded Gaze Backbone weights from {self.cfg.model.pretraining.gaze_backbone}.", TERM_COLOR))

            # Delete checkpoints
            del gaze_backbone_ckpt, gaze_backbone_weights
        
        # Freeze weights
        self.freeze()

    
    def _set_batchnorm_eval(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

            
    def _set_dropout_eval(self, module):
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()

            
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

            
    def freeze(self):
        if self.cfg.train.freeze.gaze_encoder:
            print(colored(f"Freezing the Gaze Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_encoder)
        if self.cfg.train.freeze.image_tokenizer:
            print(colored(f"Freezing the Image Tokenizer layers.", TERM_COLOR))
            self.freeze_module(self.model.image_tokenizer)
        if self.cfg.train.freeze.image_encoder:
            if self.image_encoder_lora_enabled:
                print(colored("Image encoder LoRA enabled: keeping adapter params trainable.", TERM_COLOR))
            else:
                print(colored(f"Freezing the Image Encoder layers.", TERM_COLOR))
                self.freeze_module(self.model.encoder)
        if self.cfg.train.freeze.gaze_decoder:
            print(colored(f"Freezing the Gaze Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_decoder)


    def forward(self, batch, return_alignment=False):
        return self.model(batch, return_alignment=return_alignment, align_layer_index=self.align_layer_index)
        
    
    def compute_loss(
        self, 
        gaze_heatmap_gt, 
        gaze_vec_gt, 
        gaze_label_emb_gt,
        gaze_label_id_gt,
        inout_gt, 
        gaze_heatmap_pred, 
        gaze_vec_pred, 
        gaze_label_emb_pred, 
    ):

        device = gaze_heatmap_pred.device
        
        heatmap_loss = torch.tensor(0.0, device=device)
        label_loss = torch.tensor(0.0, device=device)
        angular_loss = torch.tensor(0.0, device=device)

        if torch.sum(inout_gt) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
            heatmap_loss = compute_heatmap_loss(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt)
            if self.label_objective in {"legacy_infonce", "batch_local_infonce"}:
                label_loss = compute_info_nce_loss_batch_local(
                    gaze_label_emb_pred, gaze_label_emb_gt, inout_gt, self.logit_scale
                )
            else:
                self._ensure_vocab_embeddings()
                label_loss = compute_info_nce_loss(
                    emb_pred=gaze_label_emb_pred,
                    label_id_gt=gaze_label_id_gt,
                    io_gt=inout_gt,
                    logit_scale=self.logit_scale,
                    vocab_emb=self.vocab_emb,
                    margin_type=self.label_margin_type,
                    margin=self.label_margin,
                    easy_margin=self.label_easy_margin,
                )
            angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, inout_gt)
        
        total_loss = (
            self.cfg.loss.weight_heatmap * heatmap_loss +
            self.cfg.loss.weight_angular * angular_loss +
            self.cfg.loss.weight_label * label_loss
        )

        logs = {
            "heatmap_loss": heatmap_loss.detach(),
            "label_loss": label_loss.detach(),
            "angular_loss": angular_loss.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, logs

    
    def configure_optimizers(self):
        # Optimizer
        base_lr = float(self.cfg.optimizer.lr)
        head_lr_mult = float(_cfg_get(self.cfg, "alignment.head_lr_mult", 1.0))
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer_params = trainable_params
        if self.align_enabled and (head_lr_mult != 1.0):
            align_head_param_ids = {id(p) for p in self.model.alignment_head.parameters() if p.requires_grad}
            align_head_params = [p for p in trainable_params if id(p) in align_head_param_ids]
            base_params = [p for p in trainable_params if id(p) not in align_head_param_ids]
            if len(align_head_params) > 0:
                optimizer_params = []
                if len(base_params) > 0:
                    optimizer_params.append({"params": base_params, "lr": base_lr})
                optimizer_params.append({"params": align_head_params, "lr": base_lr * head_lr_mult})

        optimizer = optim.AdamW(
            optimizer_params,
            lr=base_lr,
            weight_decay=self.cfg.optimizer.weight_decay
        ) 
        
        # Scheduler: Cosine Annealing with Warmup or None
        if self.cfg.scheduler.type == "cosine_warmup":
            warmup_steps = self.cfg.scheduler.warmup_epochs * self.num_steps_in_epoch
            max_steps = self.cfg.train.epochs * self.num_steps_in_epoch
            scheduler = _get_cosine_schedule_with_warmup_torch(optimizer, warmup_steps, max_steps)
            scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return optimizer
            
                
    def on_fit_start(self):
        if self.label_objective not in {"legacy_infonce", "batch_local_infonce"}:
            self._ensure_vocab_embeddings()
        # Define metrics
        if self.cfg.wandb.log:
            if self.dataset == "gazefollow":
                wandb.define_metric('metric/test/dist_to_avg', summary='min')
                wandb.define_metric('metric/test/avg_dist', summary='min')
                wandb.define_metric('metric/test/min_dist', summary='min')
                wandb.define_metric('metric/test/auc', summary='max')
                wandb.define_metric('metric/test/multi_acc@1', summary='max')
            elif self.dataset == "gazehoi":
                wandb.define_metric('metric/val/gaze_acc', summary='max')
                wandb.define_metric('metric/test/gaze_acc', summary='max')
                wandb.define_metric('metric/test/dist', summary='min')
            
            wandb.define_metric('loss/train_epoch', summary='min')
            wandb.define_metric('loss/val', summary='min')
            wandb.define_metric('metric/val/dist', summary='min')
            wandb.define_metric('metric/test/acc@1', summary='max')
            wandb.define_metric('metric/test/acc@3', summary='max')

    def _ensure_vocab_embeddings(self):
        if self.vocab_emb.numel() > 0:
            return
        vocab2id_path = os.path.join(self.cfg.project.root, f"data/{self.dataset}/vocab2id.json")
        with open(vocab2id_path, "r") as f:
            vocab2id = json.load(f)
        vocab_size = len(vocab2id)
        id2vocab = [None] * vocab_size
        for label, idx in vocab2id.items():
            idx = int(idx)
            if idx < 0 or idx >= vocab_size:
                raise ValueError(f"Invalid vocab id {idx} for label `{label}`.")
            id2vocab[idx] = label
        if any(v is None for v in id2vocab):
            raise ValueError("vocab2id ids are not contiguous from 0..V-1.")

        vocab_emb = []
        for label in id2vocab:
            label_emb_path = os.path.join(self.cfg.project.root, f"data/{self.dataset}/label-embeds/{label}-emb.pt")
            label_emb = torch.load(label_emb_path, map_location="cpu", weights_only=False)
            label_emb = F.normalize(label_emb.to(torch.float32), p=2, dim=-1)
            vocab_emb.append(label_emb)

        self.vocab = id2vocab
        self.vocab_emb = torch.stack(vocab_emb, dim=0).to(self.device)

            
    def on_train_epoch_start(self):
        # Set BN layers to eval mode for frozen modules
        if self.cfg.train.freeze.gaze_encoder:
            self.model.gaze_encoder.apply(self._set_batchnorm_eval)
            self.model.gaze_encoder.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.image_tokenizer:
            self.model.image_tokenizer.apply(self._set_batchnorm_eval)
            self.model.image_tokenizer.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.image_encoder:
            if not self.image_encoder_lora_enabled:
                self.model.encoder.apply(self._set_batchnorm_eval)
                self.model.encoder.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.gaze_decoder:
            self.model.gaze_decoder.apply(self._set_batchnorm_eval)
            self.model.gaze_decoder.apply(self._set_dropout_eval)
            
            
    def training_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())
        inout_mask = batch["inout"] > 0.5
        reason_valid_mask = batch["reason_valid"] > 0.5
        align_valid_mask = reason_valid_mask & inout_mask
        align_path_active = self.align_enabled and ((not self.align_train_only) or self.training)
        
        # Forward pass
        if align_path_active:
            gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred, align_feat_pred = self(batch, return_alignment=True)
            align_feat_pred = align_feat_pred[:, -1, ...]  # select last annotated person token
        else:
            gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred = self(batch, return_alignment=False)
            align_feat_pred = None
        gaze_vec_pred = gaze_vec_pred[:, -1, ...] # (b, n, 64, 64) >> (b, 64, 64) / select last (annotated) person
        gaze_heatmap_pred = gaze_heatmap_pred[:, -1, ...] # (b, n, 64, 64) >> (b, 64, 64)
        gaze_label_emb_pred = gaze_label_emb_pred[:, -1, ...] # (b, n, 512) >> (b, 512)
                                
        # Compute loss
        loss, logs = self.compute_loss(
            batch["gaze_heatmap"], 
            batch["gaze_vec"], 
            batch["gaze_label_emb"],
            batch["gaze_label_id"],
            batch["inout"],
            gaze_heatmap_pred, 
            gaze_vec_pred, 
            gaze_label_emb_pred, 
        )

        align_loss = torch.tensor(0.0, device=loss.device)
        if align_path_active and align_feat_pred is not None:
            align_loss = compute_alignment_loss(
                emb_pred=align_feat_pred,
                emb_gt=batch["reason_emb"],
                valid_mask=align_valid_mask,
                loss_type=self.align_loss_type,
                logit_scale=self.align_logit_scale,
            )
            loss = loss + self.align_weight * align_loss

        # Logging losses
        self.log("loss/train/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/label", logs["label_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/align", align_loss.detach(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/base", logs["total_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/inout_ratio", inout_mask.float().mean(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/reason_valid_ratio", reason_valid_mask.float().mean(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/align_mask_ratio", align_valid_mask.float().mean(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/align_mask_count", align_valid_mask.float().sum(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", loss.detach(), batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}
    
    
    def on_after_backward(self):
        # Clipping temperature value
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0., 4.6052) # 4.6 = log(100)
        self.align_logit_scale.data = torch.clamp(self.align_logit_scale.data, 0., 4.6052)

        
    def validation_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())
        
        # Forward pass
        gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred = self(batch, return_alignment=False)
        gaze_vec_pred = gaze_vec_pred[:, -1, ...] # (b, n, 64, 64) >> (b, 64, 64) / select last (annotated) person
        gaze_heatmap_pred = gaze_heatmap_pred[:, -1, ...]  # (b, 1, 64, 64) >> (b, 64, 64)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)
        gaze_label_emb_pred = gaze_label_emb_pred[:, -1, ...] # (b, n, 512) >> (b, 512)
        
        # Compute loss
        loss, logs = self.compute_loss(
            batch["gaze_heatmap"], 
            batch["gaze_vec"], 
            batch["gaze_label_emb"],
            batch["gaze_label_id"],
            batch["inout"], 
            gaze_heatmap_pred, 
            gaze_vec_pred, 
            gaze_label_emb_pred
            )

        # Update metrics
        self.metrics["val_dist"].update(gaze_pt_pred, batch["gaze_pt"], batch["inout"])
        if "val_gaze_acc" in self.metrics:
            self.metrics["val_gaze_acc"].update(gaze_pt_pred, batch["obj_bbox"])
            self.log("metric/val/gaze_acc", self.metrics["val_gaze_acc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)

        # Logging losses
        self.log("loss/val/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/label", logs["label_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val", logs["total_loss"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Logging metrics
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        
    
    def on_test_start(self):        
        self._ensure_vocab_embeddings()
        
    
    def test_step(self, batch, batch_idx):
        
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())

        device = batch["gaze_pt"].device
        vocab_size = self.vocab_emb.size(0)
                
        # Forward pass
        gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred = self(batch, return_alignment=False)
        gaze_heatmap_pred = gaze_heatmap_pred[:, -1, ...]  # (b, 1, 64, 64) >> (b, 64, 64) / select last person
        gaze_pt_pred = dark_coordinate_decoding(gaze_heatmap_pred, kernel_size=self.cfg.data.heatmap_sigma * 3, normalize=True)       
        gaze_label_emb_pred = gaze_label_emb_pred[:, -1, ...] # (b, n, 512) >> (b, 512)
        gaze_label_logit_pred = gaze_label_emb_pred @ self.vocab_emb.T * self.logit_scale.exp() # (b, vocab_size)
            
        # Logging dataset-specific metrics
        if self.dataset == "gazefollow":
            test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, batch["gaze_pt"])
            self.metrics["test_auc"].update(gaze_heatmap_pred, batch["gaze_pt"])
            self.metrics["test_multi_acc@1"].update(gaze_label_logit_pred, batch["gaze_label_ids"])
            
            self.log("metric/test/auc", self.metrics["test_auc"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/avg_dist", test_avg_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/min_dist", test_min_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/multi_acc@1", self.metrics["test_multi_acc@1"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        
        elif self.dataset == "gazehoi":
            self.metrics["test_dist"].update(gaze_pt_pred, batch["gaze_pt"], batch["inout"])
            self.metrics["test_gaze_acc"].update(gaze_pt_pred, batch["obj_bbox"])
            
            self.log("metric/test/dist", self.metrics["test_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/gaze_acc", self.metrics["test_gaze_acc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        
        # Update and log common metrics
        self.metrics["test_acc@1"].update(gaze_label_logit_pred, batch["gaze_label_id"])
        self.metrics["test_acc@3"].update(gaze_label_logit_pred, batch["gaze_label_id"])

        self.log("metric/test/acc@1", self.metrics["test_acc@1"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        self.log("metric/test/acc@3", self.metrics["test_acc@3"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
 


# ==================================================================================================================== #
#                                                  SEMGAZE ARCHITECTURE                                                #
# ==================================================================================================================== #
class SemGaze(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        token_dim: int = 768,
        gaze_vec_dim: int = 2,
        image_encoder_name: str = "facebook/dinov3-base",
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        decoder_label_emb_dim: int = 512,
        alignment_feature_dim: int = 1024,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size # Added this line

        self.gaze_encoder = GazeEncoder(
            token_dim=token_dim, 
            feature_dim=512, 
            gaze_vec_dim=gaze_vec_dim
        )

        self.encoder = DINOv3ViTModel.from_pretrained(image_encoder_name)

        self.gaze_decoder = GazeDecoder(
            token_dim=token_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            label_emb_dim=decoder_label_emb_dim
        )
        align_hidden_dim = (token_dim + alignment_feature_dim) // 2
        self.alignment_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, align_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(align_hidden_dim, alignment_feature_dim),
        )


    def forward(self, sample, return_alignment: bool = False, align_layer_index: int = 1):
        # Expected sample = {"image": image, "heads": heads, "head_bboxes": head_bboxes}
        
        # Encode Gaze Tokens ===================================================
        gaze_tokens, gaze_vec = self.gaze_encoder(sample["heads"], sample["head_bboxes"])  # (b, n, d), (b, n, 2)
        
        # Encode Image =====================================================
        image_tokens = self.encoder(pixel_values=sample["image"]).last_hidden_state  # (b, t+1, d)
        image_tokens = image_tokens[:, (1 + self.encoder.config.num_register_tokens):, :] # (b, t, d), remove cls token and register tokens
        b, t, d = image_tokens.shape
        
        s = int(math.sqrt(t)) # This s should now be equal to s_spatial
        
        image_tokens = image_tokens.permute(0, 2, 1).view(b, d, s, s) # (b, d, t) >> (b, d, s, s)
        
        # Decode Gaze Target =====================================================
        if return_alignment:
            gaze_heatmap, gaze_label_emb, align_tokens = self.gaze_decoder(
                image_tokens,
                gaze_tokens,
                return_alignment_tokens=True,
                align_layer_index=align_layer_index,
            )
            align_feat = self.alignment_head(align_tokens)
            align_feat = F.normalize(align_feat, p=2, dim=-1)
            return gaze_heatmap, gaze_vec, gaze_label_emb, align_feat

        gaze_heatmap, gaze_label_emb = self.gaze_decoder(image_tokens, gaze_tokens)  # (b, n, hm_h, hm_w), (b, n, 512)
        return gaze_heatmap, gaze_vec, gaze_label_emb
