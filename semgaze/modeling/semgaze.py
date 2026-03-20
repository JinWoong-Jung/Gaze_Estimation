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
import omegaconf

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
        # Persist full resolved runtime config into checkpoints so eval-only runs
        # can rebuild the exact training-time architecture/configuration.
        try:
            cfg_serialized = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
            self.save_hyperparameters({"cfg": cfg_serialized})
        except Exception as exc:
            print(colored(f"Warning: failed to serialize cfg into checkpoint hyper_parameters ({exc}).", TERM_COLOR))
        self.image_encoder_name = str(
            _cfg_get(
                cfg,
                "model.semgaze.image_encoder.name",
                _cfg_get(cfg, "model.semgaze.image_encoder_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            )
        )
        self.image_encoder_freeze = bool(
            _cfg_get(
                cfg,
                "model.semgaze.image_encoder.freeze",
                _cfg_get(cfg, "train.freeze.image_encoder", False),
            )
        )
        self.image_encoder_unfreeze_last_n_blocks = max(
            0,
            int(_cfg_get(cfg, "model.semgaze.image_encoder.unfreeze_last_n_blocks", 0)),
        )
        self.test_tta_enabled = bool(_cfg_get(cfg, "test.tta.enabled", False))
        self.test_tta_hflip = bool(_cfg_get(cfg, "test.tta.hflip", True))
        self.test_l2_eval_mode = str(_cfg_get(cfg, "test.l2_eval_mode", "argmax")).lower()
        if self.test_l2_eval_mode not in ("argmax", "dark"):
            raise ValueError(
                f"Expected `test.l2_eval_mode` to be one of ['argmax', 'dark'], got {self.test_l2_eval_mode}."
            )

        self.model = SemGaze(
            image_size=cfg.model.semgaze.image_size,
            patch_size=cfg.model.semgaze.patch_size, 
            token_dim=cfg.model.semgaze.token_dim, 
            gaze_vec_dim=cfg.model.semgaze.gaze_vec_dim, 
            image_encoder_name=self.image_encoder_name,
            use_image_to_decoder_proj=bool(
                _cfg_get(
                    cfg,
                    "model.semgaze.image_to_decoder_proj.enabled",
                    _cfg_get(cfg, "model.semgaze.use_image_to_decoder_proj", True),
                )
            ),
            decoder_depth=cfg.model.semgaze.decoder_depth, 
            decoder_num_heads=cfg.model.semgaze.decoder_num_heads, 
            decoder_label_emb_dim=512,
            alignment_feature_dim=_cfg_get(
                cfg, "alignment_reasoning.feature_dim", _cfg_get(cfg, "alignment.feature_dim", 768)
            ),
            object_alignment_feature_dim=_cfg_get(
                cfg,
                "alignment_object.feature_dim",
                _cfg_get(cfg, "alignment_reasoning.feature_dim", _cfg_get(cfg, "alignment.feature_dim", 768)),
            ),
            reasoning_alignment_head_type=str(
                _cfg_get(cfg, "alignment_reasoning.head_type", _cfg_get(cfg, "alignment.head_type", "mlp"))
            ).lower(),
            object_alignment_head_type=str(
                _cfg_get(
                    cfg,
                    "alignment_object.head_type",
                    _cfg_get(
                        cfg,
                        "alignment_reasoning.head_type",
                        _cfg_get(cfg, "alignment.head_type", "mlp"),
                    ),
                )
            ).lower(),
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
        self.align_logit_scale = nn.Parameter(
            torch.log(
                torch.tensor(
                    1
                    / _cfg_get(
                        cfg,
                        "alignment_reasoning.temp_init_value",
                        _cfg_get(cfg, "alignment.temp_init_value", 0.07),
                    )
                )
            )
        )
        self.object_align_logit_scale = nn.Parameter(
            torch.log(
                torch.tensor(
                    1
                    / _cfg_get(
                        cfg,
                        "alignment_object.temp_init_value",
                        _cfg_get(
                            cfg,
                            "alignment_reasoning.temp_init_value",
                            _cfg_get(cfg, "alignment.temp_init_value", 0.07),
                        ),
                    )
                )
            )
        )
        self.align_enabled = bool(_cfg_get(cfg, "alignment_reasoning.enabled", _cfg_get(cfg, "alignment.enabled", False)))
        self.align_train_only = bool(
            _cfg_get(cfg, "alignment_reasoning.train_only", _cfg_get(cfg, "alignment.train_only", True))
        )
        self.align_layer_index = int(
            _cfg_get(cfg, "alignment_reasoning.layer_index", _cfg_get(cfg, "alignment.layer_index", 1))
        )
        self.align_loss_type = str(
            _cfg_get(cfg, "alignment_reasoning.loss_type", _cfg_get(cfg, "alignment.loss_type", "cosine"))
        ).lower()
        self.align_weight = float(_cfg_get(cfg, "loss.weight_align_reasoning", _cfg_get(cfg, "loss.weight_align", 0.0)))
        self.object_align_enabled = bool(_cfg_get(cfg, "alignment_object.enabled", False))
        self.object_align_train_only = bool(_cfg_get(cfg, "alignment_object.train_only", True))
        self.object_align_loss_type = str(
            _cfg_get(
                cfg,
                "alignment_object.loss_type",
                _cfg_get(cfg, "alignment_reasoning.loss_type", _cfg_get(cfg, "alignment.loss_type", "cosine")),
            )
        ).lower()
        self.object_align_weight = float(_cfg_get(cfg, "loss.weight_align_object", 0.0))
        self.heatmap_loss_fn = str(_cfg_get(cfg, "loss.heatmap_loss_fn", "mse")).lower()
        if self.heatmap_loss_fn not in {"mse", "bce"}:
            raise ValueError(
                f"Expected `loss.heatmap_loss_fn` to be one of ['mse', 'bce'], got {self.heatmap_loss_fn}."
            )
        self.label_objective = str(_cfg_get(cfg, "loss.label_objective", "legacy_infonce")).lower()
        self.label_margin_type = str(_cfg_get(cfg, "loss.label_margin_type", "none")).lower()
        self.label_margin = float(_cfg_get(cfg, "loss.label_margin", 0.0))
        self.label_easy_margin = bool(_cfg_get(cfg, "loss.label_easy_margin", False))
        self.out_of_frame_log_enabled = bool(
            _cfg_get(cfg, "train.out_of_frame_logging.enabled", self.dataset == "gazehoi")
        )
        self.out_of_frame_log_dir = str(
            _cfg_get(
                cfg,
                "train.out_of_frame_logging.dir",
                os.path.join(cfg.project.root, "logs", "gazehoi_out_of_frame"),
            )
        )
        self.out_of_frame_log_max_paths = int(
            _cfg_get(cfg, "train.out_of_frame_logging.max_paths_per_epoch", 500)
        )
        self._oof_epoch_count = 0
        self._oof_epoch_total = 0
        self._oof_epoch_paths = []
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
        enabled = bool(
            _cfg_get(
                self.cfg,
                "model.semgaze.image_encoder.lora.enabled",
                _cfg_get(self.cfg, "model.semgaze.image_encoder_lora.enabled", False),
            )
        )
        if not enabled:
            return
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise ImportError("LoRA is enabled but `peft` is not installed. Install with `pip install peft`.")

        target_modules_cfg = _cfg_get(
            self.cfg,
            "model.semgaze.image_encoder.lora.target_modules",
            _cfg_get(self.cfg, "model.semgaze.image_encoder_lora.target_modules", ["auto"]),
        )
        target_modules = self._resolve_image_encoder_lora_targets(self.model.encoder, target_modules_cfg)
        lora_cfg = LoraConfig(
            r=int(
                _cfg_get(
                    self.cfg,
                    "model.semgaze.image_encoder.lora.r",
                    _cfg_get(self.cfg, "model.semgaze.image_encoder_lora.r", 8),
                )
            ),
            lora_alpha=int(
                _cfg_get(
                    self.cfg,
                    "model.semgaze.image_encoder.lora.alpha",
                    _cfg_get(self.cfg, "model.semgaze.image_encoder_lora.alpha", 16),
                )
            ),
            lora_dropout=float(
                _cfg_get(
                    self.cfg,
                    "model.semgaze.image_encoder.lora.dropout",
                    _cfg_get(self.cfg, "model.semgaze.image_encoder_lora.dropout", 0.05),
                )
            ),
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

    def _get_image_encoder_blocks(self):
        candidates = []
        for name, module in self.model.encoder.named_modules():
            if not isinstance(module, nn.ModuleList):
                continue
            if len(module) == 0:
                continue
            if name.split(".")[-1] not in {"layer", "layers", "blocks"}:
                continue
            candidates.append((name, module))

        if len(candidates) == 0:
            return []

        candidates.sort(key=lambda x: (len(x[1]), x[0].count(".")), reverse=True)
        return list(candidates[0][1])

    def _unfreeze_image_encoder_last_blocks(self, num_blocks: int):
        if num_blocks <= 0:
            return

        blocks = self._get_image_encoder_blocks()
        if len(blocks) == 0:
            print(colored("Could not locate image encoder transformer blocks. Skipping partial unfreeze.", TERM_COLOR))
            return

        num_to_unfreeze = min(num_blocks, len(blocks))
        for block in blocks[-num_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True
        print(colored(f"Unfroze last {num_to_unfreeze}/{len(blocks)} image encoder blocks.", TERM_COLOR))

            
    def freeze(self):
        if self.cfg.train.freeze.gaze_encoder:
            print(colored(f"Freezing the Gaze Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_encoder)
        if self.cfg.train.freeze.image_tokenizer:
            print(colored(f"Freezing the Image Tokenizer layers.", TERM_COLOR))
            self.freeze_module(self.model.image_tokenizer)
        if self.image_encoder_freeze:
            if self.image_encoder_lora_enabled:
                print(colored("Image encoder LoRA enabled: keeping adapter params trainable.", TERM_COLOR))
            else:
                print(colored(f"Freezing the Image Encoder layers.", TERM_COLOR))
                self.freeze_module(self.model.encoder)
            if self.image_encoder_unfreeze_last_n_blocks > 0:
                self._unfreeze_image_encoder_last_blocks(self.image_encoder_unfreeze_last_n_blocks)
        if self.cfg.train.freeze.gaze_decoder:
            print(colored(f"Freezing the Gaze Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_decoder)


    def forward(self, batch, return_alignment=False, return_object_alignment=False):
        return self.model(
            batch,
            return_alignment=return_alignment,
            align_layer_index=self.align_layer_index,
            return_object_alignment=return_object_alignment,
        )

    def _build_hflip_batch(self, batch):
        batch_flip = dict(batch)
        batch_flip["image"] = torch.flip(batch["image"], dims=[-1])

        if "heads" in batch and torch.is_tensor(batch["heads"]):
            batch_flip["heads"] = torch.flip(batch["heads"], dims=[-1])

        if "head_bboxes" in batch and torch.is_tensor(batch["head_bboxes"]):
            head_bboxes = batch["head_bboxes"].clone()
            head_bboxes[..., [0, 2]] = 1.0 - head_bboxes[..., [2, 0]]
            batch_flip["head_bboxes"] = head_bboxes
        return batch_flip

    def _forward_test(self, batch):
        gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred = self(batch, return_alignment=False)
        if not (self.test_tta_enabled and self.test_tta_hflip):
            return gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred

        batch_flip = self._build_hflip_batch(batch)
        gaze_heatmap_pred_flip, _, _ = self(batch_flip, return_alignment=False)
        gaze_heatmap_pred_flip = torch.flip(gaze_heatmap_pred_flip, dims=[-1])
        gaze_heatmap_pred = 0.5 * (gaze_heatmap_pred + gaze_heatmap_pred_flip)
        return gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred
        
    
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
            heatmap_loss = compute_heatmap_loss(
                gaze_heatmap_pred,
                gaze_heatmap_gt,
                inout_gt,
                loss_fn=self.heatmap_loss_fn,
            )
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
        reason_head_lr_mult = float(
            _cfg_get(self.cfg, "alignment_reasoning.head_lr_mult", _cfg_get(self.cfg, "alignment.head_lr_mult", 1.0))
        )
        object_head_lr_mult = float(_cfg_get(self.cfg, "alignment_object.head_lr_mult", 1.0))
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer_params = trainable_params
        param_groups = []
        grouped_param_ids = set()

        if self.align_enabled and (reason_head_lr_mult != 1.0):
            align_head_param_ids = {id(p) for p in self.model.alignment_head.parameters() if p.requires_grad}
            align_head_params = [p for p in trainable_params if id(p) in align_head_param_ids]
            if len(align_head_params) > 0:
                param_groups.append({"params": align_head_params, "lr": base_lr * reason_head_lr_mult})
                grouped_param_ids.update(align_head_param_ids)

        if self.object_align_enabled and (object_head_lr_mult != 1.0):
            object_head_param_ids = {id(p) for p in self.model.object_alignment_head.parameters() if p.requires_grad}
            object_head_params = [p for p in trainable_params if id(p) in object_head_param_ids]
            if len(object_head_params) > 0:
                param_groups.append({"params": object_head_params, "lr": base_lr * object_head_lr_mult})
                grouped_param_ids.update(object_head_param_ids)

        if len(param_groups) > 0:
            base_params = [p for p in trainable_params if id(p) not in grouped_param_ids]
            optimizer_params = []
            if len(base_params) > 0:
                optimizer_params.append({"params": base_params, "lr": base_lr})
            optimizer_params.extend(param_groups)

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
        if self.image_encoder_freeze:
            if (not self.image_encoder_lora_enabled) and (self.image_encoder_unfreeze_last_n_blocks <= 0):
                self.model.encoder.apply(self._set_batchnorm_eval)
                self.model.encoder.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.gaze_decoder:
            self.model.gaze_decoder.apply(self._set_batchnorm_eval)
            self.model.gaze_decoder.apply(self._set_dropout_eval)
        self._oof_epoch_count = 0
        self._oof_epoch_total = 0
        self._oof_epoch_paths = []
        if self.out_of_frame_log_enabled and self.dataset == "gazehoi":
            if self.trainer is not None and self.trainer.is_global_zero:
                os.makedirs(self.out_of_frame_log_dir, exist_ok=True)
            
    def _select_target_person_token(self, token_tensor: torch.Tensor, batch: Dict) -> torch.Tensor:
        if token_tensor.ndim < 2:
            return token_tensor

        target_head_idx = batch.get("target_head_idx", None)
        if target_head_idx is None:
            return token_tensor[:, -1, ...]

        if not torch.is_tensor(target_head_idx):
            target_head_idx = torch.tensor(target_head_idx, device=token_tensor.device)
        target_head_idx = target_head_idx.to(device=token_tensor.device, dtype=torch.long).view(-1)
        if target_head_idx.numel() != token_tensor.size(0):
            return token_tensor[:, -1, ...]

        target_head_idx = target_head_idx.clamp(min=0, max=token_tensor.size(1) - 1)
        gather_shape = [token_tensor.size(0), 1] + [1] * (token_tensor.ndim - 2)
        gather_index = target_head_idx.view(*gather_shape).expand(-1, 1, *token_tensor.shape[2:])
        return token_tensor.gather(1, gather_index).squeeze(1)
            
            
    def training_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())
        inout_mask = batch["inout"] > 0.5
        if self.out_of_frame_log_enabled and self.dataset == "gazehoi":
            out_mask = ~inout_mask
            out_count = int(out_mask.sum().item())
            self._oof_epoch_count += out_count
            self._oof_epoch_total += int(n)

            if out_count > 0 and len(self._oof_epoch_paths) < self.out_of_frame_log_max_paths:
                batch_paths = batch.get("path", None)
                if isinstance(batch_paths, (list, tuple)):
                    out_indices = torch.nonzero(out_mask, as_tuple=False).flatten().tolist()
                    remaining = self.out_of_frame_log_max_paths - len(self._oof_epoch_paths)
                    for i in out_indices[:remaining]:
                        if 0 <= i < len(batch_paths):
                            self._oof_epoch_paths.append(str(batch_paths[i]))
                        else:
                            self._oof_epoch_paths.append("UNKNOWN_PATH")
        reasoning_valid_mask = (
            (batch["reasoning_valid"] > 0.5)
            if "reasoning_valid" in batch
            else ((batch["reason_valid"] > 0.5) if "reason_valid" in batch else torch.zeros_like(inout_mask, dtype=torch.bool))
        )
        object_valid_mask = (
            (batch["object_valid"] > 0.5)
            if "object_valid" in batch
            else torch.zeros_like(inout_mask, dtype=torch.bool)
        )
        align_valid_mask = reasoning_valid_mask & inout_mask
        object_align_valid_mask = object_valid_mask & inout_mask
        align_path_active = (
            self.align_enabled
            and ((not self.align_train_only) or self.training)
            and (("reasoning_emb" in batch) or ("reason_emb" in batch))
            and (("reasoning_valid" in batch) or ("reason_valid" in batch))
        )
        object_align_path_active = (
            self.object_align_enabled
            and ((not self.object_align_train_only) or self.training)
            and ("object_emb" in batch)
            and ("object_valid" in batch)
        )
        
        # Forward pass
        if align_path_active and object_align_path_active:
            gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred, align_feat_pred, object_align_feat_pred = self(
                batch,
                return_alignment=True,
                return_object_alignment=True,
            )
            align_feat_pred = self._select_target_person_token(align_feat_pred, batch)
            object_align_feat_pred = self._select_target_person_token(object_align_feat_pred, batch)
        elif align_path_active:
            gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred, align_feat_pred = self(
                batch,
                return_alignment=True,
                return_object_alignment=False,
            )
            align_feat_pred = self._select_target_person_token(align_feat_pred, batch)
            object_align_feat_pred = None
        elif object_align_path_active:
            gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred, object_align_feat_pred = self(
                batch,
                return_alignment=False,
                return_object_alignment=True,
            )
            align_feat_pred = None
            object_align_feat_pred = self._select_target_person_token(object_align_feat_pred, batch)
        else:
            gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred = self(
                batch,
                return_alignment=False,
                return_object_alignment=False,
            )
            align_feat_pred = None
            object_align_feat_pred = None
        gaze_vec_pred = self._select_target_person_token(gaze_vec_pred, batch)
        gaze_heatmap_pred = self._select_target_person_token(gaze_heatmap_pred, batch)
        gaze_label_emb_pred = self._select_target_person_token(gaze_label_emb_pred, batch)
                                
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
            reasoning_emb_gt = batch["reasoning_emb"] if "reasoning_emb" in batch else batch["reason_emb"]
            align_loss = compute_alignment_loss(
                emb_pred=align_feat_pred,
                emb_gt=reasoning_emb_gt,
                valid_mask=align_valid_mask,
                loss_type=self.align_loss_type,
                logit_scale=self.align_logit_scale,
            )
            loss = loss + self.align_weight * align_loss

        object_align_loss = torch.tensor(0.0, device=loss.device)
        if object_align_path_active and object_align_feat_pred is not None:
            object_align_loss = compute_alignment_loss(
                emb_pred=object_align_feat_pred,
                emb_gt=batch["object_emb"],
                valid_mask=object_align_valid_mask,
                loss_type=self.object_align_loss_type,
                logit_scale=self.object_align_logit_scale,
            )
            loss = loss + self.object_align_weight * object_align_loss

        # Logging losses
        self.log("loss/train/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/label", logs["label_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/align_reasoning", align_loss.detach(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/align_object", object_align_loss.detach(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/inout_ratio", inout_mask.float().mean(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/reasoning_valid_ratio", reasoning_valid_mask.float().mean(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/object_valid_ratio", object_valid_mask.float().mean(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/reasoning_align_mask_ratio", align_valid_mask.float().mean(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/reasoning_align_mask_count", align_valid_mask.float().sum(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/object_align_mask_ratio", object_align_valid_mask.float().mean(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("stats/train/object_align_mask_count", object_align_valid_mask.float().sum(), batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", loss.detach(), batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}


    def on_train_epoch_end(self):
        if not (self.out_of_frame_log_enabled and self.dataset == "gazehoi"):
            return

        total = int(self._oof_epoch_total)
        out = int(self._oof_epoch_count)
        ratio = (float(out) / float(total)) if total > 0 else 0.0
        self.log("stats/train/out_of_frame_count", float(out), prog_bar=False, on_step=False, on_epoch=True)
        self.log("stats/train/out_of_frame_ratio", ratio, prog_bar=False, on_step=False, on_epoch=True)

        if self.trainer is None or (not self.trainer.is_global_zero):
            return

        os.makedirs(self.out_of_frame_log_dir, exist_ok=True)
        epoch_idx = int(self.current_epoch)
        log_path = os.path.join(self.out_of_frame_log_dir, f"epoch_{epoch_idx:03d}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"epoch: {epoch_idx}\n")
            f.write(f"dataset: {self.dataset}\n")
            f.write(f"total_train_samples_seen: {total}\n")
            f.write(f"out_of_frame_count: {out}\n")
            f.write(f"out_of_frame_ratio: {ratio:.8f}\n")
            f.write(f"max_paths_per_epoch: {self.out_of_frame_log_max_paths}\n")
            f.write(f"logged_paths_count: {len(self._oof_epoch_paths)}\n")
            f.write("paths:\n")
            for p in self._oof_epoch_paths:
                f.write(f"{p}\n")
    
    
    def on_after_backward(self):
        # Clipping temperature value
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0., 4.6052) # 4.6 = log(100)
        self.align_logit_scale.data = torch.clamp(self.align_logit_scale.data, 0., 4.6052)
        self.object_align_logit_scale.data = torch.clamp(self.object_align_logit_scale.data, 0., 4.6052)

        
    def validation_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())
        
        # Forward pass
        gaze_heatmap_pred, gaze_vec_pred, gaze_label_emb_pred = self(batch, return_alignment=False)
        gaze_vec_pred = self._select_target_person_token(gaze_vec_pred, batch)
        gaze_heatmap_pred = self._select_target_person_token(gaze_heatmap_pred, batch)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)
        gaze_label_emb_pred = self._select_target_person_token(gaze_label_emb_pred, batch)
        
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

        # Forward pass
        gaze_heatmap_pred, _, gaze_label_emb_pred = self._forward_test(batch)
        gaze_heatmap_pred = self._select_target_person_token(gaze_heatmap_pred, batch)
        gaze_label_emb_pred = self._select_target_person_token(gaze_label_emb_pred, batch)
        gaze_label_logit_pred = gaze_label_emb_pred @ self.vocab_emb.T * self.logit_scale.exp() # (b, vocab_size)
            
        # Logging dataset-specific metrics
        if self.dataset == "gazefollow":
            if self.test_l2_eval_mode == "dark":
                gaze_pt_pred = dark_coordinate_decoding(
                    gaze_heatmap_pred,
                    kernel_size=self.cfg.data.heatmap_sigma * 3,
                    normalize=True,
                )
            else:
                gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)

            test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, batch["gaze_pt"])
            self.metrics["test_auc"].update(gaze_heatmap_pred, batch["gaze_pt"], batch["img_size"])
            self.metrics["test_multi_acc@1"].update(gaze_label_logit_pred, batch["gaze_label_ids"])
            
            self.log("metric/test/auc", self.metrics["test_auc"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/avg_dist", test_avg_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/min_dist", test_min_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/multi_acc@1", self.metrics["test_multi_acc@1"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        
        elif self.dataset == "gazehoi":
            gaze_pt_pred = dark_coordinate_decoding(
                gaze_heatmap_pred,
                kernel_size=self.cfg.data.heatmap_sigma * 3,
                normalize=True,
            )
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
        use_image_to_decoder_proj: bool = True,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        decoder_label_emb_dim: int = 512,
        alignment_feature_dim: int = 1024,
        object_alignment_feature_dim: int = 1024,
        reasoning_alignment_head_type: str = "mlp",
        object_alignment_head_type: str = "mlp",
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
        self.image_encoder_dim = int(getattr(self.encoder.config, "hidden_size", token_dim))
        self.use_image_to_decoder_proj = bool(use_image_to_decoder_proj)
        self.image_to_decoder_proj = nn.Linear(self.image_encoder_dim, token_dim)
        if self.image_encoder_dim == token_dim:
            nn.init.eye_(self.image_to_decoder_proj.weight)
            nn.init.zeros_(self.image_to_decoder_proj.bias)
        if (not self.use_image_to_decoder_proj) and (self.image_encoder_dim != token_dim):
            raise ValueError(
                "image_to_decoder_proj is disabled, but image encoder dim "
                f"({self.image_encoder_dim}) != token_dim ({token_dim}). "
                "Set model.semgaze.image_to_decoder_proj.enabled=True, "
                "or make token_dim match the image encoder hidden size."
            )

        self.gaze_decoder = GazeDecoder(
            token_dim=token_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            label_emb_dim=decoder_label_emb_dim
        )
        self.alignment_head = self._build_alignment_head(
            token_dim=token_dim,
            output_dim=alignment_feature_dim,
            head_type=reasoning_alignment_head_type,
        )
        self.object_alignment_head = self._build_alignment_head(
            token_dim=token_dim,
            output_dim=object_alignment_feature_dim,
            head_type=object_alignment_head_type,
        )

    @staticmethod
    def _build_alignment_head(token_dim: int, output_dim: int, head_type: str) -> nn.Module:
        head_type = str(head_type).lower()
        if head_type == "projection":
            return nn.Linear(token_dim, output_dim)
        if head_type == "mlp":
            hidden_dim = (token_dim + output_dim) // 2
            return nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, output_dim),
            )
        raise ValueError(f"Unsupported alignment head_type: {head_type}. Use one of {{'projection', 'mlp'}}.")


    def forward(
        self,
        sample,
        return_alignment: bool = False,
        align_layer_index: int = 1,
        return_object_alignment: bool = False,
    ):
        # Expected sample = {"image": image, "heads": heads, "head_bboxes": head_bboxes}
        
        # Encode Gaze Tokens ===================================================
        gaze_tokens, gaze_vec = self.gaze_encoder(sample["heads"], sample["head_bboxes"])  # (b, n, d), (b, n, 2)
        
        # Encode Image =====================================================
        image_tokens = self.encoder(pixel_values=sample["image"]).last_hidden_state  # (b, t+1, d)
        image_tokens = image_tokens[:, (1 + self.encoder.config.num_register_tokens):, :] # (b, t, d), remove cls token and register tokens
        if self.use_image_to_decoder_proj:
            image_tokens = self.image_to_decoder_proj(image_tokens)  # (b, t, token_dim)
        b, t, d = image_tokens.shape
        
        s = int(math.sqrt(t)) # This s should now be equal to s_spatial
        
        image_tokens = image_tokens.permute(0, 2, 1).view(b, d, s, s) # (b, d, t) >> (b, d, s, s)
        
        # Decode Gaze Target =====================================================
        if return_alignment and return_object_alignment:
            gaze_heatmap, gaze_label_emb, align_tokens, object_tokens = self.gaze_decoder(
                image_tokens,
                gaze_tokens,
                return_alignment_tokens=True,
                align_layer_index=align_layer_index,
                return_object_tokens=True,
            )
            align_feat = self.alignment_head(align_tokens)
            align_feat = F.normalize(align_feat, p=2, dim=-1)
            object_align_feat = self.object_alignment_head(object_tokens)
            object_align_feat = F.normalize(object_align_feat, p=2, dim=-1)
            return gaze_heatmap, gaze_vec, gaze_label_emb, align_feat, object_align_feat

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

        if return_object_alignment:
            gaze_heatmap, gaze_label_emb, object_tokens = self.gaze_decoder(
                image_tokens,
                gaze_tokens,
                return_alignment_tokens=False,
                return_object_tokens=True,
            )
            object_align_feat = self.object_alignment_head(object_tokens)
            object_align_feat = F.normalize(object_align_feat, p=2, dim=-1)
            return gaze_heatmap, gaze_vec, gaze_label_emb, object_align_feat

        gaze_heatmap, gaze_label_emb = self.gaze_decoder(image_tokens, gaze_tokens)  # (b, n, hm_h, hm_w), (b, n, 512)
        return gaze_heatmap, gaze_vec, gaze_label_emb
