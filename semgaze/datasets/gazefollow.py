#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import os
import json
import h5py
import time
from typing import Dict, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

from semgaze.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCropSafeGaze,
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from semgaze.utils.common import pair, expand_bbox, generate_gaze_heatmap, generate_mask, get_img_size, square_bbox



IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ============================================================================= #
#                               GAZEFOLLOW DATASET                              #
# ============================================================================= #
class GazeFollowDataset(Dataset):
    def __init__(
        self,
        root,
        root_project,
        root_heads,
        split: str = "train",
        transform: Union[Compose, None] = None,
        tr: tuple = (-0.1, 0.1),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 1,
        head_thr: float = 0.5,
        return_head_mask: bool = False,
        reason_feature_root: Union[str, None] = None,
        reason_feature_preload: bool = False,
        reason_feature_dim: int = 768,
        reason_log_limit: int = 20,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), f"Expected `split` to be one of [`train`, `val`, `test`] but received `{split}` instead."
        assert (num_people == -1) or (num_people > 0), f"Expected `num_people` to be strictly positive or `-1`, but received {num_people} instead."
        assert 0 <= head_thr <= 1, f"Expected `head_thr` to be in [0, 1]. Received {head_thr} instead."

        self.root = root
        self.root_project = root_project
        self.root_heads = root_heads
        self.split = split
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask
        self.reason_feature_root = reason_feature_root
        self.reason_feature_preload = reason_feature_preload
        self.reason_feature_dim = reason_feature_dim
        self.reason_log_limit = reason_log_limit
        self.reason_warn_count = 0
        self.label_emb_cache = {}
        self.annotations, self.vocab2id = self.load_annotations()
        self.reason_feature_cache = None
        if (self.split == "train") and (self.reason_feature_root is not None) and self.reason_feature_preload:
            self.reason_feature_cache = self._build_reason_feature_cache()

    def _warn_reason(self, msg: str):
        if self.reason_warn_count < self.reason_log_limit:
            print(f"[GazeFollowDataset][reason] {msg}")
            self.reason_warn_count += 1
            if self.reason_warn_count == self.reason_log_limit:
                print("[GazeFollowDataset][reason] warning log limit reached; suppressing further messages.")

    def _get_reason_feature_path(self, image_path: str, sample_id: Union[int, str]) -> str:
        rel_dir = os.path.dirname(image_path)  # e.g. train/00000000
        basename = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{basename}_{sample_id}.pt"
        return os.path.join(self.reason_feature_root, rel_dir, filename)

    def _get_label_embedding(self, gaze_label: str) -> torch.Tensor:
        if gaze_label not in self.label_emb_cache:
            label_emb_path = os.path.join(self.root_project, f"data/gazefollow/label-embeds/{gaze_label}-emb.pt")
            label_emb = torch.load(label_emb_path, weights_only=False).to(torch.float32)
            label_emb = F.normalize(label_emb, p=2, dim=-1)
            self.label_emb_cache[gaze_label] = label_emb
        return self.label_emb_cache[gaze_label].clone()

    def _build_reason_feature_cache(self):
        reason_cache = {}
        loaded_count = 0
        missing_count = 0
        failed_count = 0
        total_count = len(self.annotations)
        start_time = time.time()
        print(
            f"[GazeFollowDataset][reason] preload start: total={total_count}",
            flush=True,
        )

        for index, item in enumerate(self.annotations.itertuples(index=False), start=1):
            reason_path = self._get_reason_feature_path(item.path, item.id)
            try:
                reason_emb = torch.load(reason_path, map_location="cpu", weights_only=False).to(torch.float32)
                reason_cache[reason_path] = F.normalize(reason_emb, p=2, dim=-1)
                loaded_count += 1
            except FileNotFoundError:
                reason_cache[reason_path] = None
                missing_count += 1
            except Exception:
                reason_cache[reason_path] = None
                failed_count += 1

            if (index % 10000 == 0) or (index == total_count):
                elapsed = time.time() - start_time
                speed = index / elapsed if elapsed > 0 else 0.0
                print(
                    f"[GazeFollowDataset][reason] preload progress: "
                    f"{index}/{total_count} ({(100.0 * index / total_count):.1f}%), "
                    f"loaded={loaded_count}, missing={missing_count}, failed={failed_count}, "
                    f"speed={speed:.1f} files/s",
                    flush=True,
                )

        total_elapsed = time.time() - start_time
        print(
            f"[GazeFollowDataset][reason] preload complete: "
            f"loaded={loaded_count}, missing={missing_count}, failed={failed_count}, "
            f"elapsed={total_elapsed:.1f}s"
        )
        return reason_cache

    def load_annotations(self) -> pd.DataFrame:
        annotations = pd.DataFrame()
        if self.split == "test":
            columns = ["path", "id", "body_x", "body_y", "body_w", "body_h", "eye_x", "eye_y", "gaze_x", "gaze_y", 
                       "head_xmin", "head_ymin", "head_xmax", "head_ymax", "origin", "meta"]
            annotations = pd.read_csv(
                os.path.join(self.root, "test_annotations_release.txt"),
                sep=",",
                names=columns,
                index_col=False,
                encoding="utf-8-sig",
            )
            # Add inout col for consistency (ie. missing from test set)
            annotations["inout"] = 1
            # Each test image is annotated by multiple people (around 10 on avg.)
            self.image_paths = annotations.path.unique().tolist()
            self.length = len(self.image_paths)

        elif self.split in ("train", "val"):
            columns = ["path", "id", "body_x", "body_y", "body_w", "body_h", "eye_x", "eye_y", "gaze_x", "gaze_y", 
                       "head_xmin", "head_ymin", "head_xmax", "head_ymax", "inout", "origin", "meta"]
            annotations = pd.read_csv(
                os.path.join(self.root_project, f"data/gazefollow/{self.split}_annotations_new.txt"), # reprocessed train/val head bboxes
                sep=",",
                names=columns,
                index_col=False,
                encoding="utf-8-sig",
            )
            # Clean annotations (e.g. remove invalid ones)
            annotations = self._clean_annotations(annotations)
            self.length = len(annotations)
            
        #  Load Gaze Labels and merge with annotations
        df_label = pd.read_csv(os.path.join(self.root_project, f"data/gazefollow/gaze-labels-{self.split}.csv"))
        merge_on = ["path", "id"] if self.split in ["train", "val"] else ["path"]
        annotations = pd.merge(annotations, df_label, how="left", on=merge_on)
        
        # Each test image is annotated by multiple people (around 10 on avg.)
        self.image_paths = sorted(annotations.path.unique())
        self.length = len(self.image_paths) if self.split == "test" else len(annotations)
        
        # Load vocab2id
        with open(os.path.join(self.root_project, 'data/gazefollow/vocab2id.json'), 'r') as f:
            vocab2id = json.load(f)

        return annotations, vocab2id


    def _clean_annotations(self, annotations):
        # Only keep "in" and "out". (-1 is invalid)
        annotations = annotations[annotations.inout != -1]
        # Discard instances where max in bbox coordinates is smaller than min
        annotations = annotations[annotations.head_xmin < annotations.head_xmax]
        annotations = annotations[annotations.head_ymin < annotations.head_ymax]
        return annotations.reset_index(drop=True)

    def __getitem__(self, index: int) -> Dict:
        if self.split in ("train", "val"):
            item = self.annotations.iloc[index]
            gaze_pt = torch.tensor([item["gaze_x"], item["gaze_y"]], dtype=torch.float)
            gaze_label = item["gaze_pseudo_label"]
            gaze_labels = [gaze_label]
            gaze_label_id = torch.tensor(item["label_id"])
            gaze_label_ids = torch.tensor([gaze_label_id])
            idx = item["id"]
        elif self.split == "test":
            image_path = self.image_paths[index]
            p_annotations = self.annotations[self.annotations.path == image_path]
            gaze_pt = torch.from_numpy(p_annotations[["gaze_x", "gaze_y"]].values).float()
            p = 20 - len(gaze_pt)
            gaze_pt = F.pad(gaze_pt, (0, 0, 0, p), value=-1.0)
            idx = p_annotations["id"].values.tolist() + [-1] * p
            item = p_annotations.iloc[0]
            gaze_label = item.gaze_gt_label
            gaze_labels = item.gaze_gt_labels
            gaze_label_id = torch.tensor(item.test_label_id)
            gaze_label_ids = torch.tensor([gaze_label_id])
            if gaze_label_id != -1:
                gaze_label_ids = torch.tensor([self.vocab2id[label] for label in gaze_labels.split('-')])
            l = 5 - len(gaze_label_ids)
            gaze_label_ids = F.pad(gaze_label_ids, (0, l), value=-1)

        inout = torch.tensor(item["inout"], dtype=torch.float)
        path = item["path"]

        # --- Original, slow path: Load raw image and process ---
        raw_image = Image.open(os.path.join(self.root, path)).convert("RGB")
        img_w, img_h = raw_image.size
        
        target_head_bbox = item[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]]
        target_head_bbox = torch.from_numpy(target_head_bbox.values.astype(np.float32)).unsqueeze(0)
        target_head_bbox = expand_bbox(target_head_bbox, img_w, img_h, k=0.1)

        context_head_bboxes = torch.zeros((0, 4))
        if (self.num_people == -1) or (self.num_people > 1):
            split_from_path, partition, basename_with_ext = path.split('/')
            basename, _ = os.path.splitext(basename_with_ext)
            det_file = f"{split_from_path}/{partition}/{basename}-head-detections.npy"
            det_path = os.path.join(self.root_heads, det_file)
            if os.path.exists(det_path):
                detections = np.load(det_path)
                if len(detections) > 0:
                    scores = torch.tensor(detections[:, -1])
                    context_head_bboxes = torch.tensor(detections[(scores >= self.head_thr).tolist(), :-1])
                    ious = box_iou(context_head_bboxes, target_head_bbox).flatten()
                    context_head_bboxes = context_head_bboxes[ious <= 0.5]
            if self.split == "train":
                perm_indices = torch.randperm(context_head_bboxes.size(0))
                context_head_bboxes = context_head_bboxes[perm_indices]
            num_context_heads = len(context_head_bboxes)
            num_keep = num_context_heads if self.num_people == -1 else self.num_people - 1
            context_head_bboxes = context_head_bboxes[:num_keep]

        head_bboxes = torch.concat([context_head_bboxes, target_head_bbox], dim=0).to(torch.float)
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

        head_bboxes = square_bbox(head_bboxes, img_w, img_h)
        heads = []
        for head_bbox in head_bboxes:
            heads.append(raw_image.crop(head_bbox.int().tolist()))

        head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        head_bboxes = torch.clamp(head_bboxes, min=0.0, max=1.0)
        
        image = raw_image

        if pd.isnull(gaze_label):
            gaze_label = gaze_labels = ""
            gaze_label_emb = torch.zeros(512, dtype=torch.float32)
        else:
            gaze_label_emb = self._get_label_embedding(gaze_label)

        reason_emb = torch.zeros(self.reason_feature_dim, dtype=torch.float32)
        reason_valid = torch.tensor(0.0, dtype=torch.float32)
        if (self.split == "train") and (self.reason_feature_root is not None):
            reason_path = self._get_reason_feature_path(path, idx)
            if self.reason_feature_cache is not None:
                cached_reason_emb = self.reason_feature_cache.get(reason_path)
                if cached_reason_emb is not None:
                    reason_emb = cached_reason_emb
                    reason_valid = torch.tensor(1.0, dtype=torch.float32)
            else:
                try:
                    reason_emb = torch.load(reason_path, map_location="cpu", weights_only=False).to(torch.float32)
                    reason_emb = F.normalize(reason_emb, p=2, dim=-1)
                    reason_valid = torch.tensor(1.0, dtype=torch.float32)
                except FileNotFoundError:
                    self._warn_reason(f"missing feature file: {reason_path}")
                except Exception as exc:
                    self._warn_reason(f"failed loading feature: {reason_path} ({exc})")

        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pt": gaze_pt,
            "gaze_label": gaze_label,
            "gaze_label_id": gaze_label_id,
            "gaze_labels": gaze_labels,
            "gaze_label_ids": gaze_label_ids,
            "gaze_label_emb": gaze_label_emb,
            "inout": inout,
            "reason_emb": reason_emb,
            "reason_valid": reason_valid,
            "id": idx,
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": path,
        }

        if self.transform:
            sample = self.transform(sample)

        num_heads_current = len(sample["heads"])
        num_target_people = self.num_people if self.num_people != -1 else num_heads_current
        
        num_missing_heads = num_target_people - num_heads_current

        if num_missing_heads > 0:
            pad_bbox = (0, 0, num_missing_heads, 0)
            sample["head_bboxes"] = F.pad(sample["head_bboxes"], pad_bbox, mode="constant", value=0.)
            
            single_head_shape = sample["heads"].shape[1:] 
            blank_head_tensors = torch.zeros((num_missing_heads, *single_head_shape), dtype=sample["heads"].dtype, device=sample["heads"].device)
            sample["heads"] = torch.cat([sample["heads"], blank_head_tensors], dim=0)
        elif num_missing_heads < 0:
            sample["heads"] = sample["heads"][:num_target_people]
            sample["head_bboxes"] = sample["head_bboxes"][:num_target_people]

        sample["head_centers"] = torch.hstack(
            [
                (sample["head_bboxes"][:, [0]] + sample["head_bboxes"][:, [2]]) / 2,
                (sample["head_bboxes"][:, [1]] + sample["head_bboxes"][:, [3]]) / 2,
            ]
        )

        if sample["inout"] == 1.0:
            sample["gaze_heatmap"] = generate_gaze_heatmap(sample["gaze_pt"], sigma=self.heatmap_sigma, size=self.heatmap_size)
        else:
            sample["gaze_heatmap"] = torch.zeros((self.heatmap_size, self.heatmap_size), dtype=torch.float)

        new_img_w, new_img_h = get_img_size(sample["image"])
        gaze_vec = sample["gaze_pt"] - sample["head_centers"][-1]
        gaze_vec = gaze_vec * torch.tensor([new_img_w, new_img_h])
        sample["gaze_vec"] = F.normalize(gaze_vec, p=2, dim=-1)

        if self.return_head_mask:
            sample["head_masks"] = generate_mask(sample["head_bboxes"], new_img_w, new_img_h)

        return sample

    def __len__(self):
        return self.length


# ============================================================================= #
#                             GAZEFOLLOW DATAMODULE                             #
# ============================================================================= #
class GazeFollowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_project: str,
        root_heads: str,
        batch_size: Union[int, dict] = 32,
        image_size: Union[int, tuple[int, int]] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: Union[int, tuple[int, int]] = 64,
        num_people: dict = {"train": 1, "val": 1, "test": 1},
        return_head_mask: bool = False,
        reason_feature_root: Union[str, None] = None,
        reason_feature_preload: bool = False,
        reason_feature_dim: int = 768,
        reason_log_limit: int = 20,
    ):
        super().__init__()
        self.root = root
        self.root_project = root_project
        self.root_heads = root_heads
        self.image_size = pair(image_size)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = {stage: num_people for stage in ["train", "val", "test"]} if isinstance(num_people, int) else num_people
        self.batch_size = {stage: batch_size for stage in ["train", "val", "test"]} if isinstance(batch_size, int) else batch_size
        self.return_head_mask = return_head_mask
        self.reason_feature_root = reason_feature_root
        self.reason_feature_preload = reason_feature_preload
        self.reason_feature_dim = reason_feature_dim
        self.reason_log_limit = reason_log_limit
        
    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=1.0, p=0.8, p_safe=1.0),
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.0, 1.5), hue=None, p=0.8),
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.train_dataset = GazeFollowDataset(
                self.root,
                self.root_project,
                self.root_heads,
                "train",
                train_transform,
                tr=(-0.1, 0.1),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['train'],
                return_head_mask=self.return_head_mask,
                reason_feature_root=self.reason_feature_root,
                reason_feature_preload=self.reason_feature_preload,
                reason_feature_dim=self.reason_feature_dim,
                reason_log_limit=self.reason_log_limit,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.root_project,
                self.root_heads,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['val'],
                return_head_mask=self.return_head_mask,
                reason_feature_root=None,
                reason_feature_preload=False,
                reason_feature_dim=self.reason_feature_dim,
                reason_log_limit=self.reason_log_limit,
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.root_project,
                self.root_heads,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['val'],
                return_head_mask=self.return_head_mask,
                reason_feature_root=None,
                reason_feature_preload=False,
                reason_feature_dim=self.reason_feature_dim,
                reason_log_limit=self.reason_log_limit,
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.test_dataset = GazeFollowDataset(
                self.root,
                self.root_project,
                self.root_heads,
                "test",
                test_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people['test'],
                return_head_mask=self.return_head_mask,
                reason_feature_root=None,
                reason_feature_preload=False,
                reason_feature_dim=self.reason_feature_dim,
                reason_log_limit=self.reason_log_limit,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        return dataloader
