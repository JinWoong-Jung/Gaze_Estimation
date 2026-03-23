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
        gaze_align_feature_root: Union[str, None] = None,
        gaze_align_feature_preload: bool = False,
        gaze_align_feature_dim: int = 768,
        object_align_feature_root: Union[str, None] = None,
        object_align_feature_preload: bool = False,
        object_align_feature_dim: int = 768,
        image_align_feature_root: Union[str, None] = None,
        image_align_feature_preload: bool = False,
        image_align_feature_dim: int = 768,
        reasoning_feature_root: Union[str, None] = None,
        reasoning_feature_preload: bool = False,
        reasoning_feature_dim: int = 768,
        object_feature_root: Union[str, None] = None,
        object_feature_preload: bool = False,
        object_feature_dim: int = 768,
        reasoning_log_limit: int = 20,
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
        # New per-path feature roots (with backward-compatible fallbacks).
        self.gaze_align_feature_root = (
            gaze_align_feature_root if gaze_align_feature_root is not None else reasoning_feature_root
        )
        self.gaze_align_feature_preload = bool(gaze_align_feature_preload or reasoning_feature_preload)
        self.gaze_align_feature_dim = int(gaze_align_feature_dim if gaze_align_feature_dim is not None else reasoning_feature_dim)

        self.object_align_feature_root = (
            object_align_feature_root if object_align_feature_root is not None else object_feature_root
        )
        self.object_align_feature_preload = bool(object_align_feature_preload or object_feature_preload)
        self.object_align_feature_dim = int(
            object_align_feature_dim if object_align_feature_dim is not None else object_feature_dim
        )

        self.image_align_feature_root = image_align_feature_root
        self.image_align_feature_preload = bool(image_align_feature_preload)
        self.image_align_feature_dim = int(image_align_feature_dim)

        # Backward-compatible aliases used by existing code paths.
        self.reasoning_feature_root = self.gaze_align_feature_root
        self.reasoning_feature_preload = self.gaze_align_feature_preload
        self.reasoning_feature_dim = self.gaze_align_feature_dim
        self.object_feature_root = self.object_align_feature_root
        self.object_feature_preload = self.object_align_feature_preload
        self.object_feature_dim = self.object_align_feature_dim
        self.reasoning_log_limit = reasoning_log_limit
        self.reasoning_warn_count = 0
        self.object_warn_count = 0
        self.image_warn_count = 0
        self.label_emb_cache = {}
        self.test_groups = None
        self.test_group_keys = []
        self.image_paths = []
        self.length = 0
        self.annotations, self.vocab2id = self.load_annotations()
        self.reasoning_feature_h5_path = None
        self.reasoning_feature_h5 = None
        self.reasoning_feature_index = None
        if (self.split == "train") and (self.gaze_align_feature_root is not None):
            self.reasoning_feature_h5_path = os.path.join(self.gaze_align_feature_root, f"{self.split}.h5")
        self.object_feature_h5_path = None
        self.object_feature_h5 = None
        self.object_feature_index = None
        if (self.split == "train") and (self.object_align_feature_root is not None):
            self.object_feature_h5_path = os.path.join(self.object_align_feature_root, f"{self.split}.h5")
        self.image_feature_h5_path = None
        self.image_feature_h5 = None
        self.image_feature_index = None
        if (self.split == "train") and (self.image_align_feature_root is not None):
            self.image_feature_h5_path = os.path.join(self.image_align_feature_root, f"{self.split}.h5")

        # Optional eager preload: open h5 + build key->row index once at startup.
        if self.gaze_align_feature_preload and (self.reasoning_feature_h5_path is not None):
            self.reasoning_feature_index = self._build_reasoning_feature_index()
        if self.object_align_feature_preload and (self.object_feature_h5_path is not None):
            self.object_feature_index = self._build_object_feature_index()
        if self.image_align_feature_preload and (self.image_feature_h5_path is not None):
            self.image_feature_index = self._build_image_feature_index()

    def _warn_reasoning(self, msg: str):
        if self.reasoning_warn_count < self.reasoning_log_limit:
            print(f"[GazeFollowDataset][reasoning] {msg}")
            self.reasoning_warn_count += 1
            if self.reasoning_warn_count == self.reasoning_log_limit:
                print("[GazeFollowDataset][reasoning] warning log limit reached; suppressing further messages.")

    def _warn_object(self, msg: str):
        if self.object_warn_count < self.reasoning_log_limit:
            print(f"[GazeFollowDataset][object] {msg}")
            self.object_warn_count += 1
            if self.object_warn_count == self.reasoning_log_limit:
                print("[GazeFollowDataset][object] warning log limit reached; suppressing further messages.")

    def _warn_image(self, msg: str):
        if self.image_warn_count < self.reasoning_log_limit:
            print(f"[GazeFollowDataset][image] {msg}")
            self.image_warn_count += 1
            if self.image_warn_count == self.reasoning_log_limit:
                print("[GazeFollowDataset][image] warning log limit reached; suppressing further messages.")

    def _get_reasoning_feature_key(self, image_path: str, sample_id: Union[int, str]) -> str:
        rel_dir = os.path.dirname(image_path)  # e.g. train/00000000
        split_prefix = f"{self.split}/"
        if rel_dir.startswith(split_prefix):
            rel_dir = rel_dir[len(split_prefix):]
        basename = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{basename}_{sample_id}"
        return os.path.join(rel_dir, filename)

    def _get_label_embedding(self, gaze_label: str) -> torch.Tensor:
        if gaze_label not in self.label_emb_cache:
            label_emb_path = os.path.join(self.root_project, f"data/gazefollow/label-embeds/{gaze_label}-emb.pt")
            label_emb = torch.load(label_emb_path, weights_only=False).to(torch.float32)
            label_emb = F.normalize(label_emb, p=2, dim=-1)
            self.label_emb_cache[gaze_label] = label_emb
        return self.label_emb_cache[gaze_label].clone()

    def _ensure_reasoning_feature_h5(self):
        if self.reasoning_feature_h5 is not None:
            return True
        if self.reasoning_feature_h5_path is None:
            return False
        if not os.path.exists(self.reasoning_feature_h5_path):
            self._warn_reasoning(f"missing h5 feature file: {self.reasoning_feature_h5_path}")
            return False
        try:
            self.reasoning_feature_h5 = h5py.File(self.reasoning_feature_h5_path, "r")
            return True
        except Exception as exc:
            self._warn_reasoning(f"failed opening h5 feature file: {self.reasoning_feature_h5_path} ({exc})")
            self.reasoning_feature_h5 = None
            return False

    def _build_reasoning_feature_index(self):
        if not self._ensure_reasoning_feature_h5():
            return {}
        start_time = time.time()
        index = {}
        keys_ds = self.reasoning_feature_h5.get("keys")
        if keys_ds is None:
            self._warn_reasoning(f"missing dataset `keys` in h5: {self.reasoning_feature_h5_path}")
            return index
        for i, key in enumerate(keys_ds):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            index[str(key)] = i
        elapsed = time.time() - start_time
        print(
            f"[GazeFollowDataset][reasoning] preload index complete: "
            f"entries={len(index)}, elapsed={elapsed:.1f}s"
        )
        return index

    def _load_reasoning_feature_from_h5(self, reasoning_key: str):
        if not self._ensure_reasoning_feature_h5():
            return None
        if self.reasoning_feature_index is None:
            self.reasoning_feature_index = self._build_reasoning_feature_index()
        emb_ds = self.reasoning_feature_h5.get("embeddings")
        if emb_ds is None:
            self._warn_reasoning(f"missing dataset `embeddings` in h5: {self.reasoning_feature_h5_path}")
            return None
        row_idx = self.reasoning_feature_index.get(reasoning_key)
        if row_idx is None:
            return None
        try:
            emb = torch.from_numpy(emb_ds[row_idx]).to(torch.float32)
            emb = F.normalize(emb, p=2, dim=-1)
            return emb
        except Exception as exc:
            self._warn_reasoning(f"failed loading h5 feature: key={reasoning_key} ({exc})")
            return None

    def _ensure_object_feature_h5(self):
        if self.object_feature_h5 is not None:
            return True
        if self.object_feature_h5_path is None:
            return False
        if not os.path.exists(self.object_feature_h5_path):
            self._warn_object(f"missing h5 feature file: {self.object_feature_h5_path}")
            return False
        try:
            self.object_feature_h5 = h5py.File(self.object_feature_h5_path, "r")
            return True
        except Exception as exc:
            self._warn_object(f"failed opening h5 feature file: {self.object_feature_h5_path} ({exc})")
            self.object_feature_h5 = None
            return False

    def _build_object_feature_index(self):
        if not self._ensure_object_feature_h5():
            return {}
        start_time = time.time()
        index = {}
        keys_ds = self.object_feature_h5.get("keys")
        if keys_ds is None:
            self._warn_object(f"missing dataset `keys` in h5: {self.object_feature_h5_path}")
            return index
        for i, key in enumerate(keys_ds):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            index[str(key)] = i
        elapsed = time.time() - start_time
        print(
            f"[GazeFollowDataset][object] preload index complete: "
            f"entries={len(index)}, elapsed={elapsed:.1f}s"
        )
        return index

    def _load_object_feature_from_h5(self, object_key: str):
        if not self._ensure_object_feature_h5():
            return None
        if self.object_feature_index is None:
            self.object_feature_index = self._build_object_feature_index()
        emb_ds = self.object_feature_h5.get("embeddings")
        if emb_ds is None:
            self._warn_object(f"missing dataset `embeddings` in h5: {self.object_feature_h5_path}")
            return None
        row_idx = self.object_feature_index.get(object_key)
        if row_idx is None:
            return None
        try:
            emb = torch.from_numpy(emb_ds[row_idx]).to(torch.float32)
            emb = F.normalize(emb, p=2, dim=-1)
            return emb
        except Exception as exc:
            self._warn_object(f"failed loading h5 feature: key={object_key} ({exc})")
            return None

    def _ensure_image_feature_h5(self):
        if self.image_feature_h5 is not None:
            return True
        if self.image_feature_h5_path is None:
            return False
        if not os.path.exists(self.image_feature_h5_path):
            self._warn_image(f"missing h5 feature file: {self.image_feature_h5_path}")
            return False
        try:
            self.image_feature_h5 = h5py.File(self.image_feature_h5_path, "r")
            return True
        except Exception as exc:
            self._warn_image(f"failed opening h5 feature file: {self.image_feature_h5_path} ({exc})")
            self.image_feature_h5 = None
            return False

    def _build_image_feature_index(self):
        if not self._ensure_image_feature_h5():
            return {}
        start_time = time.time()
        index = {}
        keys_ds = self.image_feature_h5.get("keys")
        if keys_ds is None:
            self._warn_image(f"missing dataset `keys` in h5: {self.image_feature_h5_path}")
            return index
        for i, key in enumerate(keys_ds):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            index[str(key)] = i
        elapsed = time.time() - start_time
        print(
            f"[GazeFollowDataset][image] preload index complete: "
            f"entries={len(index)}, elapsed={elapsed:.1f}s"
        )
        return index

    def _load_image_feature_from_h5(self, image_key: str):
        if not self._ensure_image_feature_h5():
            return None
        if self.image_feature_index is None:
            self.image_feature_index = self._build_image_feature_index()
        emb_ds = self.image_feature_h5.get("embeddings")
        if emb_ds is None:
            self._warn_image(f"missing dataset `embeddings` in h5: {self.image_feature_h5_path}")
            return None
        row_idx = self.image_feature_index.get(image_key)
        if row_idx is None:
            return None
        try:
            emb = torch.from_numpy(emb_ds[row_idx]).to(torch.float32)
            emb = F.normalize(emb, p=2, dim=-1)
            return emb
        except Exception as exc:
            self._warn_image(f"failed loading h5 feature: key={image_key} ({exc})")
            return None

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
        
        if self.split == "test":
            # Group test samples by target person identity, not just image path.
            self.test_groups = annotations.groupby(["path", "eye_x"], sort=False)
            self.test_group_keys = list(self.test_groups.groups.keys())
            self.image_paths = sorted(annotations.path.unique())
            self.length = len(self.test_group_keys)
        else:
            self.image_paths = sorted(annotations.path.unique())
            self.length = len(annotations)
        
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
            if pd.isnull(gaze_label):
                gaze_label_id = torch.tensor(-1, dtype=torch.long)
            else:
                gaze_label_id = torch.tensor(int(self.vocab2id.get(gaze_label, -1)), dtype=torch.long)
            gaze_label_ids = torch.tensor([int(gaze_label_id.item())], dtype=torch.long)
            idx = item["id"]
        elif self.split == "test":
            group_key = self.test_group_keys[index]
            p_annotations = self.test_groups.get_group(group_key)
            gaze_pt = torch.from_numpy(p_annotations[["gaze_x", "gaze_y"]].values).float()
            p = max(0, 20 - len(gaze_pt))
            if p > 0:
                gaze_pt = F.pad(gaze_pt, (0, 0, 0, p), value=-1.0)
            elif gaze_pt.size(0) > 20:
                gaze_pt = gaze_pt[:20]

            idx = p_annotations["id"].values.tolist()
            if p > 0:
                idx = idx + [-1] * p
            elif len(idx) > 20:
                idx = idx[:20]

            item = p_annotations.iloc[0]
            gaze_label = item.gaze_gt_label
            gaze_labels = item.gaze_gt_labels
            if pd.isnull(gaze_label):
                gaze_label_id = torch.tensor(-1, dtype=torch.long)
            else:
                gaze_label_id = torch.tensor(int(self.vocab2id.get(gaze_label, -1)), dtype=torch.long)
            gaze_label_ids_list = [int(gaze_label_id.item())]
            if (not pd.isnull(gaze_labels)) and (int(gaze_label_id.item()) != -1):
                parsed_ids = [int(self.vocab2id.get(label, -1)) for label in str(gaze_labels).split("-")]
                parsed_ids = [x for x in parsed_ids if x >= 0]
                if len(parsed_ids) > 0:
                    gaze_label_ids_list = parsed_ids
            gaze_label_ids = torch.tensor(gaze_label_ids_list, dtype=torch.long)
            l = 5 - len(gaze_label_ids)
            if l > 0:
                gaze_label_ids = F.pad(gaze_label_ids, (0, l), value=-1)
            elif l < 0:
                gaze_label_ids = gaze_label_ids[:5]

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

        gaze_align_emb = torch.zeros(self.gaze_align_feature_dim, dtype=torch.float32)
        gaze_align_valid = torch.tensor(0.0, dtype=torch.float32)
        if (self.split == "train") and (self.gaze_align_feature_root is not None):
            reasoning_key = self._get_reasoning_feature_key(path, idx)
            loaded_reasoning_emb = self._load_reasoning_feature_from_h5(reasoning_key)
            if loaded_reasoning_emb is not None:
                gaze_align_emb = loaded_reasoning_emb
                gaze_align_valid = torch.tensor(1.0, dtype=torch.float32)
            else:
                self._warn_reasoning(f"missing h5 feature key: {reasoning_key}")

        object_align_emb = torch.zeros(self.object_align_feature_dim, dtype=torch.float32)
        object_align_valid = torch.tensor(0.0, dtype=torch.float32)
        if (self.split == "train") and (self.object_align_feature_root is not None):
            object_key = self._get_reasoning_feature_key(path, idx)
            loaded_object_emb = self._load_object_feature_from_h5(object_key)
            if loaded_object_emb is not None:
                object_align_emb = loaded_object_emb
                object_align_valid = torch.tensor(1.0, dtype=torch.float32)
            else:
                self._warn_object(f"missing h5 feature key: {object_key}")

        image_align_emb = torch.zeros(self.image_align_feature_dim, dtype=torch.float32)
        image_align_valid = torch.tensor(0.0, dtype=torch.float32)
        if (self.split == "train") and (self.image_align_feature_root is not None):
            image_key = self._get_reasoning_feature_key(path, idx)
            loaded_image_emb = self._load_image_feature_from_h5(image_key)
            if loaded_image_emb is not None:
                image_align_emb = loaded_image_emb
                image_align_valid = torch.tensor(1.0, dtype=torch.float32)
            else:
                self._warn_image(f"missing h5 feature key: {image_key}")

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
            "gaze_align_emb": gaze_align_emb,
            "gaze_align_valid": gaze_align_valid,
            "object_align_emb": object_align_emb,
            "object_align_valid": object_align_valid,
            "image_align_emb": image_align_emb,
            "image_align_valid": image_align_valid,
            # Backward-compatible aliases
            "reasoning_emb": gaze_align_emb,
            "reasoning_valid": gaze_align_valid,
            "object_emb": object_align_emb,
            "object_valid": object_align_valid,
            "id": idx,
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": path,
        }

        if self.transform:
            sample = self.transform(sample)

        num_heads_current = len(sample["heads"])
        num_target_people = self.num_people if self.num_people != -1 else num_heads_current
        target_head_idx = num_heads_current - 1
        
        num_missing_heads = num_target_people - num_heads_current

        if num_missing_heads > 0:
            pad_bbox = (0, 0, num_missing_heads, 0)
            sample["head_bboxes"] = F.pad(sample["head_bboxes"], pad_bbox, mode="constant", value=0.)
            
            single_head_shape = sample["heads"].shape[1:] 
            blank_head_tensors = torch.zeros((num_missing_heads, *single_head_shape), dtype=sample["heads"].dtype, device=sample["heads"].device)
            sample["heads"] = torch.cat([sample["heads"], blank_head_tensors], dim=0)
        elif num_missing_heads < 0:
            num_context_keep = max(0, num_target_people - 1)
            sample["heads"] = torch.cat([sample["heads"][:num_context_keep], sample["heads"][-1:]], dim=0)
            sample["head_bboxes"] = torch.cat([sample["head_bboxes"][:num_context_keep], sample["head_bboxes"][-1:]], dim=0)
            target_head_idx = num_target_people - 1

        target_head_idx = min(target_head_idx, len(sample["heads"]) - 1)
        sample["target_head_idx"] = torch.tensor(target_head_idx, dtype=torch.long)

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
        gaze_vec = sample["gaze_pt"] - sample["head_centers"][target_head_idx]
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
        gaze_align_feature_root: Union[str, None] = None,
        gaze_align_feature_preload: bool = False,
        gaze_align_feature_dim: int = 768,
        object_align_feature_root: Union[str, None] = None,
        object_align_feature_preload: bool = False,
        object_align_feature_dim: int = 768,
        image_align_feature_root: Union[str, None] = None,
        image_align_feature_preload: bool = False,
        image_align_feature_dim: int = 768,
        reasoning_feature_root: Union[str, None] = None,
        reasoning_feature_preload: bool = False,
        reasoning_feature_dim: int = 768,
        object_feature_root: Union[str, None] = None,
        object_feature_preload: bool = False,
        object_feature_dim: int = 768,
        reasoning_log_limit: int = 20,
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
        self.gaze_align_feature_root = (
            gaze_align_feature_root if gaze_align_feature_root is not None else reasoning_feature_root
        )
        self.gaze_align_feature_preload = bool(gaze_align_feature_preload or reasoning_feature_preload)
        self.gaze_align_feature_dim = int(gaze_align_feature_dim if gaze_align_feature_dim is not None else reasoning_feature_dim)
        self.object_align_feature_root = (
            object_align_feature_root if object_align_feature_root is not None else object_feature_root
        )
        self.object_align_feature_preload = bool(object_align_feature_preload or object_feature_preload)
        self.object_align_feature_dim = int(
            object_align_feature_dim if object_align_feature_dim is not None else object_feature_dim
        )
        self.image_align_feature_root = image_align_feature_root
        self.image_align_feature_preload = bool(image_align_feature_preload)
        self.image_align_feature_dim = int(image_align_feature_dim)
        # Backward-compatible aliases.
        self.reasoning_feature_root = self.gaze_align_feature_root
        self.reasoning_feature_preload = self.gaze_align_feature_preload
        self.reasoning_feature_dim = self.gaze_align_feature_dim
        self.object_feature_root = self.object_align_feature_root
        self.object_feature_preload = self.object_align_feature_preload
        self.object_feature_dim = self.object_align_feature_dim
        self.reasoning_log_limit = reasoning_log_limit
        
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
                gaze_align_feature_root=self.gaze_align_feature_root,
                gaze_align_feature_preload=self.gaze_align_feature_preload,
                gaze_align_feature_dim=self.gaze_align_feature_dim,
                object_align_feature_root=self.object_align_feature_root,
                object_align_feature_preload=self.object_align_feature_preload,
                object_align_feature_dim=self.object_align_feature_dim,
                image_align_feature_root=self.image_align_feature_root,
                image_align_feature_preload=self.image_align_feature_preload,
                image_align_feature_dim=self.image_align_feature_dim,
                reasoning_feature_root=self.reasoning_feature_root,
                reasoning_feature_preload=self.reasoning_feature_preload,
                reasoning_feature_dim=self.reasoning_feature_dim,
                object_feature_root=self.object_feature_root,
                object_feature_preload=self.object_feature_preload,
                object_feature_dim=self.object_feature_dim,
                reasoning_log_limit=self.reasoning_log_limit,
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
                gaze_align_feature_root=None,
                gaze_align_feature_preload=False,
                gaze_align_feature_dim=self.gaze_align_feature_dim,
                object_align_feature_root=None,
                object_align_feature_preload=False,
                object_align_feature_dim=self.object_align_feature_dim,
                image_align_feature_root=None,
                image_align_feature_preload=False,
                image_align_feature_dim=self.image_align_feature_dim,
                reasoning_feature_root=None,
                reasoning_feature_preload=False,
                reasoning_feature_dim=self.reasoning_feature_dim,
                object_feature_root=None,
                object_feature_preload=False,
                object_feature_dim=self.object_feature_dim,
                reasoning_log_limit=self.reasoning_log_limit,
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
                gaze_align_feature_root=None,
                gaze_align_feature_preload=False,
                gaze_align_feature_dim=self.gaze_align_feature_dim,
                object_align_feature_root=None,
                object_align_feature_preload=False,
                object_align_feature_dim=self.object_align_feature_dim,
                image_align_feature_root=None,
                image_align_feature_preload=False,
                image_align_feature_dim=self.image_align_feature_dim,
                reasoning_feature_root=None,
                reasoning_feature_preload=False,
                reasoning_feature_dim=self.reasoning_feature_dim,
                object_feature_root=None,
                object_feature_preload=False,
                object_feature_dim=self.object_feature_dim,
                reasoning_log_limit=self.reasoning_log_limit,
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
                gaze_align_feature_root=None,
                gaze_align_feature_preload=False,
                gaze_align_feature_dim=self.gaze_align_feature_dim,
                object_align_feature_root=None,
                object_align_feature_preload=False,
                object_align_feature_dim=self.object_align_feature_dim,
                image_align_feature_root=None,
                image_align_feature_preload=False,
                image_align_feature_dim=self.image_align_feature_dim,
                reasoning_feature_root=None,
                reasoning_feature_preload=False,
                reasoning_feature_dim=self.reasoning_feature_dim,
                object_feature_root=None,
                object_feature_preload=False,
                object_feature_dim=self.object_feature_dim,
                reasoning_log_limit=self.reasoning_log_limit,
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
