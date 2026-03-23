#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import json
import os
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
from semgaze.utils.common import generate_gaze_heatmap, generate_mask, get_img_size, pair, square_bbox


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


class VATDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Union[Compose, None] = None,
        tr: tuple = (-0.1, 0.1),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 1,
        return_head_mask: bool = False,
    ):
        super().__init__()
        assert split in ("train", "test"), f"Expected split in ['train', 'test'], got {split}."
        assert (num_people == -1) or (num_people > 0), f"Expected num_people > 0 or -1, got {num_people}."

        self.root = root
        self.split = split
        self.transform = transform
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.return_head_mask = return_head_mask

        self.frames = []
        self.samples = []
        self._load_preprocessed()

    def _load_preprocessed(self):
        preprocessed_path = os.path.join(self.root, f"{self.split}_preprocessed.json")
        if not os.path.exists(preprocessed_path):
            raise FileNotFoundError(
                f"Missing VAT preprocessed file: {preprocessed_path}. "
                "Run `python data_prep/preprocess_vat.py --data_path <vat_root>` first."
            )
        with open(preprocessed_path, "r") as f:
            sequences = json.load(f)

        for seq in sequences:
            seq_width = int(seq["width"])
            seq_height = int(seq["height"])
            for frame in seq["frames"]:
                frame_path = str(frame["path"])
                raw_heads = frame.get("heads", [])
                valid_heads = []
                for head in raw_heads:
                    bbox_norm = head.get("bbox_norm", None)
                    if bbox_norm is None or len(bbox_norm) != 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox_norm]
                    if (x2 <= x1) or (y2 <= y1):
                        continue
                    valid_heads.append(head)
                if len(valid_heads) == 0:
                    continue

                frame_idx = len(self.frames)
                self.frames.append(
                    {
                        "path": frame_path,
                        "width": seq_width,
                        "height": seq_height,
                        "heads": valid_heads,
                    }
                )
                for target_idx in range(len(valid_heads)):
                    self.samples.append((frame_idx, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        frame_idx, target_head_idx_raw = self.samples[index]
        frame_record = self.frames[frame_idx]

        path = frame_record["path"]
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        img_w, img_h = image.size

        heads_meta = frame_record["heads"]
        all_head_bboxes = torch.tensor(
            [[float(v) for v in h["bbox_norm"]] for h in heads_meta],
            dtype=torch.float32,
        )
        all_head_bboxes = torch.clamp(all_head_bboxes, min=0.0, max=1.0)

        # Keep target head as the last token to match current transform assumptions.
        reorder = [i for i in range(len(heads_meta)) if i != target_head_idx_raw] + [target_head_idx_raw]
        head_bboxes = all_head_bboxes[reorder]
        target_head_idx = len(reorder) - 1

        if self.split == "train" and target_head_idx > 0:
            perm = torch.randperm(target_head_idx)
            head_bboxes = torch.cat([head_bboxes[:target_head_idx][perm], head_bboxes[target_head_idx:]], dim=0)

        head_bboxes_px = head_bboxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        if self.split == "train":
            head_bboxes_px = self.jitter_bbox(head_bboxes_px, img_w, img_h)
        head_bboxes_px = square_bbox(head_bboxes_px, img_w, img_h)

        heads = [image.crop(bbox.int().tolist()) for bbox in head_bboxes_px]

        head_bboxes = head_bboxes_px / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        head_bboxes = torch.clamp(head_bboxes, min=0.0, max=1.0)

        target_meta = heads_meta[target_head_idx_raw]
        inout = float(target_meta["inout"])
        if inout >= 0.5:
            gaze_x = float(target_meta["gazex_norm"][0])
            gaze_y = float(target_meta["gazey_norm"][0])
            gaze_pt = torch.tensor([gaze_x, gaze_y], dtype=torch.float32).clamp(min=0.0, max=1.0)
        else:
            gaze_pt = torch.tensor([-1.0, -1.0], dtype=torch.float32)

        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pt": gaze_pt,
            "gaze_label": "",
            "gaze_label_id": torch.tensor(-1, dtype=torch.long),
            "gaze_labels": "",
            "gaze_label_ids": torch.full((5,), -1, dtype=torch.long),
            "gaze_label_emb": torch.zeros(512, dtype=torch.float32),
            "inout": torch.tensor(inout, dtype=torch.float32),
            "id": f"{path}:{target_head_idx_raw}",
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": path,
        }

        if self.transform:
            sample = self.transform(sample)

        num_heads_current = len(sample["heads"])
        num_target_people = self.num_people if self.num_people != -1 else num_heads_current
        num_missing_heads = num_target_people - num_heads_current

        if num_missing_heads > 0:
            pad_bboxes = torch.zeros(
                (num_missing_heads, 4),
                dtype=sample["head_bboxes"].dtype,
                device=sample["head_bboxes"].device,
            )
            sample["head_bboxes"] = torch.cat([sample["head_bboxes"], pad_bboxes], dim=0)

            if torch.is_tensor(sample["heads"]):
                pad_heads = torch.zeros(
                    (num_missing_heads, *sample["heads"].shape[1:]),
                    dtype=sample["heads"].dtype,
                    device=sample["heads"].device,
                )
                sample["heads"] = torch.cat([sample["heads"], pad_heads], dim=0)
            else:
                sample["heads"] = sample["heads"] + [Image.new("RGB", (224, 224)) for _ in range(num_missing_heads)]
        elif num_missing_heads < 0:
            num_context_keep = max(0, num_target_people - 1)
            if torch.is_tensor(sample["heads"]):
                sample["heads"] = torch.cat([sample["heads"][:num_context_keep], sample["heads"][-1:]], dim=0)
            else:
                sample["heads"] = sample["heads"][:num_context_keep] + sample["heads"][-1:]
            sample["head_bboxes"] = torch.cat(
                [sample["head_bboxes"][:num_context_keep], sample["head_bboxes"][-1:]], dim=0
            )
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
            sample["gaze_heatmap"] = generate_gaze_heatmap(
                sample["gaze_pt"],
                sigma=self.heatmap_sigma,
                size=self.heatmap_size,
            )
        else:
            sample["gaze_heatmap"] = torch.zeros((self.heatmap_size, self.heatmap_size), dtype=torch.float32)

        new_img_w, new_img_h = get_img_size(sample["image"])
        gaze_vec = sample["gaze_pt"] - sample["head_centers"][target_head_idx]
        gaze_vec = gaze_vec * torch.tensor([new_img_w, new_img_h], dtype=torch.float32)
        sample["gaze_vec"] = F.normalize(gaze_vec, p=2, dim=-1)

        if self.return_head_mask:
            sample["head_masks"] = generate_mask(sample["head_bboxes"], new_img_w, new_img_h)

        return sample


class VATDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_project: str,
        image_size: Union[int, tuple[int, int]] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: Union[int, dict] = 1,
        return_head_mask: bool = False,
        batch_size: Union[int, dict] = 32,
    ):
        super().__init__()
        self.root = root
        self.root_project = root_project
        self.image_size = pair(image_size)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = (
            {stage: num_people for stage in ["train", "val", "test"]}
            if isinstance(num_people, int)
            else num_people
        )
        self.return_head_mask = return_head_mask
        self.batch_size = (
            {stage: batch_size for stage in ["train", "val", "test"]}
            if isinstance(batch_size, int)
            else batch_size
        )

    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=1.0, p=0.8, p_safe=1.0),
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.0, 1.5),
                        hue=None,
                        p=0.8,
                    ),
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            eval_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.train_dataset = VATDataset(
                root=self.root,
                split="train",
                transform=train_transform,
                tr=(-0.1, 0.1),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people["train"],
                return_head_mask=self.return_head_mask,
            )
            # Use VAT test split as validation split.
            self.val_dataset = VATDataset(
                root=self.root,
                split="test",
                transform=eval_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people["val"],
                return_head_mask=self.return_head_mask,
            )

        elif stage == "validate":
            eval_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = VATDataset(
                root=self.root,
                split="test",
                transform=eval_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people["val"],
                return_head_mask=self.return_head_mask,
            )

        elif stage == "test":
            eval_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.test_dataset = VATDataset(
                root=self.root,
                split="test",
                transform=eval_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people["test"],
                return_head_mask=self.return_head_mask,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
