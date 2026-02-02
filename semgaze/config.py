#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

from dataclasses import dataclass
from typing import Union


@dataclass
class GazeFollowData:
    root: str
    root_annotations: str
    root_heads: str
    num_train_samples: int
    vocab_size: int

@dataclass
class GazeHOIData:
    root: str
    num_train_samples: int
    vocab_size: int

@dataclass
class Data:
    num_people: int
    image_size: int
    heatmap_size: int
    heatmap_sigma: int
    return_head_mask: bool
    gf: GazeFollowData
    gazehoi: GazeHOIData


@dataclass
class Model:
    init_weights: Union[str, None]


@dataclass
class Optimizer:
    type: str


@dataclass
class Train:
    seed: int
    lr: float
    batch_size: int
    epochs: int
    device: str
    resume: bool
    resume_from: Union[str, None]


@dataclass
class Val:
    checkpoint: str
    batch_size: int
    device: str


@dataclass
class Test:
    checkpoint: str
    batch_size: int
    device: str


@dataclass
class Predict:
    file: str
    checkpoint: str
    batch_size: int
    device: str


@dataclass
class WandB:
    name: Union[str, None]
    log: bool


@dataclass
class Project:
    name: str
    version: str
    description: str


@dataclass
class Experiment:
    name: str
    description: str
    task: str


@dataclass
class MyConfig:
    project: Project
    experiment: Experiment
    data: Data
    model: Model
    optimizer: Optimizer
    train: Train
    val: Val
    test: Test
    predict: Predict
    wandb: WandB
