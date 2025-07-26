import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.fabric.strategies import DDPStrategy
from modelvshuman.utils import load_dataset
from modelvshuman.models.wrappers import PytorchModel
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from solo.methods import METHODS, BaseMethod, LinearModel


@torch.no_grad()
def extract_images(loader: DataLoader) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """
    images, labels= [], []
    for im, lab in tqdm(loader):
        images.append(im)
        labels.append(lab)
    return torch.cat(images), torch.cat(labels)

class ModelWrapper(nn.Module):
    def __init__(self, backbone, backbone_name):
        super().__init__()
        self.backbone = backbone
        if backbone_name.startswith("resnet"):
            self.num_features = self.backbone.inplanes
        else:
            self.num_features = backbone.num_features

    def forward(self, input):
        out = self.backbone(input)
        return out

class LinearWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input):
        out = self.backbone(input)
        out = out["logits"]

        lr = 0.4
        layer_name="backbone.avgpool"
        clf_str = f"classifier-layer={layer_name}-lr_{lr:.8f}".replace(".", ":")
        out = out[clf_str]
        return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--devices", default=0, type=int)


    args = parser.parse_args()


    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if "last" in ckpt][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    cfg = OmegaConf.create(method_args)
    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    backbone = ModelWrapper(backbone, cfg.backbone.name)

    cfg.grid.layer_names = [l for l in cfg.grid.layer_names if not "head" in l ]
    cfg.grid.lr = [0.4]
    model = LinearModel(backbone, loss_func=torch.nn.CrossEntropyLoss(), mixup_func=False, cfg=cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model = LinearWrapper(model)


    all_datasets = ['sketch', 'stylized', 'edge', 'silhouette', 'cue-conflict']
    # all_datasets = list(datasets.list_datasets().keys())
    # all_datasets.remove("imagenet_validation")
    # all_datasets.remove("original")
    all_datasets = [load_dataset(dataset) for dataset in all_datasets]
    fabric = L.Fabric(accelerator=cfg.accelerator, devices=cfg.devices if not args.devices else args.devices, precision=cfg.precision)
    fabric.launch()
    model = fabric.setup(model)
    model.eval()
    all_values = []
    all_names = []


    model = PytorchModel(model, "nymeria")

    for dataset in all_datasets:
        for metric in dataset.metrics:
            metric.reset()
        dataset.loader = fabric.setup_dataloaders(dataset.loader)

        for images, target, paths in tqdm(dataset.loader):
            # images = images.to(device())
            logits = model.forward_batch(images)
            softmax_output = model.softmax(logits)
            if isinstance(target, torch.Tensor):
                batch_targets = model.to_numpy(target)
            else:
                batch_targets = target
            predictions = dataset.decision_mapping(softmax_output)
            for metric in dataset.metrics:
                metric.update(predictions, batch_targets,paths)
        values = [metric.value for metric in dataset.metrics]
        names = [f"{dataset.name}_{metric.name}" for metric in dataset.metrics]
        values = fabric.all_gather(torch.tensor(values))#.flatten(0, 1).mean()
        if args.devices != 1:
            values = values.sum(dim=0) / len(dataset)
        all_values.append(values.view(-1))
        all_names.extend(names)

    all_values = torch.cat(all_values).cpu().numpy()*100
    print(all_values)
    if fabric.global_rank == 0:
        writer = csv.writer(open(str(ckpt_dir / f"robustness.csv"), "w"))
        writer.writerow(all_names)
        writer.writerow(all_values)



if __name__ == "__main__":
    main()