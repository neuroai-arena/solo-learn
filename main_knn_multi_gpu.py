# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightning.fabric import Fabric

from solo.args.knn import parse_args_knn
from solo.data.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from solo.methods import METHODS
from solo.utils.knn import WeightedKNNClassifier


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module, fabric: Fabric, momentum_model: bool = False) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    backbone_features, proj_features, labels = [], [], []
    bar = tqdm(loader) if fabric.local_rank == 0 else loader
    for im, lab in bar:
        if momentum_model:
            outs = model.momentum_forward(im)
            proj_features.append(outs["k"])
        else:
            outs = model(im)
            proj_features.append(outs["z"])

        backbone_features.append(outs["feats"].detach())
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    proj_features = torch.cat(proj_features)
    labels = torch.cat(labels)
    return backbone_features, proj_features, labels


@torch.no_grad()
def run_knn(
        train_features: torch.Tensor,
        train_targets: torch.Tensor,
        test_features: torch.Tensor,
        test_targets: torch.Tensor,
        k: int,
        T: float,
        distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )

    # add features
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    acc1, acc5 = knn.compute()

    # free up memory
    del knn

    return acc1, acc5


def main():
    torch.set_float32_matmul_precision('medium')

    args = parse_args_knn()

    fabric = Fabric(devices="auto")
    fabric.launch()

    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = sorted([ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")])

    # try to get last checkpoint
    last = list(filter(lambda x: "ep=last" in str(x), ckpt_path))
    if last:
        ckpt_path = last[0]
    else:
        ckpt_path = ckpt_path[-1]

    print("Using checkpoint", ckpt_path)
    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)
    cfg = OmegaConf.create(method_args)

    # build the model
    model = METHODS[method_args["method"]].load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)

    # prepare data
    _, T = prepare_transforms(args.dataset)
    train_dataset, val_dataset = prepare_datasets(
        args.dataset,
        T_train=T,
        T_val=T,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format,
        **cfg.data.dataset_kwargs
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = fabric.setup(model)
    if args.momentum_model:
        model.mark_forward_method('momentum_forward')

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # extract train features
    train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model, fabric, args.momentum_model)
    train_features = {"backbone": train_features_bb, "projector": train_features_proj}

    # extract test features
    test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model, fabric, args.momentum_model)
    test_features = {"backbone": test_features_bb, "projector": test_features_proj}

    # synchronize
    fabric.barrier()

    result = []
    total_hp = len(args.feature_type) * len(args.k) * len(args.distance_function) * len(args.temperature)
    bar = tqdm(total=total_hp) if fabric.local_rank == 0 else None
    # run k-nn for all possible combinations of parameters
    for feat_type in args.feature_type:
        for k in args.k:
            for distance_fx in args.distance_function:
                temperatures = args.temperature if distance_fx == "cosine" else [None]
                for T in temperatures:
                    acc1, acc5 = run_knn(
                        train_features=train_features[feat_type],
                        train_targets=train_targets,
                        test_features=test_features[feat_type],
                        test_targets=test_targets,
                        k=k,
                        T=T,
                        distance_fx=distance_fx,
                    )
                    result.append({"feat_type": feat_type, "distance_fx": distance_fx, "k": k, "T": T, "acc1": acc1,
                                   "acc5": acc5})
                    if bar is not None:
                        bar.update(1)

    if fabric.local_rank == 0:
        result = pd.DataFrame(result)
        print(result)
        name = f"res/knn/knn_{args.dataset}_{ckpt_dir.stem}{'_momentum' if args.momentum_model else ''}.csv"
        result.to_csv(name, index=False)


if __name__ == "__main__":
    main()
