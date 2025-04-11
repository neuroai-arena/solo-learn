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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from solo.args.knn import parse_args_knn
from solo.data.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from solo.methods import METHODS
from solo.utils.knn import WeightedKNNClassifier


@torch.no_grad()
def extract_features_1(loader: DataLoader, model: nn.Module) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    patches, patches_norm, prefixes, prefixes_norm, labels = [], [], [], [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)

        patch, prefix = model.backbone.get_intermediate_layers(
            im,
            n=1,
            return_prefix_tokens=True,
            norm=False,
        )[0]

        norm_layer = model.backbone.norm

        # average pool
        patch_norm = torch.mean(norm_layer(patch), dim=1)
        patch = torch.mean(patch, dim=1)  # (B, N, D) -> (B, D)
        patches.append(patch.detach())
        patches_norm.append(patch_norm.detach())

        # if CLS token is present
        if not prefix.numel() == 0:
            prefix_norm = norm_layer(prefix).reshape(prefix.shape[0], -1)
            prefix = prefix.reshape(prefix.shape[0], -1)  # (B, N, D) -> (B, N*D)
            prefixes.append(prefix.detach())
            prefixes_norm.append(prefix_norm.detach())

        labels.append(lab)

    model.train()
    patches = torch.cat(patches)
    patches_norm = torch.cat(patches_norm)
    labels = torch.cat(labels)

    if prefixes and prefixes_norm:
        prefixes = torch.cat(prefixes)
        prefixes_norm = torch.cat(prefixes_norm)

    return patches, patches_norm, prefixes, prefixes_norm, labels


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    patches, prefixes, cats, labels = [], [], [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)

        patch, prefix = model.backbone.get_intermediate_layers(
            im,
            n=1,
            return_prefix_tokens=True,
            norm=False,
        )[0]

        norm_layer = model.backbone.norm

        # average pool
        patch = torch.mean(patch, dim=1)  # (B, N, D) -> (B, D)
        patches.append(patch.detach())

        # if CLS token is present
        if not prefix.numel() == 0:
            prefix = prefix.reshape(prefix.shape[0], -1)  # (B, N, D) -> (B, N*D)
            prefixes.append(prefix.detach())

            cat = torch.cat([patch, prefix], dim=1) # (B, D) + (B, N*D) -> (B, D + N*D)
            cats.append(cat.detach())
        labels.append(lab)

    model.train()
    patches = torch.cat(patches)
    labels = torch.cat(labels)

    if prefixes:
        prefixes = torch.cat(prefixes)
        cats = torch.cat(cats)

    return patches, prefixes, cats, labels


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
    args = parse_args_knn()

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
    # train_dataset = Subset(train_dataset, range(1000))


    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # extract train features
    # tf_patch, tf_patch_norm, tf_prefix, tf_prefix_norm,  train_targets = extract_features(train_loader, model)
    # train_features = {"patch": tf_patch, "patch_norm": tf_patch_norm, "prefix": tf_prefix, "prefix_norm": tf_prefix_norm}

    tf_patch, tf_prefix, tf_cats,  train_targets = extract_features(train_loader, model)
    train_features = {"patch": tf_patch, "prefix": tf_prefix, "concat": tf_cats}

    print("Train")
    for k, v in train_features.items():
        print(k, v.shape)

    # extract test features
    # test_patch, test_patch_norm, test_prefix, test_prefix_norm, test_targets = extract_features(val_loader, model)
    # test_features = {"patch": test_patch, "patch_norm": test_patch_norm, "prefix": test_prefix, "prefix_norm": test_prefix_norm}

    test_patch, test_prefix, test_cats, test_targets = extract_features(val_loader, model)
    test_features = {"patch": test_patch, "prefix": test_prefix, "concat": test_cats}

    print("Test")
    for k, v in test_features.items():
        print(k, v.shape)

    result = []
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
    result = pd.DataFrame(result)
    print(result)


    import datetime
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result.to_csv(f"res/knn/knn_{ckpt_dir.stem}_{now}.csv", index=False)


if __name__ == "__main__":
    main()
