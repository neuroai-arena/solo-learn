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
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


from solo.data.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from solo.methods import METHODS, BaseMethod, LinearModel
import foolbox as fb
from foolbox import accuracy

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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="imagenet_im")
    parser.add_argument("--train_data_path", type=Path)
    parser.add_argument("--val_data_path", type=Path)
    parser.add_argument("--data_format", default="h5", choices=["image_folder", "dali", "h5"])
    parser.add_argument("--devices", default=0, type=int)


    args = parser.parse_args()


    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir/ ckpt for ckpt in os.listdir(ckpt_dir) if "last" in ckpt][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    cfg = OmegaConf.create(method_args)
    cfg.finetune=True #Needed to allow gradient flow
    # build the model
    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    backbone = ModelWrapper(backbone, cfg.backbone.name)

    cfg.grid.layer_names = [l for l in cfg.grid.layer_names if not "head" in l ]
    cfg.grid.lr = [0.4]
    model = LinearModel(backbone, loss_func=torch.nn.CrossEntropyLoss(), mixup_func=False, cfg=cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model = LinearWrapper(model)


    # prepare data
    _, T = prepare_transforms(args.dataset)
    del T.transforms[3]
    print("build dataset")
    train_dataset, val_dataset = prepare_datasets(
        args.dataset,
        T_train=T,
        T_val=T,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )



    # extract test features
    fabric = L.Fabric(accelerator=cfg.accelerator, devices=cfg.devices if not args.devices else args.devices, precision=cfg.precision)
    fabric.launch()
    val_loader = fabric.setup_dataloaders(val_loader)
    model = fabric.setup(model)
    model.eval()
    # test_features_bb, test_targets = extract_features(val_loader, model)


    # fmodel = fb.PyTorchModel(model_lin, bounds=(torch.min(test_images).item(), torch.max(test_images).item()))
    preprocessing = dict(mean=torch.tensor([0.485, 0.456, 0.406],device=fabric.device), std=torch.tensor([0.229, 0.224, 0.225],device=fabric.device), axis=-3)
    # bounds_pre = (0 - preprocessing["mean"][0])/ preprocessing["std"][1], (1 - preprocessing["mean"][2])/ preprocessing["std"][1]
    fmodel = fb.PyTorchModel(model, bounds=(0 , 1), preprocessing=preprocessing, device=fabric.device)



    # fmodel = fb.PyTorchModel(model, bounds=bounds_pre, preprocessing=preprocessing)
    # fmodel = fmodel.transform_bounds((0, 1))

    print("build attacks")

    attacks = []
    rsss = [1/40, 1/10]
    # steps = [1, 5, 20]
    # steps = [1, 5]
    steps = 1, 5
    for rss in rsss:
        for step in steps:
            attacks.append(fb.attacks.LinfPGD(rel_stepsize=rss, steps=step))  # 1 , 5, 20, 40 defaut ou 5, 20, 100
    # epsilons = [0.003, 0.01, 0.03, 0.1, 0.3]
    epsilons = [0.003, 0.01, 0.03, 0.1]



    #relative step size 1/40 or 1/10
    rows = []
    # attack2 = fb.attacks.L2ProjectedGradientDescentAttack() # 5, 20, 100 ?
    # epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    c = 0
    print("Start")

    # test_images, test_targets = extract_images(val_loader)

    with tqdm(total=len(val_loader)) as pbar:
        for images, targets in val_loader:
            row = []
            c += 1
            # print(images.min(), fmodel.bounds.lower)
            with torch.no_grad():
                inputs_, labels_ = images, targets
                out = fmodel(inputs_)


                predictions = out.argmax(axis=-1)
                clean_acc = (predictions == labels_).float().sum()
                # clean_acc = accuracy(fmodel, images, targets)
            # if fabric.global_rank == 0 and c % 20 == 1:
            #     print(c, clean_acc)
            row.append(clean_acc)

            for attack in attacks:
                images2 = torch.clone(images)
                 #maximum step size per pixel
                _, clipped_advs, success = attack(fmodel,  images2, targets, epsilons=epsilons)
                clipped_advs = [clip.detach() for clip in clipped_advs]
                with torch.no_grad():
                    for eps, advs_ in zip(epsilons, clipped_advs):
                        # acc2 = accuracy(fmodel, advs_, test_targets)
                        inputs_, labels_ = advs_, targets
                        del advs_

                        predictions = fmodel(inputs_).argmax(axis=-1)
                        acc = (predictions == labels_).float().sum()
                        row.append(acc)

            row.append(torch.tensor(images.shape[0], device=fabric.device))
            row = fabric.all_gather(torch.stack(row))#.flatten(0, 1).mean()
            if args.devices != 1:
                row = row.sum(dim=0)
            rows.append(row)
            pbar.update(row[-1].item())
    rows = torch.stack(rows)
    if args.devices != 1:
        rows = rows.sum(dim=0)
    rows = rows / rows[-1].clone()
    rows = rows.cpu().numpy() * 100
    print(rows)
    if fabric.global_rank == 0:
        writer = csv.writer(open(str(ckpt_dir / f"adversarial.csv"), "w"))
        writer.writerow([f"{rss}_{step}_{eps}" for eps in epsilons for step in steps for rss in rsss]+["size"])
        writer.writerow(rows)



if __name__ == "__main__":
    main()