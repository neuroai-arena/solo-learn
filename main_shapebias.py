import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import lightning as L
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2 as trv2, InterpolationMode

from tqdm import tqdm

from solo.data.callbacks.shapebias import TripletDataset, eval_fc
from solo.methods import METHODS


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
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        if model.backbone_name.startswith("resnet"):
            self.num_features = self.backbone.inplanes
        else:
            self.num_features = self.backbone.num_features
        self.projector = model.projector

    def forward(self, input):
        out = self.backbone(input)
        out = self.projector(out)
        return out

def modify_cfg(option, cfg):
    if option == "none":
        cfg.method_kwargs.layers = 2
        cfg.method_kwargs.layers_pred = 2
    if option =="vit":
        cfg.backbone.name = "vit_base"
        cfg.method_kwargs.layers = 3
        cfg.method_kwargs.layers_pred = 2
        cfg.backbone.kwargs.patch_size = 16
        cfg.backbone.kwargs.global_pool = "avg"

def main():
    parser = argparse.ArgumentParser()

    def str2table(v):
        return v.split(',')
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--pretrained_checkpoint_file", type=str, default="")
    parser.add_argument("--path", type=str, default="/home/autolearn/aubret/triplets/")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--layer_names", default=["backbone.avgpool"], type=str2table)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--option", default="none", type=str)


    args = parser.parse_args()


    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    cfg = OmegaConf.create(method_args)
    modify_cfg(args.option, cfg)

    model = METHODS[cfg.method](cfg)

    # backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    # backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    # model = ModelWrapper(backbone, cfg.backbone.name)
    if not args.pretrained_checkpoint_file:
        ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if "last" in ckpt][0]
    else:
        ckpt_path = args.pretrained_checkpoint_file
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model = ModelWrapper(model)
    model = create_feature_extractor(model, return_nodes=args.layer_names)

    mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
    preprocess = trv2.Compose(
        [trv2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), trv2.ToImage(),
         trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])

    dataset_common = TripletDataset(os.path.join(args.path, "shape_simpletext"), transform=preprocess)
    dataloader = DataLoader(dataset_common, batch_size=32, shuffle=False, pin_memory=True)
    fabric = L.Fabric(accelerator=cfg.accelerator, devices=cfg.devices if not args.devices else args.devices, precision=cfg.precision)
    fabric.launch()
    model = fabric.setup(model)
    dataloader = fabric.setup_dataloaders(dataloader)
    model.eval()

    shape_acc = eval_fc(dataloader, model, fabric.device)
    print(shape_acc)





if __name__ == "__main__":
    main()