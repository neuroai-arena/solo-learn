import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from PIL import Image
from omegaconf import OmegaConf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms


from solo.data.classification_dataloader import (
    prepare_transforms,
)
from solo.data.pretrain_dataloader import prepare_dataloader, prepare_datasets
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


class DoubleTransform:
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, x: Image, x2: Image, action = None):
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        out = []
        out.append(self.transforms(x))
        out.append(self.transforms(x2))
        if action is not None:
            out.append(action)
        return out

    def __repr__(self) -> str:
        return str(self.transforms)


def overlay_saliency_on_image(image, saliency, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Args:
        image: H x W x 3 RGB image (uint8)
        saliency: H x W saliency map (float or uint8)
        alpha: float in [0, 1], transparency of overlay
        colormap: OpenCV colormap to apply to saliency
    Returns:
        Blended image (RGB)
    """
    # Normalize saliency to [0, 255]
    # Apply colormap (result is BGR)
    heatmap = cv2.applyColorMap(saliency, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend heatmap with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="cuda")

    parser.add_argument("--dataset", type=str, default="nymeria")
    parser.add_argument("--train_data_path", type=Path, default="/scratch_shared/aubret/nymeria/")
    parser.add_argument("--val_data_path", type=Path, default="/scratch_shared/aubret/nymeria/")
    parser.add_argument("--data_format", default="h5", choices=["image_folder", "dali", "h5"])
    parser.add_argument("--scale", type=float, default=0.99)

    # parser.add_argument("--shift", default=100, type=int)


    args = parser.parse_args()


    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    cfg = OmegaConf.create(method_args)
    cfg.devices = args.num_devices
    cfg.accelerator = args.accelerator

    # build the model
    # if not args.random:
    model = METHODS[method_args["method"]].load_from_checkpoint(ckpt_path, strict=True, cfg=cfg)
    # else:
    #     model = METHODS[method_args["method"]](cfg=cfg)
    # prepare data
    T = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ]
    )
    # del T.transforms[3]
    T = DoubleTransform(T)
    print(T)



    train_dataset = prepare_datasets(
        args.dataset,
        T,
        train_data_path=args.train_data_path,
        data_format=args.data_format
    )


    train_loader = prepare_dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )



    # extract test features
    # strategy = DDPStrategy(broadcast_buffers=False) #if args.device != "cpu" else "ddp_cpu"
    fabric = L.Fabric(accelerator=cfg.accelerator, devices=cfg.devices, strategy="ddp", precision=cfg.precision)
    fabric.launch()

    # model = model.backbone
    model = fabric.setup(model)
    train_loader = fabric.setup_dataloaders(train_loader)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=fabric.device).view(-1,1,1).expand((3, 224, 224))
    std = torch.tensor([0.229, 0.224, 0.225], device=fabric.device).view(-1,1,1).expand((3, 224, 224))

    path = f"/scratch_shared/aubret/results/visualizations/{cfg.method}/"

    for p in model.parameters():
        p.requires_grad_(False)

    for c, data, targets in train_loader:
        image0 = data[0]
        image1 = data[1]

        image0.requires_grad_()
        image1.requires_grad_()
        loss = model.training_step((c, data, targets), (np.arange(args.batch_size)))
        loss.backward(retain_graph=True)

        saliency0, _ = torch.max(image0.grad.data.abs().clone().detach(), dim=1)
        saliency1, _ = torch.max(image1.grad.data.abs().clone().detach(), dim=1)
        saliency0_norm = (saliency0 / torch.amax(saliency0, dim=(1, 2), keepdim=True)).cpu().numpy()
        saliency1_norm = (saliency1 / torch.amax(saliency1, dim=(1, 2), keepdim=True)).cpu().numpy()

        for i in range(data[0].shape[0]):
            npim0 = np.einsum('hw->hw', saliency0_norm[i] * 255).astype(np.uint8)
            npim1 = np.einsum('hw->hw',saliency1_norm[i] * 255).astype(np.uint8)

            os.makedirs(os.path.join(path, "sal"), exist_ok=True)

            denorm_img0 = ((image0[i] * std + mean).detach().cpu().numpy() * 255).astype(np.uint8)
            denorm_img1 = ((image1[i] * std + mean).detach().cpu().numpy() * 255).astype(np.uint8)
            denorm_img0 = np.einsum('chw->hwc', denorm_img0)
            denorm_img1 = np.einsum('chw->hwc', denorm_img1)

            # zero_channel = np.zeros_like(npim0)
            # npim0 = np.stack([npim0, zero_channel, zero_channel], axis=2)
            # npim1 = np.stack([npim1, zero_channel, zero_channel], axis=2)


            denorm_img0 = overlay_saliency_on_image(denorm_img0, npim0, alpha=0.4)
            denorm_img1 = overlay_saliency_on_image(denorm_img1, npim1, alpha=0.4)


            Image.fromarray(denorm_img0).save(os.path.join(path, "sal", f"image{i}_0.png"))
            Image.fromarray(denorm_img1).save(os.path.join(path, "sal", f"image{i}_1.png"))
        break



    # if fabric.global_rank == 0:
    #     writer.writerows(rows)



if __name__ == "__main__":
    main()