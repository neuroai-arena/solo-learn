import warnings

# Suppress the specific warning
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting .* in registry with .*\. This is because the name being registered conflicts with an existing name.*",
    category=UserWarning
)
import logging

import hydra
import h5py
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pathlib import Path
import numpy as np
from lightning.fabric import Fabric
from solo.data.classification_dataloader import prepare_data
from solo.methods.base import BaseMethod
from timm.models.vision_transformer import VisionTransformer
import timm


class ViTFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        if not isinstance(self.backbone, VisionTransformer):
            raise ValueError("Backbone must be a VisionTransformer")

    @property
    def output_shape(self):
        return self.backbone.num_features

    @torch.no_grad()
    def forward(self, x):
        x = self.backbone.forward_features(x)

        cls = x[:, 0]
        avgpool = x[:, self.backbone.num_prefix_tokens:].mean(dim=1)

        return cls, avgpool


def merge_features(world_size, output_dir, meta):
    """Merge features from all GPU processes into a single file"""
    # Get total number of samples across all processes
    total_samples = 0
    feature_dim = None

    # First, determine the total number of samples and feature dimension
    for rank in range(world_size):
        input_file = output_dir / f"features_{rank}.h5"
        with h5py.File(input_file, 'r') as h5f:
            features = h5f['cls_token']
            total_samples += features.shape[0]
            if feature_dim is None:
                feature_dim = features.shape[1]

    # Create merged output file
    merged_file = output_dir / "features.h5"
    with h5py.File(merged_file, 'w') as merged_h5f:
        cls_token_dataset = merged_h5f.create_dataset('cls_token',
                                                      shape=(total_samples, feature_dim),
                                                      dtype=np.float32)
        avgpool_dataset = merged_h5f.create_dataset('avgpool',
                                                    shape=(total_samples, feature_dim),
                                                    dtype=np.float32)
        labels_dataset = merged_h5f.create_dataset('labels',
                                                   shape=(total_samples,),
                                                   dtype=np.int64)

        # Copy data from each rank's file
        start_idx = 0
        for rank in range(world_size):
            input_file = output_dir / f"features_{rank}.h5"
            with h5py.File(input_file, 'r') as h5f:
                cls_features = h5f['cls_token'][:]
                avgpool_features = h5f['avgpool'][:]
                labels = h5f['labels'][:]

                # Get number of samples in this file
                n_samples = features.shape[0]
                end_idx = start_idx + n_samples

                # Copy data to merged file
                cls_token_dataset[start_idx:end_idx] = cls_features
                avgpool_dataset[start_idx:end_idx] = avgpool_features
                labels_dataset[start_idx:end_idx] = labels

                # Update start index
                start_idx = end_idx

            # remove the individual file
            input_file.unlink()

        # Add metadata
        merged_h5f.attrs['num_samples'] = total_samples
        merged_h5f.attrs['feature_dim'] = feature_dim
        for k, v in meta.items():
            merged_h5f.attrs[k] = v

    print(f"Successfully merged features from {world_size} GPUs. Total samples: {total_samples}")


def load_custom_model(cfg):
    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()
        cifar = cfg.data.dataset in ["cifar10", "cifar100"]
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    if cfg.pretrained_feature_extractor is not None:
        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state.keys()):
            if "encoder" in k:
                state[k.replace("encoder", "backbone")] = state[k]
                logging.warn(
                    "You are using an older checkpoint. Use a new one as some issues might arrise."
                )
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]
        backbone.load_state_dict(state, strict=False)
        logging.info(f"Loaded {ckpt_path}")
    return backbone




@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    fabric = Fabric(accelerator="cuda", devices=cfg.devices, strategy="ddp", precision=32)
    fabric.launch()


    timm_model = "vit_base_patch16_rpn_224" # vit_base_patch16_gap_224
    backbone = timm.create_model(timm_model, pretrained=True)
    # backbone = load_custom_model(cfg=cfg)


    _, loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=cfg.data.format,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = ViTFeatureExtractor(backbone=backbone)

    model = fabric.setup_module(model)
    loader = fabric.setup_dataloaders(loader)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"features_{fabric.global_rank}.h5"

    # Get total number of samples this process will handle
    num_samples = len(loader) * cfg.data.batch_size
    fabric.print(f"Extracting features for {num_samples} samples per GPU...")

    with h5py.File(output_file, 'w') as h5f:
        cls_token_dataset = h5f.create_dataset('cls_token',
                                               shape=(num_samples, model.output_shape),
                                               dtype=np.float32)
        avgpool_dataset = h5f.create_dataset('avgpool',
                                             shape=(num_samples, model.output_shape),
                                             dtype=np.float32)
        labels_dataset = h5f.create_dataset('labels',
                                            shape=(num_samples,),
                                            dtype=np.int64)

        idx = 0
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"GPU {fabric.global_rank}")):
            cls_features, avgpool_features = model(images)

            current_batch_size = images.size(0)

            end_idx = idx + current_batch_size
            cls_token_dataset[idx:end_idx] = cls_features.cpu().numpy()
            avgpool_dataset[idx:end_idx] = avgpool_features.cpu().numpy()
            labels_dataset[idx:end_idx] = labels.cpu().numpy()

            idx = end_idx

    fabric.barrier()

    if fabric.global_rank == 0:
        name = Path(cfg.pretrained_feature_extractor).name if cfg.pretrained_feature_extractor is not None else "model"
        print("Merging features from all GPUs...")
        merge_features(fabric.world_size, output_dir, meta={'dataset': cfg.data.dataset, 'ckpt': name})
        print("Feature extraction completed successfully!")


if __name__ == "__main__":
    main()
