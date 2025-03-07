import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import OmegaConf
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.functional import to_pil_image

from solo.data.classification_dataloader import (
    prepare_datasets,
)
from solo.methods import METHODS


class Denormalize(T.Normalize):
    def __init__(self, mean, std):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)


class InferAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_store = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        self.attn_store = attn
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def replace_attn(model):
    for block in model.backbone.blocks:
        device, dtpye = block.attn.qkv.weight.device, block.attn.qkv.weight.dtype
        new_attn = InferAttention(
            dim=block.attn.qkv.in_features,
            num_heads=block.attn.num_heads,
            qkv_bias=block.attn.qkv.bias is not None,
            qk_norm=not isinstance(block.attn.q_norm, nn.Identity),
            attn_drop=block.attn.attn_drop.p,
            proj_drop=block.attn.proj_drop.p,
            norm_layer=type(block.attn.q_norm)
        )
        new_attn.to(device=device, dtype=dtpye)
        new_attn.load_state_dict(block.attn.state_dict())

        block.attn = new_attn
    return model


def load_model(pretrained_checkpoint_dir):
    ckpt_dir = Path(pretrained_checkpoint_dir)
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
    model.eval()

    return model


def load_dataset(dataset_name, train_data_path, val_data_path, img_size=512):
    transform = T.Compose([
        T.Resize((img_size, img_size)),  # resize shorter
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    reverse_transform = T.Compose([
        Denormalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # Denormalize
        T.Lambda(lambda x: to_pil_image(x.clamp(0, 1)))  # Convert tensor to PIL Image
    ])

    _, val_dataset = prepare_datasets(
        dataset_name,
        T_train=transform,
        T_val=transform,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
    )

    return val_dataset, reverse_transform
