from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics.segmentation import MeanIoU
# from torchvision.models.feature_extraction import create_feature_extractor

from solo.methods.linear import LinearModel


class SegmentationHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 width: int,
                 height: int,
                 num_classes: int,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.height = height
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.height, self.width, self.in_channels)  # (bs, num_tokens, c) -> (bs, h, w, c)
        x = x.permute(0, 3, 1, 2)  # (bs, h, w, c) -> (bs, c, h, w)
        return self.classifier(x)  # (bs, c, h, w) -> (bs, num_classes, h, w)


class SegmentationModel(LinearModel):
    def __init__(self, backbone: nn.Module, cfg: DictConfig):
        super().__init__(backbone, cfg)

        if hasattr(self.backbone, "patch_embed"):  # transformer backbone
            feat_heigth, feat_width = getattr(self.backbone.patch_embed, "grid_size")
        else:
            raise ValueError("Backbone type not supported for segmentation")

        # self.backbone = create_feature_extractor(backbone, cfg.data.dataset, use_patches=True)

        if not hasattr(self.backbone, "forward_features"):
            raise ValueError("Backbone forward_features method not implemented")

        self.classifier = SegmentationHead(in_channels=self.features_dim,
                                           num_classes=cfg.data.num_classes,
                                           width=feat_width, height=feat_heigth)

        self.train_miou = MeanIoU(num_classes=cfg.data.num_classes, include_background=False)
        self.val_miou = MeanIoU(num_classes=cfg.data.num_classes, include_background=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """

        if not self.no_channel_last and not self.pre_extract_feats:
            X = X.to(memory_format=torch.channels_last)

        if not self.pre_extract_feats or self.trainer.sanity_checking:
            with torch.set_grad_enabled(self.finetune):
                feats = self.backbone.forward_features(X)
        else:
            feats = X

        if feats.ndim == 3:
            feats = feats[:, 1:, :]  # exclude the [CLS] token
        else:
            raise ValueError("Features shape not supported for segmentation")

        logits = self.classifier(feats)
        # interpolate logits to the original image size
        logits = F.interpolate(logits, size=X.shape[2:], mode="bilinear", align_corners=False)

        return logits

    def shared_step(self, batch: Tuple, batch_idx: int, training: bool = True) -> torch.Tensor:
        """Performs operations that are shared between the training and validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]: a dict containing
                batch size, loss, mIoU.
        """
        mode = "train" if training else "val"
        if training and not self.finetune:
            self.backbone.eval()

        X, target = batch
        target = target.long().squeeze()

        out = self(X)

        loss = F.cross_entropy(out.squeeze(), target, ignore_index=0)
        self.log(f"{mode}/loss", loss, on_step=training, on_epoch=not training, sync_dist=True)

        getattr(self, f"{mode}_miou")(out.argmax(dim=1), target)
        self.log(f"{mode}/mIoU", getattr(self, f"{mode}_miou"), on_step=training, on_epoch=not training, sync_dist=True)

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, training=True)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, training=False)

    def on_validation_epoch_end(self):
        pass
