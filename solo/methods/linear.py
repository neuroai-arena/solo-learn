# Copyright 2023 solo-learn development team.
import gc
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import lightning.pytorch as pl
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from solo.utils.lars import LARS
from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.misc import (
    omegaconf_select,
    param_groups_layer_decay,
    remove_bias_and_norm_from_weight_decay,
)


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        assert len(self.features) == len(self.labels), "Features and labels must have the same length."

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


@torch.inference_mode()
def extract_features(loader: DataLoader, module: pl.LightningModule) -> Tuple[torch.Tensor, torch.Tensor]:
    module.eval()
    features, labels = [], []
    bar = tqdm(loader, desc="Extracting features") if module.trainer.is_global_zero else loader
    for i, (image, target) in enumerate(bar):
        image = image.cuda(non_blocking=True, memory_format=torch.channels_last)
        outs = module.backbone(image)

        features.append(outs)
        labels.append(target)
    features = torch.cat(features)
    labels = torch.cat(labels)

    features = module.all_gather(features).view(-1, features.shape[-1])
    labels = module.all_gather(labels).view(-1)
    # print("RANK", module.trainer.local_rank, ":", features.shape, labels.shape)

    module.train()
    return features, labels


class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
            self,
            backbone: nn.Module,
            cfg: omegaconf.DictConfig,
            loss_func: Callable = None,
            mixup_func: Callable = None,
    ):
        """Implements linear and finetune evaluation.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        backbone (nn.Module): backbone architecture for feature extraction.
        Cfg basic structure:
            data:
                num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.

            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                kwargs (Dict): extra named arguments for the optimizer.
            scheduler:
                name (str): name of the scheduler.
                min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
                warmup_start_lr (float): initial learning rate for warmup scheduler.
                    Defaults to 0.00003.
                warmup_epochs (float): number of warmup epochs. Defaults to 10.
                lr_decay_steps (Sequence, optional): steps to decay the learning rate
                    if scheduler is step. Defaults to None.
                interval (str): interval to update the lr scheduler. Defaults to 'step'.

            finetune (bool): whether or not to finetune the backbone. Defaults to False.

            performance:
                disable_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        loss_func (Callable): loss function to use (for mixup, label smoothing or default).
        Defaults to None mixup_func (Callable, optional). function to convert data and targets
        with mixup/cutmix. Defaults to None.
        """

        super().__init__()

        # add default values and assert that config has the basic needed settings
        cfg = self.add_and_assert_specific_cfg(cfg)
        self.cfg = cfg

        # backbone
        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features

        # classifier
        self.classifier = nn.Linear(features_dim, cfg.data.num_classes)  # type: ignore

        # mixup/cutmix function
        self.mixup_func: Callable = mixup_func

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()
        self.loss_func = loss_func

        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: Union[int, None] = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd
        self.layer_decay: float = cfg.optimizer.layer_decay

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]
        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # if finetuning the backbone
        self.finetune: bool = cfg.finetune

        # if pre-extracting features
        self.pre_extract_feats: bool = cfg.pre_extract_feats

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last

        if not self.finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # keep track of validation metrics
        self.validation_step_outputs = []
        self.max_val_acc_top1 = torch.tensor(0.0)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # default parameters for optimizer
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(
            cfg, "optimizer.exclude_bias_n_norm_wd", False
        )
        # default for extra optimizer kwargs (use pytorch's default if not available)
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        cfg.optimizer.layer_decay = omegaconf_select(cfg, "optimizer.layer_decay", 0.0)

        # whether or not to finetune the backbone
        cfg.finetune = omegaconf_select(cfg, "finetune", False)
        cfg.pre_extract_feats = omegaconf_select(cfg, "pre_extract_feats", False)

        if cfg.pre_extract_feats and cfg.finetune:
            raise ValueError("Cannot pre-extract features and finetune the backbone at the same time.")

        # default for acc grad batches
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", 1)

        # default parameters for the scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        return cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        if self.layer_decay > 0:
            assert self.finetune, "Only with use layer weight decay with finetune on."
            msg = (
                "Method should implement no_weight_decay() that returns "
                "a set of parameter names to ignore from weight decay"
            )
            assert hasattr(self.backbone, "no_weight_decay"), msg

            learnable_params = param_groups_layer_decay(
                self.backbone,
                self.weight_decay,
                no_weight_decay_list=self.backbone.no_weight_decay(),
                layer_decay=self.layer_decay,
            )
            learnable_params.append({"name": "classifier", "params": self.classifier.parameters()})
        else:
            learnable_params = (
                self.classifier.parameters()
                if not self.finetune
                else [
                    {"name": "backbone", "params": self.backbone.parameters()},
                    {"name": "classifier", "params": self.classifier.parameters()},
                ]
            )

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def on_train_start(self) -> None:
        if self.pre_extract_feats:
            print("Extracting train features", self.trainer.local_rank)
            train_feat, train_lab = extract_features(self.trainer.train_dataloader, self)
            # print(f"RANK {self.trainer.local_rank}: {train_feat.shape=} {train_lab.shape=}")

            print("Extracting val features", self.trainer.local_rank)
            val_feat, val_lab = extract_features(self.trainer.val_dataloaders, self)
            # print(f"RANK {self.trainer.local_rank}: {val_feat.shape=} {val_lab.shape=}")

            self.backbone.cpu()
            torch.cuda.empty_cache()
            gc.collect()

            train_dataloader = DataLoader(FeatureDataset(train_feat.detach().cpu(), train_lab.detach().cpu()),
                                          batch_size=self.cfg.optimizer.batch_size,
                                          shuffle=True,
                                          num_workers=self.cfg.data.num_workers)

            val_dataloader = DataLoader(FeatureDataset(val_feat.detach().cpu(), val_lab.detach().cpu()),
                                        batch_size=self.cfg.optimizer.batch_size,
                                        shuffle=False,
                                        num_workers=self.cfg.data.num_workers)

            self.trainer._data_connector.attach_data(self,
                                                     train_dataloaders=train_dataloader,
                                                     val_dataloaders=val_dataloader)
            self.trainer._data_connector.prepare_data()
            self.trainer.fit_loop.setup_data()
            self.trainer.fit_loop.reset()
            self.trainer.fit_loop.epoch_loop.val_loop.setup_data()

            # print(f"RANK {self.trainer.local_rank}: {self.trainer.train_dataloader.dataset}")
            # print(f"RANK {self.trainer.local_rank}: {self.trainer.train_dataloader.sampler}")
            # print(f"RANK {self.trainer.local_rank}: Done")
            self.trainer.strategy.barrier()

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
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
                feats = self.backbone(X)
        else:
            feats = X
        logits = self.classifier(feats)
        return {"logits": logits, "feats": feats}

    def shared_step(
            self, batch: Tuple, batch_idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        X, target = batch
        target = target.long()

        metrics = {"batch_size": X.size(0)}
        if self.training and self.mixup_func is not None:
            X, target = self.mixup_func(X, target)
            out = self(X)["logits"]
            loss = self.loss_func(out, target)
            metrics.update({"loss": loss})
        else:
            out = self(X)["logits"]
            loss = F.cross_entropy(out, target)
            acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
            metrics.update({"loss": loss, "acc1": acc1, "acc5": acc5})

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        if not self.finetune:
            self.backbone.eval()

        out = self.shared_step(batch, batch_idx)

        log = {"train_loss": out["loss"]}
        if self.mixup_func is None:
            log.update({"train_acc1": out["acc1"], "train_acc5": out["acc5"]})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        out = self.shared_step(batch, batch_idx)

        metrics = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        """

        val_loss = weighted_mean(self.validation_step_outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(self.validation_step_outputs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(self.validation_step_outputs, "val_acc5", "batch_size")
        self.validation_step_outputs.clear()

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        self.max_val_acc_top1 = torch.max(self.max_val_acc_top1, torch.tensor(val_acc1))
        self.log("max_val_acc_top1", self.max_val_acc_top1, sync_dist=True)

        self.log_dict(log, sync_dist=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a checkpoint.

        Args:
            checkpoint (Dict[str, Any]): checkpoint to save.
        """
        # remove backbone from checkpoint
        if not self.finetune:
            for key in list(checkpoint["state_dict"].keys()):
                if key.startswith("backbone"):
                    del checkpoint["state_dict"][key]