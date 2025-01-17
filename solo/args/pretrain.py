import os

import omegaconf
from omegaconf import OmegaConf, ListConfig
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import omegaconf_select
from solo.args.linear import _SUPPORTED_DATASETS as _CLF_SUPPORTED_DATASETS

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule
except ImportError:
    _dali_available = False
else:
    _dali_available = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

_N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet2": 1000,
    "imagenet100": 100,
    "imagenet2_100": 100,
    "tiny": 200,
    "ego4d": 0,
}

_SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    "stl10",
    "imagenet",
    "imagenet100",
    "custom",
    "imagenet2",
    "imagenet2_100",
    "tiny",
    "ego4d"
]


def add_and_assert_dataset_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for dataset config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    assert not OmegaConf.is_missing(cfg, "data.dataset")
    assert not OmegaConf.is_missing(cfg, "data.train_path")

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    # if validation path is not available, assume that we want to skip eval
    cfg.data.val_path = omegaconf_select(cfg, "data.val_path", None)
    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.data.no_labels = omegaconf_select(cfg, "data.no_labels", False)
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)
    cfg.debug_augmentations = omegaconf_select(cfg, "debug_augmentations", False)
    cfg.data.dataset_kwargs = omegaconf_select(cfg, "data.dataset_kwargs", {})

    return cfg


def add_and_assert_knn_clb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for KNN callback config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    assert not OmegaConf.is_missing(cfg, "knn_clb.dataset")
    assert not OmegaConf.is_missing(cfg, "knn_clb.train_path")
    assert not OmegaConf.is_missing(cfg, "knn_clb.val_path")
    assert cfg.knn_clb.dataset in _CLF_SUPPORTED_DATASETS

    cfg.knn_clb.format = omegaconf_select(cfg, "knn_clb.format", "image_folder")
    cfg.knn_clb.batch_size = omegaconf_select(cfg, "knn_clb.batch_size", 32)
    cfg.knn_clb.num_workers = omegaconf_select(cfg, "knn_clb.num_workers", 4)

    k = omegaconf_select(cfg, "knn_clb.k", 20)
    k = (k,) if isinstance(k, int) else k
    cfg.knn_clb.k = k

    cfg.knn_clb.T = omegaconf_select(cfg, "knn_clb.T", 0.07)
    cfg.knn_clb.distance_fx = omegaconf_select(cfg, "knn_clb.distance_fx", "cosine")

    cfg.knn_clb.perform_on_validation = omegaconf_select(cfg, "knn_clb.perform_on_validation", True)
    cfg.knn_clb.perform_on_test = omegaconf_select(cfg, "knn_clb.perform_on_test", False)
    cfg.knn_clb.delay_epochs = omegaconf_select(cfg, "knn_clb.delay_epochs", 0)
    cfg.knn_clb.perform_every_n_batches = omegaconf_select(cfg, "knn_clb.perform_every_n_batches", None)

    if cfg.knn_clb.perform_every_n_batches is not None:
        assert isinstance(cfg.knn_clb.perform_every_n_batches,
                          (int, float)), "perform_every_n_batches must be an int or float"
        if isinstance(cfg.knn_clb.perform_every_n_batches, float):
            assert cfg.knn_clb.perform_every_n_batches > 0, "perform_every_n_batches must be greater than 0"
            assert cfg.knn_clb.perform_every_n_batches < 1, "perform_every_n_batches must be less than 1"

    cfg.knn_clb.verbose = omegaconf_select(cfg, "knn_clb.verbose", False)

    return cfg


def add_and_assert_wandb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for wandb config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "solo-learn")
    cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", False)
    cfg.wandb.group = omegaconf_select(cfg, "wandb.group", None)
    cfg.wandb.job_type = omegaconf_select(cfg, "wandb.job_type", None)

    return cfg


def add_and_assert_lightning_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for Pytorch Lightning config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.seed = omegaconf_select(cfg, "seed", 5)
    cfg.resume_from_checkpoint = omegaconf_select(cfg, "resume_from_checkpoint", None)
    cfg.strategy = omegaconf_select(cfg, "strategy", None)

    return cfg


def parse_cfg(cfg: omegaconf.DictConfig):
    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for knn_clb
    cfg = add_and_assert_knn_clb_cfg(cfg)

    # default values for dali
    if _dali_available:
        cfg = PretrainDALIDataModule.add_and_assert_specific_cfg(cfg)

    # default values for auto_umap
    if _umap_available:
        cfg = AutoUMAP.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    cfg = add_and_assert_lightning_cfg(cfg)

    # extra processing
    if cfg.data.dataset in _N_CLASSES_PER_DATASET:
        cfg.data.num_classes = _N_CLASSES_PER_DATASET[cfg.data.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        cfg.data.num_classes = max(
            1,
            sum(entry.is_dir() for entry in os.scandir(cfg.data.train_path)),
        )

    # find number of big/small crops
    big_size = cfg.augmentations[0].crop_size
    num_large_crops = num_small_crops = 0
    for pipeline in cfg.augmentations:
        if big_size == pipeline.crop_size:
            num_large_crops += pipeline.num_crops
        else:
            num_small_crops += pipeline.num_crops
    cfg.data.num_large_crops = num_large_crops
    cfg.data.num_small_crops = num_small_crops

    if cfg.data.format == "dali":
        assert cfg.data.dataset in ["imagenet100", "imagenet", "custom"]

    # adjust lr according to batch size
    cfg.num_nodes = omegaconf_select(cfg, "num_nodes", 1)
    tl = len(cfg.devices) if isinstance(cfg.devices, ListConfig) else cfg.devices
    scale_factor = cfg.optimizer.batch_size * tl * cfg.num_nodes / 256
    print(f"Scaling learning rate by {scale_factor}")
    cfg.optimizer.lr = cfg.optimizer.lr * scale_factor
    if cfg.data.val_path is not None:
        assert not OmegaConf.is_missing(cfg, "optimizer.classifier_lr")
        cfg.optimizer.classifier_lr = cfg.optimizer.classifier_lr * scale_factor

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
    elif cfg.optimizer.name == "lars":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
        cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
        cfg.optimizer.kwargs.exclude_bias_n_norm = omegaconf_select(
            cfg,
            "optimizer.kwargs.exclude_bias_n_norm",
            False,
        )
    elif cfg.optimizer.name == "adamw":
        cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

    cfg.no_validation = omegaconf_select(cfg, "no_validation", False)

    return cfg
