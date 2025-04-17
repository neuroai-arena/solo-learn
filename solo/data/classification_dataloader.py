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

import os
from pathlib import Path
import shutil
from typing import Callable, Optional, Tuple, Union

import torch
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from solo.data.custom.imagenet import ImgnetDataset, ImgNetDataset_42, ImageNetS
from solo.data.custom.base import H5ClassificationDataset
from solo.data.custom.core50 import Core50, Core50ForBGClassification
from solo.data.custom.tinyimgnet import TinyDataset
from solo.data.pretrain_dataloader import GaussianBlur

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True


class FeatureDataset(Dataset):
    def __init__(self, root: Union[str, Path]):
        self.features = torch.load(str(root).format(type='feat'))
        self.labels = torch.load(str(root).format(type='target'))

        assert len(self.features) == len(self.labels), "Features and labels must have the same length."

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """

    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }
    return pipeline


def prepare_transforms(dataset: str, **kwargs) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    cifar_scale_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline_224 = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        )
    }

    core50_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        )
    }

    toybox_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        )
    }

    tiny_pipeline = {
        "T_train": transforms.Compose(
            [
                # transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomResizedCrop(size=64, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                # transforms.Resize(256),  # resize shorter
                transforms.Resize(74),  # resize shorter
                # transforms.CenterCrop(224),  # take center crop
                transforms.CenterCrop(64),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
    }

    coil100_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(224),  # resize shorter
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "cifar10_224": cifar_scale_pipeline,
        "cifar100_224": cifar_scale_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "imagenet2_100": imagenet_pipeline,
        "imagenet2": imagenet_pipeline,
        "imagenet_42": imagenet_pipeline,
        "imagenet100_42": imagenet_pipeline,
        "ego4d": imagenet_pipeline,
        "tiny": tiny_pipeline,
        'core50': core50_pipeline,
        'core50_bg': core50_pipeline,
        "custom": custom_pipeline,
        'DTD': imagenet_pipeline,
        'Flowers102': imagenet_pipeline,
        'FGVCAircraft': imagenet_pipeline,
        'Food101': imagenet_pipeline,
        'OxfordIIITPet': imagenet_pipeline,
        'Places365': imagenet_pipeline,
        'StanfordCars': imagenet_pipeline,
        "STL10": stl_pipeline,
        "STL10_224": stl_pipeline_224,
        "STL10_FG_224": stl_pipeline_224,
        "STL10_FG": stl_pipeline,
        "Places365_h5": imagenet_pipeline,
        "SUN397": imagenet_pipeline,
        "imagenet1pct_42": imagenet_pipeline,
        "imagenet10pct_42": imagenet_pipeline,
        "toybox": toybox_pipeline,
        "feat": imagenet_pipeline,  # this is a placeholder
        'COIL100': coil100_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    if kwargs.get("global_gaussian_blur", None) is not None:
        sigma = kwargs["global_gaussian_blur"]["sigma"]
        T_train.transforms.insert(0, GaussianBlur(sigma=sigma))
        T_val.transforms.insert(0, GaussianBlur(sigma=sigma))

    print("T_train", T_train)
    print("T_val", T_val)
    return T_train, T_val


def prepare_datasets(
        dataset: str,
        T_train: Callable,
        T_val: Callable,
        train_data_path: Optional[Union[str, Path]] = None,
        val_data_path: Optional[Union[str, Path]] = None,
        data_format: Optional[str] = "image_folder",
        download: bool = True,
        data_fraction: float = -1.0,
        **dataset_kwargs
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if val_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        val_data_path = sandbox_folder / "datasets"

    assert dataset in [
        "cifar10", "cifar100", "stl10", "imagenet", "imagenet100", "custom", "imagenet2", "imagenet2_100", "ego4d",
        "tiny", "cifar10_224", "cifar100_224", "imagenet_42", "imagenet100_42", 'core50', "DTD", 'Flowers102',
        'FGVCAircraft', 'Food101', 'OxfordIIITPet', 'Places365', 'StanfordCars', "STL10","STL10_224", "Places365_h5", "SUN397",
        "Caltech101", "imagenet1pct_42", "imagenet10pct_42", "toybox", 'core50_bg', "feat", "COIL100", "STL10_FG_224",
        "STL10_FG"
    ]

    if dataset in ["cifar10", "cifar100", "cifar10_224", "cifar100_224"]:
        if dataset == "cifar10_224": dataset = "cifar10"
        if dataset == "cifar100_224": dataset = "cifar100"
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            train_data_path,
            train=True,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            val_data_path,
            train=False,
            download=download,
            transform=T_val,
        )
    elif dataset in ['DTD', 'Flowers102', 'FGVCAircraft', 'Food101', 'OxfordIIITPet', 'Places365', 'StanfordCars',
                     'STL10', 'SUN397', 'Caltech101', 'STL10_224']:
        if dataset == "STL10_224": dataset = "STL10"
        DatasetClass = vars(torchvision.datasets)[dataset]

        if dataset == "StanfordCars" and download:
            download = False
            if not (Path(train_data_path) / 'stanford_cars').exists():
                import kagglehub
                kagglehub.login()
                path = kagglehub.dataset_download("rickyyyyyyy/torchvision-stanford-cars", force_download=False)
                ds_path = Path(path) / 'stanford_cars'
                shutil.move(ds_path, train_data_path)

        # some datasets have different names for their train split
        if dataset == "Places365":
            train_split = "train-standard"
        elif dataset == "OxfordIIITPet":
            train_split = "trainval"
        else:
            train_split = "train"

        train_dataset = DatasetClass(
            train_data_path,
            split=train_split,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            val_data_path,
            split="test",
            download=download,
            transform=T_val,
        )
    elif dataset in ["Places365_h5"]:
        train_dataset = H5ClassificationDataset(root=Path(train_data_path) / 'Places365', transform=T_train,
                                                split="train")
        val_dataset = H5ClassificationDataset(root=Path(val_data_path) / 'Places365', transform=T_val, split="val")
    elif dataset in ["COIL100"]:
        train_dataset = H5ClassificationDataset(root=Path(train_data_path) / 'coil100', transform=T_train,
                                                split="train")
        val_dataset = H5ClassificationDataset(root=Path(val_data_path) / 'coil100', transform=T_val, split="val")
    elif dataset in ["STL10_FG_224", "STL10_FG"]:
        train_dataset = H5ClassificationDataset(root=Path(train_data_path) / 'STL10-FG', transform=T_train,
                                                split="train")
        val_dataset = H5ClassificationDataset(root=Path(val_data_path) / 'STL10-FG', transform=T_val, split="val")
    elif dataset in ["tiny"]:
        train_dataset = TinyDataset(train_data_path, "train", T_train)
        val_dataset = TinyDataset(val_data_path, "val", T_val)
    elif dataset in ["ego4d"]:
        train_dataset = ImgnetDataset(val_data_path, "val", T_val,
                                      True)  # ImgnetDataset(train_data_path, "train", T_train, True)
        val_dataset = ImgnetDataset(val_data_path, "val", T_val, True)
    elif dataset in ["imagenet2", "imagenet2_100"]:
        train_dataset = ImgnetDataset(train_data_path, "train", T_train, dataset == "imagenet2_100")
        val_dataset = ImgnetDataset(val_data_path, "val", T_val, dataset == "imagenet2_100")
    elif dataset in ["imagenet_42", "imagenet100_42", "imagenet1pct_42", "imagenet10pct_42"]:
        if dataset == "imagenet100_42":
            subset = "imgnet100"
        elif dataset == "imagenet1pct_42":
            subset = "1pct"
        elif dataset == "imagenet10pct_42":
            subset = "10pct"
        else:
            subset = None

        train_dataset = ImgNetDataset_42(Path(train_data_path) / 'ImageNet/h5', T_train, split="train", subset=subset)
        val_dataset = ImgNetDataset_42(Path(val_data_path) / 'ImageNet/h5', T_val, split="val", subset=subset)
    elif dataset in ["imagenet", "imagenet100", "custom"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = H5Dataset(dataset, train_data_path, T_train)
            val_dataset = H5Dataset(dataset, val_data_path, T_val)
        else:
            train_dataset = ImageFolder(train_data_path, T_train)
            val_dataset = ImageFolder(val_data_path, T_val)
    elif dataset == 'core50':
        assert 'train_backgrounds' in dataset_kwargs.keys()
        assert 'val_backgrounds' in dataset_kwargs.keys()
        train_dataset = Core50(h5_path=Path(train_data_path) / 'core50_350x350/core50_arr.h5', transform=T_train,
                               backgrounds=dataset_kwargs['train_backgrounds'])
        val_dataset = Core50(h5_path=Path(val_data_path) / 'core50_350x350/core50_arr.h5', transform=T_val,
                             backgrounds=dataset_kwargs['val_backgrounds'])
    elif dataset == 'core50_bg':
        train_dataset = Core50ForBGClassification(h5_path=Path(train_data_path) / 'core50_350x350/core50_arr.h5',
                                                  split="train", transform=T_train)
        val_dataset = Core50ForBGClassification(h5_path=Path(val_data_path) / 'core50_350x350/core50_arr.h5',
                                                split="test", transform=T_val)
    elif dataset == "toybox":
        train_dataset = H5ClassificationDataset(Path(train_data_path) / 'ToyBox/h5', split="train", transform=T_train)
        val_dataset = H5ClassificationDataset(Path(val_data_path) / 'ToyBox/h5', split="test", transform=T_val)
    elif dataset == "feat":
        train_dataset = FeatureDataset(train_data_path)
        val_dataset = FeatureDataset(val_data_path)
    elif dataset.startswith("imagenet_s"):
        mode = ""
        train_dataset = ImageNetS(Path(train_data_path) / 'ImageNet-S/ImageNetS919/training', transform=T_train, mode=mode)
        val_dataset = ImageNetS(Path(val_data_path) / 'ImageNet-S/ImageNetS919/validation', transform=T_val, mode=mode)
    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        )
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset, val_dataset


def prepare_dataloaders(
        train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, samplers=(None, None)
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=samplers[0]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=samplers[1]
    )
    return train_loader, val_loader


def prepare_data(
        dataset: str,
        train_data_path: Optional[Union[str, Path]] = None,
        val_data_path: Optional[Union[str, Path]] = None,
        data_format: Optional[str] = "image_folder",
        batch_size: int = 64,
        num_workers: int = 4,
        download: bool = True,
        data_fraction: float = -1.0,
        auto_augment: bool = False,
        transform_kwargs: Optional[dict] = None,
        **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
        auto_augment (bool, optional): use auto augment following timm.data.create_transform.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader.
    """

    T_train, T_val = prepare_transforms(dataset, **(transform_kwargs or {}))
    if auto_augment:
        T_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=None,  # don't use color jitter when doing random aug
            auto_augment="rand-m9-mstd0.5-inc1",  # auto augment string
            interpolation="bicubic",
            re_prob=0.25,  # random erase probability
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        download=download,
        data_fraction=data_fraction,
        **dataset_kwargs
    )
    # samplers = (None, None)
    # if sampler == "distributed":
    #     samplers = (DistributedSampler(train_dataset), DistributedSampler(val_dataset))
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader
