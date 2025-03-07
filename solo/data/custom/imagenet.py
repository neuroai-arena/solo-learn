import io
import json
import os
from pathlib import Path
from typing import Union, Callable, Optional, Tuple
import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from solo.data.custom.base import H5ClassificationDataset


class ImgnetDataset(Dataset):

    def __init__(self, data_root, split, transform, imgnet100=True):
        # super().__init__()
        self.mode = "train" if split == "train" else "val"
        driver = "core" if self.mode == "train" and imgnet100 else None
        self.hdf5_file = h5py.File(os.path.join(data_root, f'data2_{self.mode}.h5'), "r", driver=driver)
        self.dataset = pd.read_csv(os.path.join(data_root, f'dataset2_{self.mode}.csv'), header=None)
        self.dataset.columns.astype(str)
        self.dataset.columns = ["index", "0", "1", "2", "label", "4"] if self.mode == "train" else ["0", "1", "2",
                                                                                                    "label", "4"]
        self.transform = transform
        self.n_classes = 1000

        self.imgnet100 = imgnet100
        if imgnet100:
            self.n_classes = 100
            name_id_map = json.load(open(os.path.join(data_root, "imagenet_class_index.json"), "r"))
            mapping = {v[0]: int(k) for k, v in name_id_map.items()}

            classes_file = "solo/data/dataset_subset/imagenet100_classes.txt"
            with open(classes_file) as f:
                self.classes = f.readline().strip().split()
            self.id_classes = sorted(self.classes)

            catfilter = list(map(lambda x: mapping[x], self.id_classes))
            self.category_mapping = {cat: i for i, cat in enumerate(catfilter)}

            self.dataset = self.dataset.query("label in @catfilter")
            self.dataset = self.dataset.reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        h5_index, label, begin = self.dataset.loc[idx, "0"], self.dataset.loc[idx, "label"], self.dataset.loc[idx, '4']

        if self.imgnet100:
            label = self.category_mapping[label]

        if self.mode == "val":
            img = Image.open(io.BytesIO(self.hdf5_file.get(f"data2_{begin}")[h5_index]))
        else:
            h5_index = h5_index % 50000
            img = Image.open(io.BytesIO(self.hdf5_file.get(str(begin)).get("data")[h5_index]))
        x = self.transform(img)
        return x, label


class ImgNetDataset_42(H5ClassificationDataset):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            split: str = "train",
            subset: Optional[str] = None
    ):
        super().__init__(root, transform, split, driver="core" if split == "train" and subset is not None else None)
        if subset == '1pct' and split == "train":
            subset_df = pd.read_csv("solo/data/dataset_subset/imagenet_1percent.txt", names=['filename'])
            self.mapper = self.mapper.query("filename.isin(@subset_df.filename)").copy().reset_index(drop=True)
            print("Using IN 1%")
        elif subset == '10pct' and split == "train":
            subset_df = pd.read_csv("solo/data/dataset_subset/imagenet_10percent.txt", names=['filename'])
            self.mapper = self.mapper.query("filename.isin(@subset_df.filename)").copy().reset_index(drop=True)
            print("Using IN 10%")
        elif subset == "imgnet100":
            with open(self.root / 'imagenet100_classes.txt') as f:
                imgnet100_classes = sorted(f.readline().strip().split())
            imgnet100_class_wn_2_class_index = {class_wn: class_index for class_index, class_wn in
                                                enumerate(imgnet100_classes)}

            self.mapper = self.mapper.query('wn_name in @imgnet100_classes').reset_index(drop=True)
            self.mapper['target'] = self.mapper['wn_name'].apply(lambda x: imgnet100_class_wn_2_class_index[x])

        self.n_classes = self.mapper['target'].nunique()
        self.target_2_class_name = self.mapper[['target', 'class_name']].drop_duplicates().set_index('target')[
            'class_name'].to_dict()


class ImageNetOODDataset(Dataset):
    VERSION = ['colour',
               'contrast',
               'cue-conflict',
               'edge',
               'eidolonI',
               'eidolonII',
               'eidolonIII',
               'false-colour',
               'high-pass',
               'low-pass',
               'phase-scrambling',
               'power-equalisation',
               'rotation',
               'silhouette',
               'sketch',
               'stylized',
               'uniform-noise']

    def __init__(self,
                 path: str,
                 version: str,
                 return_classname: bool = False,
                 transform: Optional[Callable] = None):
        self.version = version
        self.transform = transform
        self.return_classname = return_classname
        self.h5_file = h5py.File(path, "r", driver="core")

        self.available_versions = list(self.h5_file.keys())
        if not self.version in self.available_versions:
            raise ValueError(f"Version {self.version} not available, choose one of {self.available_versions}")

    def __len__(self) -> int:
        return self.h5_file.get(self.version).get('targets').shape[0]

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Union[str, np.ndarray]]:
        version = self.h5_file.get(self.version)
        image = Image.fromarray(version.get("images")[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_classname:
            targets = version.get("classnames")[idx]
        else:
            targets = version.get("targets")[idx]

        return image, targets


class ImageNetS(ImageFolder):
    def __init__(self, root, transform=None, mode: str = 'full'):
        super().__init__(root, transform)
        mask_p = root + '-segmentation/'
        self.masks = [Path(mask_p + '/'.join(x[0].split('/')[-2:])).with_suffix('.png') for x in self.samples]
        self.mode = mode
        if self.mode not in ['full', 'fg', 'bg', 'bg_rec']:
            raise ValueError(f"Mode {self.mode} not available, choose one of ['full', 'fg', 'bg', 'bg_rec']")

    def __getitem__(self, index: int):
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.mode == "fg":
            mask = np.array(Image.open(self.masks[index]))[..., 0] > 0
            sample = Image.fromarray(np.array(sample) * mask[..., None])
        elif self.mode == "bg":
            mask = 1 - (np.array(Image.open(self.masks[index]))[..., 0] > 0).astype(np.uint8)
            sample = Image.fromarray(np.array(sample) * mask[..., None])
        elif self.mode == 'bg_rec':
            mask = np.array(Image.open(self.masks[index]))[..., 0]
            bbox = cv2.boundingRect(mask)
            mask = np.ones_like(mask)
            mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 0
            sample = Image.fromarray(np.array(sample) * mask[..., None])

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
