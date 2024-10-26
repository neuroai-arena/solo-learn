import io
import json
import os
from pathlib import Path
from typing import Union, Callable, Optional, Tuple

import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


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


class ImgNetDataset_42(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            split: str = "train",
            imgnet100: bool = False
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        assert self.split in ["train", "val"]

        self.h5_file = h5py.File(root / f"ImageNet_{split}.h5", "r")
        self.mapper = pd.read_parquet(root / f"ImageNet_{split}_mapper.parquet")

        if imgnet100:
            with open(root / 'imagenet100_classes.txt') as f:
                imgnet100_classes = sorted(f.readline().strip().split())
            imgnet100_class_wn_2_class_index = {class_wn: class_index for class_index, class_wn in
                                                enumerate(imgnet100_classes)}

            self.mapper = self.mapper.query('wn_name in @imgnet100_classes').reset_index(drop=True)
            self.mapper['target'] = self.mapper['wn_name'].apply(lambda x: imgnet100_class_wn_2_class_index[x])

        self.n_classes = self.mapper['target'].nunique()
        self.target_2_class_name = self.mapper[['target', 'class_name']].drop_duplicates().set_index('target')[
            'class_name'].to_dict()

    def __len__(self) -> int:
        return len(self.mapper)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        dp = self.mapper.loc[idx]
        image = Image.open(io.BytesIO(self.h5_file.get("images")[dp["h5_index"]]))

        if self.transform:
            image = self.transform(image)

        return image, dp["target"]
