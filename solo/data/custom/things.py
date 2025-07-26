from pathlib import Path
from typing import Union

import PIL
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class Things(Dataset):
    def __init__(self, root: Union[str, Path], transforms):
        self.root = root #/ "object_images"
        self.csv_file = pd.read_csv(root / "image-paths.csv", names=["image"])
        self.csv_file['nameID'] = self.csv_file['image'].apply(lambda x: x.split('/')[1])
        self.csv_file['ID'] = pd.factorize(self.csv_file['nameID'])[0]
        self.csv_file.sort_values(by=["ID", 'image'])
        self.transforms = transforms

        print(len(self))

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        row = self.csv_file.iloc[idx]
        path, concept_id = row.loc[["image","ID"]]

        return self.transforms(PIL.Image.open(self.root / path)), concept_id