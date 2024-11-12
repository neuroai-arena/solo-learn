import os.path
import sys
import pandas
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

sys.path.append("../..")

import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision import transforms
from solo.data.pretrain_dataloader import NCropAugmentation, GaussianBlur, Solarization, Equalization
import copy
import io
import os
import random

import h5py
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from solo.data.foveation import foveation

class Ego4d(Dataset):

    corrupted = [(24,14), (60, 16), (61, 13), (64, 12), (65,9), (40,8)]
    readded = [71,56,67,74]
    def __init__(self, data_root, transform,gaze_size=224, time_window=15, center_crop=False, resize_gs=False, foveation=None, **kwargs):
        super().__init__()
        assert gaze_size in  [114, 160, 224, 313, 440, 540]

        self.data_root = data_root
        self.transform = transform
        self.time_window = time_window
        self.center_crop = center_crop
        self.gaze_size = gaze_size
        self.resize_gs = resize_gs
        self.foveation = foveation

        self.hdf5_file = h5py.File(os.path.join(self.data_root, f"data0.hdf5"), "r")
        self.dataset = pandas.read_csv(os.path.join(self.data_root, f"dataset0.csv"), header=None)

        self.gaze_index = (9, 10)

        self.size = len(self.dataset)
        print("Length:", self.size)


    def __len__(self):
        return self.size

    def open_image(self, row):
        index, number= int(row[6]), int(row[11])
        # print(partition, number, index, flush=True)

        gaze_size = self.gaze_size
        if gaze_size == -1:
            gaze_size = random.choice([114, 160, 224, 313, 439, 540])

        binimg = self.hdf5_file.get("frames").get(f"images540_{str(number)}")[index]
        img = Image.open(io.BytesIO(binimg))



        if self.center_crop:
            img = torchvision.transforms.functional.center_crop(img, (self.gaze_size, self.gaze_size))
        elif gaze_size == 540:
            if self.resize_gs:
                img = torchvision.transforms.functional.resize(img, 224, InterpolationMode.BICUBIC)
        elif self.foveation:
            ### We extract the gaze location in the image
            imtsr = torchvision.transforms.functional.to_tensor(img)
            gaze_x, gaze_y = row[self.gaze_index[0]], row[self.gaze_index[1]]
            img = foveation(img, (gaze_y, gaze_x), **self.foveation)
            img = torchvision.transforms.functional.to_pil_image(img)
        else:
            ### We control the gaze the boundaries of the gaze to not go beyond the image boundaries
            gaze_x += - max(0,gaze_x + gaze_size//2 - 540) - min(0, gaze_x - gaze_size//2)
            gaze_y += - max(0,gaze_y + gaze_size//2 - 540) - min(0, gaze_y - gaze_size//2)

            img = torchvision.transforms.functional.crop(img,
                                                             gaze_y - gaze_size//2,
                                                             gaze_x - gaze_size//2,
                                                             gaze_size,
                                                             gaze_size,
                                                         )

        return img

    def __getitem__(self, idx):
        self.idx = idx
        # idx = self.clear_frames[idx]
        r = self.dataset.iloc[idx]
        image, video_name = self.open_image(r), r[0]

        if self.time_window == 0:
            return self.transform(image, image), -1

        new_video_name, new_idx, try_cpt = "", idx, 0
        # while video_name != new_video_name or not self.bool_clear_frames[new_idx]:
        while video_name != new_video_name:
            new_idx = idx + random.randint(-self.time_window,self.time_window)
            new_idx = max(0,min(new_idx, self.size-1))
            if try_cpt > 5:
                new_idx = idx
            rn = self.dataset[new_idx]
            new_video_name = rn[0]
            try_cpt += 1


        image_pair = self.open_image(rn) if new_idx != idx else image
        return self.transform(image, image_pair), -1





def build_transform_pipeline():
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    augmentations = []
    augmentations.append(
        transforms.RandomResizedCrop(
            224,
            scale=(0.2, 1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
    )


    augmentations.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    0.4,
                    0.4,
                    0.2,
                    0.1,
                )
            ],
            p=0.8,
        ),
    )

    augmentations.append(transforms.RandomGrayscale(p=0.1))
    augmentations.append(transforms.RandomApply([GaussianBlur()], p=0.1))
    augmentations.append(transforms.RandomApply([Solarization()], p=0.2))
    augmentations.append(transforms.RandomApply([Equalization()], p=0))
    augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
    augmentations.append(transforms.ToTensor())
    # augmentations.append(transforms.Normalize(mean=mean, std=std))

    augmentations = transforms.Compose(augmentations)
    return augmentations



# t= NCropAugmentation( transforms.Compose([transforms.RandomResizedCrop((224,224),interpolation=InterpolationMode.BICUBIC), transforms.ToTensor()]), 2)

t =NCropAugmentation(build_transform_pipeline(), 2)
# t= NCropAugmentation( transforms.Compose([transforms.CenterCrop((540,540))]), 2)
dataset_v1 = Ego4d("/scratch/autolearn/aubret/ego4dv2/", t, gaze_size=224, time_window=0, foveation={"kerW_coef":0.04})
dataloader_v1 = DataLoader(dataset_v1, shuffle=False, batch_size=1, num_workers=1)

torch.manual_seed(0)
itv1 = iter(dataloader_v1)


if not os.path.exists("/scratch/autolearn/aubret/ego4d/samples_fov"):
    os.makedirs("/scratch/autolearn/aubret/ego4d/samples_fov")


for i, i1 in enumerate(itv1):
    torchvision.utils.save_image(i1[0][0], f"/scratch/autolearn/aubret/ego4d/samples_fov/{i}_1.png")
    torchvision.utils.save_image(i1[0][1], f"/scratch/autolearn/aubret/ego4d/samples_fov/{i}_2.png")
    # torchvision.utils.save_image(i2[0][0], f"/scratch/autolearn/aubret/ego4d/samplesv2/{i}.png")
    if i > 30:
        break