import io
import os
import random

import h5py
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class Ego4d(Dataset):

    data=None

    def __init__(self, data_root, transform, *args, gaze_size=224, time_window=30, center_crop=False, ego4d_subset=1, **kwargs):
        super().__init__(*args, **kwargs)
        assert gaze_size in  [114, 160, 224, 313, 440, 540]
        assert abs(ego4d_subset) <= 1

        self.data_root = data_root
        self.transform = transform
        self.gaze_size = gaze_size
        self.time_window = time_window
        self.center_crop = center_crop

        self.hdf5_file = h5py.File(os.path.join(self.data_root, f"data_all95.h5"), "r")
        self.dataset = h5py.File(os.path.join(self.data_root, f"dataset_all95.h5"), "r")
        # self.crop_index = 7 + np.argmin(np.abs(np.array([114, 160, 224, 313, 440, 540]) - gaze_size))*2

        blurred_frames = np.load(os.path.join(self.data_root, f"unblur_all.npy"),)
        self.bool_clear_frames = blurred_frames == 0
        self.clear_frames = np.where(self.bool_clear_frames)[0]

        if ego4d_subset != 1:
            self.clear_frames = self.clear_frames[:int(len(self.clear_frames)*ego4d_subset)]
        self.size = len(self.clear_frames)

        print("Length:", len(self))


    def __len__(self):
        return self.size

    def open_image(self, row):
        index, number, partition = int(row[6]), int(row[19]), str(int(row[5]))
        gaze_size = self.gaze_size
        if gaze_size == -1:
            gaze_size = random.choice([114, 160, 224, 313, 439, 540])

        binary_img = self.hdf5_file.get(partition).get(f"images{gaze_size}_{str(number)}")[index]
        pil_img = Image.open(io.BytesIO(binary_img))
        if self.center_crop:
            pil_img = torchvision.transforms.functional.center_crop(pil_img, (224, 244))
        return pil_img

    def __getitem__(self, idx):
        idx = self.clear_frames[idx]
        r = self.dataset.get("data")[idx]
        image, video_name = self.open_image(r), r[0]

        if self.time_window == 0:
            return self.transform(image, image)

        new_video_name, new_idx, try_cpt = "", idx, 0
        while video_name != new_video_name or not self.bool_clear_frames[new_idx]:
            new_idx = idx + random.randint(-self.time_window,self.time_window)
            new_idx = max(0,min(new_idx, len(self)-1))
            if try_cpt > 5:
                new_idx = idx
            rn = self.dataset.get("data")[new_idx]
            new_video_name = rn[0]
            try_cpt += 1


        image_pair = self.open_image(rn)
        x = self.transform(image, image_pair)
        return x, -1