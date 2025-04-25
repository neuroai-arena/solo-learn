import io
import os
import random

import h5py
import numpy as np
import pandas as pd
import scipy
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from solo.data.cortical_magnification import radial_quad_isotrop_gridfun, img_cortical_magnif_tsr
from solo.data.foveation import foveation


class Nymeria(Dataset):
    gaze_sizes = (112, 224, 336, 448, 540)
    def __init__(self, data_root, transform,gaze_size=224, time_window=3, center_crop=False, resize_gs=False, resolution=512, fps=1, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.time_window = time_window
        self.center_crop = center_crop
        self.gaze_size = gaze_size
        self.resize_gs = resize_gs
        self.resolution = resolution
        self.adj_resolution = int(self.resolution*0.9)


        self.hdf5_file = h5py.File(os.path.join(self.data_root, f"data_fps{fps}_res{resolution}.h5"), "r")
        self.dataset = pd.read_csv(os.path.join(self.data_root, f"egodata_fps{fps}_res{resolution}.csv"))

        self.action_size = 9
        self.size = len(self.dataset)
        print("Length:", self.size)


    def __len__(self):
        return self.size

    def open_image(self, row):
        # index, number, partition = int(row[6]), int(row[11]), str(int(row[5]))
        recording = str(row["file_id"])
        index = int(row["index"])
        # img = Image.open(io.BytesIO(self.hdf5_file.get(recording)[index]))
        img = Image.fromarray(self.hdf5_file.get(recording)[index].reshape(self.resolution,self.resolution,3))
        img = torchvision.transforms.functional.center_crop(img, (self.adj_resolution, self.adj_resolution))

        if self.center_crop:
            img = torchvision.transforms.functional.center_crop(img, (self.gaze_size, self.gaze_size))
            return img, (row["gaze_x"], row["gaze_y"])
        elif self.gaze_size == 540:
            if self.resize_gs:
                img = torchvision.transforms.functional.resize(img, 224, InterpolationMode.BICUBIC)
            return img, (row["gaze_x"], row["gaze_y"])
        else:
            adj_gaze_x, adj_gaze_y = row["gaze_x"], row["gaze_y"]
            ### We control the gaze the boundaries of the gaze to not go beyond the image boundaries
            adj_gaze_x += - max(0,adj_gaze_x + self.gaze_size//2 - self.adj_resolution) - min(0, adj_gaze_x - self.gaze_size//2)
            adj_gaze_y += - max(0,adj_gaze_y + self.gaze_size//2 - self.adj_resolution) - min(0, adj_gaze_y - self.gaze_size//2)

            img = torchvision.transforms.functional.crop(img,
                                                             adj_gaze_y - self.gaze_size//2,
                                                             adj_gaze_x - self.gaze_size//2,
                                                             self.gaze_size,
                                                             self.gaze_size,
                                                         )
            return img, (adj_gaze_x, adj_gaze_y)

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def get_action(self, row_before, row_after, gaze_before, gaze_after):
        bef_rot = row_before.loc[["quatw", "quatx", "quaty", "quatz"]].values
        aft_rot = row_after.loc[["quatw", "quatx", "quaty", "quatz"]].values

        bef_trans = row_before.loc[["tx", "ty", "tz"]].values
        aft_trans = row_after.loc[["tx", "ty", "tz"]].values

        # bef_rot = bef_rot / np.linalg.norm(bef_rot)
        # aft_rot = aft_rot / np.linalg.norm(aft_rot)

        assert np.abs(np.linalg.norm(bef_rot) - 1) < 0.05
        assert np.abs(np.linalg.norm(aft_rot) -1) < 0.05

        camera_rot = self.quaternion_multiply((aft_rot[0],-aft_rot[1],-aft_rot[2],-aft_rot[3]), (bef_rot[0],bef_rot[1],bef_rot[1],bef_rot[3])).squeeze()
        camera_rot = np.concatenate((camera_rot[1:4], camera_rot[0:1]), axis=0)

        translation = aft_trans- bef_trans
        translation_rot = np.transpose(scipy.spatial.transform.Rotation.from_quat(quat=camera_rot).as_matrix())
        translation = np.matmul(translation_rot, translation)

        gaze_mouvement = np.array(gaze_after) - np.array(gaze_before)

        action = torch.tensor(np.concatenate(
            (gaze_mouvement.astype(np.float32),
            camera_rot.astype(np.float32),
            translation.astype(np.float32)), axis=0
        ))
        return action




    def __getitem__(self, idx):

        r = self.dataset.iloc[idx]
        video_name = r["file_id"]


        image, adj_gaze_before = self.open_image(r)

        if self.time_window == 0:
            return self.transform(image, image), -1

        new_video_name, new_idx, try_cpt = "", idx, 0
        while video_name != new_video_name:
            new_idx = idx + random.randint(-self.time_window,self.time_window)
            new_idx = max(0,min(new_idx, self.size-1))
            if try_cpt > 10:
                new_idx = idx
            rn = self.dataset.iloc[new_idx]
            new_video_name = rn["file_id"]
            try_cpt += 1


        image_pair, adj_gaze_after = self.open_image(rn) if new_idx != idx else (image, adj_gaze_before)
        action = self.get_action(r, rn, adj_gaze_before, adj_gaze_after)
        return self.transform(image, image_pair, action), -1