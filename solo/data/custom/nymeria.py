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
    def __init__(self, data_root, transform,gaze_size=224, min_gaze_size=0, time_window=3, aa_time_window=25, center_crop=False, resolution=512,
                 fps=1,  normalize=False, version=1, distinct_action=False, size_gaze_aware=True, reversed_gaze_size=False, fixations=None,
                 cam_frame=False, as_euler=False, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.time_window = time_window
        self.center_crop = center_crop
        self.gaze_size = gaze_size
        self.min_gaze_size = gaze_size if not min_gaze_size else min_gaze_size
        self.min_gaze_size = self.min_gaze_size if self.min_gaze_size != -1 else 0
        self.resolution = resolution
        self.normalize = normalize
        self.version = version
        self.distinct_action = distinct_action
        self.aa_time_window=aa_time_window
        self.adj_resolution = int(self.resolution*0.9)
        self.size_gaze_aware = size_gaze_aware
        self.reversed_gaze_size = reversed_gaze_size
        self.cam_frame = cam_frame
        self.as_euler = as_euler
        if fixations:
            self.fixations = np.load(fixations)
            self.fixations = np.concatenate([[self.fixations[0]],self.fixations])
        else:
            self.fixations = None

        v = f"v{self.version}" if self.version >= 2 else ""


        self.hdf5_file = h5py.File(os.path.join(self.data_root, f"data{v}_fps{fps}_res{resolution}.h5"), "r")
        # self.hdf5_file_name = os.path.join(self.data_root, f"data{v}_fps{fps}_res{resolution}.h5")
        try:
            self.dataset = pd.read_csv(os.path.join(self.data_root, f"egodata_depth_fps{fps}_res{resolution}.csv"))
        except:
            self.dataset = pd.read_csv(os.path.join(self.data_root, f"egodata{v}_fps{fps}_res{resolution}.csv"))


        self.action_size = 9
        self.size = len(self.dataset)

        if self.cam_frame:
            # self.means = np.array([ 0.869, -0.654,  0.005,  0.014,  0.036,  0.927,  0, 0, 0],dtype=np.float32)
            # self.stds = np.array([50.698, 40.,  0.099,  0.101,  0.258,  0.23, 0.22,  0.73,  0.68],dtype=np.float32)
            if self.as_euler:
                self.means = np.zeros((11,), dtype=np.float32)
                self.stds = np.array([50.698, 40.,  1,1,1 ,1,1,1, -1.5358144e-03,
                                      -6.2901266e-02,  6.9679245e-02],dtype=np.float32)
            else:
                self.means = np.zeros((9,),dtype=np.float32)
                self.stds = np.array([50.698, 40.,  -6.6378959e-03,  6.5328823e-03,  9.0809369e-01,4.6217211e-02,  -1.5358144e-03,
                                      -6.2901266e-02,  6.9679245e-02],dtype=np.float32)
        else:
            self.means = np.array([ 0.869, -0.654,  0.005,  0.014,  0.036,  0.927,  5.975, -4.339,  0.538],dtype=np.float32)
            self.stds = np.array([50.698, 40.,  0.099,  0.101,  0.258,  0.23,  43.243, 40.499, 12.283],dtype=np.float32)
        print("Length:", self.size)


    def __len__(self):
        return self.size

    def open_image(self, row, gaze_size):
        # index, number, partition = int(row[6]), int(row[11]), str(int(row[5]))
        # img = Image.open(io.BytesIO(self.hdf5_file.get(recording)[index]))
        img = Image.open(io.BytesIO(self.hdf5_file.get("data")[int(row["new_index"])]))



        if self.center_crop:
            img = torchvision.transforms.functional.center_crop(img, (gaze_size, gaze_size))
            return img, (row["gaze_x"], row["gaze_y"])
        else:
            adj_gaze_x, adj_gaze_y = row["gaze_x"], row["gaze_y"]
            ### We control the gaze the boundaries of the gaze to not go beyond the image boundaries
            gap_resolution = self.resolution - self.adj_resolution
            adj_gaze_x += - max(0,adj_gaze_x + gaze_size//2 - self.adj_resolution) - min(0, adj_gaze_x - gaze_size//2)
            adj_gaze_y += - max(0,adj_gaze_y + gaze_size//2 - self.adj_resolution) - min(0, adj_gaze_y - gaze_size//2)
            # adj_gaze_x += - max(0,adj_gaze_x + self.gaze_size//2 - self.adj_resolution) - min(0, adj_gaze_x - self.gaze_size//2 - gap_resolution)
            # adj_gaze_y += - max(0,adj_gaze_y + self.gaze_size//2 - self.adj_resolution) - min(0, adj_gaze_y - self.gaze_size//2 - gap_resolution)
            img = torchvision.transforms.functional.crop(img,
                                                             adj_gaze_y - gaze_size//2,
                                                             adj_gaze_x - gaze_size//2,
                                                             gaze_size,
                                                             gaze_size,
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
        bef_rot = row_before.loc[["quatx", "quaty", "quatz","quatw"]].values
        aft_rot = row_after.loc[["quatx", "quaty", "quatz","quatw"]].values

        bef_trans = row_before.loc[["tx", "ty", "tz"]].values
        aft_trans = row_after.loc[["tx", "ty", "tz"]].values

        assert np.abs(np.linalg.norm(bef_rot) - 1) < 0.05
        assert np.abs(np.linalg.norm(aft_rot) -1) < 0.05

        # if self.version == 2:
        return self.get_action_v2(bef_rot, aft_rot, bef_trans, aft_trans, gaze_before, gaze_after)
        #
        # camera_rot = self.quaternion_multiply((aft_rot[0],-aft_rot[1],-aft_rot[2],-aft_rot[3]), (bef_rot[0],bef_rot[1],bef_rot[1],bef_rot[3])).squeeze()
        # camera_rot = np.concatenate((camera_rot[1:4], camera_rot[0:1]), axis=0)
        #
        # translation = aft_trans- bef_trans
        # translation_rot = np.transpose(scipy.spatial.transform.Rotation.from_quat(camera_rot).as_matrix())
        # translation = np.matmul(translation_rot, translation)
        #
        # gaze_movement = np.array(gaze_after) - np.array(gaze_before)
        # gaze_movement /= self.adj_resolution - self.gaze_size
        #
        # action = torch.tensor(np.concatenate(
        #     (gaze_movement.astype(np.float32),
        #     camera_rot.astype(np.float32),
        #     translation.astype(np.float32)), axis=0
        # ))
        # return action

    def compute_relative_pose(self, R1, t1, R2, t2):
        # Compute relative rotation
         # or np.dot(R2, R1.T)

        # Compute relative translation
        if self.cam_frame:
            # R_rel = R2 @ R1.T
            # t_rel = R1.T @ (t2 - t1)
            R_rel = R2.T @ R1
            t_rel = R1.T @ (t2 - t1)
        elif self.cam_frame == 2:
            # R_rel = R2 @ R1.T
            # t_rel = R1.T @ (t2 - t1)
            R_rel = R2 @ R1.T
            t_rel = R1 @ (t2 - t1)
        else:
            R_rel = R2 @ R1.T
            t_rel = t2 - R_rel @ t1

        return R_rel, t_rel

    def get_action_v2(self, r1, r2, t1, t2, g1, g2):
        r1 = scipy.spatial.transform.Rotation.from_quat(r1).as_matrix()
        r2 = scipy.spatial.transform.Rotation.from_quat(r2).as_matrix()

        r_rel, t_rel = self.compute_relative_pose(r1, t1, r2, t2)

        gaze_movement = np.array(g2) - np.array(g1)
        # if self.normalize:
        #     gaze_movement /= self.adj_resolution - self.gaze_size
        if self.as_euler:
            r_euler = scipy.spatial.transform.Rotation.from_matrix(r_rel).as_euler("xyz")
            r_rel = np.concatenate((np.cos(r_euler), np.sin(r_euler)), axis=0)
        else:
            r_rel = scipy.spatial.transform.Rotation.from_matrix(r_rel).as_quat()

        action = torch.tensor(np.concatenate(
            (gaze_movement.astype(np.float32),
            r_rel.astype(np.float32),
            t_rel.astype(np.float32)), axis=0
        ))

        action = (action - self.means)/self.stds
        return action

    def get_gaze_sizes(self, row):
        if self.min_gaze_size != 0:
            g = random.randint(self.min_gaze_size, self.gaze_size)
        else:
            depth_value = row["corrected_gaze_depth"]
            depth_value = depth_value if depth_value != -1 else 1
            if not self.reversed_gaze_size:
                g = depth_value * self.gaze_size / 4
            else:
                g = self.gaze_size / depth_value

        return g
    def __getitem__(self, idx):
        # if not hasattr(self, 'hdf5_file'):
        #     self.hdf5_file = h5py.File(self.hdf5_file_name, 'r')


        r = self.dataset.iloc[idx]
        video_name = r["file_id"]

        g1 = self.get_gaze_sizes(r)
        image, adj_gaze_before = self.open_image(r, g1)

        if self.time_window == 0:
            return self.transform(image, image), -1

        new_video_name, new_idx, try_cpt = "", idx, 0

        keep_searching=True
        while keep_searching:
            new_idx = idx + random.randint(-self.time_window,self.time_window)
            new_idx = max(0,min(new_idx, self.size-1))
            if try_cpt > 10:
                new_idx = idx
            rn = self.dataset.iloc[new_idx]
            new_video_name = rn["file_id"]

            if try_cpt > 10:
                break
            try_cpt += 1
            keep_searching = (video_name != new_video_name)
            if self.fixations is not None:
                same_fixation = self.fixations[idx] == self.fixations[new_idx]
                keep_searching = keep_searching or not same_fixation



        g2 = self.get_gaze_sizes(rn)
        image_pair, adj_gaze_after = self.open_image(rn, g2) if new_idx != idx or g1 != g2 else (image, adj_gaze_before)

        image_pair_action = None
        if self.distinct_action:
            # Action sample
            new_video_name, new_idx, try_cpt = "", idx, 0
            keep_searching=True
            while keep_searching:
                new_idx = idx + random.randint(-self.aa_time_window, self.aa_time_window)
                new_idx = max(0, min(new_idx, self.size - 1))
                if try_cpt > 10:
                    new_idx = idx
                rn = self.dataset.iloc[new_idx]
                new_video_name = rn["file_id"]
                if try_cpt > 10:
                    break
                try_cpt += 1
                keep_searching = (video_name != new_video_name)
                if self.fixations is not None:
                    same_fixation = self.fixations[idx] == self.fixations[new_idx]
                    keep_searching = keep_searching or not same_fixation
            g2 = self.get_gaze_sizes(rn)
            image_pair_action, adj_gaze_after = self.open_image(rn, g2) if new_idx != idx or g1 != g2 else (image, adj_gaze_before)

        action = self.get_action(r, rn, adj_gaze_before, adj_gaze_after)
        if self.min_gaze_size != self.gaze_size and self.size_gaze_aware:
            action = torch.cat((action, torch.tensor([float(g2 - g1) / (self.gaze_size - self.min_gaze_size)])))
        return self.transform(image, image_pair, image_pair_action, action), -1


