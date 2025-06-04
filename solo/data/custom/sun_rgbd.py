
import os
from glob import glob

import PIL.Image

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class SunRGBD(Dataset):

    def __init__(self, data_root, split, transform, transform_depth):
        self.transform = transform
        self.transform_depth = transform_depth

        images = []
        depths = []
        for data_source in ["kv1", "kv2"]:
            ds = os.path.join(data_root, data_source)
            for ds2 in sorted(os.listdir(ds)):
                if ds2.startswith("."):
                    continue
                dataset = os.path.join(ds, ds2)
                for img_dir in sorted(os.listdir(dataset)):
                    if img_dir.startswith("."):
                        continue
                    scene_dir = os.path.join(dataset, img_dir)
                    images.append(glob(f"{scene_dir}/image/*")[0])
                    ## Get depth map file path from scene directory
                    depths.append(glob(f"{scene_dir}/depth_bfx/*")[0])

        train_images, test_images, train_depths, test_depths = train_test_split(images, depths, test_size=0.25, random_state=0)
        if split == "train":
            self.images = train_images
            self.depths = train_depths
        else:
            self.images = test_images
            self.depths = test_depths

        print("Dataset size:", len(self))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.transform(PIL.Image.open(self.images[idx]))
        # d = torchvision.transforms.functional.to_tensor(PIL.Image.open(self.depths[idx]))
        d = self.transform_depth(PIL.Image.open(self.depths[idx]).convert("L"))
        return x, d
