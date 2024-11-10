import io
from typing import Tuple, Callable, Optional
from pathlib import Path
import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Core50(Dataset):
    def __init__(self,
                 h5_path: str,
                 backgrounds: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None
                 ):
        self.transform = transform
        print(h5_path, Path(h5_path).exists())

        self.h5_file = h5py.File(h5_path, "r")

        avail_bgs = set(self.h5_file.keys())
        self.backgrounds = backgrounds if backgrounds is not None else avail_bgs

        df = []
        for bg in self.backgrounds:
            if bg not in avail_bgs:
                raise ValueError(f"Background class {bg} not found. Use {avail_bgs}.")
            for i in range(self.h5_file.get(bg).get("images").shape[0]):
                df.append({'h5_index': i, 'bg': bg})
        self.df = pd.DataFrame(df)
        print(f"Using {self.backgrounds} with {len(df)} images.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        dp = self.df.iloc[idx]
        bg = self.h5_file.get(dp.bg)

        # image = Image.open(io.BytesIO(bg.get("images")[dp.h5_index])).convert("RBG")
        image = Image.fromarray(bg.get("images")[dp.h5_index])
        target = bg.get("targets")[dp.h5_index]

        if self.transform is not None:
            image = self.transform(image)

        # print(image, target)

        return image, target







# import h5py
# import pandas as pd
# from pathlib import Path
# from typing import Callable, Optional, Tuple
# from PIL import Image
# import io
# from torch.utils.data import Dataset
#
# class Core50(Dataset):
#     def __init__(self,
#                  h5_path: str,
#                  backgrounds: Optional[Tuple[str, ...]] = None,
#                  transform: Optional[Callable] = None):
#         self.h5_path = h5_path
#         self.transform = transform
#
#         # Ensure the HDF5 file path exists
#         print(h5_path, Path(h5_path).exists())
#
#         # Only open the file once in __getitem__
#         self.h5_file = None
#
#         # Load metadata for indexing
#         with h5py.File(h5_path, "r") as h5_file:
#             avail_bgs = set(h5_file.keys())
#             self.backgrounds = backgrounds if backgrounds is not None else avail_bgs
#
#             # Build a dataframe with indexing information
#             df = []
#             for bg in self.backgrounds:
#                 if bg not in avail_bgs:
#                     raise ValueError(f"Background class {bg} not found. Use {avail_bgs}.")
#                 for i in range(h5_file[bg]["images"].shape[0]):
#                     df.append({'h5_index': i, 'bg': bg})
#             self.df = pd.DataFrame(df)
#             print(f"Using {self.backgrounds} with {len(df)} images.")
#
#     def __len__(self) -> int:
#         return len(self.df)
#
#     def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
#         # Open the HDF5 file in each worker if it hasn't been opened already
#         if self.h5_file is None:
#             self.h5_file = h5py.File(self.h5_path, "r", driver="core")
#
#         # Access the required data
#         dp = self.df.iloc[idx]
#
#         # Read the image and target
#         image = Image.open(
#             io.BytesIO(
#                 self.h5_file.get(dp.bg).get("images")[dp.h5_index]
#             )
#         ).convert("RGB")
#
#         target = self.h5_file.get(dp.bg).get("targets")[dp.h5_index]
#
#         # Apply any transformations
#         if self.transform is not None:
#             image = self.transform(image)
#
#         return image, target
#
#     def __del__(self):
#         # Ensure the file is properly closed when the dataset is deleted
#         if self.h5_file is not None:
#             self.h5_file.close()
#
#
# if __name__ == '__main__':
#     from torchvision.transforms import Compose, ToTensor
#
#     ds = Core50(h5_path='/Volumes/CVAI-SSD-0/datasets/core50/core50.h5',
#                 transform=Compose([ToTensor()]))
#     dl = DataLoader(ds, batch_size=10, num_workers=8)
#
#     for i in dl:
#         print(i)
#         break
