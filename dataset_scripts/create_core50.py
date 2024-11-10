from pathlib import Path
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    dir = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/core50_350x350")
    corr = [Path('/s3/o43/C_03_43_209.png')]

    total = len(list(filter(lambda x: not x.stem.startswith("."), dir.rglob("*.png"))))

    with h5py.File(dir / "core50_arr.h5", "w") as h5_file, tqdm(total=total) as bar:
        for bg_p in filter(lambda x: x.is_dir() and not x.stem.startswith("."), dir.iterdir()):
            bg_group = h5_file.create_group(bg_p.stem)

            images = list(filter(lambda x: not x.stem.startswith('.'), bg_p.rglob("*.png")))

            total_images = len(images)
            img_dataset = bg_group.create_dataset("images",
                                                  # shape=(len(images),),
                                                  # dtype=h5py.vlen_dtype(np.dtype('uint8')),
                                                  shape=(total_images, 350, 350, 3),
                                                  dtype=np.uint8
                                                  )
            target_dataset = bg_group.create_dataset("targets",
                                                     shape=(total_images,),
                                                     dtype=np.int32)
            for i, images_p in enumerate(images):
                # img_dataset[i] = np.fromfile(images_p, dtype=np.uint8)
                try:
                    img_dataset[i] = np.array(Image.open(images_p))
                except Exception as e:
                    print("Corrupted", images_p)
                    continue
                    # raise e

                target_dataset[i] = int(images_p.parent.stem[1:]) - 1
                bar.update(n=1)