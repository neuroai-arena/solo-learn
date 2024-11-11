from pathlib import Path

import h5py
import numpy as np
from torchvision.datasets import SUN397
from tqdm import tqdm
from sklearn.utils import shuffle

if __name__ == '__main__':
    dataset_root = Path("/scratch/autolearn/aubret/SUN397/")
    output_dir = dataset_root
    output_dir.mkdir(exist_ok=True, parents=True)



    data_dir = Path(dataset_root) / "SUN397"
    with open(data_dir / "ClassName.txt") as f:
        classes = [c[3:].strip() for c in f]

    class_to_idx = dict(zip(classes, range(len(classes))))
    imgs = list(data_dir.rglob("sun_*.jpg"))
    labels = [
        class_to_idx["/".join(path.relative_to(data_dir).parts[1:-1])] for path in imgs
    ]

    imgs, labels = shuffle(imgs, labels, random_state=0)

    train_size = int(0.8*len(imgs))
    val_size = len(imgs) - train_size


    train_h5 = h5py.File(output_dir / "train.h5", "w")
    val_h5 = h5py.File(output_dir / "val.h5", "w")
    train_images = train_h5.create_dataset(f"images",shape=(train_size,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    train_targets = train_h5.create_dataset(f"targets",shape=(train_size,),dtype=np.int32)

    val_images = val_h5.create_dataset(f"images",shape=(val_size,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    val_targets = val_h5.create_dataset(f"targets",shape=(val_size,),dtype=np.int32)

    bar = tqdm(range(len(imgs)), total=len(imgs), desc=f"process")
    for i in bar:
        file, target = imgs[i], labels[i]
        if i < train_size:
            train_images[i] = np.fromfile(file, dtype=np.uint8)
            train_targets[i] = target
        else:
            val_images[i - train_size] = np.fromfile(file, dtype=np.uint8)
            val_targets[i - train_size] = target
    train_h5.close()
    val_h5.close()




