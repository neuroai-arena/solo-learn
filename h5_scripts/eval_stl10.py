import json
import os
from typing import Optional, Callable, cast, Tuple, Any

import PIL
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset, CIFAR10
from torchvision.datasets.utils import verify_str_arg, check_integrity, download_and_extract_archive
from torchvision.models import resnet50
from torch import nn
from torchvision import transforms
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode


def main():
    # dataset_name = "cifar10"
    # dataset_name = "stl10"
    dataset_name = "imgnet"

    # strmodels = 'ffm'
    strmodels = 'cvm'

    setting = "all"
    black_mask = True

    imgnet_mapping = json.load(open("/home/fias/postdoc/datasets/imgnet/imagenet_class_index.json", "r"))

    imgnetclasses = ["airliner", "wagon", "humming_bird", "siamese_cat", "ox", "golden_retriever", "tailed_frog", "zebra",
     "container_ship", "trailer_truck"]
    imgnetids = {imgnet_mapping[str(i)][1] :i for i in range(1000) if imgnet_mapping[str(i)][1] in imgnetclasses}
    print(imgnet_mapping["0"])
    print(imgnetids)
    imgnetmapping = [imgnetids[ic] for ic in imgnetclasses]
    allids = imgnetids.keys()

    # cifar10classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    central_vision_models = {
        'model_ckpt': 'iiiqnrs0/mocov3nf_lr16-iiiqnrs0-ep=last.ckpt',
        'imgnet': 'iiiqnrs0/linlast_mocov3nf_lr16_fullIN-sc7poe6z-ep=last.ckpt',
        'stl10': 'iiiqnrs0/v2_STL10_mocov3nf_lr16_lr1.0-45xzm2dt-ep=last-stp=last.ckpt',
        'cifar10': 'iiiqnrs0/v2_cifar10_224_mocov3nf_lr16_lr1.0-0ovrj9io-ep=last-stp=last.ckpt'
    }

    # time_window = 15, gaze_size = 540, center_crop = False, resize_gs = False
    full_frame_models = {
        'model_ckpt': 'ba1e2na0/mocov3nf_lr16_gs540-ba1e2na0-ep=last.ckpt',
        'imgnet': 'ba1e2na0/linlast_mocov3nf_lr16_gs540_fullIN-nn14c1wf-ep=last.ckpt',
        'stl10': 'ba1e2na0/v2_STL10_mocov3nf_lr16_gs540_lr1.0-2lm9md9t-ep=last-stp=last.ckpt',
        'cifar10': 'ba1e2na0/v2_cifar10_224_mocov3nf_lr16_gs540_lr1.0-6ja1gb2m-ep=last-stp=last.ckpt'
    }

    models = central_vision_models if strmodels == "cvm" else full_frame_models


    central_vision_models = torch.load(f"/home/fias/postdoc/papiers/CVPR25/gradcam/{models['model_ckpt']}",map_location="cpu")["state_dict"]
    central_vision_linear = torch.load(f"/home/fias/postdoc/papiers/CVPR25/gradcam/{models[dataset_name]}",map_location="cpu")["state_dict"]
    cvm = {}

    for k, v in central_vision_models.items():
        if "momentum" in k:
            continue
        if "backbone" in k:
            k2 = ".".join(k.split(".")[1:])
            cvm[k2] = v

    cvm["fc.weight"] = central_vision_linear["classifier.weight"]
    cvm["fc.bias"] = central_vision_linear["classifier.bias"]

    model = resnet50()
    model.fc = nn.Linear(2048, 10 if dataset_name != "imgnet" else 1000)
    model.load_state_dict(cvm)
    model.eval()
    model = model.to("cuda:0")

    t = transforms.Compose(
                [
                    # transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),# if dataset_name != "stl10" else 96),
                    transforms.Resize(224),# if dataset_name != "stl10" else 96),
                    # transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261))
                ]
    )

    normalize = transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261))

    if dataset_name in ["cifar10","cifar100","imgnet"]:
        d = CIFAR10(f"/home/fias/postdoc/datasets/cifar10", train=False, transform=t)
    else:
        d = STL10(f"/home/fias/postdoc/datasets/stl10", split="test", transform = t, setting=setting, black_mask=black_mask)
    loader = DataLoader(d, batch_size=64, shuffle=False, pin_memory=True)
    all_succ = 0
    all_cpt = 0
    with torch.no_grad():
        # for it, (images, target, id) in enumerate(loader):
        for it, (images, target) in enumerate(loader):
            if it < 10:
                torchvision.utils.save_image(images[0], f"/home/fias/postdoc/gym_results/test_images/stl10/test_mask{it}.png")
            images = normalize(images)
            out = model(images.to("cuda:0"))
            if dataset_name == "imgnet":
                out = out.gather(imgnetmapping, dim=1)


            outputs = torch.argmax(out, dim=1)
            success = (outputs == target.to("cuda:0")).to(torch.float32)

            all_succ += success.sum(dim=0)
            all_cpt += images.shape[0]

            if it % 64*10 == 0:
                print(it, (all_succ/all_cpt).item())


    print(it, (all_succ/all_cpt).item())







class STL10(VisionDataset):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly, dataset is selected.
        folds (int, optional): One of {0-9} or None.
            For training, loads one of the 10 pre-defined folds of 1k samples for the
            standard evaluation procedure. If no value is passed, loads the 5k samples.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "stl10_binary"
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
    class_names_file = "class_names.txt"
    folds_list_file = "fold_indices.txt"
    train_list = [
        ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
        ["train_y.bin", "5a34089d4802c674881badbb80307741"],
        ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
    ]

    test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"], ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        setting = "all",
        black_mask=True
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        if self.split == "train":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)

        elif self.split == "train+unlabeled":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate((self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == "unlabeled":
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

        self.setting = setting
        self.black_mask = black_mask


    def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = "Value for argument folds should be in the range [0, 10), but got {}."
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.setting in ["fg","bg","maskonly","boundedbg","boundedfg"]:
            try:
                mask = PIL.Image.open(f"/home/fias/postdoc/datasets/stl10/val/masks/{index}.png")
                mask = torchvision.transforms.functional.resize(mask, (224,224))
                mask = torchvision.transforms.functional.to_tensor(mask)

                if self.setting in ["boundedbg","boundedfg"]:
                    masky = mask.max(dim=2).values.squeeze()
                    maskx = mask.max(dim=1).values.squeeze()
                    y_ind = masky.nonzero().squeeze()
                    x_ind = maskx.nonzero().squeeze()

                    imgrec = Image.new("RGB", (224, 224))
                    img1 = ImageDraw.Draw(imgrec)
                    img1.rectangle([(x_ind[0].item(), y_ind[0].item()), (x_ind[-1].item(), y_ind[-1].item())], fill="white", outline="white")
                    imgrec = self.transform(imgrec)
                    img = torch.minimum(img, (1 - imgrec) if self .setting == "boundedbg" else imgrec)

                elif self.setting == "maskonly":
                    img = mask.repeat(3,1,1).squeeze()
                elif self.black_mask:
                    if self.setting == "fg":
                        img = mask * img
                    elif self.setting == "bg":
                        img = (1-mask) * img
                else:
                    mask = mask.repeat(3,1,1).squeeze()
                    if self.setting == "bg":
                        img = torch.maximum(mask, img)
                    elif self.setting == "fg":
                        img = torch.maximum((1-mask), img)
            except Exception as e:
                # print(e)
                pass
        if self.setting == "square":
            imgrec = Image.new("RGB", (224, 224))
            img1 = ImageDraw.Draw(imgrec)
            img1.rectangle([(32,32), (224-32, 224-32)], fill="white", outline="white")
            imgrec = self.transform(imgrec)
            img = torch.minimum(img, 1-imgrec)







        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]

    def __loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = os.path.join(self.root, self.base_folder, labels_file)
            with open(path_to_labels, "rb") as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, "rb") as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds: Optional[int]) -> None:
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds) as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=" ")
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]


if __name__ == "__main__":
    main()