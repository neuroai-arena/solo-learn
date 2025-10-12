import os

import PIL
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from torchvision.transforms import v2 as trv2, InterpolationMode
from lightning.pytorch.callbacks import Callback

class TripletDataset(Dataset):

    def __init__(self, root_dir,  transform=None):

        self.root_dir = root_dir
        self.transform = transform
        # self.whitebg = whitebg
        self.triplets = list(os.listdir(self.root_dir))


    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet_root = os.path.join(self.root_dir, self.triplets[idx])
        images = list(sorted(os.listdir(triplet_root)))
        images = [PIL.Image.open(os.path.join(triplet_root,im)) for im in images]

        if self.transform:
            return self.transform(images[0]), self.transform(images[1]), self.transform(images[2])
        return images[0], images[1], images[2]


def feature(f):
    k = list(f.keys())[0]
    return torch.flatten(f[k], 1)

def eval_fc(dataloader, backbone, device):
    shape_decision = 0
    total_decision = 0

    for im1, im2, im3 in dataloader:
        f1, f2, f3 = backbone(im3.to(device)), backbone(im2.to(device)), backbone(im1.to(device))
        if isinstance(f1, dict):
            f1, f2, f3 = feature(f1), feature(f2), feature(f3)
        # lf1, lf2, lf3 = classifier(f1), classifier(f2), classifier(f3)

        f1_f3 = torch.nn.functional.cosine_similarity(f1, f3, dim=1)
        f2_f3 = torch.nn.functional.cosine_similarity(f2, f3, dim=1)
        shape_decision += (f1_f3 > f2_f3).float().sum()
        total_decision += im1.shape[0]

        # lf1_f3 = torch.nn.functional.cosine_similarity(lf1, lf3, dim=1)
        # lf2_f3 = torch.nn.functional.cosine_similarity(lf2, lf3, dim=1)
        # shape_decision_lin += (lf1_f3 > lf2_f3).float().sum()
        # total_decision_lin += im1.shape[0]

    return shape_decision / total_decision

class ShapeBiasCallback(Callback):
    def __init__(self, cfg):
        """
        Args:
            eval_fn (callable): a function that takes (model, dataloader) and returns a float metric
            log_name (str): name under which to log the metric
        """
        super().__init__()
        self.log_name = "shapebias"
        self.freq_epochs = cfg.freq_epochs
        self.cfg=cfg


    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        mean, std, image_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
        preprocess = trv2.Compose([trv2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC), trv2.ToImage(),
                             trv2.ToDtype(torch.float32, scale=True), trv2.Normalize(mean=mean, std=std)])

        dataset_common = TripletDataset(os.path.join(self.cfg.path, "shape_simpletext"), transform=preprocess)
        self.dataloader_common = DataLoader(dataset_common, batch_size=32, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset_common, shuffle=False))

        dataset_common2 = TripletDataset(os.path.join(self.cfg.path, "shape_simpletext2"), transform=preprocess)
        self.dataloader_common2 = DataLoader(dataset_common2, batch_size=32, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset_common2, shuffle=False))


        dataset_novel = TripletDataset(os.path.join(self.cfg.path, "simpleshape_simpletext"), transform=preprocess)
        self.dataloader_novel = DataLoader(dataset_novel, batch_size=32, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset_novel, shuffle=False))

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch != 0:
            return
        common_acc = eval_fc(self.dataloader_common, pl_module.backbone, pl_module.device)
        novel_acc = eval_fc(self.dataloader_novel, pl_module.backbone, pl_module.device)
        common_acc_cat = eval_fc(self.dataloader_common2, pl_module.backbone, pl_module.device)
        pl_module.log_dict({"shape_acc": common_acc,
                            "novel_shape_acc": novel_acc,
                            "shape_category_acc": common_acc_cat
                            }, sync_dist=True, on_epoch=True)


    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.freq_epochs:
            return
        common_acc = eval_fc(self.dataloader_common, pl_module.backbone, pl_module.device)
        novel_acc = eval_fc(self.dataloader_novel, pl_module.backbone, pl_module.device)
        common_acc_cat = eval_fc(self.dataloader_common2, pl_module.backbone, pl_module.device)
        pl_module.log_dict({"shape_acc": common_acc,
                            "novel_shape_acc": novel_acc,
                            "shape_category_acc": common_acc_cat
                            }, sync_dist=True, on_epoch=True)
                            # "novel_acc": novel_acc, "novel_acc_lin": novel_acc_lin})









