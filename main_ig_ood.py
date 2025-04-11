from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from solo.data.classification_dataloader import prepare_transforms
from solo.data.custom.imagenet import ImageNetOODDataset
from solo.methods.base import BaseMethod
from solo.utils.misc import make_contiguous


def collate(batch):
    images = torch.stack([i[0] for i in batch])
    labels = [i[1] for i in batch]
    return images, labels


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(0)
    device = "cuda"

    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        backbone.fc = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")

    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            print(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)
    make_contiguous(backbone)
    backbone.to(device)

    clf = nn.Linear(backbone.inplanes if hasattr(backbone, 'inplanes') else backbone.num_features, 1000)
    state = torch.load(cfg.classifier_ckpt, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if k.startswith("classifier."):
            state[k.replace("classifier.", "")] = state[k]
        del state[k]

    clf.load_state_dict(state, strict=True)
    make_contiguous(clf)
    clf.to(device)

    _, transform = prepare_transforms("imagenet")

    result = []
    with torch.inference_mode():
        for version in ImageNetOODDataset.VERSION:
            hit, total = 0, 0
            dataset = ImageNetOODDataset(path=cfg.data.path,
                                         version=version,
                                         transform=transform)
            dl = DataLoader(dataset, collate_fn=collate, shuffle=False,
                            num_workers=cfg.data.num_workers, batch_size=cfg.data.batch_size)
            for batch in tqdm(dl, desc=version):
                image, target = batch
                image = image.to(device)

                out = clf(backbone(image))  # (bs, cls)
                out = out.argmax(-1)  # (bs, )

                for i, pred in enumerate(out):
                    if pred.detach().item() in target[i]:
                        hit += 1
                    total += 1
            result.append({'version': version, 'accuracy': hit / total, 'ckpt_path': ckpt_path})
            print(hit / total, hit, total)

    result = pd.DataFrame(result)
    result = pd.concat([result, pd.DataFrame([
        {'version': 'Average', 'accuracy': result['accuracy'].mean(), 'ckpt_path': Path(ckpt_path).name}])])
    print(result)
    result.to_csv(f"results_IGOOD_{Path(ckpt_path).stem}.csv", index=False)


if __name__ == '__main__':
    main()
