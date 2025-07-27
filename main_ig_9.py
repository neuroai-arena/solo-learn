from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import requests
from copy import deepcopy
import torch
import torch.nn as nn
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from solo.data.classification_dataloader import prepare_transforms
from solo.methods.base import BaseMethod
from solo.utils.misc import make_contiguous
from omegaconf import DictConfig, OmegaConf
import json
from solo.utils.multi_linear import AllClassifiers
from solo.methods.linear import LinearModel



def collate(batch):
    images = torch.stack([i[0] for i in batch])
    labels = [i[1] for i in batch]
    return images, labels


class ImageNet9(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

        url = "https://raw.githubusercontent.com/MadryLab/backgrounds_challenge/refs/heads/master/in_to_in9.json"
        cls_map = requests.get(url).json()

        idx_2_cls = {int(k): v for k, v in cls_map.items()}
        cls_2_idx = defaultdict(list)
        for key, value in idx_2_cls.items():
            cls_2_idx[value].append(key)

        class_mapping = list(map(lambda x: cls_2_idx[int(x[:2])], self.classes))
        classes = lambda x: class_mapping[x]
        self.target_transform = classes



class Dummy(nn.Module):
    def __init__(self, clf):
        super().__init__()
        self.linear = clf
    def forward(self, x):
        return self.linear(x)

def resolve_multi_clf(state, clf):
    clf_names = []
    for name in state:
        print(name)
        clf_names.append(name.split('.')[2])
    clf_names = set(clf_names)
    print(clf_names)
    return AllClassifiers({name: Dummy(clf()) for name in clf_names})



@hydra.main(version_base="1.2")
def main(cfg: DictConfig):

    res_dir = Path("img9_results")
    res_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(0)
    device = "cuda"

    ckpt_path = Path(cfg.classifier_ckpt)
    args_path = ckpt_path.parent / "args.json"

    with open(args_path) as f:
        method_args = json.load(f)
    linear_cfg = OmegaConf.create(method_args)

    backbone_model = BaseMethod._BACKBONES[linear_cfg.backbone.name]
    backbone = backbone_model(method=linear_cfg.pretrain_method, **linear_cfg.backbone.kwargs)
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

    # linear_cfg.grid.use_avgpool = deepcopy(linear_cfg.grid.avgpool)
    # linear_cfg.grid.use_n_blocks = deepcopy(linear_cfg.grid.n_blocks)
    # linear_cfg.grid.use_cls_token = [True]

    model = LinearModel(backbone, loss_func=torch.nn.CrossEntropyLoss(), mixup_func=None, cfg=linear_cfg)
    state = torch.load(cfg.classifier_ckpt, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if k.startswith("classifier."):
            # base = "classifiers_dict.classifier-use_avgpool=True-use_cls_token=True-use_n_blocks=1"
            # split = k.split('.')
            # lr_str = split[2].split("lr_")[-1].replace("_", ".")
            # lr = float(lr_str)
            # base += f"-lr={lr:.8f}".replace(".", ":")
            # res = ".".join([base, split[-2], split[-1]])
            name = k.replace("classifier.", "")

            state[name] = state[k]
        del state[k]
    model.classifier.load_state_dict(state, strict=True)

    make_contiguous(model)
    model.to(device)
    model.eval()



    # backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    # initialize backbone
    # backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    # if cfg.backbone.name.startswith("resnet"):
    #     backbone.fc = nn.Identity()
    #
    # ckpt_path = cfg.pretrained_feature_extractor
    # assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    #
    # state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # for k in list(state.keys()):
    #     if "encoder" in k:
    #         state[k.replace("encoder", "backbone")] = state[k]
    #         print(
    #             "You are using an older checkpoint. Use a new one as some issues might arrise."
    #         )
    #     if "backbone" in k:
    #         state[k.replace("backbone.", "")] = state[k]
    #     del state[k]
    # backbone.load_state_dict(state, strict=False)
    # make_contiguous(backbone)
    # backbone.to(device)
    # backbone.eval()
    #
    # clf = nn.Linear(backbone.inplanes if hasattr(backbone, 'inplanes') else backbone.num_features, 1000)
    # state = torch.load(cfg.classifier_ckpt, map_location="cpu")["state_dict"]
    #
    # for k in list(state.keys()):
    #     if k.startswith("classifier."):
    #         state[k.replace("classifier.", "")] = state[k]
    #     del state[k]
    #
    # clf.load_state_dict(state, strict=True)
    # make_contiguous(clf)
    # clf.to(device)
    # clf.eval()

    # _, transform = prepare_transforms("imagenet")

    transform = transforms.Compose(
        [
            transforms.Resize(224),  # resize shorter
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ]
    )

    result = []
    with torch.inference_mode():
        for version in [
            "original",
            "mixed_rand", "mixed_same", "no_fg", "only_bg_b", "only_bg_t", "only_fg", "mixed_next",
        ]:


            dataset = ImageNet9(root=Path(cfg.data.path) / F"bg_challenge/{version}/val",
                                transform=transform)
            dl = DataLoader(dataset, collate_fn=collate, shuffle=False,
                            num_workers=cfg.data.num_workers, batch_size=cfg.data.batch_size)

            predictions = defaultdict(list)
            targets = []
            for batch in tqdm(dl, desc=version):
                image, target = batch
                image = image.to(device)
                targets.extend(target)

                # out = clf(backbone(image))

                out = model(image)['logits'] # (bs, cls)
                if isinstance(out, dict):
                    for classifier, logits in out.items():
                        scores = logits.argmax(-1)  # (bs, )
                        predictions[classifier].append(scores.detach().cpu())
                else:
                    scores = out.argmax(-1)

            sub_result = []
            for clf_name, scores in predictions.items():
                scores = torch.cat(scores, dim=0)

                hit, total = 0, 0
                classwise_hit, classwise_total = defaultdict(int), defaultdict(int)
                for i, pred in enumerate(scores):

                    # print(pred.detach().item(), target[i], pred.detach().item() in target[i]),
                    if pred in targets[i]:
                        hit += 1
                        classwise_hit[tuple(targets[i])] += 1
                    total += 1
                    classwise_total[tuple(targets[i])] += 1

                # average_classwise acc
                classwise_acc = {k: v / classwise_total[k] for k, v in classwise_hit.items()}
                average_classwise_acc = sum(classwise_acc.values()) / len(classwise_acc)
                sub_result.append(hit / total)

            result.append(
                {'version': version, 'accuracy': max(sub_result), 'ckpt_path': ckpt_path, 'classwise_acc': classwise_acc,
                 'average_classwise_acc': average_classwise_acc, 'clf': clf_name})

    result = pd.DataFrame(result)
    result = pd.concat([result, pd.DataFrame([
        {'version': 'Average', 'accuracy': result['accuracy'].mean(), 'ckpt_path': Path(ckpt_path).name}])])
    print(result)
    result.to_csv(res_dir / f"results_IG9_{Path(ckpt_path).stem}.csv", index=False)


if __name__ == '__main__':
    main()
