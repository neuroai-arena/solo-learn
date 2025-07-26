import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import scipy
import torch
import torch.nn as nn
import lightning as L
from omegaconf import OmegaConf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from solo.data.custom.things import Things
from solo.methods import METHODS
from torchvision import transforms



class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        if model.backbone_name.startswith("resnet"):
            self.num_features = self.backbone.inplanes
        else:
            self.num_features = self.backbone.num_features
        self.projector = model.projector

    def forward(self, input):
        out = self.backbone(input)
        out = self.projector(out)
        return out


@torch.no_grad()
def extract_features(model, args, fabric, layer_names):
    t = transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
    )
    dataset = Things(Path(args.things_path), transforms=t)
    dataloader = DataLoader(dataset, batch_size=64, pin_memory=True, shuffle=False, num_workers=1)
    dataloader = fabric.setup_dataloaders(dataloader)
    all_embeddings = [[] for _ in range(len(layer_names))]
    all_concepts = []
    for images, conceptid in tqdm(dataloader):
        out = model(images)
        for i, l in enumerate(layer_names):
            all_embeddings[i].append(out[l])
        all_concepts.append(conceptid)
    return [torch.cat(e) for e in all_embeddings], torch.cat(all_concepts)


#from thingsvision
def correlate_rdms(
    rdm_1,
    rdm_2,
    correlation: str = "pearson",
) -> float:
    """Correlate the upper triangular parts of two distinct RDMs.

    Parameters
    ----------
    rdm_1 : ndarray
        First RDM.
    rdm_2 : ndarray
        Second RDM.
    correlation : str
        Correlation coefficient (e.g., Spearman, Pearson).

    Returns
    -------
    output : float
        Returns the correlation coefficient of the two RDMs.
    """
    triu_inds = np.triu_indices(len(rdm_1), k=1)
    corr_func = getattr(scipy.stats, "".join((correlation, "r")))
    rdm1, rdm2 = rdm_1[triu_inds], rdm_2[triu_inds]
    valididx = np.logical_not(np.logical_or(np.isnan(rdm1), np.isnan(rdm2)))
    rdm1, rdm2 = rdm1[valididx], rdm2[valididx]

    rho = corr_func(rdm1, rdm2)[0]
    return rho


def human_rdm(path):
    data = np.loadtxt(path, dtype=int)  # Or use pd.read_csv with sep=' ' if needed

    # Get all unique item IDs
    items = np.unique(data)
    # item_to_idx = {item: idx for idx, item in enumerate(items)}
    n = len(items)

    # Initialize RDM with zeros
    rdm = np.zeros((n, n))
    counts = np.zeros((n, n))

    # Fill the RDM
    for row in tqdm(data):
        a, b, o = row
        # ia, ib, io = item_to_idx[a], item_to_idx[b], item_to_idx[odd]

        # Similar pair → dissimilarity = 0
        # rdm[a, b] = rdm[b, a] = 0

        # Odd vs. each → dissimilarity = 1
        rdm[a, o] += 1
        rdm[o, a] += 1
        rdm[b, o] +=1
        rdm[o, b] += 1

        counts[o, a]+=1
        counts[o, b]+=1
        counts[a, o]+=1
        counts[b, o]+=1
        counts[b, a]+=1
        counts[a, b]+=1

    with np.errstate(divide='ignore', invalid='ignore'):
        rdm = np.true_divide(rdm, counts)
        rdm[~np.isfinite(rdm)] = np.nan  # Replace inf/-inf/nan with np.nan
    return rdm

@torch.no_grad()
def machine_rdm(features_local, concept_ids_local, features, concept_ids):

    n = len(torch.unique(concept_ids))
    rdm = torch.zeros((n, n), device=features.device)
    counts = torch.zeros((n, n), device=features.device)

    batch_size=64
    print("start")
    cpt = 0
    for f, cid in tqdm(zip(features_local.split(batch_size), concept_ids_local.split(batch_size)), total=len(features_local)//batch_size + 1):
        rand_ids = torch.randint(0, features.shape[0], (1000,), device=features.device)
        rand_feats = features[rand_ids]
        rand_cid = concept_ids[rand_ids]
        cpt += 1
        dis_matrix = torch.nn.functional.cosine_similarity(f.unsqueeze(1), rand_feats.unsqueeze(0), dim=2)
        dis_matrix = 1 - dis_matrix

        a = cid.unsqueeze(1).expand(-1, rand_cid.size(0)).reshape(-1)
        b = rand_cid.unsqueeze(0).expand(cid.size(0), -1).reshape(-1)
        vals = dis_matrix.reshape(-1)

        rdm.index_put_((a, b), vals, accumulate=True)
        rdm.index_put_((b, a), vals, accumulate=True)
        counts.index_put_((a, b), torch.ones_like(vals), accumulate=True)
        counts.index_put_((b, a), torch.ones_like(vals), accumulate=True)

    return rdm, counts

def modify_cfg(option, cfg):
    if option == "none":
        cfg.method_kwargs.layers = 2
        cfg.method_kwargs.layers_pred = 2
    if option =="vit":
        cfg.backbone.name = "vit_base"
        cfg.method_kwargs.layers = 3
        cfg.method_kwargs.layers_pred = 2
        cfg.backbone.kwargs.patch_size = 16
        cfg.backbone.kwargs.global_pool = "avg"



def main():
    parser = argparse.ArgumentParser()

    def str2table(v):
        return v.split(',')

    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--pretrained_checkpoint_file", type=str, default="")
    parser.add_argument("--things_path", default="/home/aubret/Documents/postdoc/datasets/things", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--layer_names", default=["backbone.avgpool", "projector.2"], type=str2table)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--option", default="none", type=str)


    args = parser.parse_args()


    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    cfg = OmegaConf.create(method_args)
    modify_cfg(args.option, cfg)

    model = METHODS[cfg.method](cfg)

    # backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    # backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    # model = ModelWrapper(backbone, cfg.backbone.name)
    if not args.pretrained_checkpoint_file:
        ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if "last" in ckpt][0]
    else:
        ckpt_path = args.pretrained_checkpoint_file
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model = ModelWrapper(model)
    model = create_feature_extractor(model, return_nodes=args.layer_names)

    fabric = L.Fabric(accelerator=args.accelerator, devices=cfg.devices if not args.devices else args.devices, precision=cfg.precision)
    fabric.launch()
    model = fabric.setup(model)
    model.eval()


    features, conceptids = extract_features(model, args, fabric, args.layer_names)
    try:
        hum_rdm = np.load(Path(args.things_path) / "hum_rdm.npy")
    except:
        hum_rdm = human_rdm(Path(args.things_path) / "triplet_dataset" / "trainset.txt")
        np.save(Path(args.things_path) / "hum_rdm.npy", hum_rdm)
    print("Number of layers", len(features))
    for i, f in enumerate(features):
        f = f.squeeze()
        features_all, conceptids_all = fabric.all_gather((f, conceptids))
        print(features_all.shape)
        features_all = features_all.flatten(0,1) if fabric.world_size != 1 else features_all

        model_rdm, count_rdm = machine_rdm(f, conceptids, features_all, conceptids_all)
        model_rdm, count_rdm = fabric.all_gather((model_rdm, count_rdm))
        if fabric.world_size != 1:
            model_rdm = model_rdm.flatten(0,1).sum(dim=0)
            count_rdm = count_rdm.flatten(0,1).sum(dim=0)
        model_rdm = model_rdm / count_rdm
        model_rdm = model_rdm.cpu().numpy()

        print(f"layer name: {args.layer_names[i]}", correlate_rdms(model_rdm, hum_rdm))

    # if fabric.global_rank == 0:
    #     writer = csv.writer(open(str(ckpt_dir / f"robustness.csv"), "w"))
    #     writer.writerow(all_names)
    #     writer.writerow(all_values)



if __name__ == "__main__":
    main()