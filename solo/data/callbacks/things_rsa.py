import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, DistributedSampler
from lightning.pytorch.utilities.types import STEP_OUTPUT

from solo.data.custom.things import Things
from torchvision import transforms
from lightning.pytorch.callbacks import Callback
import pandas as pd
import torch.distributed as dist

import torch
import torch.distributed as dist

def gather_to_rank0(tensor: torch.Tensor, dst=0):
    """
    Gather tensors from all ranks to rank `dst`. Only rank `dst` gets the result.
    Other ranks return None.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Allocate list to receive tensors on dst rank
    if rank == dst:
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gathered = None

    # Do the gather
    dist.gather(tensor, gather_list=gathered, dst=dst)

    if rank == dst:
        return torch.stack(gathered, dim=0)
    else:
        return None


@torch.no_grad()
def extract_features(dataloader, pl_module, layers):
    all_embeddings = [[] for _ in range(len(layers)+1)]
    all_concepts = []
    max_layer = max(layers)
    for images, conceptid in dataloader:
        all_concepts.append(conceptid.to(pl_module.device,  non_blocking=True))
        out = pl_module.backbone(images.to(pl_module.device, non_blocking=True))
        store_id = 0
        all_embeddings[store_id].append(out)
        store_id += 1

        if hasattr(pl_module, "projector"):
            for l in range(len(pl_module.projector)):
                if l > max_layer:
                    break
                out = pl_module.projector[l](out)
                if l in layers:
                    all_embeddings[store_id].append(out)
                    store_id += 1
    return [torch.cat(e) for e in all_embeddings], torch.cat(all_concepts)


def rearrange(matrix, classes_src, classes_dest, to_remove_name=[]):
    removed = []
    if not to_remove_name:
        to_remove = (matrix.sum(1) == 0)
        for i in range(len(classes_src)):
            if to_remove[i]:
                classes_dest.remove(classes_src[i])
                removed.append(classes_src[i])
                print("remove", classes_src[i])
    else:
        for r in to_remove_name:
            if r in classes_dest:
                classes_dest.remove(r)

    new_matrix = torch.zeros((len(classes_dest), len(classes_dest)), device=matrix.device)
    for i in range(new_matrix.shape[0]):
        for j in range(new_matrix.shape[1]):
            old_i = classes_src.index(classes_dest[i])
            old_j = classes_src.index(classes_dest[j])
            new_matrix[i,j] = matrix[old_i, old_j]

    return new_matrix, classes_dest, removed


class ThingsCallback(Callback):
    def __init__(self, cfg):
        """
        Args:
            eval_fn (callable): a function that takes (model, dataloader) and returns a float metric
            log_name (str): name under which to log the metric
        """
        super().__init__()
        self.log_name = "things"
        self.cfg=cfg
        self.ordered_classes = pd.read_csv(Path(self.cfg.path) / "ordered_classes.csv", header=None)[0].tolist()

        #with open(Path(self.cfg.path) / "mapping.json", 'r') as f:
            # self.concept_mapping = json.load(f)


    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        t = transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        print(pl_module.device)
        dataset = Things(Path(self.cfg.path), transforms=t)
        self.dataloader = DataLoader(dataset, batch_size=32, pin_memory=True, sampler=DistributedSampler(dataset, shuffle=False), drop_last=True)
        self.unordered_classes = dataset.codes.tolist()
        self.missing_ego4d_classes = ["elephant", "giraffe", "bottle"]
        self.ego4d_cooccurrences = torch.tensor(np.load(Path(self.cfg.path) / "co_matrix4.npy"), device=pl_module.device, dtype=torch.float32)
        self.human_rdm = torch.tensor(np.load(Path(self.cfg.path) / "hum_rdm.npy"), device=pl_module.device, dtype=torch.float32)
        self.cfg.perform_every_n_batches = int(trainer.estimated_stepping_batches * self.cfg.perform_every_n_batches / trainer.max_epochs)

        print("device", pl_module.device)
        if dist.is_initialized():
            print("rank", dist.get_rank(), dist.get_world_size())
        else:
            print("not initialized")
        trainer.strategy.barrier()

    # def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     if trainer.current_epoch != 0:
    #         return
    #     pl_module.eval()
    #     correlations = self.eval_fc(self.dataloader, pl_module)
    #     if trainer.is_global_zero:
    #         pl_module.log_dict(correlations)
    #     pl_module.train()

    # def on_train_epoch_end(self, trainer, pl_module):
    #     if trainer.current_epoch % self.freq_epochs:
    #         return
    #     pl_module.eval()
    #     correlations = self.eval_fc(self.dataloader, pl_module)
    #     if trainer.is_global_zero:
    #         pl_module.log_dict(correlations)
    #     pl_module.train()


    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        if self.cfg.perform_every_n_batches is not None and batch_idx % self.cfg.perform_every_n_batches == 0:
            pl_module.eval()
            correlations = self.eval_fc_rank0(trainer, self.dataloader, pl_module)
            if trainer.is_global_zero:
                #print(correlations)
                pl_module.log_dict(correlations, sync_dist=False)
            pl_module.train()





    @torch.no_grad()
    def eval_fc(self, trainer, dataloader, pl_module):
        features, conceptids = extract_features(dataloader, pl_module, self.cfg.layers)

        correlates = {}
        for i, f in enumerate(features):
            f = f.squeeze()
            print(conceptids.shape, f.shape, pl_module.device)
            trainer.strategy.barrier()
            features_all, conceptids_all = pl_module.all_gather((f, conceptids))
            features_all, conceptids_all = features_all.flatten(0,1), conceptids_all.flatten(0,1)

            # print(features_all.shape, conceptids_all.shape, conceptids.shape)
            model_rdm, count_rdm = self.machine_rdm(f, conceptids, features_all, conceptids_all)

            #RSA with human judgements
            trainer.strategy.barrier()
            model_rdms, count_rdms = pl_module.all_gather((model_rdm, count_rdm))
            model_rdms, count_rdms = model_rdms.sum(0), count_rdms.sum(0)
            model_rdms = model_rdms / count_rdms
            name = "rsa/backbone" if i == 0 else f"rsa/projector{self.cfg.layers[i-1]}"
            correlates[name] = self.correlate_rdms(model_rdms, self.human_rdm)

            #RSA with Ego4D co-occurrences
            reordered_rdm, labels, _ = rearrange(model_rdms, self.unordered_classes,self.ordered_classes,self.missing_ego4d_classes)
            name_ego4d = "ego4d-co/backbone" if i == 0 else f"ego4d-co/projector{self.cfg.layers[i-1]}"
            correlates[name_ego4d] = self.correlate_rdms(self.ego4d_cooccurrences, reordered_rdm)

        return correlates

    @torch.no_grad()
    def eval_fc_rank0(self, trainer, dataloader, pl_module):
        features, conceptids = extract_features(dataloader, pl_module, self.cfg.layers)

        correlates = {}
        for i, f in enumerate(features):
            f = f.squeeze()

            trainer.strategy.barrier()
            features_all = gather_to_rank0(f)
            conceptids_all = gather_to_rank0(conceptids)
            if dist.get_rank() != 0:
                continue

            features_all, conceptids_all = features_all.flatten(0,1), conceptids_all.flatten(0,1)

            # print(features_all.shape, conceptids_all.shape, conceptids.shape)
            model_rdms, count_rdms = self.machine_rdm(features_all, conceptids_all, features_all, conceptids_all)
            #RSA with human judgements
            model_rdms = model_rdms / count_rdms
            name = "rsa/backbone" if i == 0 else f"rsa/projector{self.cfg.layers[i-1]}"
            correlates[name] = self.correlate_rdms(model_rdms, self.human_rdm)

            #RSA with Ego4D co-occurrences
            reordered_rdm, labels, _ = rearrange(model_rdms, self.unordered_classes,self.ordered_classes,self.missing_ego4d_classes)
            name_ego4d = "ego4d-co/backbone" if i == 0 else f"ego4d-co/projector{self.cfg.layers[i-1]}"
            correlates[name_ego4d] = self.correlate_rdms(self.ego4d_cooccurrences, reordered_rdm)

        return correlates


    @torch.no_grad()
    def machine_rdm(self, features_local, concept_ids_local, features, concept_ids):
        n = len(torch.unique(concept_ids))
        rdm = torch.zeros((n, n), device=features.device, dtype=torch.float32)
        counts = torch.zeros((n, n), device=features.device, dtype=torch.float32)

        batch_size = 64
        cpt = 0
        for f, cid in zip(features_local.split(batch_size), concept_ids_local.split(batch_size)):
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

    @torch.no_grad()
    def correlate_rdms(self,
            rdm_1: torch.Tensor,
            rdm_2: torch.Tensor,
            correlation: str = "pearson",
    ) -> float:
        assert rdm_1.shape[0] == rdm_1.shape[1], f"RDMs must be square {rdm_1.shape[1]} {rdm_1.shape[0]}"

        n = rdm_1.shape[0]
        # Get upper triangle indices, excluding the diagonal
        triu_indices = torch.triu_indices(n, n, offset=1)

        # Extract upper triangular values
        rdm1_vals = rdm_1[triu_indices[0], triu_indices[1]]
        rdm2_vals = rdm_2[triu_indices[0], triu_indices[1]]

        # Remove NaNs
        valid_mask = ~(torch.isnan(rdm1_vals) | torch.isnan(rdm2_vals))
        rdm1_vals = rdm1_vals[valid_mask]
        rdm2_vals = rdm2_vals[valid_mask]
        assert rdm1_vals.shape  == rdm2_vals.shape, f"RDMs must have the same shape {rdm1_vals.shape} {rdm2_vals.shape}"

        if correlation.lower() == "pearson":
            rdm1_vals = rdm1_vals - rdm1_vals.mean()
            rdm2_vals = rdm2_vals - rdm2_vals.mean()
            rho = torch.sum(rdm1_vals * rdm2_vals) / (
                    torch.sqrt(torch.sum(rdm1_vals ** 2)) * torch.sqrt(torch.sum(rdm2_vals ** 2)) + 1e-8
            )
        elif correlation.lower() == "spearman":
            rdm1_vals = rdm1_vals.argsort().argsort().float()
            rdm2_vals = rdm2_vals.argsort().argsort().float()
            rdm1_vals = rdm1_vals - rdm1_vals.mean()
            rdm2_vals = rdm2_vals - rdm2_vals.mean()
            rho = torch.sum(rdm1_vals * rdm2_vals) / (
                    torch.sqrt(torch.sum(rdm1_vals ** 2)) * torch.sqrt(torch.sum(rdm2_vals ** 2)) + 1e-8
            )
        else:
            raise ValueError(f"Unsupported correlation type: {correlation}")

        return rho




