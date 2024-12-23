from typing import Dict, Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from solo.data.classification_dataloader import prepare_datasets, prepare_transforms
from solo.utils.knn import WeightedKNNClassifier


class KNNCallback(pl.Callback):
    def __init__(self, cfg: DictConfig, ):
        self.cfg = cfg
        self.train_loader, self.test_loader = None, None
        self.knn = WeightedKNNClassifier(k=self.cfg.k[0], T=self.cfg.T, distance_fx=self.cfg.distance_fx)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        T_train, T_val = prepare_transforms(self.cfg.dataset)

        train_dataset, val_dataset = prepare_datasets(
            self.cfg.dataset,
            T_train,
            T_val,
            train_data_path=self.cfg.train_path,
            val_data_path=self.cfg.val_path,
            data_format=self.cfg.format,

        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(train_dataset, shuffle=False)
        )
        self.test_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=DistributedSampler(val_dataset, shuffle=False)
        )

        if isinstance(self.cfg.perform_every_n_batches, float):
            print("Estimated stepping batches", trainer.estimated_stepping_batches)
            self.cfg.perform_every_n_batches = int(
                trainer.estimated_stepping_batches * self.cfg.perform_every_n_batches)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.cfg.perform_on_validation and trainer.current_epoch >= self.cfg.delay_epochs:
            self._run(trainer, pl_module)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.cfg.perform_on_test:
            self._run(trainer, pl_module)

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        if self.cfg.perform_every_n_batches is not None and batch_idx % self.cfg.perform_every_n_batches == 0 and batch_idx != 0:
            self._run(trainer, pl_module)

    def _run(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if not trainer.sanity_checking and not trainer.fast_dev_run:
            torch.cuda.empty_cache()
            pl_module.eval()

            result = self.run(trainer, pl_module)

            torch.cuda.empty_cache()
            pl_module.train()

            for k, value in result.items():
                if hasattr(trainer.logger, 'log_metrics'):
                    trainer.logger.log_metrics({
                        f'knn/{self.cfg.dataset}_{k}_top1': value[0],
                        f'knn/{self.cfg.dataset}_{k}_top5': value[1]
                    }, step=trainer.global_step)
                else:
                    raise ValueError("Please use a logger that supports `log_metrics`")

    @torch.no_grad()
    def extract_features(self, loader: DataLoader, model: pl.LightningModule, mode: str = "train") -> None:
        bar = tqdm(loader, desc=f'{mode} KNN',
                   total=len(loader)) if self.cfg.verbose and model.local_rank == 0 else loader
        for batch in bar:
            X, y = batch
            X = X.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)

            outs = model(X)
            inputs = {f"{mode}_features": outs["feats"].detach(), f"{mode}_targets": y.detach()}
            self.knn(**inputs)

    def run(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict:
        # extract train and test features
        self.extract_features(self.train_loader, pl_module, mode="train")
        self.extract_features(self.test_loader, pl_module, mode="test")

        # barrier to make sure all features are extracted
        trainer.strategy.barrier()

        result = {}
        for k in self.cfg.k:
            self.knn.set_k(k)
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            result[k] = (val_knn_acc1, val_knn_acc5)

        self.knn.set_k(self.cfg.k[0])

        print("Result: ", result)
        return result
