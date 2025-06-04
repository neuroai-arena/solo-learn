# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn

from solo.losses import simclr_loss_func
from solo.methods import MoCoV3, SimCLR, BaseMethod
from solo.utils.misc import omegaconf_select


class AASimCLR(SimCLR):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        cfg.method_kwargs.aa_layers= omegaconf_select(cfg, "method_kwargs.aa_layers", 2)
        cfg.method_kwargs.aa_hidden_dim = omegaconf_select(cfg, "method_kwargs.aa_hidden_dim", 4096)
        cfg.method_kwargs.aa_weight = omegaconf_select(cfg, "method_kwargs.aa_weight", 1)
        cfg.method_kwargs.aa_temperature = omegaconf_select(cfg, "method_kwargs.aa_temperature", 0.1)

        self.action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                9,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim
                                                )


        self.vis_action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                self.features_dim * 2,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                )

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "action_projector", "params": self.action_projector.parameters()},
            {"name": "vis_action_projector", "params": self.vis_action_projector.parameters()},
        ]
        return super().learnable_params + extra_learnable_params


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """
        # SimCLR stuff
        indexes = batch[0]

        out = BaseMethod.training_step(self, batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )

        # AA stuff
        v1 = out["feats"][0]
        v2 = out["feats"][1]
        vis_action_proj = self.vis_action_projector(torch.cat((v1, v2), dim=1))

        _, X, targets = batch
        action = X[-1]
        action_proj = self.action_projector(action)

        aa_nce_loss = simclr_loss_func(
            torch.cat((vis_action_proj, action_proj)),
            indexes=indexes,
            temperature=self.cfg.method_kwargs.aa_temperature
        )

        metrics = {
            "train_constrastive_loss": nce_loss,
            "train_aa_constrastive_loss": aa_nce_loss
        }
        for i in range(action.shape[1]):
            metrics[f"a{i}"] = action[0,i]

        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)


        return nce_loss + class_loss + self.cfg.method_kwargs.aa_weight * aa_nce_loss

