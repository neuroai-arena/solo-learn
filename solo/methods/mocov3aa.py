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

from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
from solo.losses.mocov3 import mocov3_loss_func
from solo.methods import MoCoV3
from solo.methods.base import BaseMomentumMethod
from solo.utils.misc import omegaconf_select
from solo.utils.momentum import initialize_momentum_params


class AAMoCoV3(MoCoV3):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        cfg.method_kwargs.action_layers = omegaconf_select(cfg, "method_kwargs.aa_layers", 2)
        cfg.method_kwargs.action_hidden_dim = omegaconf_select(cfg, "method_kwargs.aa_hidden_dim", 4096)
        cfg.method_kwargs.action_weight = omegaconf_select(cfg, "method_kwargs.aa_weight", 1)
        cfg.method_kwargs.aa_temperature= omegaconf_select(cfg, "method_kwargs.aa_temperature", 0.2)

        self.action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                9,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=True
                                                )

        self.momentum_action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                9,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=True
                                                )

        self.action_predictor = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                cfg.method_kwargs.proj_output_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=False
                                                )



        self.vis_action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                self.features_dim * 2,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=True
                                                )

        self.momentum_vis_action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                         self.features_dim * 2,
                                                         cfg.method_kwargs.aa_hidden_dim,
                                                         cfg.method_kwargs.proj_output_dim,
                                                         last_bn=True
                                                         )

        self.vis_action_predictor = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                cfg.method_kwargs.proj_output_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=False
                                                )
        initialize_momentum_params(self.action_projector, self.momentum_action_projector)
        initialize_momentum_params(self.vis_action_projector, self.momentum_vis_action_projector)

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "action_projector", "params": self.action_projector.parameters()},
            {"name": "action_predictor", "params": self.action_predictor.parameters()},
            {"name": "vis_action_projector", "params": self.vis_action_projector.parameters()},
            {"name": "vis_action_predictor", "params": self.vis_action_predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [
            (self.action_projector, self.momentum_action_projector),
            (self.vis_action_projector, self.momentum_vis_action_projector)
                                ]
        return super().momentum_pairs + extra_momentum_pairs



    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = BaseMomentumMethod.training_step(self, batch, batch_idx)

        # MoCoV3 stuff
        class_loss = out["loss"]
        Q = out["q"]
        K = out["momentum_k"]

        contrastive_loss = mocov3_loss_func(
            Q[0], K[1], temperature=self.temperature
        ) + mocov3_loss_func(Q[1], K[0], temperature=self.temperature)



        # AA stuff
        v1 = out["feats"][0]
        v2 = out["feats"][1]
        vis_action_proj = self.vis_action_projector(torch.cat((v1, v2), dim=1))
        vis_action_pred = self.vis_action_predictor(vis_action_proj)

        mom_v1 = out["momentum_feats"][0]
        mom_v2 = out["momentum_feats"][1]


        _, X, targets = batch
        action = X[-1]
        action_proj = self.action_projector(action)
        action_pred = self.action_predictor(action_proj)

        with torch.no_grad():
            mom_vis_action_proj = self.momentum_vis_action_projector(torch.cat((mom_v1, mom_v2), dim=1))
            mom_action_proj = self.momentum_action_projector(action)

        aa_contrastive_loss =  mocov3_loss_func(
            vis_action_pred, mom_action_proj, temperature=self.cfg.method_kwargs.aa_temperature
        ) + mocov3_loss_func(action_pred, mom_vis_action_proj, temperature=self.cfg.method_kwargs.aa_temperature)


        metrics = {
            "train_contrastive_loss": contrastive_loss,
            "train_aa_constrastive_loss": aa_contrastive_loss
        }
        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)


        return contrastive_loss + class_loss + self.cfg.method_kwargs.action_weight * aa_contrastive_loss

