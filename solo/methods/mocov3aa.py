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

from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

from solo.losses.mocov3 import mocov3_loss_func
from solo.methods import MoCoV3
from solo.methods.base import BaseMomentumMethod
from solo.utils.misc import omegaconf_select
from solo.utils.momentum import initialize_momentum_params


class AAMoCoV3(MoCoV3):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        cfg.method_kwargs.aa_layers= omegaconf_select(cfg, "method_kwargs.aa_layers", 2)
        cfg.method_kwargs.aa_hidden_dim = omegaconf_select(cfg, "method_kwargs.aa_hidden_dim", 4096)
        cfg.method_kwargs.aa_weight = omegaconf_select(cfg, "method_kwargs.aa_weight", 1)
        cfg.method_kwargs.tt_weight = omegaconf_select(cfg, "method_kwargs.tt_weight", 1)
        cfg.method_kwargs.aa_temperature = omegaconf_select(cfg, "method_kwargs.aa_temperature", 0.2)
        cfg.method_kwargs.layer_names = omegaconf_select(cfg, "method_kwargs.layer_names", ["avgpool"])

        self.cfg.method_kwargs.dorsal = omegaconf_select(self.cfg, "method_kwargs.dorsal", {})
        self.cfg.method_kwargs.dorsal.in_planes = omegaconf_select(self.cfg, "method_kwargs.dorsal.in_planes", 2048)
        self.cfg.method_kwargs.dorsal.strides = omegaconf_select(self.cfg, "method_kwargs.dorsal.strides", [])
        self.cfg.method_kwargs.dorsal.layers = omegaconf_select(self.cfg, "method_kwargs.dorsal.layers", 0)
        self.cfg.method_kwargs.dorsal.last_kernel = omegaconf_select(self.cfg, "method_kwargs.dorsal.last_kernel", 7)

        self.cfg.method_kwargs.dorsal.layers = max(self.cfg.method_kwargs.dorsal.layers,len(self.cfg.method_kwargs.dorsal.strides))
        self.cfg.method_kwargs.dorsal.strides = self.cfg.method_kwargs.dorsal.strides + [1] * (self.cfg.method_kwargs.dorsal.layers - len(self.cfg.method_kwargs.dorsal.strides))

        self.backbone = create_feature_extractor(self.backbone, return_nodes=list(cfg.method_kwargs.layer_names))
        self.momentum_backbone = create_feature_extractor(self.momentum_backbone, return_nodes=list(cfg.method_kwargs.layer_names))

        self.dorsal = self.create_dorsal_stream()
        self.momentum_dorsal = self.create_dorsal_stream()

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
        initialize_momentum_params(self.dorsal, self.momentum_dorsal)


    def _make_layer(
        self,
        inplanes,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        strides: int,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None

        layers = []

        for i in range(0, blocks):
            if i > 0:
                inplanes = planes * block.expansion

            if strides[i] != 1 or inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(inplanes, planes * block.expansion, strides[i]),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = None
            layers.append(
                block(
                    inplanes,
                    planes,
                    strides[i],
                    downsample,
                    base_width=64,
                    norm_layer=norm_layer,
                )
            )
        # layers.append(self.avgpool = nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(inplanes, inplanes, self.cfg.method_kwargs.dorsal.last_kernel))
        layers.append(nn.Flatten())
        layers.append(nn.BatchNorm1d(inplanes))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def create_dorsal_stream(self):

        cfg_ds = self.cfg.method_kwargs.dorsal

        if len(self.cfg.method_kwargs.layer_names) == 1:
            return nn.Flatten()

        dorsal = self._make_layer(cfg_ds.in_planes, Bottleneck, 512, cfg_ds.layers, strides=cfg_ds.strides)
        for m in dorsal.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        # test_image = torch.rand((2, 3, 224, 224), device="cpu")
        # out = self.backbone(test_image)[self.cfg.method_kwargs.layer_names[-1]]
        # print(out.shape)
        # test_out = dorsal(out)
        # print(test_out.shape)
        return dorsal

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
            {"name": "dorsal", "params": self.dorsal.parameters()},
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
            (self.vis_action_projector, self.momentum_vis_action_projector),
            (self.dorsal, self.momentum_dorsal)
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
        v1 = self.dorsal(out[self.cfg.method_kwargs.layer_names[-1]][0])
        v2 = self.dorsal(out[self.cfg.method_kwargs.layer_names[-1]][1])




        vis_action_proj = self.vis_action_projector(torch.cat((v1, v2), dim=1))
        vis_action_pred = self.vis_action_predictor(vis_action_proj)

        mom_v1 = self.momentum_dorsal(out[self.cfg.method_kwargs.layer_names[-1]][0])
        mom_v2 = self.momentum_dorsal(out[self.cfg.method_kwargs.layer_names[-1]][1])


        step , X, targets = batch
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
        for i in range(action.shape[1]):
            metrics[f"a{i}"] = action[0,i]

        # img = X[0]
        # img2 = X[1]
        # mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        # unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        #
        # for i in range(20):
        #     imgi = unnormalize(img[i:i+1].cpu())
        #     imgi2 = unnormalize(img2[i:i+1].cpu())
        #     torchvision.utils.save_image(imgi, f"/home/aubret/test_images/same/{step[i].item()}_1.png")
        #     torchvision.utils.save_image(imgi2, f"/home/aubret/test_images/same/{step[i].item()}_2.png")

        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)


        return self.cfg.method_kwargs.tt_weight * contrastive_loss + class_loss + self.cfg.method_kwargs.aa_weight * aa_contrastive_loss

