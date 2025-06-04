import numpy as np
import torch
from torch import nn
from x_transformers import Decoder

from solo.utils.misc import omegaconf_select


class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()

        self.predictor = Decoder(dim=embed_dim, depth=depth, heads=num_heads)

    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim=1)
        x = self.predictor(x)
        # return last len(target_masks) tokens
        l = x.shape[1]
        return x[:, l - target_masks.shape[1]:, :]

class IJEPA(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        cfg.method_kwargs.num_heads = omegaconf_select(cfg, "method_kwargs.num_heads", 6)
        cfg.method_kwargs.depth = omegaconf_select(cfg, "method_kwargs.depth", 8)

        self.predictor = Predictor(self.features_dim, cfg.method_kwargs.num_heads, cfg.method_kwargs.depth)


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(IJEPA, IJEPA).add_and_assert_specific_cfg(cfg)
        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @torch.no_grad()
    def get_target_block(self, target_encoder, x, patch_dim, aspect_ratio, scale, M):
        # get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x)
        x = self.norm(x)
        # get the patch dimensions
        patch_h, patch_w = patch_dim
        # get the number of patches
        num_patches = patch_h * patch_w
        # get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        # get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        # get the patches in the target block
        target_block = torch.zeros((M, x.shape[0], block_h * block_w, x.shape[2]))
        target_patches = []
        all_patches = []
        for z in range(M):
            # get the starting patch
            start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w

            patches = []
            # get the patches in the target block
            for i in range(block_h):
                for j in range(block_w):
                    patches.append(start_patch + i * patch_w + j)
                    if start_patch + i * patch_w + j not in all_patches:
                        all_patches.append(start_patch + i * patch_w + j)

            # get the target block
            target_patches.append(patches)
            target_block[z] = x[:, patches, :]
        return target_block.cuda(), target_patches, all_patches


    def get_context_block(self, x, patch_dim, aspect_ratio, scale, target_patches):
        patch_h, patch_w = patch_dim
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the starting patch
        start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        #get the patches in the context_block
        patches = []
        for i in range(block_h):
            for j in range(block_w):
                if start_patch + i * patch_w + j not in target_patches: #remove the target patches
                    patches.append(start_patch + i * patch_w + j)
        return x[:, patches, :]

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        # if mode is test, we get return full embedding:
        if self.mode == 'test':
            return self.backbone(x)

        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])

        # #get target embeddings
        target_blocks, target_patches, all_patches = self.get_target_block(self.teacher_encoder, x, self.patch_dim,
                                                                           target_aspect_ratio, target_scale, self.M)
        m, b, n, e = target_blocks.shape
        # get context embedding

        context_block = self.get_context_block(x, self.patch_dim, context_aspect_ratio, context_scale, all_patches)
        context_encoding = self.student_encoder(context_block)
        context_encoding = self.norm(context_encoding)

        prediction_blocks = torch.zeros((m, b, n, e)).cuda()
        # get the prediction blocks, predict each target block separately
        for i in range(m):
            target_masks = self.mask_token.repeat(b, n, 1)
            target_pos_embedding = self.pos_embedding[:, target_patches[i], :]
            target_masks = target_masks + target_pos_embedding
            prediction_blocks[i] = self.predictor(context_encoding, target_masks)

        return prediction_blocks, target_blocks
        return out


    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
