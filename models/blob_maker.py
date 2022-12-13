from __future__ import annotations
import copy

from utils.utils import instantiate_from_config

__all__ = ["BlobGAN"]

import random
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Tuple, Dict

import einops
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
# from cleanfid import fid
from einops import rearrange, repeat
from matplotlib import cm
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torchvision.utils import make_grid
from tqdm import trange

from .blob_utils.misc import pyramid_resize, splat_features_from_scores, rotation_matrix

# SPLAT_KEYS = ['spatial_style', 'xs', 'ys', 'covs', 'sizes']
SPLAT_KEYS = ['spatial_style', 'scores_pyramid']
_ = Image
_ = make_grid


# @dataclass
# class Lossλs:
#     D_real: float = 1
#     D_fake: float = 1
#     D_R1: float = 5
#     G: float = 1
#     G_path: float = 2

#     G_feature_mean: float = 10
#     G_feature_variance: float = 10

#     def __getitem__(self, key):
#         return super().__getattribute__(key)

@dataclass(eq=False)
class BlobMaker(torch.nn.Module):
    # Modules
    # generator: FromConfig[nn.Module]
    layout_net_config: any
    # discriminator: FromConfig[nn.Module]
    # Module parameters
    dim: int = 256
    noise_dim: int = 512
    resolution: int = 128
    p_mixing_noise: float = 0.0
    n_ema_sample: int = 8
    freeze_G: bool = False
    
    decoder_size_in: int = None
    decoder_size: int = 256
 
    # Input feature generation
    n_features_min: int = 10
    n_features_max: int = 10
    spatial_style: bool = False
    ab_norm: float = 0.02
    feature_jitter_xy: float = 0.0
    feature_jitter_shift: float = 0.0
    feature_jitter_angle: float = 0.0

    def __post_init__(self):
        super().__init__()

        self.layout_net_config.params.n_features_max = self.n_features_max

        self.layout_net = instantiate_from_config(self.layout_net_config)
        self.layout_net_ema = copy.deepcopy(self.layout_net)

    def forward(self, z=None, layout=None, ema=False, norm_img=False, ret_layout=False, ret_latents=False, noise=None, decoder_size_in=None, decoder_size=None, 
            **kwargs):
        assert (z is not None) or (layout is not None)
        if layout is not None and 'covs_raw' not in kwargs:
            kwargs['covs_raw'] = False
        layout = self.generate_layout(z, layout=layout, ret_layout=ret_layout, ema=ema, **kwargs)
        gen_input = {
            'input': layout['feature_grid'],
            'styles': {k: layout[k] for k in SPLAT_KEYS} if self.spatial_style else z,
            'return_image_only': not ret_latents,
            'return_latents': ret_latents,
            'noise': noise
        }

        if 'feature_img' in layout:
            gen_input['feature_img'] = layout['feature_img']

        return gen_input

    # Training and evaluation
    @torch.no_grad()
    def visualize_features(self, xs, ys, viz_size, features=None, scores=None, feature_img=None,
                           c_border=-1, c_fill=1, sz=5, viz_entropy=False, viz_centers=False, viz_colors=None,
                           feature_center_mask=None, **kwargs) -> Dict[str, Tensor]:
        if feature_img is None:
            rand_colors = viz_colors is None
            viz_colors = (viz_colors if not rand_colors else torch.rand_like(features[..., :3])).to(xs.device)
            if viz_colors.ndim == 2:
                # viz colors should be [Kmax, 3]
                viz_colors = viz_colors[:features.size(1)][None].repeat_interleave(len(features), 0)
            elif viz_colors.ndim == 3:
                # viz colors should be [Nbatch, Kmax, 3]
                viz_colors = viz_colors[:, :features.size(1)]
            else:
                viz_colors = torch.rand_like(features[..., :3])
            img = splat_features_from_scores(scores, viz_colors, viz_size)
            if rand_colors:
                imax = img.amax((2, 3))[:, :, None, None]
                imin = img.amin((2, 3))[:, :, None, None]
                feature_img = img.sub(imin).div((imax - imin).clamp(min=1e-5)).mul(2).sub(1)
            else:
                feature_img = img
        imgs_flat = rearrange(feature_img, 'n c h w -> n c (h w)')
        if viz_centers:
            centers = torch.stack((xs, ys), -1).mul(viz_size).round()
            centers[..., 1].mul_(viz_size)
            centers = centers.sum(-1).long()
            if feature_center_mask is not None:
                fill_center = centers[torch.arange(len(centers)), feature_center_mask.int().argmax(1)]
                centers[~feature_center_mask] = fill_center.repeat_interleave((~feature_center_mask).sum(1), dim=0)
            offsets = (-sz // 2, sz // 2 + 1)
            offsets = (torch.arange(*offsets)[None] + torch.arange(*offsets).mul(viz_size)[:, None])
            border_mask = torch.zeros_like(offsets).to(bool)
            border_mask[[0, -1]] = border_mask[:, [0, -1]] = True
            offsets_border = offsets[border_mask].flatten()
            offsets_center = offsets[~border_mask].flatten()
            nonzero_features = scores[..., :-1].amax((1, 2)) > 0
            # draw center
            pixels = (centers[..., None] + offsets_center[None, None].to(self.device)) \
                .clamp(min=0, max=imgs_flat.size(-1) - 1)
            pixels = pixels.flatten(start_dim=1)
            pixels = repeat(pixels, 'n m -> n c m', c=3)
            empty_img = torch.ones_like(imgs_flat)
            imgs_flat.scatter_(dim=-1, index=pixels, value=c_fill)
            empty_img.scatter_(dim=-1, index=pixels, value=c_fill)
            # draw borders
            pixels = (centers[..., None] + offsets_border[None, None].to(self.device)) \
                .clamp(min=0, max=imgs_flat.size(-1) - 1)
            pixels = pixels.flatten(start_dim=1)
            pixels = repeat(pixels, 'n m -> n c m', c=3)
            imgs_flat.scatter_(dim=-1, index=pixels, value=c_border)
            empty_img.scatter_(dim=-1, index=pixels, value=c_border)
        out = {
            'feature_img': imgs_flat.reshape_as(feature_img)
        }
        if viz_centers:
            out['just_centers'] = empty_img.reshape_as(feature_img)
        if scores is not None and viz_entropy:
            img = (-scores.log2() * scores).sum(-1).nan_to_num(0)
            imax = img.amax((1, 2))[:, None, None]
            imin = img.amin((1, 2))[:, None, None]
            img = img.sub(imin).div((imax - imin).clamp(min=1e-5)).mul(256).int().cpu()
            h = w = img.size(-1)
            img = torch.from_numpy(cm.plasma(img.flatten())).mul(2).sub(1)[:, :-1]
            out['entropy_img'] = rearrange(img, '(n h w) c -> n c h w', h=h, w=w)
        return out

    def splat_features(self, xs: Tensor, ys: Tensor, features: Tensor, covs: Tensor, sizes: Tensor, size: int,
                       score_size: int, viz_size: Optional[int] = None, viz: bool = False,
                       ret_layout: bool = True,
                       covs_raw: bool = True, pyramid: bool = True, no_jitter: bool = False,
                       no_splat: bool = False, viz_score_fn=None,
                       **kwargs) -> Dict:
        """
        Args:
            xs: [N, M] X-coord location in [0,1]
            ys: [N, M] Y-coord location in [0,1]
            features: [N, M+1, dim] feature vectors to splat (and bg feature vector)
            covs: [N, M, 2, 2] xy covariance matrices for each feature
            sizes: [N, M+1] distributions of per feature (and bg) weights
            size: output grid size
            score_size: size at which to render score grid before downsampling to size
            viz_size: visualized grid in RGB dimension
            viz: whether to visualize
            covs_raw: whether covs already processed or not
            ret_layout: whether to return dict with layout info
            viz_score_fn: map from raw score to new raw score for generating blob maps. if you want to artificially enlarge blob borders, e.g., you can send in lambda s: s*1.5
            no_splat: return without computing scores, can be useful for visualizing
            no_jitter: manually disable jittering. useful for consistent results at test if model trained with jitter
            pyramid: generate score pyramid
            **kwargs: unused

        Returns: dict with requested information
        """
        if self.feature_jitter_xy and not no_jitter:
            xs = xs + torch.empty_like(xs).uniform_(-self.feature_jitter_xy, self.feature_jitter_xy)
            ys = ys + torch.empty_like(ys).uniform_(-self.feature_jitter_xy, self.feature_jitter_xy)
        if covs_raw:
            a, b = covs[..., :2].sigmoid().unbind(-1)
            ab_norm = 1
            if self.ab_norm is not None:
                ab_norm = self.ab_norm * (a * b).rsqrt()
            basis_i = covs[..., 2:]
            basis_i = F.normalize(basis_i, p=2, dim=-1)
            if self.feature_jitter_angle and not no_jitter:
                with torch.no_grad():
                    theta = basis_i[..., 0].arccos()
                    theta = theta + torch.empty_like(theta).uniform_(-self.feature_jitter_angle,
                                                                     self.feature_jitter_angle)
                    basis_i_jitter = (rotation_matrix(theta)[..., 0] - basis_i).detach()
                basis_i = basis_i + basis_i_jitter
            basis_j = torch.stack((-basis_i[..., 1], basis_i[..., 0]), -1)
            R = torch.stack((basis_i, basis_j), -1)
            covs = torch.zeros_like(R)
            covs[..., 0, 0] = a * ab_norm
            covs[..., -1, -1] = b * ab_norm
            covs = torch.einsum('...ij,...jk,...lk->...il', R, covs, R)
            covs = covs + torch.eye(2)[None, None].to(covs.device) * 1e-5

        if no_splat:
            return {'xs': xs, 'ys': ys, 'covs': covs, 'sizes': sizes, 'features': features}

        feature_coords = torch.stack((xs, ys), -1).mul(score_size)  # [n, m, 2]
        grid_coords = torch.stack(
            (torch.arange(score_size).repeat(score_size), torch.arange(score_size).repeat_interleave(score_size))).to(
            xs.device)  # [2, size*size]
        delta = (grid_coords[None, None] - feature_coords[..., None]).div(score_size)  # [n, m, 2, size*size]

        sq_mahalanobis = (delta * torch.linalg.solve(covs, delta)).sum(2)
        sq_mahalanobis = einops.rearrange(sq_mahalanobis, 'n m (s1 s2) -> n s1 s2 m', s1=score_size)

        # [n, h, w, m]
        shift = sizes[:, None, None, 1:]
        if self.feature_jitter_shift and not no_jitter:
            shift = shift + torch.empty_like(shift).uniform_(-self.feature_jitter_shift, self.feature_jitter_shift)
        scores = sq_mahalanobis.div(-1).add(shift).sigmoid()

        bg_scores = torch.ones_like(scores[..., :1])
        scores = torch.cat((bg_scores, scores), -1)  # [n, h, w, m+1]

        # alpha composite
        rev = list(range(scores.size(-1) - 1, -1, -1))  # flip, but without copy
        d_scores = (1 - scores[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * scores
        d_scores[..., -1] = scores[..., -1]

        ret = {}

        if pyramid:
            score_img = einops.rearrange(d_scores, 'n h w m -> n m h w')

            ret['scores_pyramid'] = pyramid_resize(score_img, cutoff=self.decoder_size_in)

        feature_grid = splat_features_from_scores(ret['scores_pyramid'][size], features, size, channels_last=False)
        ret.update({'feature_grid': feature_grid, 'feature_img': None, 'entropy_img': None})
        if ret_layout:
            layout = {'xs': xs, 'ys': ys, 'covs': covs, 'raw_scores': scores, 'sizes': sizes,
                        'composed_scores': d_scores, 'features': features}
            ret.update(layout)
        if viz:
            if viz_score_fn is not None:
                viz_posterior = viz_score_fn(scores)
                scores_viz = (1 - viz_posterior[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * viz_posterior
                scores_viz[..., -1] = viz_posterior[..., -1]
            else:
                scores_viz = d_scores
            ret.update(self.visualize_features(xs, ys, viz_size, features, scores_viz, **kwargs))
        return ret

    def generate_layout(self, z: Optional[Tensor] = None, ret_layout: bool = False, ema: bool = False,
                        size: Optional[int] = None, viz: bool = False,
                        num_features: Optional[int] = None,
                        layout: Optional[Dict[str, Tensor]] = None,
                        mlp_idx: Optional[int] = None,
                        score_size: Optional[int] = None,
                        viz_size: Optional[int] = None,
                        truncate: Optional[float] = None,
                        **kwargs) -> Dict[str, Tensor]:
        """
        Args:
            z: [N x D] tensor of noise
            mlp_idx: idx at which to split layout net MLP used for truncating
            num_features: how many features if not drawn randomly
            ema: use EMA version or not
            size: H, W output for feature grid
            viz: return RGB viz of feature grid
            ret_layout: if true, return an RGB image demonstrating feature placement
            score_size: size at which to render score grid before downsampling to size
            viz_size: visualized grid in RGB dimension
            truncate: if not None, use this as factor for computing truncation. requires self.mean_latent to be set. 0 = no truncation. 1 = full truncation.
            layout: output in format returned by ret_layout, can be used to generate instead of fwd pass
        Returns: [N x C x H x W] tensor of input, optionally [N x 3 x H_out x W_out] visualization of feature spread
        """
        if num_features is None:
            num_features = random.randint(self.n_features_min, self.n_features_max)
        if layout is None:
            layout_net = self.layout_net_ema if ema else self.layout_net
            assert z is not None
            if truncate is not None:
                mlp_idx = -1
                z = layout_net.mlp[:mlp_idx](z)
                try:
                    z = (self.mean_latent * truncate) + (z * (1 - truncate))
                except AttributeError:
                    self.get_mean_latent(ema=ema)
                    z = (self.mean_latent * truncate) + (z * (1 - truncate))
            layout = layout_net(z, num_features, mlp_idx)

        ret = self.splat_features(**layout, size=size or self.decoder_size_in, viz_size=viz_size or self.decoder_size,
                                  viz=viz, ret_layout=ret_layout, score_size=score_size or (size or self.decoder_size),
                                  pyramid=True,
                                  **kwargs)

        if self.spatial_style:
            ret['spatial_style'] = layout['spatial_style']
        if 'noise' in layout:
            ret['noise'] = layout['noise']
        if 'h_stdev' in layout:
            ret['h_stdev'] = layout['h_stdev']
        return ret

    def get_mean_latent(self, n_trunc: int = 10000, ema=True):
        Z = torch.randn((n_trunc, 512)).to(self.device)
        layout_net = self.layout_net_ema if ema else self.layout_net
        latents = [layout_net.mlp[:-1](Z[_]) for _ in trange(n_trunc, desc='Computing mean latent')]
        mean_latent = self.mean_latent = torch.stack(latents, 0).mean(0)
        return mean_latent
