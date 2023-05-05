# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule


@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding3D(BaseModule):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding3D, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            n_embed = (n_embed + self.offset) / \
                      (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, :, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_n = n_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, N, H, W = mask.size()
        pos_n = torch.stack((pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
        pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),
                        dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class FourierEncoding(BaseModule):

    def __init__(self, num_bands, max_resolution, sine_only=False, concat_pos=True, proj_dim=-1, init_cfg=None):
        super(FourierEncoding, self).__init__(init_cfg)
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.sine_only = sine_only
        self.concat_pos = concat_pos
        self.proj_dim = proj_dim

        if proj_dim > 0:
            self.proj_layer = nn.Linear(self.fourier_embed_dim, proj_dim)
        else:
            self.proj_layer = nn.Identity()

    @property
    def fourier_embed_dim(self):
        num_dims = len(self.max_resolution)
        embed_dim = self.num_bands * num_dims

        if not self.sine_only:
            embed_dim *= 2
        if self.concat_pos:
            embed_dim += len(self.max_resolution)

        return embed_dim

    @property
    def embed_dim(self):
        if self.proj_dim > 0:
            return self.proj_dim
        return self.fourier_embed_dim

    def forward(self, pos):
        min_freq = 1.0
        # Nyquist frequency at the target resolution:
        freq_bands = [
            torch.linspace(start=min_freq, end=res / 2, steps=self.num_bands, device=pos.device)
            for res in self.max_resolution
        ]
        freq_bands = torch.stack(freq_bands, dim=0).repeat(*pos.shape[:-1], 1, 1)

        # Get frequency bands for each spatial dimension.
        # Output is size [n, d * num_bands]
        per_pos_features = pos[..., None] * freq_bands
        per_pos_features = per_pos_features.flatten(-2)

        if self.sine_only:
            # Output is size [n, d * num_bands]
            per_pos_features = torch.sin(np.pi * (per_pos_features))
        else:
            # Output is size [n, 2 * d * num_bands]
            per_pos_features = torch.cat([torch.sin(np.pi * per_pos_features),
                                          torch.cos(np.pi * per_pos_features)],
                                         dim=-1)
        # Concatenate the raw input positions.
        if self.concat_pos:
            # Adds d bands to the encoding.
            per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

        per_pos_features = self.proj_layer(per_pos_features)

        return per_pos_features


@POSITIONAL_ENCODING.register_module()
class FourierMLPEncoding(BaseModule):

    def __init__(self,
                 input_channels=10,
                 hidden_dims=[1024],
                 embed_dim=256,
                 fourier_type='exponential',
                 fourier_channels=-1,
                 temperature=10000,
                 max_frequency=64,
                 init_cfg=None):
        super(FourierMLPEncoding, self).__init__(init_cfg)
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.fourier_type = fourier_type
        self.fourier_channels = fourier_channels
        self.temperature = temperature
        self.max_frequency = max_frequency

        start_channels = fourier_channels if fourier_channels > 0 else input_channels

        mlp = []
        for l, (in_channel, out_channel) in enumerate(zip([start_channels] + hidden_dims, hidden_dims + [embed_dim])):
            mlp.append(nn.Linear(in_channel, out_channel))
            if l < len(hidden_dims):
                mlp.append(nn.GELU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, pos):
        if self.fourier_channels > 0:
            pos = pos2posemb3d(
                pos,
                num_pos_feats=self.fourier_channels // self.input_channels,
                fourier_type=self.fourier_type,
                temperature=self.temperature,
                max_freq=self.max_frequency)
        return self.mlp(pos)


def pos2posemb3d(pos, num_pos_feats=128, fourier_type='exponential', temperature=10000, max_freq=8):
    if fourier_type == 'exponential':
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        pos_embed = []
        for i in range(pos.shape[-1]):
            pos_x = pos[..., i, None] / dim_t
            pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_embed.append(pos_x)

        pos_embed = torch.cat(pos_embed, dim=-1)

    elif fourier_type == 'linear':
        min_freq = 1.0
        # Nyquist frequency at the target resolution:
        if isinstance(max_freq, int):
            max_freq = [max_freq for _ in range(pos.shape[-1])]
        else:
            assert len(max_freq) == pos.shape[-1]
        freq_bands = [
            torch.linspace(start=min_freq, end=freq, steps=num_pos_feats // 2, device=pos.device) for freq in max_freq
        ]
        freq_bands = torch.stack(freq_bands, dim=0).repeat(*pos.shape[:-1], 1, 1)

        # Get frequency bands for each spatial dimension.
        # Output is size [n, d * num_bands]
        pos_embed = pos[..., None] * freq_bands
        pos_embed = pos_embed.flatten(-2)

        # Output is size [n, 2 * d * num_bands]
        pos_embed = torch.cat([torch.sin(np.pi * pos_embed), torch.cos(np.pi * pos_embed)], dim=-1)
    return pos_embed
