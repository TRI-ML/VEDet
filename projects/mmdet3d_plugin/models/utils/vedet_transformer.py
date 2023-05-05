# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule


@TRANSFORMER.register_module()
class VETransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 det_decoder=None,
                 seg_decoder=None,
                 use_iterative_refinement=False,
                 reduction='ego',
                 init_cfg=None):
        super(VETransformer, self).__init__(init_cfg=init_cfg)

        self.det_decoders = None
        if det_decoder is not None:
            self.det_decoders = build_transformer_layer_sequence(det_decoder)

        self.seg_decoders = None
        if seg_decoder is not None:
            self.seg_decoders = build_transformer_layer_sequence(seg_decoder)

        assert reduction in {'ego', 'mean'}
        self.reduction = reduction
        self.use_iterative_refinement = use_iterative_refinement

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self,
                x,
                mask,
                x_pos,
                init_det_points,
                init_det_points_mtv,
                init_seg_points,
                pos_encoder,
                pos_seg_encoder,
                reg_branch=None,
                num_decode_views=2,
                **kwargs):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, hw, c = x.shape
        x = x.reshape(bs, n * hw, c)
        x_pos = x_pos.reshape(bs, n * hw, -1)

        mask = mask.view(bs, -1)  # [bs, n, h*w] -> [bs, n*h*w]

        # segmentation decoders
        seg_outputs = []
        if self.seg_decoders is not None:
            query_points = init_seg_points.flatten(1, -2)
            # query_embeds = pos_encoder(query_points)
            query_embeds = pos_seg_encoder(query_points)
            query = torch.zeros_like(query_embeds)

            seg_outputs = self.seg_decoders(
                query=query.transpose(0, 1),
                key=x.transpose(0, 1),
                value=x.transpose(0, 1),
                key_pos=None,
                query_pos=query_embeds.transpose(0, 1),
                key_padding_mask=None,
                reg_branch=None)
            seg_outputs = seg_outputs.transpose(1, 2)
            seg_outputs = torch.nan_to_num(seg_outputs)

        # detection decoders
        det_outputs, regs = [], []
        if self.det_decoders is not None:
            memory = x.transpose(0, 1)
            attn_masks = [None, None]
            num_query = init_det_points.shape[-2]
            total_num = num_query * (1 + num_decode_views)
            self_attn_mask = memory.new_ones((total_num, total_num))
            for i in range(1 + num_decode_views):
                self_attn_mask[i * num_query:(i + 1) * num_query, i * num_query:(i + 1) * num_query] = 0
            attn_masks[0] = self_attn_mask
            det_outputs, regs = self.decode_bboxes(init_det_points, init_det_points_mtv, memory, x_pos.transpose(0, 1),
                                                   mask, attn_masks, pos_encoder, reg_branch, num_decode_views)

        return det_outputs, regs, seg_outputs

    def decode_bboxes(self, init_det_points, init_det_points_mtv, memory, key_pos, mask, attn_masks, pos_encoder,
                      reg_branch, num_decode_views):
        if init_det_points_mtv is not None:
            # append queries from virtual views
            query_points = torch.cat([init_det_points, init_det_points_mtv], dim=1).flatten(1, 2)
        else:
            query_points = init_det_points.flatten(1, 2)

        query_embeds = pos_encoder(query_points)
        query = torch.zeros_like(query_embeds)

        regs = []
        # output from layers' won't update next's layer's ref points
        det_outputs = self.det_decoders(
            query=query.transpose(0, 1),
            key=memory,
            value=memory,
            key_pos=key_pos,
            query_pos=query_embeds.transpose(0, 1),
            key_padding_mask=mask,
            attn_masks=attn_masks,
            reg_branch=reg_branch)
        det_outputs = det_outputs.transpose(1, 2)
        det_outputs = torch.nan_to_num(det_outputs)

        for reg_brch, output in zip(reg_branch, det_outputs):

            reg = reg_brch(output)
            reference = inverse_sigmoid(query_points[..., :3].clone())
            reg[..., 0:2] += reference[..., 0:2]
            reg[..., 0:2] = reg[..., 0:2].sigmoid()
            reg[..., 4:5] += reference[..., 2:3]
            reg[..., 4:5] = reg[..., 4:5].sigmoid()

            regs.append(reg)

        L, B, _, C = det_outputs.shape
        # (L, B, V + 1, M, C)
        det_outputs = det_outputs.reshape(L, B, num_decode_views + 1, -1, C)
        # (L, B, V + 1, M, 10)
        regs = torch.stack(regs).reshape(L, B, num_decode_views + 1, init_det_points.shape[-2], -1)

        # ego decode + mtv center decode, (L, B, M, V * 10)
        regs = regs.permute(0, 1, 3, 2, 4).flatten(-2)

        return det_outputs, regs
