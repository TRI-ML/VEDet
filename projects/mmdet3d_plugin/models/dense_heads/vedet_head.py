# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply, reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
import numpy as np
from pytorch3d import transforms as tfms


@HEADS.register_module()
class VEDetHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 in_channels,
                 num_classes=10,
                 num_query=900,
                 det_transformer=None,
                 det_feat_idx=0,
                 sync_cls_avg_factor=False,
                 grid_offset=0.0,
                 input_ray_encoding=None,
                 input_pts_encoding=None,
                 output_det_encoding=None,
                 output_seg_encoding=None,
                 code_weights=None,
                 num_decode_views=2,
                 reg_channels=None,
                 bbox_coder=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_iou=None,
                 loss_seg=None,
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 valid_range=[0.0, 1.0],
                 init_cfg=None,
                 shared_head=True,
                 cls_hidden_dims=[],
                 reg_hidden_dims=[],
                 with_time=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.num_decode_views = num_decode_views
        self.reg_channels = reg_channels if reg_channels else self.code_size
        if loss_cls is not None:
            self.bg_cls_weight = 0
            self.sync_cls_avg_factor = sync_cls_avg_factor
            class_weight = loss_cls.get('class_weight', None)
            if class_weight is not None and (self.__class__ is VEDetHead):
                assert isinstance(class_weight, float), 'Expected ' \
                    'class_weight to have type float. Found ' \
                    f'{type(class_weight)}.'
                # NOTE following the official DETR rep0, bg_cls_weight means
                # relative classification weight of the no-object class.
                bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
                assert isinstance(bg_cls_weight, float), 'Expected ' \
                    'bg_cls_weight to have type float. Found ' \
                    f'{type(bg_cls_weight)}.'
                class_weight = torch.ones(num_classes + 1) * class_weight
                # set background class as the last indice
                class_weight[num_classes] = bg_cls_weight
                loss_cls.update({'class_weight': class_weight})
                if 'bg_cls_weight' in loss_cls:
                    loss_cls.pop('bg_cls_weight')
                self.bg_cls_weight = bg_cls_weight

            if train_cfg:
                assert 'assigner' in train_cfg, 'assigner should be provided '\
                    'when train_cfg is set.'
                self.assigner = build_assigner(train_cfg['assigner'])
                # DETR sampling=False, so use PseudoSampler
                sampler_cfg = dict(type='PseudoSampler')
                self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.last_timestamp = None
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.valid_range = valid_range
        super(VEDetHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg)

        self.loss_cls = build_loss(loss_cls) if loss_cls else None
        self.loss_bbox = build_loss(loss_bbox) if loss_bbox else None
        self.loss_iou = build_loss(loss_iou) if loss_iou else None
        self.loss_seg = build_loss(loss_seg) if loss_seg else None

        if self.loss_cls is not None and self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.grid_offset = grid_offset
        self.input_ray_encoding = build_positional_encoding(input_ray_encoding) if input_ray_encoding else None
        self.input_pts_encoding = build_positional_encoding(input_pts_encoding) if input_pts_encoding else None
        self.output_det_encoding = build_positional_encoding(output_det_encoding) if output_det_encoding else None
        self.output_seg_encoding = build_positional_encoding(output_seg_encoding) if output_seg_encoding else None

        self.det_feat_idx = det_feat_idx

        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder) if bbox_coder else None
        self.pc_range = self.bbox_coder.pc_range

        self.embed_dims = 256
        self.det_transformer = build_transformer(det_transformer) if det_transformer is not None else None
        self.with_time = with_time

        if self.det_transformer is not None:
            assert self.loss_bbox is not None
            query_points = nn.Parameter(torch.rand((num_query, 3)))
            self.register_parameter('query_points', query_points)

            num_layers = len(self.det_transformer.det_decoders.layers)

            # classification branch
            cls_branch = []
            for lyr, (in_channel, out_channel) in enumerate(
                    zip([self.output_det_encoding.embed_dim] + cls_hidden_dims,
                        cls_hidden_dims + [self.cls_out_channels])):
                cls_branch.append(nn.Linear(in_channel, out_channel))
                if lyr < len(cls_hidden_dims):
                    cls_branch.append(nn.LayerNorm(out_channel))
                    cls_branch.append(nn.ReLU(inplace=True))
            cls_branch = nn.Sequential(*cls_branch)
            if shared_head:
                self.cls_branch = nn.ModuleList([cls_branch for _ in range(num_layers)])
            else:
                self.cls_branch = nn.ModuleList([deepcopy(cls_branch) for _ in range(num_layers)])

            # regression branch
            reg_branch = []
            for lyr, (in_channel, out_channel) in enumerate(
                    zip([self.output_det_encoding.embed_dim] + reg_hidden_dims, reg_hidden_dims + [self.reg_channels])):
                reg_branch.append(nn.Linear(in_channel, out_channel))
                if lyr < len(reg_hidden_dims):
                    reg_branch.append(nn.ReLU())
            reg_branch = nn.Sequential(*reg_branch)
            if shared_head:
                self.reg_branch = nn.ModuleList([reg_branch for _ in range(num_layers)])
            else:
                self.reg_branch = nn.ModuleList([deepcopy(reg_branch) for _ in range(num_layers)])
        else:
            self.query_points = None
            self.cls_branch = None
            self.reg_branch = None

        if self.loss_seg is not None:
            # TODO: add semantic branch
            pass

    def _init_layers(self):
        pass
        """Initialize layers of the transformer head."""

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        if self.det_transformer is not None:
            self.det_transformer.init_weights()
        if self.loss_cls is not None and self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for cls_branch in self.cls_branch:
                nn.init.constant_(cls_branch[-1].bias, bias_init)

    def generate_rays(self, pad_h, pad_w, H, W, img_feat, img_metas):
        B, N = img_feat.shape[:2]
        coords_h = (torch.arange(H, device=img_feat.device).float() + self.grid_offset) * pad_h / H
        coords_w = (torch.arange(W, device=img_feat.device).float() + self.grid_offset) * pad_w / W
        coords = torch.stack(torch.meshgrid([coords_w, coords_h]))
        coords = torch.cat([coords, torch.ones_like(coords)], dim=0).permute(2, 1, 0)
        # (B, N, H, W, 4, 1)
        coords = coords.view(1, 1, H, W, 4, 1).repeat(B, N, 1, 1, 1, 1)

        inv_intrinsics, extrinsics = [], []
        for img_meta in img_metas:
            for i in range(N):
                inv_intrinsics.append(np.linalg.inv(img_meta['intrinsics'][i]))
                extrinsics.append(img_meta['extrinsics'][i])
        inv_intrinsics = coords.new_tensor(np.asarray(inv_intrinsics)).view(B, N, 1, 1, 4, 4)
        extrinsics = coords.new_tensor(np.asarray(extrinsics)).view(B, N, 1, 1, 4, 4)

        inv_intrinsics = inv_intrinsics.repeat(1, 1, H, W, 1, 1)
        extrinsics = extrinsics.repeat(1, 1, H, W, 1, 1)

        coords3d = torch.matmul(inv_intrinsics, coords)
        rays = F.normalize(coords3d[..., :3, :], dim=-2)

        rays = torch.matmul(extrinsics[..., :3, :3], rays).squeeze(-1)

        return rays, extrinsics

    def position_embedding(self, img_feat, img_metas, mask, depth_maps=None, use_cache_depth=False):
        """Produce position embedding for one feature level
        
        scenario #1: ctrs & end points, depth_maps from sup/self-sup prediction
        scenario #2: ctrs & unit rays, depth_maps is None
        """

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, _, H, W = img_feat.shape
        rays, extrinsics = self.generate_rays(pad_h, pad_w, H, W, img_feat, img_metas)
        rg = self.pc_range
        divider = torch.tensor([rg[3] - rg[0], rg[4] - rg[1], rg[5] - rg[2]], device=extrinsics.device)
        subtract = torch.tensor([rg[0], rg[1], rg[2]], device=extrinsics.device)

        ctrs = extrinsics[..., :3, 3]
        ctrs = (ctrs - subtract) / divider

        # pytorch3d uses row-major, so transpose the R first
        quats = tfms.matrix_to_quaternion(extrinsics[..., :3, :3].transpose(-1, -2))
        ctrs = torch.cat([ctrs, quats], dim=-1)

        geometry = torch.cat([ctrs, rays], dim=-1)
        camera_embedding = self.input_ray_encoding(geometry)

        return camera_embedding, mask

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is VEDetHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                          unexpected_keys, error_msgs)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        batch_size, num_cams = mlvl_feats[0].shape[:2]
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks_full = mlvl_feats[0].new_ones((batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks_full[img_id, cam_id, :img_h, :img_w] = 0

        if self.input_ray_encoding is not None:
            masks = F.interpolate(masks_full, size=mlvl_feats[self.det_feat_idx].shape[-2:]).to(torch.bool)
            pos_embeds, masks = self.position_embedding(mlvl_feats[self.det_feat_idx], img_metas, masks)
        feats = mlvl_feats[self.det_feat_idx].permute(0, 1, 3, 4, 2).flatten(2, 3)

        # detection & segmentation
        cls_scores, bbox_preds, seg_scores = None, None, None
        if self.det_transformer is not None:
            init_det_points = self.query_points.repeat(batch_size, 1, 1, 1) if self.query_points is not None else None
            # transform query points to local viewpoints
            init_det_points_mtv = self.get_mtv_points_local(init_det_points, img_metas)
            init_det_points, init_det_points_mtv = self.add_pose_info(init_det_points, init_det_points_mtv, img_metas)

            # TODO: seg points
            init_seg_points = None

            # transformer decode
            num_decode_views = init_det_points_mtv.shape[1] if init_det_points_mtv is not None else 0
            det_outputs, regs, seg_outputs = self.det_transformer(feats, masks, pos_embeds, init_det_points,
                                                                  init_det_points_mtv, init_seg_points,
                                                                  self.output_det_encoding, self.output_seg_encoding,
                                                                  self.reg_branch, num_decode_views)

            # detection from queries
            if len(det_outputs) > 0 and len(regs) > 0:
                cls_scores = torch.stack(
                    [cls_branch(output) for cls_branch, output in zip(self.cls_branch, det_outputs)], dim=0)
                if cls_scores.dim() == 5:
                    cls_scores = cls_scores[:, :, 0]

                bbox_preds = torch.stack(regs, dim=0) if isinstance(regs, list) else regs
                bbox_preds[..., 0::10] = (
                    bbox_preds[..., 0::10] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
                bbox_preds[..., 1::10] = (
                    bbox_preds[..., 1::10] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
                bbox_preds[..., 4::10] = (
                    bbox_preds[..., 4::10] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

                if self.with_time:
                    time_stamps = []
                    for img_meta in img_metas:
                        time_stamps.append(np.asarray(img_meta['timestamp']))
                    time_stamp = bbox_preds.new_tensor(time_stamps)
                    time_stamp = time_stamp.view(batch_size, -1, 6)
                    mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)
                    bbox_preds[..., 8::10] = bbox_preds[..., 8::10] / mean_time_stamp
                    bbox_preds[..., 9::10] = bbox_preds[..., 9::10] / mean_time_stamp

            # segmentation
            if len(seg_outputs) > 0:
                seg_scores = torch.stack([self.seg_branch(output) for output in seg_outputs], dim=0)

        outs = {
            'all_cls_scores': cls_scores,
            'all_bbox_preds': bbox_preds,
            'all_seg_preds': seg_scores,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def get_mtv_points_local(self, init_det_points, img_metas):
        if not self.training:
            return None

        if len(img_metas[0].get('dec_extrinsics', [])) == 0:
            return None

        extrinsics = init_det_points.new_tensor([img_meta['dec_extrinsics'] for img_meta in img_metas])
        B, N = extrinsics.shape[:2]
        M = init_det_points.shape[2]

        # bring back to metric values
        rg = self.pc_range
        divider = torch.tensor([rg[3] - rg[0], rg[4] - rg[1], rg[5] - rg[2]], device=extrinsics.device)
        subtract = torch.tensor([rg[0], rg[1], rg[2]], device=extrinsics.device)
        init_det_points = init_det_points * divider + subtract

        # (B, N, M, 3)
        init_det_points_mtv = init_det_points.repeat(1, N, 1, 1)
        init_det_points_mtv = init_det_points_mtv - extrinsics[:, :, None, :3, 3]
        Rt = extrinsics[:, :, None, :3, :3].transpose(-1, -2).repeat(1, 1, M, 1, 1)
        init_det_points_mtv = torch.matmul(Rt, init_det_points_mtv[..., None]).squeeze(-1)

        # normalize
        init_det_points_mtv = (init_det_points_mtv - subtract) / (divider + 1e-6)

        return init_det_points_mtv

    def add_pose_info(self, init_det_points, init_det_points_mtv, img_metas):
        # add identity pose to ego queries and make V copies of queries with viewing poses as mtv queries
        B = init_det_points.shape[0]
        identity_quat = init_det_points.new_zeros(B, 1, self.num_query, 4)
        identity_quat[..., 0] = 1
        # (B, 1, M, 10)
        init_det_points = torch.cat([init_det_points, identity_quat, torch.zeros_like(init_det_points)], dim=-1)

        if init_det_points_mtv is None:
            return init_det_points, None

        dec_extrinsics = init_det_points.new_tensor([img_meta['dec_extrinsics'] for img_meta in img_metas])
        dec_quat = tfms.matrix_to_quaternion(dec_extrinsics[..., :3, :3].transpose(-1, -2))
        dec_tvec = dec_extrinsics[..., :3, 3]
        # (B, V, M, 7)
        dec_pose = torch.cat([dec_quat, dec_tvec], dim=-1).unsqueeze(2).repeat(1, 1, self.num_query, 1)

        # (B, V, M, 10)
        init_det_points_mtv = torch.cat([init_det_points_mtv, dec_pose], dim=-1)

        return init_det_points, init_det_points_mtv

    def _get_target_single(self, cls_score, bbox_pred, gt_labels, gt_bboxes, gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore,
                                             self.code_weights)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(self, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list, bbox_preds_list, gt_labels_list,
                                      gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    seg_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_seg_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        loss_cls, loss_bbox, loss_seg = None, None, None
        if cls_scores is not None:
            num_imgs = cls_scores.size(0)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list,
                                               gt_bboxes_ignore_list)
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos,
             num_total_neg) = cls_reg_targets
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            # classification loss
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = num_total_pos * 1.0 + \
                num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

            cls_avg_factor = max(cls_avg_factor, 1)
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

            # Compute the average number of gt boxes accross all gpus, for
            # normalization purposes
            num_total_pos = loss_cls.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

            # regression L1 loss
            bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
            normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
            isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
            bbox_weights = bbox_weights * self.code_weights

            loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan],
                normalized_bbox_targets[isnotnan],
                bbox_weights[isnotnan],
                avg_factor=num_total_pos)

            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)

        if seg_preds is not None:
            loss_seg = self.loss_seg(seg_preds, gt_seg_list[0])
            loss_seg = torch.nan_to_num(loss_seg)

        return loss_cls, loss_bbox, loss_seg

    @force_fp32(apply_to=('preds_dicts'))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_segs,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = dict()
        if self.loss_cls is not None:
            assert gt_bboxes_ignore is None, \
                f'{self.__class__.__name__} only supports ' \
                f'for gt_bboxes_ignore setting to None.'

            all_cls_scores = preds_dicts['all_cls_scores']
            all_bbox_preds = preds_dicts['all_bbox_preds']
            enc_cls_scores = preds_dicts['enc_cls_scores']
            enc_bbox_preds = preds_dicts['enc_bbox_preds']
            all_seg_preds = preds_dicts['all_seg_preds']

            num_dec_layers = len(all_cls_scores) if all_cls_scores is not None else len(all_seg_preds)
            all_gt_bboxes_list = [None] * num_dec_layers
            all_gt_labels_list = [None] * num_dec_layers
            all_gt_bboxes_ignore_list = [None] * num_dec_layers
            all_gt_seg_list = [None] * num_dec_layers

            if all_cls_scores is not None:
                device = gt_labels_list[0].device
                if hasattr(gt_bboxes_list[0], 'mtv_targets'):
                    gt_bboxes_list = [
                        torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:], gt_bboxes.mtv_targets),
                                  dim=1).to(device) for gt_bboxes in gt_bboxes_list
                    ]
                else:
                    gt_bboxes_list = [
                        torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
                        for gt_bboxes in gt_bboxes_list
                    ]

                all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
                all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
                all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
            else:
                all_cls_scores = [None] * num_dec_layers
                all_bbox_preds = [None] * num_dec_layers

            if all_seg_preds is not None:
                all_gt_seg_list = [[gt_segs] for _ in range(num_dec_layers)]
            else:
                all_seg_preds = [None] * num_dec_layers

            losses_cls, losses_bbox, losses_seg = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds,
                                                              all_seg_preds, all_gt_bboxes_list, all_gt_labels_list,
                                                              all_gt_seg_list, all_gt_bboxes_ignore_list)

            # loss of proposal generated from encode feature map.
            if enc_cls_scores is not None:
                binary_labels_list = [torch.zeros_like(gt_labels_list[i]) for i in range(len(all_gt_labels_list))]
                enc_loss_cls, enc_losses_bbox = \
                    self.loss_single(enc_cls_scores, enc_bbox_preds,
                                    gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
                loss_dict['enc_loss_cls'] = enc_loss_cls
                loss_dict['enc_loss_bbox'] = enc_losses_bbox

            if losses_cls is not None:
                # loss from the last decoder layer
                loss_dict['loss_cls'] = losses_cls[-1]
                loss_dict['loss_bbox'] = losses_bbox[-1]

                # loss from other decoder layers
                num_dec_layer = 0
                for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
                    loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
                    loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
                    num_dec_layer += 1

            if losses_seg[0] is not None:
                # loss from the last decoder layer
                loss_dict['loss_seg'] = losses_seg[-1]

                # loss from other decoder layers
                num_dec_layer = 0
                for loss_seg_i in losses_seg[:-1]:
                    loss_dict[f'd{num_dec_layer}.loss_seg'] = loss_seg_i
                    num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts['all_bbox_preds'] = preds_dicts['all_bbox_preds'][..., :10]
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
