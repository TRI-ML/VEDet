_base_ = [
    '/workspace/mmlab/mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '/workspace/mmlab/mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs={'project': 'mmdet3d'},
            interval=10,
        )
    ])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian',
    'traffic_cone'
]
input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False)
bands, max_freq = 64, 8
num_views = 2
code_weights = [1.0] * 10 + [0.0] * 10 * num_views
code_weights[8] = 0.2
code_weights[9] = 0.2
virtual_weights = 0.2
for i in range(1, num_views + 1):
    code_weights[i * 10] = virtual_weights  # x
    code_weights[i * 10 + 1] = virtual_weights  # y
    code_weights[i * 10 + 4] = virtual_weights  # z
    code_weights[i * 10 + 6] = virtual_weights  # sin(yaw)
    code_weights[i * 10 + 7] = virtual_weights  # cos(yaw)
    code_weights[i * 10 + 2] = virtual_weights  # w
    code_weights[i * 10 + 3] = virtual_weights  # l
    code_weights[i * 10 + 5] = virtual_weights  # h
    code_weights[i * 10 + 8] = 0.2 * virtual_weights  # vx
    code_weights[i * 10 + 9] = 0.2 * virtual_weights  # vy
model = dict(
    type='VEDet',
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=(
            'stage4',
            'stage5',
        )),
    img_neck=dict(type='CPFPN', in_channels=[768, 1024], out_channels=256, num_outs=2),
    gt_depth_sup=False,  # use cache to supervise
    pts_bbox_head=dict(
        type='VEDetHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        position_range=point_cloud_range,
        reg_hidden_dims=[512, 512],
        code_size=(num_views + 1) * 10,
        code_weights=code_weights,
        reg_channels=10,
        num_decode_views=num_views,
        with_time=True,
        det_transformer=dict(
            type='VETransformer',
            det_decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='PETRMultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        input_ray_encoding=dict(
            type='FourierMLPEncoding',
            input_channels=10,
            hidden_dims=[int(1.5 * 10 * 2 * bands)],
            embed_dim=256,
            fourier_type='linear',
            fourier_channels=10 * 2 * bands,
            max_frequency=max_freq),
        output_det_encoding=dict(
            type='FourierMLPEncoding',
            input_channels=10,
            hidden_dims=[int(1.5 * 10 * 2 * bands)],
            embed_dim=256,
            fourier_type='linear',
            fourier_channels=10 * 2 * bands,
            max_frequency=max_freq),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost',
                              weight=0.0),  # Fake cost. This is just to make it compatible with DETR head. 
                align_with_loss=True,
                pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')
ida_aug_conf = {
    "resize_lim": (0.94, 1.25),
    "final_dim": (640, 1600),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
meta_keys = ('filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
             'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
             'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
             'intrinsics', 'extrinsics', 'scale_ratio', 'dec_extrinsics', 'timestamp')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=1,
        to_float32=True,
        pad_empty_sweeps=True,
        test_mode=False,
        time_range=3,
        sweep_range=[3, 27]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImageFull3D', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(type='ComputeMultiviewTargets', local_frame=True, visible_only=False, use_virtual=True, num_views=num_views),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'], meta_keys=meta_keys)
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadMultiViewImageFromMultiSweepsFiles',
        sweeps_num=1,
        to_float32=True,
        pad_empty_sweeps=True,
        time_range=3,
        sweep_range=[3, 27]),
    dict(type='ResizeCropFlipImageFull3D', data_aug_conf=ida_aug_conf, training=False),
    dict(type='ComputeMultiviewTargets', local_frame=True, visible_only=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img'], meta_keys=meta_keys)
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl',
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl',
        classes=class_names,
        modality=input_modality))

optimizer = dict(
    type='AdamW', lr=2e-4, paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }), weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 24
evaluation = dict(interval=2, pipeline=test_pipeline, metric=['bbox'])
checkpoint_config = dict(interval=24)
find_unused_parameters = False

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from = None
