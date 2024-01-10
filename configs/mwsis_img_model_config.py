_base_ = [
    './mwsis_img_dataset_config.py',
    './default_runtime.py',
]

seg_voxel_size = (0.25, 0.25, 0.2)
point_cloud_range = [-80, -80, -2, 80, 80, 4]
class_names = ['Car', 'Pedestrian', 'Cyclist']
semantic_class_names = class_names
num_classes = len(class_names)
seg_score_thresh = (0.3, 0.25, 0.25) # , 0.25
gt_box_type = 2  # 1 is 3d,2 is 2d

pts_segmentor = dict(
    type='VoteSegmentor',
    need_full_seg=False,

    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,  # -1 is use dynamic voxel
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    middle_encoder=dict(
        type='PseudoMiddleEncoderForSpconvFSD',
    ),

    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, 640, 640],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)), # decoder paddings seem useless in SubMConv
    ),

    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=67,
        hidden_dims=[128, 128],
        num_classes=num_classes,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='naiveSyncBN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(  # detect
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=100.0),
        loss_segment=dict(  # semantic segment
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=100.0),
        loss_lova=dict(
            type='LovaszLoss_',  # 对mean IOU loss进行的优化
            per_image=False,
            reduction='mean',
            loss_weight=1.0),
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),
        need_full_seg=False,
        num_classes_full=len(semantic_class_names),
        ignore_illegal_label=True,
        # segment_range=[-50, 50],
    ),

    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for training log
        class_names=class_names, # for training log
        centroid_offset=False,
    ),
)

use_ema=True
geometry_loss_enabled=False  # ignore
use_weight_loss=False        # discard
use_refine_pseudo_mask=True
kd_loss_weight=0.5
kd_loss_weight_3d=2
pairwise_geo_thresh=0.2
geometry_loss_weight=0.2
gt_box_type=2  # 1 is 3d,2 is 2d
pseudo_loss_iters = 17772 # pseudo bsz=6
start_kd_loss_iters = 20000 # 21518  # 41468 20000 is used for mask3d
use_his_labels_iters = 35544
pairwise_warmup = 8886
pseudo_loss_weight = 1.0

# model settings
model = dict(
    type='MWSIS',
    with_pts_branch=True,
    with_img_branch=True,
    gt_box_type=gt_box_type,
    use_2d_mask=True,   # 2d to 3d
    use_ema=use_ema,    # ema
    run_seg=False,      # run seg
    use_dynamic=False,  # dynamic
    use_refine_pseudo_mask=use_refine_pseudo_mask, # pseudo label
    use_weight_loss=use_weight_loss,
    kd_loss_weight=kd_loss_weight,
    kd_loss_weight_3d=kd_loss_weight_3d,
    pseudo_loss_weight=pseudo_loss_weight,
    num_classes=num_classes,
    warmup_iters=pseudo_loss_iters,
    start_kd_loss_iters=start_kd_loss_iters,
    use_his_labels_iters=use_his_labels_iters,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),  # Sync
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    img_bbox_head=dict(
        type='CondInstBoxHead',
        num_classes=num_classes,
        in_channels=256,
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    img_mask_branch=dict(
        type='CondInstMaskBranch',
        in_channels=256,
        in_indices=[0, 1, 2],
        strides=[8, 16, 32],
        branch_convs=4,
        branch_channels=128,
        branch_out_channels=16),
    img_mask_head=dict(
        type='CondInstMaskHead',
        in_channels=16,
        in_stride=8,
        out_stride=4,
        dynamic_convs=3,
        dynamic_channels=8,
        disable_rel_coors=False,
        bbox_head_channels=256,
        sizes_of_interest=[64, 128, 256, 512, 1024],
        max_proposals=-1,
        topk_per_img=64,
        boxinst_enabled=True,
        bottom_pixels_removed=10,
        pairwise_size=3,  # 3*3大小 9-1=8
        pairwise_dilation=2,
        pairwise_color_thresh=0.3,
        pairwise_warmup=pairwise_warmup,  # 10000
        start_kd_loss_iters=start_kd_loss_iters,
        points_enabled=True,
        use_enlarge_bbox=False,
        geometry_loss_enabled=geometry_loss_enabled,
        use_weight_loss=use_weight_loss,
        kd_loss_weight=kd_loss_weight,
        pairwise_geo_thresh=pairwise_geo_thresh,
        geometry_loss_weight=geometry_loss_weight),

    pts_segmentor = pts_segmentor,

    # training and testing settings
    train_cfg=dict(
        img=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
    ),

    test_cfg=dict(
        img=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        ),
    ),
)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
    
custom_hooks = []

# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=90000)
runner = dict(type='EpochBasedRunner', max_epochs=24)  # NOTE
workflow = [('train', 1)] # ('val', 1)

checkpoint_config = dict(
    interval=1,  # interval = 1 if by_epoch else 5000
    by_epoch=True,
    max_keep_ckpts=20)

custom_imports = dict(
    imports=['mwsis.mwsis_pulgin'],
    allow_failed_imports=False)

# multi-task optimizer config
optimizer_config = dict(
    type='MultiTaskOptimizerHook',
    grad_clip=[
        None,
        dict(max_norm=10, norm_type=2)]
    )

pts_optimizer = dict(
    type='AdamW',
    lr=1e-5,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    )

img_optimizer = dict(
    type='SGD', 
    lr=0.01, 
    momentum=0.9,
    weight_decay=0.0001)

optimizer = dict(
        type = {'img':'SGD', 'pts':'AdamW'},  # img, pts
        constructor='MultiTaskOptimizerConstructor',
        multi_task = True,
        img = img_optimizer,
        pts = pts_optimizer,
        paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            }),
)

pts_lr_config = dict(
    policy='cyclic',  # class CyclicLrUpdaterHook(LrUpdaterHook):
    by_epoch=False,    # 若对齐，则为False
    target_ratio=(100, 1e-3), # 相对于初始LR 最大LR和最小值LR的值，LR_max=LR*100 (100, 1e-3),
    cyclic_times=1,  # 训练期间循环次数
    step_ratio_up=0.1,  # 整个LR循环中每次学习率变化比例，基于LR,到达最高点时是总iters的0.1
)

img_lr_config = dict(
    policy='step',
    warmup='linear',
    by_epoch=True,
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[25, 25])  # [60000, 80000] not fall

lr_config = dict(
    policy='multi',  # class MultiLrUpdaterHook(LrUpdaterHook):
    img = img_lr_config,
    pts = pts_lr_config,
)

# only img config
# optimizer = dict(
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(
#             custom_keys={
#                 'norm': dict(decay_mult=0.),
#                 }),
#     )
# optimizer_config = dict(grad_clip=None)
# # optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# # learning policy gamma(default:0.1)
# lr_config = dict(
#     policy='step',
#     by_epoch=False,
#     warmup='linear',
#     warmup_iters=1000,  # 1000 iters base_lr = 0.01
#     warmup_ratio=0.001,# = base_lr * warmup_ratio
#     step=[60000, 80000])

# only pts config
# optimizer = dict(
#     type='AdamW',
#     lr=3e-5,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#             custom_keys={
#                 'norm': dict(decay_mult=0.),
#                 'img_backbone':dict(lr_mult=0.3, decay_mult=0.2),  # 初始学习率的基础上乘一个放缩倍数
#                 'img_neck':dict(lr_mult=0.3, decay_mult=0.2),
#                 'img_bbox_head':dict(lr_mult=0.3, decay_mult=0.2),
#                 'img_mask_branch':dict(lr_mult=0.3, decay_mult=0.2),
#                 'img_mask_head':dict(lr_mult=0.3, decay_mult=0.2),
#                 }),
#     )

# lr_config = dict(
#     policy='cyclic',  # class CyclicLrUpdaterHook(LrUpdaterHook):
#     by_epoch=False,    # 若对齐，则为False
#     target_ratio=(100, 1e-3), # 相对于初始LR 最大LR和最小值LR的值，LR_max=LR*100 (100, 1e-3),
#     cyclic_times=1,  # 训练期间循环次数
#     step_ratio_up=0.1,  # 整个LR循环中每次学习率变化比例，基于LR,到达最高点时是总iters的0.1
# )