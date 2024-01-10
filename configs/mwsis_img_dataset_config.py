dataset_type = 'MwsisWaymoDataset'

pts_panseg_tainval_pth = "data/waymo/kitti_format/training/lidar_panseg_label"
img_panseg_tainval_pth = "data/waymo/kitti_format/training"  # data/waymo/kitti_format/training/img_panseg_label_0
pts_trainval_pth = "data/waymo/kitti_format/training/velodyne"
img_trainval_pth = "data/waymo/kitti_format/training"  # data/waymo/kitti_format/training/image_0

img_tain_info_index = "data/waymo/kitti_format/training/img_panseg_index_train.pkl"
img_val_info_index = "data/waymo/kitti_format/training/img_panseg_index_val.pkl"
frame_infos = "data/waymo/kitti_format/training/frame_infos"

file_client_args = dict(backend='disk')
pseudo_labels_client_args = dict(backend='disk')

# switch eval mode
eval_2d = True
eval_3d = False
eval_3d_mask = False

eval_keys=['points','img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d']

if eval_2d:
    eval_keys=['points','img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d']

elif eval_3d or eval_3d_mask:
    eval_keys=['points', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
        'pts_semantic_mask', 'pts_instance_mask', 'ccl_labels']
    ccl_mode = 'mwsis_ccl'

class_names = ['Car', 'Pedestrian', 'Cyclist']  # ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-80, -80, -2, 80, 80, 4]
input_modality = dict(use_lidar=True, use_camera=True)
gt_box_type = 2  # 1 use 3D Box 2 use 2D Box
semantic_class_names = class_names
# the image loaded is rgb so not convert to rgb, and mean std need exchange dimension
img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadImages',
        file_client_args=file_client_args),
    dict(
        type='LoadAnnos3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=False,
        with_mask_3d=False,
        file_client_args=file_client_args),
    # not complete this part of the code
    # dict(
    #     type='LoadPseudoLabel',
    #     coord_type='LIDAR',
    #     file_client_args=pseudo_labels_client_args,
    #     ccl_mode=ccl_mode,
    #     load_ccl=True),
    # not complete this part of the code
    dict(
        type='LoadAnnos',
        with_bbox=True,
        with_label=True,
        with_mask=True,
        with_seg=True,
        file_client_args=file_client_args),
    # dict(
    #     type='LoadAnnosLwsis',
    #     file_client_args=pseudo_labels_client_args,
        # prefix='training'),
    dict(
        type='FilterAndReLabel',
        filter_calss_name=class_names,
        with_mask=True,
        with_seg=True,
        with_mask_3d=False,
        with_seg_3d=False,
        filter_class_name=class_names),
    # Image Augment
    dict(
        type='ResizeMultiViewImage',
        # Target size (w, h)
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)], # default
        # img_scale=[(480, 320),(480, 320),(480, 320),(480, 320),(480, 320)],  # debug
        multiscale_mode='value',
        keep_ratio=True),
    dict(
        type='RandomFlipMultiImage',
        flip_ratio=0.5,
        direction='horizontal'),
    dict(type='NormalizeMultiViewImage', **img_norm_cfg),
    dict(
        type='PadMultiViewImage',
        size=[(800, 1344), (800, 1344), (800, 1344), (800, 1344), (800, 1344)],  # default
        # size=[(320, 480),(320, 480),(320, 480),(320, 480),(320, 480)],  # debug
        ),
    # Points Augment
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),  # also filter pseudo labels if not None
    dict(
        type='FilterPoints',
        coord_type='LIDAR',
        num_classes=len(class_names)),  # get in img and top lidar points
    dict(
        type='LoadHistoryLabel',
        coord_type='LIDAR',
        use_history_labels=True,
        history_nums=4,
        num_classes=len(class_names)),  # get in img and top lidar points    
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        # translation_std=[0.2, 0.2, 0.2]
        ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='FilterPointByMultiImage',
        coord_type='LIDAR',
        use_run_seg=True,  # is use ring segment
        use_pseudo_label=True,  # is use local saved ring segments and pseudo lables generated by the SPG
        use_augment=True,
        point_cloud_range=point_cloud_range,
        ),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', 
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_semantic_seg', 'gt_masks',
              'points', 
              #'gt_bboxes_3d', 'gt_labels_3d',
              #'pseudo_labels',
              'history_labels',
              #'pts_semantic_mask','pts_instance_mask'
              ],
        meta_keys=['filename', 'img_shape', 'ori_shape', 'pad_shape',
            'scale', 'scale_factor', 'keep_ratio', 'lidar2img',
            'sample_idx', 'img_info','ann_info', 'pts_info',
            'img_norm_cfg', 'pad_fixed_size', 'pad_size_divisor',
            'sample_img_id', 'img_sample','flip', 'flip_direction',
            'transformation_3d_flow', 'crop_shape'
            ],
    )
]

test_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadAnnos3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_seg_3d=eval_3d|eval_3d_mask,
        with_mask_3d=eval_3d|eval_3d_mask,
        file_client_args=file_client_args),
    dict(
        type='LoadImages',
        file_client_args=file_client_args,),
    # dict(
    #     type='LoadAnnosLwsis',
    #     file_client_args=pseudo_labels_client_args,
    #     prefix='validation'),  # visualization
    dict(
        type='LoadAnnos',
        with_bbox=True,
        with_label=True,
        with_mask=False,
        with_seg=False,
        file_client_args=file_client_args),
    dict(
        type='LoadValRingSegLabel',
        coord_type='LIDAR',
        file_client_args=pseudo_labels_client_args,
        ccl_mode='mwsis_ccl',
        load_ccl=True),
    dict(
        type='FilterAndReLabel',
        filter_calss_name=class_names,
        with_mask=False,
        with_seg=False,
        with_mask_3d=eval_3d|eval_3d_mask,
        with_seg_3d=eval_3d|eval_3d_mask,
        filter_class_name=class_names),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='ResizeMultiViewImage',
                # Target size (w, h)
                img_scale=[(1333, 800)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='NormalizeMultiViewImage', **img_norm_cfg),
            dict(
                type='PadMultiViewImage',
                size=[(800, 1344),(800, 1344),(800, 1344),(800, 1344),(800, 1344)]),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='FilterPoints',
                coord_type='LIDAR',
                num_classes=len(class_names)),  # get in img and top lidar points
            dict(type='FilterPointByMultiImage', 
                coord_type='LIDAR', 
                training=False),
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(
                type='Collect3D', 
                keys=eval_keys)
        ])
]

data = dict(
    samples_per_gpu=2,  # bsz
    workers_per_gpu=4,  # data load process
    train=dict(
        type=dataset_type,
        info_index_path=img_tain_info_index,
        info_path=frame_infos,
        classes=class_names,
        modality=input_modality,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset
        box_type_3d='LiDAR',
        test_mode=False,
        # load one frame every five frames
        load_interval=1,
        datainfo_client_args=pts_trainval_pth,
        # 3d segmentation
        load_semseg=eval_3d,
        semseg_classes=class_names,
        semseg_info_path='training/infos',  # semantic and instance same path
        # 2d segmentation
        load_img=True,
        load_panseg=eval_2d,
        panseg_classes = class_names,
        panseg_info_path='training/infos',  # semantic and instance same path
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        info_index_path=img_tain_info_index,
        info_path=frame_infos,
        datainfo_client_args=pts_trainval_pth,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        # 3d segmentation
        load_semseg=eval_3d_mask,
        semseg_classes=class_names,
        # semseg_info_path='validation/infos',
        # 2d segmentation
        load_img=True,
        load_panseg=eval_2d,
        panseg_classes = class_names,
        # panseg_info_path='validation/infos'
        ),
    test=dict(
        type=dataset_type,
        info_index_path=img_tain_info_index,
        info_path=frame_infos,
        datainfo_client_args=pts_trainval_pth,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        # 3d segmentation
        load_semseg=eval_3d_mask,
        semseg_classes=class_names,
        # semseg_info_path='validation/infos',
        # 2d segmentation
        load_img=True,
        load_panseg=eval_2d,
        panseg_classes = class_names,
        # panseg_info_path='validation/infos'
        ),

        test_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=1))
        

evaluation = dict(interval=12,
                  pipeline=test_pipeline,
                  pc_range=point_cloud_range,
                  eval_2d=eval_2d,
                  eval_3d=eval_3d_mask,
                  eval_3dmask=eval_3d_mask,
                  seg_format_worker=0)
