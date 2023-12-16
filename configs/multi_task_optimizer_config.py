optimizer_config = dict(
    type='MultiTaskOptimizerHook',
    grad_clip=[
        dict(max_norm=1, norm_type=2),
        dict(max_norm=10, norm_type=2)]
    )

pts_optimizer = dict(
    type='AdamW',
    lr=3e-5,
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
            # 'img_backbone':dict(lr_mult=1.),  # 初始学习率的基础上乘一个放缩倍数
            # 'img_neck':dict(lr_mult=1.),
            # 'img_bbox_head':dict(lr_mult=1.),
            # 'img_mask_branch':dict(lr_mult=1.),
            # 'img_mask_head':dict(lr_mult=1.),
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
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[28, 34])  # [60000, 80000]

lr_config = dict(
    policy='multi',  # class MultiLrUpdaterHook(LrUpdaterHook):
    img = img_lr_config,
    pts = pts_lr_config,
)