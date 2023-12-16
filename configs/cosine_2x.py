lr=1e-5  # 1e-5
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    )
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',  # class CyclicLrUpdaterHook(LrUpdaterHook):
    target_ratio=(100, 1e-3), # 相对于初始LR 最大LR和最小值LR的值，LR_max=LR*100 (100, 1e-3),
    cyclic_times=1,  # 训练期间循环次数
    step_ratio_up=0.1,  # 整个LR循环中每次学习率变化比例，基于LR,到达最高点时是总iters的0.1
)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=24)