# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',     # 线性增加
    warmup_iters=500,    # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=0.001,  # 起始学习率
    step=[16, 22])       # 在第16个和22个epoch降低学习率
runner = dict(type='EpochBasedRunner', max_epochs=24)
