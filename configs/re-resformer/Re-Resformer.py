auto_scale_lr = dict(base_batch_size=256)
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

dataset_type = 'Re'

default_hooks = dict(
    checkpoint=dict(interval=20, type='CheckpointHook'),
    logger=dict(interval=1000, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'),
    custom_hooks=dict(type='ExcelLoggerHook',filename='roughness_predictions.xlsx',
                      out_dir='E:/classiyf-module/Re-Resformer/tools/work_dirs'  )
)

default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

launcher = 'none'
load_from = None
log_level = 'INFO'

model = dict(
    backbone=dict(
        depth=18,
        num_stages=4,
        out_indices=(0,1,2,3 ),
        style='pytorch',
        type='ReResformer'
    ),
    head=dict(
        in_channels=512,
        roughness_loss=dict(type='DynamicLoss', loss_weight=1),
        type='KaRHead',
        depth=18
    ),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier'
)


optim_wrapper = dict(
    type='OptimWrapper',  
    optimizer=dict(
        type='Adam',  
        lr=0.001,             
        betas=(0.9, 0.999),   
        eps=1e-8,             
        weight_decay=0.0001   

    ),
    paramwise_cfg=dict(  
        custom_keys={
            'backbone': dict(lr_mult=0.1),  
        }
    )
)

param_scheduler = dict(
    by_epoch=True,
    gamma=0.1,
    milestones=[30, 60, 90],
    type='MultiStepLR'
)

randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()

test_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='E:/classiyf-module/Re-Resformer/dataset/2Q235/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='Resize', interpolation='lanczos'), 
            dict(type='PackInputs')
        ],
        split='val',
        type='Re'
    ),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

test_evaluator = dict(type='MAPE')

train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=1)

train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='E:/classiyf-module/Re-Resformer/dataset/2Q235/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='Resize', interpolation='lanczos'), 
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs')
        ],
        split='train',
        type='Re'
    ),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='E:/classiyf-module/Re-Resformer/dataset/2Q235/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='Resize', interpolation='lanczos'),  
            dict(type='PackInputs')
        ],
        split='val',
        type='Re'
    ),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)

val_evaluator = dict(type='MAPE')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)

work_dir = './work_dirs/Re-Resformer'
