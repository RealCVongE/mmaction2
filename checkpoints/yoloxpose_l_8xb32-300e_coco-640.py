auto_scale_lr = dict(base_batch_size=256)
backend_args = dict(backend='local')
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
codec = dict(
    input_size=(
        640,
        640,
    ), type='YOLOXPoseAnnotationProcessor')
custom_hooks = [
    dict(
        new_train_pipeline=[
            dict(type='LoadImage'),
            dict(
                bbox_keep_corner=False,
                clip_border=True,
                input_size=(
                    640,
                    640,
                ),
                pad_val=(
                    114,
                    114,
                    114,
                ),
                rotate_prob=0,
                scale_prob=0,
                scale_type='long',
                shift_prob=0,
                type='BottomupRandomAffine'),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip'),
            dict(
                by_box=True,
                by_kpt=True,
                keep_empty=False,
                type='FilterAnnotations'),
            dict(
                encoder=dict(
                    input_size=(
                        640,
                        640,
                    ),
                    type='YOLOXPoseAnnotationProcessor'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        num_last_epochs=20,
        priority=48,
        type='YOLOXPoseModeSwitchHook'),
    dict(priority=48, type='SyncNormHook'),
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]
data_mode = 'bottomup'
data_root = 'data/'
dataset_coco = dict(
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_mode='bottomup',
    data_prefix=dict(img='coco/train2017/'),
    data_root='data/',
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=[
        dict(backend_args=None, type='LoadImage'),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImage'),
            ],
            type='Mosaic'),
        dict(
            bbox_keep_corner=False,
            clip_border=True,
            distribution='uniform',
            input_size=(
                640,
                640,
            ),
            pad_val=114,
            rotate_factor=10,
            scale_factor=(
                0.75,
                1.0,
            ),
            shift_factor=0.1,
            transform_mode='perspective',
            type='BottomupRandomAffine'),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImage'),
            ],
            ratio_range=(
                0.8,
                1.6,
            ),
            type='YOLOXMixUp'),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip'),
        dict(
            by_box=True,
            by_kpt=True,
            keep_empty=False,
            type='FilterAnnotations'),
        dict(
            encoder=dict(
                input_size=(
                    640,
                    640,
                ), type='YOLOXPoseAnnotationProcessor'),
            type='GenerateTarget'),
        dict(type='PackPoseInputs'),
    ],
    type='CocoDataset')
deepen_factor = 1
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_size = (
    640,
    640,
)
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(type='Swish'),
        deepen_factor=1,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=(
            2,
            3,
            4,
        ),
        spp_kernal_sizes=(
            5,
            9,
            13,
        ),
        type='CSPDarknet',
        widen_factor=1),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=1,
                random_size_range=(
                    480,
                    800,
                ),
                size_divisor=32,
                type='BatchSyncRandomResize'),
        ],
        mean=[
            0,
            0,
            0,
        ],
        pad_size_divisor=32,
        std=[
            1,
            1,
            1,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        assigner=dict(dynamic_k_indicator='oks', type='SimOTAAssigner'),
        featmap_strides=(
            8,
            16,
            32,
        ),
        head_module_cfg=dict(
            act_cfg=dict(type='Swish'),
            feat_channels=256,
            in_channels=256,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=1,
            stacked_convs=2,
            widen_factor=1),
        loss_bbox=dict(
            eps=1e-16,
            loss_weight=5.0,
            mode='square',
            reduction='sum',
            type='IoULoss'),
        loss_bbox_aux=dict(loss_weight=1.0, reduction='sum', type='L1Loss'),
        loss_cls=dict(loss_weight=1.0, reduction='sum', type='BCELoss'),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='sum',
            type='BCELoss',
            use_target_weight=True),
        loss_oks=dict(
            loss_weight=30.0,
            metainfo='configs/_base_/datasets/coco.py',
            norm_target_weight=True,
            reduction='none',
            type='OKSLoss'),
        loss_vis=dict(
            loss_weight=1.0,
            reduction='mean',
            type='BCELoss',
            use_target_weight=True),
        num_keypoints=17,
        overlaps_power=0.5,
        prior_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        type='YOLOXPoseHead'),
    init_cfg=dict(
        a=2.23606797749979,
        distribution='uniform',
        layer='Conv2d',
        mode='fan_in',
        nonlinearity='leaky_relu',
        type='Kaiming'),
    neck=dict(
        act_cfg=dict(type='Swish'),
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=256,
        type='YOLOXPAFPN',
        upsample_cfg=dict(mode='nearest', scale_factor=2),
        use_depthwise=False),
    test_cfg=dict(nms_thr=0.65, score_thr=0.01),
    type='BottomupPoseEstimator')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        type='QuadraticWarmupLR'),
    dict(
        T_max=280,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=280,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
    dict(begin=280, by_epoch=True, end=300, factor=1, type='ConstantLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='coco/val2017/'),
        data_root='data/',
        pipeline=[
            dict(type='LoadImage'),
            dict(
                input_size=(
                    640,
                    640,
                ),
                pad_val=(
                    114,
                    114,
                    114,
                ),
                type='BottomupResize'),
            dict(
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                ),
                type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='bbox',
    type='CocoMetric')
train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=10)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='coco/annotations/person_keypoints_train2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='coco/train2017/'),
        data_root='data/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImage'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImage'),
                ],
                type='Mosaic'),
            dict(
                bbox_keep_corner=False,
                clip_border=True,
                distribution='uniform',
                input_size=(
                    640,
                    640,
                ),
                pad_val=114,
                rotate_factor=10,
                scale_factor=(
                    0.75,
                    1.0,
                ),
                shift_factor=0.1,
                transform_mode='perspective',
                type='BottomupRandomAffine'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImage'),
                ],
                ratio_range=(
                    0.8,
                    1.6,
                ),
                type='YOLOXMixUp'),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip'),
            dict(
                by_box=True,
                by_kpt=True,
                keep_empty=False,
                type='FilterAnnotations'),
            dict(
                encoder=dict(
                    input_size=(
                        640,
                        640,
                    ),
                    type='YOLOXPoseAnnotationProcessor'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline_stage1 = [
    dict(backend_args=None, type='LoadImage'),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImage'),
        ],
        type='Mosaic'),
    dict(
        bbox_keep_corner=False,
        clip_border=True,
        distribution='uniform',
        input_size=(
            640,
            640,
        ),
        pad_val=114,
        rotate_factor=10,
        scale_factor=(
            0.75,
            1.0,
        ),
        shift_factor=0.1,
        transform_mode='perspective',
        type='BottomupRandomAffine'),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImage'),
        ],
        ratio_range=(
            0.8,
            1.6,
        ),
        type='YOLOXMixUp'),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(by_box=True, by_kpt=True, keep_empty=False, type='FilterAnnotations'),
    dict(
        encoder=dict(
            input_size=(
                640,
                640,
            ), type='YOLOXPoseAnnotationProcessor'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        bbox_keep_corner=False,
        clip_border=True,
        input_size=(
            640,
            640,
        ),
        pad_val=(
            114,
            114,
            114,
        ),
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        shift_prob=0,
        type='BottomupRandomAffine'),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(by_box=True, by_kpt=True, keep_empty=False, type='FilterAnnotations'),
    dict(
        encoder=dict(
            input_size=(
                640,
                640,
            ), type='YOLOXPoseAnnotationProcessor'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='coco/annotations/person_keypoints_val2017.json',
        data_mode='bottomup',
        data_prefix=dict(img='coco/val2017/'),
        data_root='data/',
        pipeline=[
            dict(type='LoadImage'),
            dict(
                input_size=(
                    640,
                    640,
                ),
                pad_val=(
                    114,
                    114,
                    114,
                ),
                type='BottomupResize'),
            dict(
                meta_keys=(
                    'id',
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'input_size',
                    'input_center',
                    'input_scale',
                ),
                type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/person_keypoints_val2017.json',
    nms_mode='none',
    score_mode='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        input_size=(
            640,
            640,
        ),
        pad_val=(
            114,
            114,
            114,
        ),
        type='BottomupResize'),
    dict(
        meta_keys=(
            'id',
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'input_size',
            'input_center',
            'input_scale',
        ),
        type='PackPoseInputs'),
]
# vis_backends = [
#     dict(type='LocalVisBackend'),
# ]
# visualizer = dict(
#     name='visualizer',
#     type='PoseLocalVisualizer',
#     vis_backends=[
#         dict(type='LocalVisBackend'),
#     ])
widen_factor = 1
# visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmpose.PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')