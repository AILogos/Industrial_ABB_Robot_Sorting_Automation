# configs/custom/mask_rcnn_sam2.py

_base_ = "../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"

# Dataset config
dataset_type = 'CocoDataset'
data_root = 'ClothesObjectsSegmentation.v1-quick_test.coco-segmentation/'
classes = ('objects', 'bag', 'bra', 'clothe', 'shoe')

metainfo = dict(classes=classes)

data = dict(
    train=dict(
        type=dataset_type,
        metainfo=metainfo,
        ann_file=data_root + 'train/_annotations.coco.json',
        img_prefix=data_root + 'train/'
    ),
    val=dict(
        type=dataset_type,
        metainfo=metainfo,
        ann_file=data_root + 'valid/_annotations.coco.json',
        img_prefix=data_root + 'valid/'
    ),
    test=dict(
        type=dataset_type,
        metainfo=metainfo,
        ann_file=data_root + 'test/_annotations.coco.json',
        img_prefix=data_root + 'test/'
    )
)

# Model config
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes))
    )
)

# Runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict()
test_cfg = dict()

deploy = False

# Logging
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(interval=1, max_keep_ckpts=3)
)

# Optimizer for fine-tuning
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.001,  # Lower learning rate for fine-tuning
        momentum=0.9,
        weight_decay=0.0001
    )
)

# Evaluation metrics
eval_cfg = dict(
    type='EvalHook',
    interval=1,
    metric=['bbox', 'segm'],
    save_best='auto'
)

# Work dir
work_dir = './work_dirs/mask_rcnn_sam2'
# configs/custom/mask_rcnn_sam2.py

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mask_rcnn/mask-rcnn_r50_fpn_1x_coco/mask-rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
